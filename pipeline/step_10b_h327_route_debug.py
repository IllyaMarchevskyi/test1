"""
10b H327 route debug.

Діагностичний крок для перевірки якості розкладів локальної підмережі
цільового закладу з RL-конфіга. Допомагає зрозуміти, чи діряві або
неповні розклади спотворюють навчання.
"""


def run() -> None:
    from config_loader import cfg
    from pathlib import Path

    import numpy as np
    import pandas as pd

    PROCESSED_DIR = Path("./data/processed")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    TARGET_FACILITY_ID = str(cfg.get("rl", {}).get("target_facility_id", "")).strip() or "H327"
    CATCHMENT_BUILDINGS = PROCESSED_DIR / "catchment_buildings_baseline.parquet"
    EASYWAY_ROUTES = Path("../gtfs_static/easyway_routes.csv")
    EASYWAY_METRO = Path("../gtfs_static/easyway_metro.csv")

    SUMMARY_CSV = PROCESSED_DIR / "h327_route_schedule_debug.csv"
    STOPS_CSV = PROCESSED_DIR / "h327_route_schedule_stops_debug.csv"

    required = [CATCHMENT_BUILDINGS, EASYWAY_ROUTES]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Відсутні входи для 10b_h327_route_debug: {missing}")

    print(f"10b_debug: аналізуємо локальні маршрути для {TARGET_FACILITY_ID}...")

    catchment = pd.read_parquet(
        CATCHMENT_BUILDINGS,
        columns=["facility_id", "peak_mode", "peak_route_id"],
    )
    catchment["facility_id"] = catchment["facility_id"].astype(str)
    catchment["peak_mode"] = catchment["peak_mode"].astype(str)
    catchment["peak_route_id"] = catchment["peak_route_id"].astype(str)

    facility_records = catchment[catchment["facility_id"] == TARGET_FACILITY_ID].copy()
    if facility_records.empty:
        raise ValueError(f"10b_debug: не знайдено записів catchment для {TARGET_FACILITY_ID}.")

    local_routes = sorted(
        facility_records[
            facility_records["peak_mode"].eq("transit")
            & facility_records["peak_route_id"].ne("nan")
            & facility_records["peak_route_id"].ne("")
        ]["peak_route_id"].unique().tolist()
    )
    if not local_routes:
        raise ValueError(f"10b_debug: для {TARGET_FACILITY_ID} не знайдено transit-маршрутів.")

    print(f"10b_debug: локальних маршрутів = {len(local_routes)}")

    def parse_times(raw: str) -> list[int]:
        times: list[int] = []
        for token in str(raw).split(","):
            token = token.strip()
            if not token or token == r"\N":
                continue
            hh, mm, ss = token.split(":")
            times.append(int(hh) * 3600 + int(mm) * 60 + int(ss))
        return sorted(times)

    def sec_to_hhmmss(value: float | int | None) -> str | None:
        if value is None or pd.isna(value):
            return None
        total = int(value)
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    easyway_parts = [pd.read_csv(EASYWAY_ROUTES)]
    if EASYWAY_METRO.exists():
        easyway_parts.append(pd.read_csv(EASYWAY_METRO))
    easyway = pd.concat(easyway_parts, ignore_index=True)
    easyway["route_id"] = easyway["route_id"].astype(str)
    easyway["stop_id"] = easyway["stop_id"].astype(str)
    easyway["direction"] = easyway["direction"].astype(str)
    easyway["calendar"] = easyway["calendar"].astype(str)
    easyway["transport"] = easyway["transport"].astype(str)
    easyway["route"] = easyway["route"].astype(str)
    easyway["stop_name"] = easyway["stop_name"].astype(str)
    easyway["index"] = pd.to_numeric(easyway["index"], errors="coerce")
    easyway["times"] = easyway["schedules"].apply(parse_times)
    easyway["n_departures"] = easyway["times"].apply(len)

    route_freq_base = (
        easyway[easyway["calendar"].str.strip().str.lower() == "weekdays"]
        .groupby("route_id", as_index=False)
        .agg(
            transport=("transport", "first"),
            total_departures=("n_departures", "sum"),
        )
        .reset_index(drop=True)
    )
    route_freq_base["current_freq"] = route_freq_base["total_departures"] / 11.0
    route_freq_base["rl_initial_freq"] = 6.0
    for _, sub_idx in route_freq_base.groupby("transport").groups.items():
        raw = np.log1p(route_freq_base.loc[sub_idx, "current_freq"].astype(float))
        min_raw = float(raw.min())
        max_raw = float(raw.max())
        if max_raw > min_raw:
            scaled = 1.0 + ((raw - min_raw) / (max_raw - min_raw) * 11.0)
        else:
            scaled = pd.Series(6.0, index=raw.index)
        route_freq_base.loc[sub_idx, "rl_initial_freq"] = scaled.round(2)
    rl_freq_map = dict(zip(route_freq_base["route_id"], route_freq_base["rl_initial_freq"]))

    route_df = easyway[easyway["route_id"].isin(local_routes)].copy()
    if route_df.empty:
        raise ValueError("10b_debug: локальні маршрути не знайдені у файлах easyway.")

    stop_rows: list[dict] = []
    summary_rows: list[dict] = []

    for route_id, route_group in route_df.groupby("route_id"):
        route_transport = str(route_group["transport"].iloc[0])
        route_name = str(route_group["route"].iloc[0])
        per_direction = route_group.groupby(["direction", "calendar"], dropna=False)

        for (direction, calendar), group in per_direction:
            if str(calendar).strip().lower() != "weekdays":
                continue
            group = group.sort_values("index").copy()
            if group.empty:
                continue

            min_idx = int(group["index"].min()) if pd.notna(group["index"]).any() else None
            max_idx = int(group["index"].max()) if pd.notna(group["index"]).any() else None
            observed_idx = (
                sorted(group["index"].dropna().astype(int).unique().tolist())
                if pd.notna(group["index"]).any()
                else []
            )
            expected_span = (max_idx - min_idx + 1) if (min_idx is not None and max_idx is not None) else len(observed_idx)
            missing_index_count = max(0, expected_span - len(observed_idx))
            coverage_ratio = (len(observed_idx) / expected_span) if expected_span > 0 else 1.0

            dep_counts = group["n_departures"].to_numpy(dtype=int)
            median_dep = float(np.median(dep_counts)) if len(dep_counts) else 0.0
            min_dep = int(dep_counts.min()) if len(dep_counts) else 0
            max_dep = int(dep_counts.max()) if len(dep_counts) else 0
            dep_range = max_dep - min_dep

            all_times = [t for seq in group["times"] for t in seq]
            first_time = min(all_times) if all_times else None
            last_time = max(all_times) if all_times else None

            first_stop_row = group.iloc[0]
            last_stop_row = group.iloc[-1]

            issue_flags: list[str] = []
            if missing_index_count > 0:
                issue_flags.append(f"missing_index:{missing_index_count}")
            if coverage_ratio < 0.9:
                issue_flags.append(f"coverage:{coverage_ratio:.2f}")
            if dep_range >= 3:
                issue_flags.append(f"dep_range:{dep_range}")
            if median_dep > 0 and min_dep < median_dep * 0.5:
                issue_flags.append("low_departures_stop")

            for row in group.itertuples(index=False):
                stop_issue_flags: list[str] = []
                n_dep = int(row.n_departures)
                if median_dep > 0 and n_dep < median_dep * 0.5:
                    stop_issue_flags.append("below_50pct_median")
                if n_dep == 0:
                    stop_issue_flags.append("zero_departures")

                times = list(row.times)
                stop_rows.append(
                    {
                        "facility_id": TARGET_FACILITY_ID,
                        "route_id": route_id,
                        "transport": route_transport,
                        "route": route_name,
                        "direction": direction,
                        "calendar": calendar,
                        "stop_id": str(row.stop_id),
                        "stop_name": str(row.stop_name),
                        "index": int(row.index) if pd.notna(row.index) else None,
                        "n_departures": n_dep,
                        "first_time": sec_to_hhmmss(min(times) if times else None),
                        "last_time": sec_to_hhmmss(max(times) if times else None),
                        "issue_flags": ",".join(stop_issue_flags),
                    }
                )

            summary_rows.append(
                {
                    "facility_id": TARGET_FACILITY_ID,
                    "route_id": route_id,
                    "transport": route_transport,
                    "route": route_name,
                    "direction": direction,
                    "calendar": calendar,
                    "n_stops": int(group["stop_id"].nunique()),
                    "total_departures": int(group["n_departures"].sum()),
                    "current_freq": float(group["n_departures"].sum() / 11.0),
                    "rl_initial_freq": float(rl_freq_map.get(route_id, 6.0)),
                    "first_stop_name": str(first_stop_row["stop_name"]),
                    "last_stop_name": str(last_stop_row["stop_name"]),
                    "first_time": sec_to_hhmmss(first_time),
                    "last_time": sec_to_hhmmss(last_time),
                    "min_departures_per_stop": min_dep,
                    "max_departures_per_stop": max_dep,
                    "median_departures_per_stop": median_dep,
                    "departure_range": dep_range,
                    "missing_index_count": missing_index_count,
                    "coverage_ratio": round(float(coverage_ratio), 4),
                    "issue_flags": ",".join(issue_flags),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["route_id", "direction", "calendar"]
    ).reset_index(drop=True)
    stops_df = pd.DataFrame(stop_rows).sort_values(
        ["route_id", "direction", "calendar", "index", "stop_name"]
    ).reset_index(drop=True)

    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    stops_df.to_csv(STOPS_CSV, index=False, encoding="utf-8")

    print(f"10b_debug: summary -> {SUMMARY_CSV}")
    print(f"10b_debug: stops   -> {STOPS_CSV}")
    print("10b_debug: weekdays current_freq -> rl_initial_freq:")
    for row in summary_df.itertuples(index=False):
        print(
            f"  {row.route_id} ({row.transport} {row.route}, {row.direction}, {row.calendar}) | "
            f"current_freq={row.current_freq:.2f} -> rl_initial_freq={float(row.rl_initial_freq):.2f}"
        )

    suspicious = summary_df[summary_df["issue_flags"].astype(str) != ""].copy()
    print(f"10b_debug: підозрілих route-direction груп = {len(suspicious)} з {len(summary_df)}")
    if not suspicious.empty:
        print("10b_debug: підозрілі маршрути:")
        for row in suspicious.itertuples(index=False):
            print(
                f"  {row.route_id} ({row.transport} {row.route}, {row.direction}, {row.calendar}) | "
                f"stops={row.n_stops}, departures={row.total_departures}, "
                f"freq={row.current_freq:.2f}, first={row.first_stop_name}, last={row.last_stop_name}, "
                f"time={row.first_time}-{row.last_time}, issues={row.issue_flags}"
            )
    else:
        print("10b_debug: явних проблем у локальних маршрутах не знайдено.")


if __name__ == "__main__":
    run()
