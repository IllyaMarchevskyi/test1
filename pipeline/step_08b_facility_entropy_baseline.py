"""
08b Facility Entropy Baseline.

Computes route diversity entropy H(f) / H_max for each facility
separately for peak and offpeak windows.

Counts named departures_* are stop-level service events at representative
nearby stops, not full route trips.
"""


def run() -> None:
    from config_loader import cfg
    import math
    import os
    import warnings

    import pandas as pd
    from tqdm.auto import tqdm

    warnings.filterwarnings("ignore")

    PROCESSED_DIR = "./data/processed"
    STOP_FAC_EXIT_PATH = f"{PROCESSED_DIR}/stop_to_fac_exit_baseline.parquet"
    EASYWAY_PATH = "../gtfs_static/easyway_routes.csv"
    SCORES_PATH = cfg["paths"]["scores"]
    OUT_PARQUET = f"{PROCESSED_DIR}/facility_entropy_baseline.parquet"
    OUT_CSV = f"{PROCESSED_DIR}/facility_entropy_baseline.csv"
    OUT_PREVIEW_CSV = f"{PROCESSED_DIR}/facility_entropy_preview_baseline.csv"

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    required = [STOP_FAC_EXIT_PATH, EASYWAY_PATH, SCORES_PATH]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Відсутні входи для 08b_base: {missing}")

    if all(os.path.exists(path) for path in [OUT_PARQUET, OUT_CSV]):
        outputs_mtime = min(os.path.getmtime(OUT_PARQUET), os.path.getmtime(OUT_CSV))
        inputs_mtime = max(os.path.getmtime(path) for path in required)
        if outputs_mtime >= inputs_mtime:
            cached = pd.read_parquet(OUT_PARQUET)
            print(f"08b_base: кеш ентропії завантажено: {len(cached):,} закладів")
            return

    def hhmm_to_sec(value: str) -> int:
        hour, minute = map(int, value.split(":"))
        return hour * 3600 + minute * 60

    peak_windows = [
        (hhmm_to_sec(cfg["peak_hours"]["morning_start"]), hhmm_to_sec(cfg["peak_hours"]["morning_end"])),
        (hhmm_to_sec(cfg["peak_hours"]["evening_start"]), hhmm_to_sec(cfg["peak_hours"]["evening_end"])),
    ]
    offpeak_start = hhmm_to_sec(cfg["offpeak_hours"]["start"])
    offpeak_end = hhmm_to_sec(cfg["offpeak_hours"]["end"])

    def in_peak(sec: int) -> bool:
        return any(start <= sec < end for start, end in peak_windows)

    def in_offpeak(sec: int) -> bool:
        return offpeak_start <= sec < offpeak_end

    def parse_schedules(value: str) -> list[int]:
        times = []
        for raw in str(value).strip().split(","):
            raw = raw.strip()
            if not raw or raw == r"\N":
                continue
            hh, mm, ss = raw.split(":")
            times.append(int(hh) * 3600 + int(mm) * 60 + int(ss))
        return sorted(times)

    def entropy_stats(route_counts: dict[str, int]) -> tuple[float, float, float, int, int]:
        positive = {route: count for route, count in route_counts.items() if count > 0}
        # Це stop-level відправлення з репрезентативних найближчих зупинок,
        # а не повні рейси маршруту від кінцевої до кінцевої.
        total_stop_departures = sum(positive.values())
        n_routes = len(positive)
        if total_stop_departures <= 0 or n_routes == 0:
            return 0.0, 0.0, 0.0, 0, 0

        entropy = 0.0
        for count in positive.values():
            p_r = count / total_stop_departures
            entropy -= p_r * math.log2(p_r)

        h_max = math.log2(n_routes) if n_routes > 1 else 0.0
        h_norm = (entropy / h_max) if h_max > 0 else 0.0
        return entropy, h_max, h_norm, n_routes, total_stop_departures

    print("08b_base: завантажуємо nearby stops для закладів...")
    stop_fac_exit = pd.read_parquet(STOP_FAC_EXIT_PATH)
    stop_fac_exit["facility_id"] = stop_fac_exit["facility_id"].astype(str)
    stop_fac_exit["stop_id"] = stop_fac_exit["stop_id"].astype(str)
    facility_meta = pd.read_csv(SCORES_PATH, usecols=["facility_id", "facility_type", "name"])
    facility_meta["facility_id"] = facility_meta["facility_id"].astype(str)

    print("08b_base: завантажуємо маршрути easyway...")
    easyway = pd.read_csv(EASYWAY_PATH)
    easyway = easyway[easyway["schedules"] != r"\N"].copy()
    easyway["stop_id"] = easyway["stop_id"].astype(str)
    easyway["route_label"] = (
        easyway["transport"].astype(str).str.strip() + " " + easyway["route"].astype(str).str.strip()
    ).str.strip()
    easyway["times"] = easyway["schedules"].apply(parse_schedules)

    # Для weekdays-моделі використовуємо тільки Weekdays та All Week.
    easyway = easyway[easyway["calendar"].isin(["Weekdays", "All Week"])].copy()

    easyway["peak_count"] = easyway["times"].apply(lambda times: sum(1 for sec in times if in_peak(sec)))
    easyway["offpeak_count"] = easyway["times"].apply(lambda times: sum(1 for sec in times if in_offpeak(sec)))

    route_counts_by_stop = (
        easyway.groupby(["stop_id", "route_label"], as_index=False)[["peak_count", "offpeak_count"]]
        .sum()
        .reset_index(drop=True)
    )

    # Для одного маршруту біля закладу беремо найближчу зупинку,
    # щоб не подвоювати один і той самий маршрут кількома близькими stop events.
    facility_stop_routes = stop_fac_exit.merge(route_counts_by_stop, on="stop_id", how="left")
    facility_stop_routes = facility_stop_routes.dropna(subset=["route_label"]).copy()
    facility_stop_routes = facility_stop_routes.sort_values(["facility_id", "route_label", "walk_min", "stop_id"])
    facility_route_representatives = facility_stop_routes.drop_duplicates(
        subset=["facility_id", "route_label"],
        keep="first",
    ).reset_index(drop=True)

    results = []
    grouped = facility_route_representatives.groupby("facility_id", sort=False)
    print(f"08b_base: рахуємо ентропію для {stop_fac_exit['facility_id'].nunique():,} закладів...")

    for facility_id in tqdm(stop_fac_exit["facility_id"].drop_duplicates().tolist(), desc="08b_base facilities"):
        if facility_id in grouped.groups:
            grp = grouped.get_group(facility_id)
            peak_counts = {
                str(row.route_label): int(row.peak_count)
                for row in grp.itertuples(index=False)
                if int(row.peak_count) > 0
            }
            offpeak_counts = {
                str(row.route_label): int(row.offpeak_count)
                for row in grp.itertuples(index=False)
                if int(row.offpeak_count) > 0
            }
        else:
            peak_counts = {}
            offpeak_counts = {}

        h_peak, hmax_peak, hnorm_peak, n_routes_peak, stop_departures_peak = entropy_stats(peak_counts)
        h_offpeak, hmax_offpeak, hnorm_offpeak, n_routes_offpeak, stop_departures_offpeak = entropy_stats(offpeak_counts)

        results.append(
            {
                "facility_id": facility_id,
                "H_peak": h_peak,
                "H_offpeak": h_offpeak,
                "Hmax_peak": hmax_peak,
                "Hmax_offpeak": hmax_offpeak,
                "Hnorm_peak": hnorm_peak,
                "Hnorm_offpeak": hnorm_offpeak,
                "n_routes_peak": n_routes_peak,
                "n_routes_offpeak": n_routes_offpeak,
                "stop_departures_peak": stop_departures_peak,
                "stop_departures_offpeak": stop_departures_offpeak,
                # Deprecated aliases for compatibility with older notebooks/reports.
                "departures_peak": stop_departures_peak,
                "departures_offpeak": stop_departures_offpeak,
            }
        )

    entropy_df = pd.DataFrame(results)
    entropy_df.to_parquet(OUT_PARQUET, index=False)
    entropy_df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    preview_df = entropy_df.merge(facility_meta, on="facility_id", how="left")
    preview_df = preview_df[
        [
            "facility_id",
            "facility_type",
            "name",
            "H_peak",
            "H_offpeak",
            "Hmax_peak",
            "Hmax_offpeak",
            "Hnorm_peak",
            "Hnorm_offpeak",
            "n_routes_peak",
            "n_routes_offpeak",
            "stop_departures_peak",
            "stop_departures_offpeak",
            "departures_peak",
            "departures_offpeak",
        ]
    ].copy()
    preview_df.to_csv(OUT_PREVIEW_CSV, index=False, encoding="utf-8")

    print(f"08b_base: ентропію збережено в {OUT_PARQUET}")
    print(f"08b_base: csv також збережено в {OUT_CSV}")
    print(f"08b_base: preview csv збережено в {OUT_PREVIEW_CSV}")
    print(f"  Закладів: {len(entropy_df):,}")
    print(f"  З Hnorm_peak > 0: {(entropy_df['Hnorm_peak'] > 0).sum():,}")
    print(f"  З Hnorm_offpeak > 0: {(entropy_df['Hnorm_offpeak'] > 0).sum():,}")
    print(
        f"  Середній Hnorm_peak: {entropy_df['Hnorm_peak'].mean():.3f} | "
        f"Середній Hnorm_offpeak: {entropy_df['Hnorm_offpeak'].mean():.3f}"
    )

    print("\nТоп-5 закладів за Hnorm_peak:")
    top_peak = preview_df.sort_values(["Hnorm_peak", "n_routes_peak"], ascending=[False, False]).head(5)
    for row in top_peak.itertuples(index=False):
        print(
            f"  {str(row.name)[:52]:<52} "
            f"{row.facility_type[:6]:<6} H={row.Hnorm_peak:.3f} "
            f"routes={int(row.n_routes_peak)} stop_dep={int(row.stop_departures_peak)}"
        )

    print("\nТоп-5 закладів за Hnorm_offpeak:")
    top_offpeak = preview_df.sort_values(["Hnorm_offpeak", "n_routes_offpeak"], ascending=[False, False]).head(5)
    for row in top_offpeak.itertuples(index=False):
        print(
            f"  {str(row.name)[:52]:<52} "
            f"{row.facility_type[:6]:<6} H={row.Hnorm_offpeak:.3f} "
            f"routes={int(row.n_routes_offpeak)} stop_dep={int(row.stop_departures_offpeak)}"
        )


if __name__ == "__main__":
    run()
