"""
10c RL recompute check.

Діагностичний крок для порівняння baseline I*_peak із 09_index
та локального recompute з 10_rl при базових частотах маршрутів.
Якщо без жодних змін частот значення суттєво розходяться,
це означає, що surrogate-модель RL не відтворює baseline.
"""


def run() -> None:
    from config_loader import cfg
    from pathlib import Path
    import json

    import numpy as np
    import pandas as pd
    from utils.dispatch_frequency import (
        apply_dispatch_peak_frequency,
        build_easyway_route_stats,
        peak_windows_from_config,
    )

    PROCESSED_DIR = Path("./data/processed")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ACCESSIBILITY_INDEX = PROCESSED_DIR / "accessibility_index_baseline.csv"
    CATCHMENT_BUILDINGS = PROCESSED_DIR / "catchment_buildings_baseline.parquet"
    FACILITY_ENTROPY = PROCESSED_DIR / "facility_entropy_baseline.parquet"
    BUILDING_WEIGHTS = PROCESSED_DIR / "building_weights_baseline.parquet"
    DISPATCH_ROUTE_STATS = PROCESSED_DIR / "dispatch_route_stats.csv"
    EASYWAY_ROUTES = Path("../gtfs_static/easyway_routes.csv")
    EASYWAY_METRO = Path("../gtfs_static/easyway_metro.csv")

    OUTPUT_CSV = PROCESSED_DIR / "rl_recompute_check_targets.csv"
    OUTPUT_JSON = PROCESSED_DIR / "rl_recompute_check_summary.json"

    required = [
        ACCESSIBILITY_INDEX,
        CATCHMENT_BUILDINGS,
        FACILITY_ENTROPY,
        BUILDING_WEIGHTS,
        EASYWAY_ROUTES,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Відсутні входи для 10c_recompute_check: {missing}")

    rl_cfg = cfg.get("rl", {})
    peak_cfg = cfg.get("peak_hours", {})
    peak_windows = peak_windows_from_config(peak_cfg)
    total_peak_hours = float(peak_cfg.get("total_peak_hours", 4))
    freq_scaling = str(rl_cfg.get("freq_scaling", "log")).strip().lower() or "log"
    target_selection = str(rl_cfg.get("target_selection", "bottom_n")).strip().lower()
    target_auto_count = max(1, int(rl_cfg.get("target_auto_count", 10)))
    target_auto_max_candidates = max(target_auto_count, int(rl_cfg.get("target_auto_max_candidates", 250)))
    target_auto_min_i_peak = max(0.0, float(rl_cfg.get("target_auto_min_i_peak", 1e-6)))
    target_facility_id = str(rl_cfg.get("target_facility_id", "")).strip() or None
    target_facility_ids_raw = rl_cfg.get("target_facility_ids", [])
    if isinstance(target_facility_ids_raw, str):
        target_facility_ids = [
            part.strip()
            for part in target_facility_ids_raw.split(",")
            if str(part).strip()
        ]
    elif isinstance(target_facility_ids_raw, (list, tuple)):
        target_facility_ids = [
            str(item).strip()
            for item in target_facility_ids_raw
            if str(item).strip()
        ]
    else:
        target_facility_ids = []
    if not target_facility_ids and target_facility_id:
        target_facility_ids = [target_facility_id]

    index_df = pd.read_csv(ACCESSIBILITY_INDEX)
    catchment = pd.read_parquet(CATCHMENT_BUILDINGS)
    entropy = pd.read_parquet(FACILITY_ENTROPY)
    weights = pd.read_parquet(BUILDING_WEIGHTS, columns=["building_id", "weight_wb"])

    easyway_parts = [pd.read_csv(EASYWAY_ROUTES)]
    if EASYWAY_METRO.exists():
        easyway_parts.append(pd.read_csv(EASYWAY_METRO))
    easyway = pd.concat(easyway_parts, ignore_index=True)

    index_df["facility_id"] = index_df["facility_id"].astype(str)
    catchment["facility_id"] = catchment["facility_id"].astype(str)
    catchment["peak_route_id"] = catchment["peak_route_id"].astype(str)
    catchment["peak_mode"] = catchment["peak_mode"].astype(str)
    weights["building_id"] = pd.to_numeric(weights["building_id"], errors="coerce").astype("Int64")
    weights["weight_wb"] = pd.to_numeric(weights["weight_wb"], errors="coerce").fillna(1.0).clip(lower=1.0)
    entropy["facility_id"] = entropy["facility_id"].astype(str)
    entropy["Hnorm_peak"] = pd.to_numeric(entropy["Hnorm_peak"], errors="coerce").fillna(0.0)

    total_city_weight = float(weights["weight_wb"].sum())

    def parse_schedules(value: str) -> list[int]:
        times = []
        for raw in str(value).strip().split(","):
            raw = raw.strip()
            if not raw or raw == r"\N":
                continue
            hh, mm, ss = raw.split(":")
            times.append(int(hh) * 3600 + int(mm) * 60 + int(ss))
        return sorted(times)

    easyway = easyway[easyway["schedules"] != r"\N"].copy()
    easyway["route_id"] = easyway["route_id"].astype(str)
    easyway["transport"] = easyway["transport"].astype(str)
    easyway["route"] = easyway["route"].astype(str)
    easyway["direction"] = easyway["direction"].astype(str)
    easyway["stop_id"] = easyway["stop_id"].astype(str)
    easyway["times"] = easyway["schedules"].apply(parse_schedules)
    easyway["n_departures"] = easyway["times"].apply(len)
    route_stats = build_easyway_route_stats(
        easyway,
        peak_windows,
        total_peak_hours,
        group_by_direction=False,
    )
    route_stats = apply_dispatch_peak_frequency(route_stats, DISPATCH_ROUTE_STATS, total_peak_hours)
    route_stats["rl_initial_freq"] = 6.0
    for _, sub_idx in route_stats.groupby("transport").groups.items():
        current_freq = route_stats.loc[sub_idx, "current_freq"].astype(float)
        raw = np.log1p(current_freq) if freq_scaling == "log" else current_freq
        min_raw = float(raw.min())
        max_raw = float(raw.max())
        if max_raw > min_raw:
            scaled = 1.0 + ((raw - min_raw) / (max_raw - min_raw) * 11.0)
        else:
            scaled = pd.Series(6.0, index=raw.index)
        route_stats.loc[sub_idx, "rl_initial_freq"] = scaled.round(2)
    base_freq_by_route = dict(zip(route_stats["route_id"], route_stats["rl_initial_freq"]))

    if not target_facility_ids:
        if target_selection not in {"bottom_n", "worst", "auto"}:
            raise ValueError(
                "10c_recompute_check: target_selection має бути bottom_n/worst/auto "
                "або потрібно явно задати target_facility_id(s)."
            )
        known_routes = set(base_freq_by_route)
        selected_ids: list[str] = []
        candidates = (
            index_df[index_df["I_peak"] > target_auto_min_i_peak]
            .sort_values("I_peak", ascending=True)
            .head(target_auto_max_candidates)
        )
        for row in candidates.itertuples(index=False):
            facility_id = str(row.facility_id)
            records = catchment[
                catchment["facility_id"].eq(facility_id)
                & catchment["peak_mode"].eq("transit")
                & catchment["peak_route_id"].isin(known_routes)
            ]
            if records.empty:
                continue
            selected_ids.append(facility_id)
            if len(selected_ids) >= target_auto_count:
                break
        if not selected_ids:
            raise ValueError("10c_recompute_check: auto target selection не знайшов закладів з transit-маршрутами.")
        target_facility_ids = selected_ids
        print(
            "10c_recompute_check: auto target selection "
            f"mode={target_selection} count={len(target_facility_ids)} "
            f"targets={', '.join(target_facility_ids)}"
        )

    print(
        "10c_recompute_check: перевіряємо baseline vs recompute для: "
        f"{', '.join(target_facility_ids)}"
    )

    catchment = catchment.merge(weights, on="building_id", how="left")
    catchment["weight_wb"] = pd.to_numeric(catchment["weight_wb"], errors="coerce").fillna(1.0).clip(lower=1.0)
    catchment = catchment.merge(entropy[["facility_id", "Hnorm_peak"]], on="facility_id", how="left")
    catchment["Hnorm_peak"] = pd.to_numeric(catchment["Hnorm_peak"], errors="coerce").fillna(0.0)

    result_rows: list[dict] = []

    for facility_id in target_facility_ids:
        rows = catchment[catchment["facility_id"] == facility_id].copy()
        if rows.empty:
            result_rows.append(
                {
                    "facility_id": facility_id,
                    "I_peak_baseline": np.nan,
                    "I_peak_recomputed_at_base_freq": np.nan,
                    "delta": np.nan,
                    "delta_pct": np.nan,
                    "n_rows": 0,
                    "n_transit_rows": 0,
                    "n_walk_rows": 0,
                    "status": "no_catchment_rows",
                }
            )
            continue

        weighted_sum = 0.0
        n_transit_rows = 0
        n_walk_rows = 0

        for row in rows.itertuples(index=False):
            total_min = getattr(row, "peak_total_min", np.nan)
            if pd.isna(total_min):
                continue

            route_id = str(getattr(row, "peak_route_id", ""))
            wait_min = getattr(row, "peak_wait_min", np.nan)
            walk_in = getattr(row, "peak_walk_in_min", np.nan)
            transit_min = getattr(row, "peak_transit_min", np.nan)
            walk_out = getattr(row, "peak_walk_out_min", np.nan)
            mode = getattr(row, "peak_mode", None)
            weight_wb = float(getattr(row, "weight_wb", 1.0))

            adjusted_total = float(total_min)
            if mode == "transit" and route_id in base_freq_by_route and pd.notna(wait_min):
                n_transit_rows += 1
                base_freq = max(float(base_freq_by_route.get(route_id, 1.0)), 1.0)
                current_freq = base_freq
                # Це той самий surrogate-recompute, що й у 10_rl,
                # але при базових частотах без жодних дій агента.
                scaled_wait = float(wait_min) * (base_freq / current_freq)
                adjusted_total = (
                    float(walk_in or 0.0)
                    + scaled_wait
                    + float(transit_min or 0.0)
                    + float(walk_out or 0.0)
                )
            else:
                n_walk_rows += 1

            weighted_sum += weight_wb * float(np.exp(-0.05 * adjusted_total))

        h_norm = float(rows["Hnorm_peak"].iloc[0]) if not rows.empty else 0.0
        recomputed = (weighted_sum / total_city_weight) * h_norm

        baseline_series = pd.to_numeric(
            index_df.loc[index_df["facility_id"] == facility_id, "I_peak"],
            errors="coerce",
        ).dropna()
        baseline_value = float(baseline_series.iloc[0]) if not baseline_series.empty else np.nan
        delta = recomputed - baseline_value if pd.notna(baseline_value) else np.nan
        delta_pct = ((delta / baseline_value) * 100.0) if pd.notna(baseline_value) and baseline_value != 0 else np.nan

        result_rows.append(
            {
                "facility_id": facility_id,
                "I_peak_baseline": baseline_value,
                "I_peak_recomputed_at_base_freq": recomputed,
                "delta": delta,
                "delta_pct": delta_pct,
                "n_rows": int(len(rows)),
                "n_transit_rows": int(n_transit_rows),
                "n_walk_rows": int(n_walk_rows),
                "status": "ok",
            }
        )

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    ok_df = result_df[result_df["status"] == "ok"].copy()
    summary = {
        "target_facility_ids": target_facility_ids,
        "mean_baseline": float(ok_df["I_peak_baseline"].mean()) if not ok_df.empty else None,
        "mean_recomputed_at_base_freq": float(ok_df["I_peak_recomputed_at_base_freq"].mean()) if not ok_df.empty else None,
        "mean_delta": float(ok_df["delta"].mean()) if not ok_df.empty else None,
        "mean_delta_pct": float(ok_df["delta_pct"].mean()) if not ok_df.empty else None,
        "max_abs_delta_pct": float(ok_df["delta_pct"].abs().max()) if not ok_df.empty else None,
        "csv": str(OUTPUT_CSV),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"10c_recompute_check: csv -> {OUTPUT_CSV}")
    print(f"10c_recompute_check: json -> {OUTPUT_JSON}")
    print("10c_recompute_check: порівняння по закладах:")
    for row in result_df.itertuples(index=False):
        if row.status != "ok":
            print(f"  {row.facility_id}: status={row.status}")
            continue
        print(
            f"  {row.facility_id}: baseline={row.I_peak_baseline:.6f}, "
            f"recomputed={row.I_peak_recomputed_at_base_freq:.6f}, "
            f"delta={row.delta:+.6f} ({row.delta_pct:+.2f}%) | "
            f"rows={row.n_rows}, transit={row.n_transit_rows}, walk={row.n_walk_rows}"
        )


if __name__ == "__main__":
    run()
