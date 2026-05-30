"""
10h Practical RL recommendations.

Converts greedy RL route-frequency changes into a transport-facing report:
estimated full scheduled trips, headway changes, carrying capacity, and
approximate vehicle resource changes for the target-facility scenario.
"""

from __future__ import annotations


def run() -> None:
    from config_loader import cfg
    import json
    import math
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from utils.rl_transfer import transfer_compatibility_for_run

    PROCESSED_DIR = Path("./data/processed")
    OUTPUTS_DIR = Path("./data/outputs")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    OPTIMAL_FREQ_CSV = PROCESSED_DIR / "optimal_frequencies_best_probe.csv"
    TARGET_BEFORE_AFTER_CSV = PROCESSED_DIR / "rl_best_probe_target_before_after.csv"
    GREEDY_RESULTS_JSON = PROCESSED_DIR / "rl_best_probe_results.json"
    DISPATCH_ROUTE_STATS = PROCESSED_DIR / "dispatch_route_stats.csv"
    EASYWAY_ROUTES = Path("../gtfs_static/easyway_routes.csv")
    EASYWAY_METRO = Path("../gtfs_static/easyway_metro.csv")
    EASYWAY_TRAM = Path("../gtfs_static/easyway_tram_data.csv")
    OSM_EASYWAY_DATA = Path("../gtfs_static/osm_easyway_data.csv")
    OSM_EASYWAY_METRO_DATA = Path("../gtfs_static/osm_easyway_metro_data.csv")

    OUT_CSV = PROCESSED_DIR / "rl_practical_recommendations.csv"
    OUT_JSON = PROCESSED_DIR / "rl_practical_recommendations.json"
    OUT_MD = OUTPUTS_DIR / "rl_recommendations_report.md"

    required = [OPTIMAL_FREQ_CSV, TARGET_BEFORE_AFTER_CSV, GREEDY_RESULTS_JSON, EASYWAY_ROUTES]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Відсутні входи для 10h_recommendations: {missing}")

    peak_cfg = cfg.get("peak_hours", {})
    morning_start = str(peak_cfg.get("morning_start", "07:00"))
    morning_end = str(peak_cfg.get("morning_end", "09:00"))
    evening_start = str(peak_cfg.get("evening_start", "17:00"))
    evening_end = str(peak_cfg.get("evening_end", "19:00"))
    total_peak_hours = float(peak_cfg.get("total_peak_hours", 4))
    total_peak_min = total_peak_hours * 60.0

    def hm_to_seconds(value: str) -> int:
        hh, mm = value.split(":")[:2]
        return int(hh) * 3600 + int(mm) * 60

    morning_start_s = hm_to_seconds(morning_start)
    morning_end_s = hm_to_seconds(morning_end)
    evening_start_s = hm_to_seconds(evening_start)
    evening_end_s = hm_to_seconds(evening_end)

    def parse_schedules(value: str) -> list[int]:
        times: list[int] = []
        for raw in str(value).strip().split(","):
            raw = raw.strip()
            if not raw or raw == r"\N":
                continue
            hh, mm, ss = raw.split(":")
            times.append(int(hh) * 3600 + int(mm) * 60 + int(ss))
        return sorted(times)

    def count_peak(times: list[int]) -> int:
        return sum(
            1
            for sec in times
            if (morning_start_s <= sec < morning_end_s) or (evening_start_s <= sec < evening_end_s)
        )

    def format_num(value: float | int | None, digits: int = 1) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "н/д"
        return f"{float(value):.{digits}f}"

    def safe_headway(period_min: float, departures: float) -> float | None:
        if departures <= 0:
            return None
        return period_min / departures

    route_changes = pd.read_csv(OPTIMAL_FREQ_CSV)
    if route_changes.empty:
        raise ValueError("10h_recommendations: optimal_frequencies_best_probe.csv порожній.")
    route_changes["route_id"] = route_changes["route_id"].astype(str)
    rl_cfg = cfg.get("rl", {})
    require_osm_mapping = bool(rl_cfg.get("require_osm_mapping", False))
    recommendation_scenario = str(rl_cfg.get("recommendation_scenario", "baseline")).strip() or "baseline"
    allow_cross_type_transfers = bool(rl_cfg.get("allow_cross_type_transfers", False))
    freq_scaling = str(rl_cfg.get("freq_scaling", "log")).strip().lower() or "log"
    transfer_matrix = transfer_compatibility_for_run(rl_cfg)
    matrix_allows_cross_type = any(
        receiver != donor
        for donor, receivers in transfer_matrix.items()
        for receiver in receivers
    )
    vehicle_capacity = {
        "bus": 80,
        "trol": 100,
        "tram": 160,
        "metro": 1000,
    }
    vehicle_capacity.update(
        {
            str(key).strip().lower(): int(value)
            for key, value in dict(rl_cfg.get("vehicle_capacity", {})).items()
        }
    )

    def normalize_route(value: str) -> str:
        table = str.maketrans(
            {
                "а": "a",
                "А": "a",
                "д": "d",
                "Д": "d",
                "к": "k",
                "К": "k",
                "т": "t",
                "Т": "t",
                "р": "r",
                "Р": "r",
            }
        )
        return str(value).replace(" ", "").translate(table).lower()

    target_effects = pd.read_csv(TARGET_BEFORE_AFTER_CSV)
    greedy_results = json.loads(GREEDY_RESULTS_JSON.read_text(encoding="utf-8"))

    easyway_parts = [pd.read_csv(EASYWAY_ROUTES)]
    if EASYWAY_METRO.exists():
        easyway_parts.append(pd.read_csv(EASYWAY_METRO))
    if EASYWAY_TRAM.exists():
        easyway_parts.append(pd.read_csv(EASYWAY_TRAM))
    easyway = pd.concat(easyway_parts, ignore_index=True)
    if require_osm_mapping:
        osm_parts = []
        if OSM_EASYWAY_DATA.exists():
            osm_parts.append(pd.read_csv(OSM_EASYWAY_DATA, usecols=["route_id"]))
        if OSM_EASYWAY_METRO_DATA.exists():
            osm_parts.append(pd.read_csv(OSM_EASYWAY_METRO_DATA, usecols=["route_id"]))
        if EASYWAY_TRAM.exists():
            osm_parts.append(pd.read_csv(EASYWAY_TRAM, usecols=["route_id"]))
        if not osm_parts:
            raise FileNotFoundError(
                "10h_recommendations: require_osm_mapping=true, але osm_easyway_data.csv не знайдено."
            )
        allowed_route_ids = set(pd.concat(osm_parts, ignore_index=True)["route_id"].astype(str).unique())
        before_count = len(route_changes)
        route_changes = route_changes[route_changes["route_id"].isin(allowed_route_ids)].copy()
        dropped_count = before_count - len(route_changes)
        if dropped_count:
            print(f"10h_recommendations: відкинуто {dropped_count} маршрут(ів) без OSM mapping.")
        if route_changes.empty:
            raise ValueError("10h_recommendations: після OSM-фільтра не лишилось маршрутів для звіту.")
        easyway = easyway[easyway["route_id"].astype(str).isin(allowed_route_ids)].copy()
    easyway = easyway[easyway["schedules"] != r"\N"].copy()
    easyway["route_id"] = easyway["route_id"].astype(str)
    easyway["transport"] = easyway["transport"].astype(str)
    easyway["route"] = easyway["route"].astype(str)
    easyway["direction"] = easyway["direction"].astype(str)
    easyway["index"] = pd.to_numeric(easyway["index"], errors="coerce")
    easyway["times"] = easyway["schedules"].apply(parse_schedules)
    easyway["n_stop_departures"] = easyway["times"].apply(len)
    easyway["n_peak_stop_departures"] = easyway["times"].apply(count_peak)

    route_stats_full = (
        easyway.groupby("route_id", as_index=False)
        .agg(
            transport=("transport", "first"),
            route=("route", "first"),
            total_stop_departures_model=("n_stop_departures", "sum"),
        )
        .reset_index(drop=True)
    )

    first_stop_rows = (
        easyway.sort_values(["route_id", "direction", "index"])
        .groupby(["route_id", "direction"], as_index=False)
        .first()
    )
    last_stop_rows = (
        easyway.sort_values(["route_id", "direction", "index"])
        .groupby(["route_id", "direction"], as_index=False)
        .last()
    )

    direction_stats = []
    for row in first_stop_rows.itertuples(index=False):
        direction_stats.append(
            {
                "route_id": str(row.route_id),
                "direction": str(row.direction),
                "weekday_stop_departures": int(row.n_stop_departures),
                "peak_stop_departures": int(row.n_peak_stop_departures),
                "first_time_s": min(row.times) if row.times else None,
                "last_time_s": max(row.times) if row.times else None,
            }
        )
    direction_stats_df = pd.DataFrame(direction_stats)

    duration_rows = []
    for first_row in first_stop_rows.itertuples(index=False):
        last_match = last_stop_rows[
            (last_stop_rows["route_id"].astype(str) == str(first_row.route_id))
            & (last_stop_rows["direction"].astype(str) == str(first_row.direction))
        ]
        if last_match.empty:
            continue
        last_times = list(last_match.iloc[0]["times"])
        first_times = list(first_row.times)
        paired = []
        for start, end in zip(first_times, last_times):
            diff = end - start
            if diff < 0:
                diff += 24 * 3600
            if 0 < diff < 4 * 3600:
                paired.append(diff / 60.0)
        if paired:
            # zip-based pairing is reliable only for the first few departures
            # (before headway changes cause alignment drift between first and last stop).
            # Using minimum of valid pairs (>= 5 min) gives actual travel time; median
            # inflates results for high-frequency routes like metro.
            valid = [p for p in paired if p >= 5.0]
            one_way = float(min(valid)) if valid else float(np.median(paired))
            duration_rows.append(
                {
                    "route_id": str(first_row.route_id),
                    "direction": str(first_row.direction),
                    "one_way_duration_min": one_way,
                }
            )
    duration_df = pd.DataFrame(duration_rows)

    route_real_stats = (
        direction_stats_df.groupby("route_id", as_index=False)
        .agg(
            directions_count=("direction", "nunique"),
            weekday_model_stop_departures_before=("weekday_stop_departures", "sum"),
            peak_model_stop_departures_before=("peak_stop_departures", "sum"),
        )
        .reset_index(drop=True)
    )
    if not duration_df.empty:
        route_duration = (
            duration_df.groupby("route_id", as_index=False)
            .agg(
                directions_with_duration=("direction", "nunique"),
                round_trip_duration_min=("one_way_duration_min", "sum"),
                avg_one_way_duration_min=("one_way_duration_min", "mean"),
            )
        )
        route_real_stats = route_real_stats.merge(route_duration, on="route_id", how="left")
        one_direction_mask = route_real_stats["directions_with_duration"].fillna(0) == 1
        route_real_stats.loc[one_direction_mask, "round_trip_duration_min"] = (
            route_real_stats.loc[one_direction_mask, "avg_one_way_duration_min"] * 2.0
        )
    else:
        route_real_stats["directions_with_duration"] = 0
        route_real_stats["round_trip_duration_min"] = np.nan
        route_real_stats["avg_one_way_duration_min"] = np.nan

    route_stats_full = route_stats_full.merge(route_real_stats, on="route_id", how="left")
    stats_by_route = route_stats_full.set_index("route_id").to_dict(orient="index")

    dispatch_by_key: dict[str, dict] = {}
    if DISPATCH_ROUTE_STATS.exists():
        dispatch_df = pd.read_csv(DISPATCH_ROUTE_STATS)
        if not dispatch_df.empty:
            dispatch_df["transport"] = dispatch_df["transport"].astype(str)
            dispatch_df["route"] = dispatch_df["route"].astype(str)
            dispatch_df["_route_norm"] = dispatch_df["route"].apply(normalize_route)
            dispatch_df["_key"] = dispatch_df["transport"] + "_" + dispatch_df["_route_norm"]
            dispatch_by_key = dispatch_df.set_index("_key").to_dict(orient="index")
    else:
        print(
            "10h_recommendations: dispatch_route_stats.csv не знайдено, "
            "звіт буде використовувати тільки easyway-оцінки."
        )

    recommendations = []
    for row in route_changes.itertuples(index=False):
        route_id = str(row.route_id)
        stats = stats_by_route.get(route_id, {})
        transport = str(getattr(row, "transport", stats.get("transport", "")))
        route_label = str(getattr(row, "route", stats.get("route", "")))
        dispatch_key = f"{transport}_{normalize_route(route_label)}"
        dispatch_stats = dispatch_by_key.get(dispatch_key)
        uses_dispatch = dispatch_stats is not None
        # RL-шкала не є кількістю рейсів. Для практичного звіту беремо
        # реальні повні рейси з диспетчерського CSV і масштабуємо їх
        # відносною зміною нормалізованого RL score. Це сценарна оцінка,
        # а не буквальний перерахунок розкладу.
        initial_rl_score = max(float(row.initial_freq), 1e-6)
        after_rl_score = max(float(row.after_freq), 1e-6)
        intensity_ratio = after_rl_score / initial_rl_score

        weekday_before = float(
            dispatch_stats.get("weekday_trips")
            if dispatch_stats is not None and not pd.isna(dispatch_stats.get("weekday_trips"))
            else stats.get("weekday_model_stop_departures_before") or 0.0
        )
        peak_before = float(
            dispatch_stats.get("peak_trips")
            if dispatch_stats is not None and not pd.isna(dispatch_stats.get("peak_trips"))
            else stats.get("peak_model_stop_departures_before") or 0.0
        )
        directions_count = max(
            1.0,
            float(
                dispatch_stats.get("directions_count")
                if dispatch_stats is not None and not pd.isna(dispatch_stats.get("directions_count"))
                else stats.get("directions_count") or 1.0
            ),
        )
        capacity_per_vehicle = int(
            dispatch_stats.get("capacity_per_vehicle")
            if dispatch_stats is not None
            and not pd.isna(dispatch_stats.get("capacity_per_vehicle"))
            and int(dispatch_stats.get("capacity_per_vehicle")) > 0
            else vehicle_capacity.get(transport, 0)
        )
        release_count_before = (
            float(dispatch_stats.get("release_count"))
            if dispatch_stats is not None and not pd.isna(dispatch_stats.get("release_count"))
            else None
        )
        weekday_scheduled_runs_before = (
            float(dispatch_stats.get("weekday_scheduled_runs"))
            if dispatch_stats is not None and not pd.isna(dispatch_stats.get("weekday_scheduled_runs"))
            else None
        )
        peak_scheduled_runs_before = (
            float(dispatch_stats.get("peak_scheduled_runs"))
            if dispatch_stats is not None and not pd.isna(dispatch_stats.get("peak_scheduled_runs"))
            else None
        )
        partial_runs_before = (
            float(dispatch_stats.get("partial_runs"))
            if dispatch_stats is not None and not pd.isna(dispatch_stats.get("partial_runs"))
            else None
        )

        weekday_after = weekday_before * intensity_ratio
        peak_after = peak_before * intensity_ratio
        peak_delta = peak_after - peak_before
        weekday_delta = weekday_after - weekday_before
        peak_capacity_before = peak_before * capacity_per_vehicle
        peak_capacity_after = peak_after * capacity_per_vehicle
        peak_capacity_delta = peak_capacity_after - peak_capacity_before

        peak_per_direction_before = peak_before / directions_count
        peak_per_direction_after = peak_after / directions_count
        headway_before = safe_headway(total_peak_min, peak_per_direction_before)
        headway_after = safe_headway(total_peak_min, peak_per_direction_after)
        headway_delta = (
            headway_after - headway_before
            if headway_before is not None and headway_after is not None
            else None
        )

        round_trip = (
            dispatch_stats.get("round_trip_duration_min")
            if dispatch_stats is not None and not pd.isna(dispatch_stats.get("round_trip_duration_min"))
            else stats.get("round_trip_duration_min")
        )
        if release_count_before is not None:
            vehicles_before = release_count_before
            vehicles_after = release_count_before * intensity_ratio
            vehicles_delta = vehicles_after - vehicles_before
        elif round_trip is not None and not pd.isna(round_trip) and headway_before and headway_after:
            vehicles_before = float(round_trip) / headway_before
            vehicles_after = float(round_trip) / headway_after
            vehicles_delta = vehicles_after - vehicles_before
        else:
            vehicles_before = vehicles_after = vehicles_delta = None

        if float(row.delta) > 0:
            action = "increase"
            unit_text = "повних рейсів" if uses_dispatch else "модельних easyway-відправлень"
            recommendation_text = (
                f"Збільшити інтенсивність маршруту {transport} {row.route}: "
                f"орієнтовно {peak_delta:+.1f} {unit_text} у пік."
            )
        else:
            action = "decrease"
            unit_text = "повних рейсів" if uses_dispatch else "модельних easyway-відправлень"
            recommendation_text = (
                f"Маршрут {transport} {row.route} може бути донором ресурсу: "
                f"орієнтовно {peak_delta:+.1f} {unit_text} у пік."
            )

        recommendations.append(
            {
                "route_id": route_id,
                "transport": transport,
                "route": route_label,
                "action": action,
                "uses_dispatch_schedule": uses_dispatch,
                "dispatch_source_file": str(dispatch_stats.get("source_file", "")) if dispatch_stats else "",
                "rl_initial_freq": float(row.initial_freq),
                "rl_after_freq": float(row.after_freq),
                "rl_delta": float(row.delta),
                "estimated_intensity_ratio": intensity_ratio,
                "intensity_conversion_method": "rl_score_ratio",
                "capacity_per_vehicle": capacity_per_vehicle,
                "release_count_before": release_count_before,
                "release_count_after_est": vehicles_after if release_count_before is not None else None,
                "release_count_delta_est": vehicles_delta if release_count_before is not None else None,
                "weekday_scheduled_runs_before": weekday_scheduled_runs_before,
                "peak_scheduled_runs_before": peak_scheduled_runs_before,
                "partial_runs_before": partial_runs_before,
                "weekday_full_trips_before": weekday_before,
                "weekday_full_trips_after_est": weekday_after,
                "weekday_full_trips_delta_est": weekday_delta,
                "peak_full_trips_before": peak_before,
                "peak_full_trips_after_est": peak_after,
                "peak_full_trips_delta_est": peak_delta,
                "peak_full_trips_delta_rounded": int(round(peak_delta)),
                "peak_capacity_before_places": peak_capacity_before,
                "peak_capacity_after_est_places": peak_capacity_after,
                "peak_capacity_delta_est_places": peak_capacity_delta,
                "peak_headway_before_min": headway_before,
                "peak_headway_after_est_min": headway_after,
                "peak_headway_delta_est_min": headway_delta,
                "round_trip_duration_min_est": float(round_trip) if round_trip is not None and not pd.isna(round_trip) else None,
                "vehicles_before_est": vehicles_before,
                "vehicles_after_est": vehicles_after,
                "vehicles_delta_est": vehicles_delta,
                "vehicles_delta_rounded": int(round(vehicles_delta)) if vehicles_delta is not None else None,
                "recommendation": recommendation_text,
            }
        )

    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    target_summary = target_effects.to_dict(orient="records")
    payload = {
        "method": "greedy_probe_practical_recommendations",
        "source": {
            "greedy_results": str(GREEDY_RESULTS_JSON),
            "optimal_frequencies": str(OPTIMAL_FREQ_CSV),
            "target_before_after": str(TARGET_BEFORE_AFTER_CSV),
        },
        "target_facility_ids": greedy_results.get("target_facility_ids", []),
        "scenario": {
            "name": recommendation_scenario,
            "allow_cross_type_transfers": allow_cross_type_transfers,
            "matrix_allows_cross_type_transfers": matrix_allows_cross_type,
            "freq_scaling": freq_scaling,
            "transfer_compatibility": transfer_matrix,
            "vehicle_capacity": vehicle_capacity,
            "interpretation": "сценарна оцінка, а не готовий диспетчерський план",
        },
        "summary": {
            "steps_applied": greedy_results.get("steps_applied"),
            "target_i_peak_before": greedy_results.get("before", {}).get("I_peak_target_mean"),
            "target_i_peak_after": greedy_results.get("after", {}).get("I_peak_target_mean"),
            "target_i_peak_delta": greedy_results.get("delta", {}).get("I_peak_target_mean"),
            "target_i_peak_delta_pct": greedy_results.get("delta", {}).get("I_peak_target_mean_pct"),
            "target_worsened_count": greedy_results.get("target_worsened_count", 0),
        },
        "route_recommendations": recommendations,
        "facility_effects": target_summary,
        "interpretation_notes": [
            "rl_freq є нормалізованою відносною шкалою, а не прямою кількістю рейсів.",
            "Рейсами у звіті називаються тільки повні проходи між першою та останньою контрольною точкою з диспетчерського CSV.",
            "Неповні виїзди/заїзди з депо рахуються окремо як scheduled_runs/partial_runs і не входять до peak_trips.",
            "Якщо для маршруту немає диспетчерського CSV, у звіті використовується easyway-оцінка і вона явно не трактується як повний рейс.",
            "Оцінка кількості транспортних одиниць є сценарною: реальний випуск масштабується за відносною зміною RL score.",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = payload["summary"]
    target_before = float(summary["target_i_peak_before"] or 0.0)
    target_after = float(summary["target_i_peak_after"] or 0.0)
    target_delta = float(summary["target_i_peak_delta"] or 0.0)
    target_delta_pct = float(summary["target_i_peak_delta_pct"] or 0.0)
    improved_count = sum(1 for item in target_summary if float(item.get("delta", 0.0)) > 0.0)
    worsened_count = int(summary.get("target_worsened_count") or 0)
    unchanged_count = max(0, len(target_summary) - improved_count - worsened_count)
    if target_delta > 0:
        conclusion_text = (
            "У межах заданої target-групи сценарій покращив середній показник "
            f"пікової доступності I*_peak на {target_delta:+.9f}, або {target_delta_pct:+.4f}%. "
            f"Покращення отримали {improved_count} із {len(target_summary)} закладів."
        )
    elif target_delta < 0:
        conclusion_text = (
            "У межах заданої target-групи сценарій не дав покращення: середній "
            f"I*_peak змінився на {target_delta:+.9f}, або {target_delta_pct:+.4f}%. "
            "Такий результат треба трактувати як відхилений сценарій."
        )
    else:
        conclusion_text = (
            "У межах заданої target-групи сценарій не змінив середній I*_peak. "
            "Це означає, що за поточних обмежень greedy/RL не знайшов корисного "
            "перерозподілу маршрутної інтенсивності."
        )

    lines = [
        "# Практичні рекомендації після RL/greedy оптимізації",
        "",
        f"Сценарій: {recommendation_scenario}",
        f"Cross-type за матрицею: {'увімкнено' if matrix_allows_cross_type else 'вимкнено'}",
        f"Матриця сумісності: {transfer_matrix}",
        f"Нормалізація частот: {freq_scaling}",
        "",
        f"Цільові заклади: {', '.join(payload['target_facility_ids'])}",
        f"Кроків greedy: {payload['summary']['steps_applied']}",
        (
            "Середній I*_peak: "
            f"{target_before:.9f} -> "
            f"{target_after:.9f} "
            f"({target_delta:+.9f}, "
            f"{target_delta_pct:+.4f}%)"
        ),
        f"Target-закладів з погіршенням: {worsened_count}",
        "",
        "## Висновок",
        "",
        conclusion_text,
        "",
        (
            f"Підсумок по target-закладах: покращено {improved_count}, "
            f"без суттєвої зміни {unchanged_count}, погіршено {worsened_count}."
        ),
        "",
        "## Рекомендовані зміни маршрутів",
        "",
    ]

    for item in recommendations:
        vehicles_delta = item["vehicles_delta_rounded"]
        vehicles_text = f"{vehicles_delta:+d} од." if vehicles_delta is not None else "н/д"
        lines.extend(
            [
                f"### {item['transport']} {item['route']} (route_id={item['route_id']})",
                "",
                item["recommendation"],
                "",
                f"- Джерело: {'диспетчерський CSV, тільки повні рейси' if item['uses_dispatch_schedule'] else 'easyway-оцінка, не повні рейси'}",
                f"- RL-шкала: {item['rl_initial_freq']:.2f} -> {item['rl_after_freq']:.2f} ({item['rl_delta']:+.2f})",
                (
                    f"- {'Повні рейси у пік' if item['uses_dispatch_schedule'] else 'Модельні easyway-відправлення у пік'}: "
                    f"{item['peak_full_trips_before']:.1f} -> "
                    f"{item['peak_full_trips_after_est']:.1f} "
                    f"({item['peak_full_trips_delta_est']:+.1f}, округлено {item['peak_full_trips_delta_rounded']:+d})"
                ),
                (
                    f"- Середній інтервал у пік на напрямок: "
                    f"{format_num(item['peak_headway_before_min'])} хв -> "
                    f"{format_num(item['peak_headway_after_est_min'])} хв "
                    f"({format_num(item['peak_headway_delta_est_min'])} хв)"
                ),
                (
                    f"- Провізна спроможність у пік: "
                    f"{item['peak_capacity_before_places']:.0f} -> "
                    f"{item['peak_capacity_after_est_places']:.0f} місць "
                    f"({item['peak_capacity_delta_est_places']:+.0f})"
                ),
                f"- Орієнтовна зміна рухомого складу: {vehicles_text}",
                "",
            ]
        )

    lines.extend(
        [
            "## Ефект по закладах",
            "",
            "| № | Заклад | Назва | I*_peak до | I*_peak після | Delta | Delta % |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for idx, item in enumerate(target_summary, start=1):
        lines.append(
            f"| {idx} | {item['facility_id']} | {item['name']} | "
            f"{float(item['I_peak_before']):.9f} | {float(item['I_peak_after']):.9f} | "
            f"{float(item['delta']):+.9f} | {float(item['delta_pct']):+.4f}% |"
        )

    lines.extend(
        [
            "",
            "## Обмеження інтерпретації",
            "",
            "- Це сценарна рекомендація, а не готовий диспетчерський план.",
            "- RL не генерує новий розклад по хвилинах; він дає сценарну зміну інтенсивності.",
            "- Перерахунок повних рейсів є наближеним: базові рейси беруться з диспетчерського CSV, а після-сценарій масштабується через RL score.",
            "- Перед практичним застосуванням потрібно перевірити водійські зміни, депо, резерв рухомого складу та пасажиропотік.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(
        "10h_recommendations: "
        f"routes={len(recommendations)} report={OUT_MD} csv={OUT_CSV}"
    )


if __name__ == "__main__":
    run()
