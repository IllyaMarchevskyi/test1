"""
10h Practical RL recommendations.

Converts greedy RL route-frequency changes into a transport-facing report:
estimated weekday/peak departures, headway changes, and approximate vehicle
resource changes for the target-facility scenario.
"""

from __future__ import annotations


def run() -> None:
    from config_loader import cfg
    import json
    import math
    from pathlib import Path

    import numpy as np
    import pandas as pd

    PROCESSED_DIR = Path("./data/processed")
    OUTPUTS_DIR = Path("./data/outputs")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    OPTIMAL_FREQ_CSV = PROCESSED_DIR / "optimal_frequencies_best_probe.csv"
    TARGET_BEFORE_AFTER_CSV = PROCESSED_DIR / "rl_best_probe_target_before_after.csv"
    GREEDY_RESULTS_JSON = PROCESSED_DIR / "rl_best_probe_results.json"
    EASYWAY_ROUTES = Path("../gtfs_static/easyway_routes.csv")
    EASYWAY_METRO = Path("../gtfs_static/easyway_metro.csv")
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

    def inverse_rl_freq(rl_freq: float, min_log: float, max_log: float) -> float:
        if max_log <= min_log:
            return float(np.expm1(min_log))
        normalized = (float(rl_freq) - 1.0) / 11.0
        normalized = min(1.0, max(0.0, normalized))
        return float(np.expm1(min_log + normalized * (max_log - min_log)))

    route_changes = pd.read_csv(OPTIMAL_FREQ_CSV)
    if route_changes.empty:
        raise ValueError("10h_recommendations: optimal_frequencies_best_probe.csv порожній.")
    route_changes["route_id"] = route_changes["route_id"].astype(str)
    require_osm_mapping = bool(cfg.get("rl", {}).get("require_osm_mapping", False))

    target_effects = pd.read_csv(TARGET_BEFORE_AFTER_CSV)
    greedy_results = json.loads(GREEDY_RESULTS_JSON.read_text(encoding="utf-8"))

    easyway_parts = [pd.read_csv(EASYWAY_ROUTES)]
    if EASYWAY_METRO.exists():
        easyway_parts.append(pd.read_csv(EASYWAY_METRO))
    easyway = pd.concat(easyway_parts, ignore_index=True)
    if require_osm_mapping:
        osm_parts = []
        if OSM_EASYWAY_DATA.exists():
            osm_parts.append(pd.read_csv(OSM_EASYWAY_DATA, usecols=["route_id"]))
        if OSM_EASYWAY_METRO_DATA.exists():
            osm_parts.append(pd.read_csv(OSM_EASYWAY_METRO_DATA, usecols=["route_id"]))
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
    easyway["n_departures"] = easyway["times"].apply(len)
    easyway["n_peak_departures"] = easyway["times"].apply(count_peak)

    route_stats_full = (
        easyway.groupby("route_id", as_index=False)
        .agg(
            transport=("transport", "first"),
            route=("route", "first"),
            total_departures_model=("n_departures", "sum"),
        )
        .reset_index(drop=True)
    )
    route_stats_full["current_freq_model"] = (route_stats_full["total_departures_model"] / 11.0).clip(lower=0.0)
    route_stats_full["log_model_freq"] = np.log1p(route_stats_full["current_freq_model"].astype(float))

    transport_scale = {}
    for transport, group in route_stats_full.groupby("transport"):
        transport_scale[str(transport)] = {
            "min_log": float(group["log_model_freq"].min()),
            "max_log": float(group["log_model_freq"].max()),
        }

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
                "weekday_departures": int(row.n_departures),
                "peak_departures": int(row.n_peak_departures),
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
            duration_rows.append(
                {
                    "route_id": str(first_row.route_id),
                    "direction": str(first_row.direction),
                    "one_way_duration_min": float(np.median(paired)),
                }
            )
    duration_df = pd.DataFrame(duration_rows)

    route_real_stats = (
        direction_stats_df.groupby("route_id", as_index=False)
        .agg(
            directions_count=("direction", "nunique"),
            weekday_departures_before=("weekday_departures", "sum"),
            peak_departures_before=("peak_departures", "sum"),
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

    recommendations = []
    for row in route_changes.itertuples(index=False):
        route_id = str(row.route_id)
        stats = stats_by_route.get(route_id, {})
        transport = str(getattr(row, "transport", stats.get("transport", "")))
        scale = transport_scale.get(transport, {"min_log": 0.0, "max_log": 0.0})

        model_before = inverse_rl_freq(float(row.initial_freq), scale["min_log"], scale["max_log"])
        model_after = inverse_rl_freq(float(row.after_freq), scale["min_log"], scale["max_log"])
        intensity_ratio = (model_after / model_before) if model_before > 0 else 1.0

        weekday_before = float(stats.get("weekday_departures_before") or 0.0)
        peak_before = float(stats.get("peak_departures_before") or 0.0)
        directions_count = max(1.0, float(stats.get("directions_count") or 1.0))

        weekday_after = weekday_before * intensity_ratio
        peak_after = peak_before * intensity_ratio
        peak_delta = peak_after - peak_before
        weekday_delta = weekday_after - weekday_before

        peak_per_direction_before = peak_before / directions_count
        peak_per_direction_after = peak_after / directions_count
        headway_before = safe_headway(total_peak_min, peak_per_direction_before)
        headway_after = safe_headway(total_peak_min, peak_per_direction_after)
        headway_delta = (
            headway_after - headway_before
            if headway_before is not None and headway_after is not None
            else None
        )

        round_trip = stats.get("round_trip_duration_min")
        if round_trip is not None and not pd.isna(round_trip) and headway_before and headway_after:
            vehicles_before = float(round_trip) / headway_before
            vehicles_after = float(round_trip) / headway_after
            vehicles_delta = vehicles_after - vehicles_before
        else:
            vehicles_before = vehicles_after = vehicles_delta = None

        if float(row.delta) > 0:
            action = "increase"
            recommendation_text = (
                f"Збільшити інтенсивність маршруту {transport} {row.route}: "
                f"орієнтовно {peak_delta:+.1f} рейсів у пік."
            )
        else:
            action = "decrease"
            recommendation_text = (
                f"Маршрут {transport} {row.route} може бути донором ресурсу: "
                f"орієнтовно {peak_delta:+.1f} рейсів у пік."
            )

        recommendations.append(
            {
                "route_id": route_id,
                "transport": transport,
                "route": str(getattr(row, "route", stats.get("route", ""))),
                "action": action,
                "rl_initial_freq": float(row.initial_freq),
                "rl_after_freq": float(row.after_freq),
                "rl_delta": float(row.delta),
                "estimated_intensity_ratio": intensity_ratio,
                "weekday_departures_before": weekday_before,
                "weekday_departures_after_est": weekday_after,
                "weekday_departures_delta_est": weekday_delta,
                "peak_departures_before": peak_before,
                "peak_departures_after_est": peak_after,
                "peak_departures_delta_est": peak_delta,
                "peak_departures_delta_rounded": int(round(peak_delta)),
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
        "summary": {
            "steps_applied": greedy_results.get("steps_applied"),
            "target_i_peak_before": greedy_results.get("before", {}).get("I_peak_target_mean"),
            "target_i_peak_after": greedy_results.get("after", {}).get("I_peak_target_mean"),
            "target_i_peak_delta": greedy_results.get("delta", {}).get("I_peak_target_mean"),
            "target_i_peak_delta_pct": greedy_results.get("delta", {}).get("I_peak_target_mean_pct"),
        },
        "route_recommendations": recommendations,
        "facility_effects": target_summary,
        "interpretation_notes": [
            "rl_freq є нормалізованою відносною шкалою, а не прямою кількістю рейсів.",
            "Оцінка рейсів отримана через відносний коефіцієнт зміни інтенсивності і розклад першої зупинки кожного напрямку.",
            "Оцінка кількості транспортних одиниць є наближеною і базується на медіанній тривалості рейсу між першою та останньою зупинкою.",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Практичні рекомендації після RL/greedy оптимізації",
        "",
        f"Цільові заклади: {', '.join(payload['target_facility_ids'])}",
        f"Кроків greedy: {payload['summary']['steps_applied']}",
        (
            "Середній I*_peak: "
            f"{payload['summary']['target_i_peak_before']:.9f} -> "
            f"{payload['summary']['target_i_peak_after']:.9f} "
            f"({payload['summary']['target_i_peak_delta']:+.9f}, "
            f"{payload['summary']['target_i_peak_delta_pct']:+.4f}%)"
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
                f"- RL-шкала: {item['rl_initial_freq']:.2f} -> {item['rl_after_freq']:.2f} ({item['rl_delta']:+.2f})",
                (
                    f"- Рейси у пік: {item['peak_departures_before']:.1f} -> "
                    f"{item['peak_departures_after_est']:.1f} "
                    f"({item['peak_departures_delta_est']:+.1f}, округлено {item['peak_departures_delta_rounded']:+d})"
                ),
                (
                    f"- Середній інтервал у пік на напрямок: "
                    f"{format_num(item['peak_headway_before_min'])} хв -> "
                    f"{format_num(item['peak_headway_after_est_min'])} хв "
                    f"({format_num(item['peak_headway_delta_est_min'])} хв)"
                ),
                f"- Орієнтовна зміна рухомого складу: {vehicles_text}",
                "",
            ]
        )

    lines.extend(
        [
            "## Ефект по закладах",
            "",
            "| Заклад | Назва | I*_peak до | I*_peak після | Delta | Delta % |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in target_summary:
        lines.append(
            f"| {item['facility_id']} | {item['name']} | "
            f"{float(item['I_peak_before']):.9f} | {float(item['I_peak_after']):.9f} | "
            f"{float(item['delta']):+.9f} | {float(item['delta_pct']):+.4f}% |"
        )

    lines.extend(
        [
            "",
            "## Обмеження інтерпретації",
            "",
            "- Це сценарна рекомендація, а не готовий диспетчерський план.",
            "- Перерахунок рейсів є наближеним, бо RL працює на нормалізованій шкалі частот.",
            "- Перед практичним застосуванням потрібно перевірити оборотність, випуск рухомого складу та вплив на пасажиропотік.",
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
