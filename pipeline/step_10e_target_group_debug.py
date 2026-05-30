"""
10e Target group debug.

Діагностичний крок для підбору target-групи перед запуском 10_rl.
Він не запускає навчання, а показує:
- які маршрути входять у поточну target-групу з config.toml;
- скільки закладів ці маршрути реально зачіпають;
- чи є простір для donor->receiver перерозподілу;
- які одиночні заклади є сильними кандидатами для наступних тестів.
"""


def run() -> None:
    from config_loader import cfg
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from utils.rl_transfer import (
        count_transfer_actions_for_routes,
        load_transfer_compatibility,
        parse_config_list,
        transfer_compatibility_for_run,
    )

    PROCESSED_DIR = Path("./data/processed")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ACCESSIBILITY_INDEX = PROCESSED_DIR / "accessibility_index_baseline.csv"
    CATCHMENT_BUILDINGS = PROCESSED_DIR / "catchment_buildings_baseline.parquet"
    EASYWAY_ROUTES = Path("../gtfs_static/easyway_routes.csv")
    EASYWAY_METRO = Path("../gtfs_static/easyway_metro.csv")
    EASYWAY_TRAM = Path("../gtfs_static/easyway_tram_data.csv")

    SUMMARY_JSON = PROCESSED_DIR / "rl_target_group_debug_summary.json"
    ROUTES_CSV = PROCESSED_DIR / "rl_target_group_routes.csv"
    AFFECTED_CSV = PROCESSED_DIR / "rl_target_group_affected_facilities.csv"
    CANDIDATES_CSV = PROCESSED_DIR / "rl_target_group_candidates.csv"
    TIME_COMPONENTS_CSV = PROCESSED_DIR / "rl_target_group_time_components.csv"

    required = [ACCESSIBILITY_INDEX, CATCHMENT_BUILDINGS, EASYWAY_ROUTES]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Відсутні входи для 10e_group_debug: {missing}")

    rl_cfg = cfg.get("rl", {})

    target_ids_raw = rl_cfg.get("target_facility_ids", [])
    target_ids = parse_config_list(target_ids_raw)
    single_target = str(rl_cfg.get("target_facility_id", "")).strip()
    if not target_ids and single_target:
        target_ids = [single_target]
    target_selection = str(rl_cfg.get("target_selection", "bottom_n")).strip().lower()
    target_auto_count = max(1, int(rl_cfg.get("target_auto_count", 10)))
    target_auto_min_actions = max(1, int(rl_cfg.get("target_auto_min_actions", 6)))
    target_auto_max_candidates = max(target_auto_count, int(rl_cfg.get("target_auto_max_candidates", 250)))
    target_auto_min_i_peak = max(0.0, float(rl_cfg.get("target_auto_min_i_peak", 1e-6)))
    excluded_transport_types = set(parse_config_list(rl_cfg.get("exclude_transport_types", [])))
    allow_cross_type_transfers = bool(rl_cfg.get("allow_cross_type_transfers", False))
    transfer_compatibility = load_transfer_compatibility(rl_cfg)

    index_df = pd.read_csv(ACCESSIBILITY_INDEX)
    index_df["facility_id"] = index_df["facility_id"].astype(str)
    if "name" not in index_df.columns:
        index_df["name"] = index_df["facility_id"]
    index_df["I_peak"] = pd.to_numeric(index_df["I_peak"], errors="coerce").fillna(0.0)

    catchment = pd.read_parquet(CATCHMENT_BUILDINGS)
    catchment["facility_id"] = catchment["facility_id"].astype(str)
    catchment["peak_route_id"] = catchment["peak_route_id"].astype(str)
    catchment["peak_mode"] = catchment["peak_mode"].astype(str)

    transit = catchment[
        catchment["peak_mode"].eq("transit")
        & catchment["peak_route_id"].notna()
        & catchment["peak_route_id"].ne("nan")
        & catchment["peak_route_id"].ne("")
    ][["facility_id", "peak_route_id"]].drop_duplicates()

    easyway_parts = [pd.read_csv(EASYWAY_ROUTES)]
    if EASYWAY_METRO.exists():
        easyway_parts.append(pd.read_csv(EASYWAY_METRO))
    if EASYWAY_TRAM.exists():
        easyway_parts.append(pd.read_csv(EASYWAY_TRAM))
    easyway = pd.concat(easyway_parts, ignore_index=True)
    easyway = easyway[easyway["schedules"] != r"\N"].copy()
    easyway["route_id"] = easyway["route_id"].astype(str)
    easyway["transport"] = easyway["transport"].astype(str)
    easyway["route"] = easyway["route"].astype(str)

    route_meta = (
        easyway.groupby("route_id", as_index=False)
        .agg(
            transport=("transport", "first"),
            route=("route", "first"),
            n_stops=("stop_id", "nunique"),
        )
    )
    route_transport = dict(zip(route_meta["route_id"], route_meta["transport"]))
    route_name = dict(zip(route_meta["route_id"], route_meta["route"]))
    route_stops = dict(zip(route_meta["route_id"], route_meta["n_stops"]))

    facility_to_routes = transit.groupby("facility_id")["peak_route_id"].apply(lambda s: set(s.astype(str))).to_dict()
    route_to_facilities = transit.groupby("peak_route_id")["facility_id"].apply(lambda s: set(s.astype(str))).to_dict()

    def action_pairs_for_routes(route_ids: set[str]) -> int:
        return count_transfer_actions_for_routes(route_ids, route_transport, transfer_compatibility)

    def eligible_routes(route_ids: set[str]) -> set[str]:
        if not excluded_transport_types:
            return set(route_ids)
        return {
            str(route_id)
            for route_id in route_ids
            if route_transport.get(str(route_id), "unknown") not in excluded_transport_types
        }

    def affected_for_routes(route_ids: set[str]) -> set[str]:
        affected: set[str] = set()
        for route_id in route_ids:
            affected.update(route_to_facilities.get(str(route_id), set()))
        return affected

    if not target_ids:
        if target_selection not in {"bottom_n", "worst", "auto"}:
            raise ValueError(
                "10e_group_debug: target_selection має бути bottom_n/worst/auto "
                "або потрібно явно задати target_facility_id(s)."
            )
        selected_ids: list[str] = []
        selected_routes: set[str] = set()
        candidates = (
            index_df[index_df["I_peak"] > target_auto_min_i_peak]
            .sort_values("I_peak", ascending=True)
            .head(target_auto_max_candidates)
        )
        for row in candidates.itertuples(index=False):
            facility_id = str(row.facility_id)
            routes = eligible_routes(facility_to_routes.get(facility_id, set()))
            if not routes:
                continue
            selected_ids.append(facility_id)
            selected_routes.update(routes)
            if len(selected_ids) >= target_auto_count and action_pairs_for_routes(selected_routes) >= target_auto_min_actions:
                break
        if not selected_ids:
            raise ValueError("10e_group_debug: auto target selection не знайшов закладів з transit-маршрутами.")
        target_ids = selected_ids
        print(
            "10e_group_debug: auto target selection "
            f"mode={target_selection} count={len(target_ids)} "
            f"routes={len(selected_routes)} actions={action_pairs_for_routes(selected_routes)} "
            f"targets={', '.join(target_ids)}"
        )

    target_set = set(target_ids)
    target_routes = set()
    for target_id in target_ids:
        target_routes.update(facility_to_routes.get(target_id, set()))
    target_routes_all = set(target_routes)
    target_routes = eligible_routes(target_routes)
    affected = affected_for_routes(target_routes)
    affected.update(target_set)
    non_target_affected = affected - target_set

    numeric_time_cols = [
        "peak_total_min",
        "peak_wait_min",
        "peak_transit_min",
        "peak_walk_in_min",
        "peak_walk_out_min",
    ]
    for col in numeric_time_cols:
        catchment[col] = pd.to_numeric(catchment[col], errors="coerce")

    all_time_records = catchment[catchment["peak_total_min"].notna()].copy()
    time_rows = []
    facility_time_stats: dict[str, dict[str, float]] = {}
    for facility_id, group in all_time_records.groupby("facility_id", sort=False):
        transit_group = group[group["peak_mode"].eq("transit")].copy()
        source = transit_group if not transit_group.empty else group
        walk_total = source["peak_walk_in_min"].fillna(0.0) + source["peak_walk_out_min"].fillna(0.0)
        avg_total = float(source["peak_total_min"].mean())
        avg_wait = float(source["peak_wait_min"].fillna(0.0).mean())
        avg_transit = float(source["peak_transit_min"].fillna(0.0).mean())
        avg_walk = float(walk_total.mean())
        wait_share = (avg_wait / avg_total) if avg_total > 0.0 else 0.0
        facility_time_stats[str(facility_id)] = {
            "avg_total_min": avg_total,
            "avg_wait_min": avg_wait,
            "avg_transit_min": avg_transit,
            "avg_walk_min": avg_walk,
            "wait_share_pct": wait_share * 100.0,
        }
        if str(facility_id) in target_set:
            time_rows.append(
                {
                    "facility_id": str(facility_id),
                    "records": int(len(group)),
                    "transit_records": int(len(transit_group)),
                    "avg_total_min": avg_total,
                    "avg_walk_min": avg_walk,
                    "avg_wait_min": avg_wait,
                    "avg_transit_min": avg_transit,
                    "wait_share_pct": wait_share * 100.0,
                }
            )

    time_components_df = pd.DataFrame(time_rows).sort_values("wait_share_pct", ascending=False)
    time_components_df.to_csv(TIME_COMPONENTS_CSV, index=False, encoding="utf-8")

    route_rows = []
    for route_id in sorted(target_routes):
        route_facilities = route_to_facilities.get(route_id, set())
        target_users = sorted(route_facilities & target_set)
        route_rows.append(
            {
                "route_id": route_id,
                "transport": route_transport.get(route_id, "unknown"),
                "route": route_name.get(route_id, ""),
                "n_stops": int(route_stops.get(route_id, 0)),
                "target_facilities_using_route": ",".join(target_users),
                "target_facilities_count": len(target_users),
                "affected_facilities_count": len(route_facilities),
                "non_target_affected_count": len(route_facilities - target_set),
            }
        )
    routes_df = pd.DataFrame(route_rows).sort_values(
        ["transport", "route", "route_id"],
        ascending=[True, True, True],
    )
    routes_df.to_csv(ROUTES_CSV, index=False, encoding="utf-8")

    affected_df = index_df[index_df["facility_id"].isin(affected)].copy()
    affected_df["is_target"] = affected_df["facility_id"].isin(target_set)
    affected_df = affected_df[["facility_id", "name", "is_target", "I_peak"]].sort_values(
        ["is_target", "I_peak"],
        ascending=[False, True],
    )
    affected_df.to_csv(AFFECTED_CSV, index=False, encoding="utf-8")

    candidate_rows = []
    for row in index_df.itertuples(index=False):
        facility_id = str(row.facility_id)
        routes = eligible_routes(facility_to_routes.get(facility_id, set()))
        if not routes:
            continue
        facility_affected = affected_for_routes(routes)
        pairs = action_pairs_for_routes(routes)
        transports = sorted({route_transport.get(route_id, "unknown") for route_id in routes})
        time_stats = facility_time_stats.get(facility_id, {})
        candidate_rows.append(
            {
                "facility_id": facility_id,
                "name": str(row.name),
                "I_peak": float(row.I_peak),
                "local_route_count": len(routes),
                "metro_route_count": sum(1 for route_id in routes if route_transport.get(route_id) == "metro"),
                "transport_types": ",".join(transports),
                "transfer_action_pairs": pairs,
                "affected_facilities_count": len(facility_affected),
                "non_target_affected_count": max(0, len(facility_affected) - 1),
                "avg_total_min": time_stats.get("avg_total_min"),
                "avg_wait_min": time_stats.get("avg_wait_min"),
                "wait_share_pct": time_stats.get("wait_share_pct"),
                # Високий score = є простір дій і зміна зачіпає не тільки один заклад.
                "candidate_score": float(pairs + np.log1p(max(0, len(facility_affected) - 1))),
            }
        )

    candidates_df = pd.DataFrame(candidate_rows)
    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values(
            ["candidate_score", "I_peak"],
            ascending=[False, True],
        ).head(50)
    candidates_df.to_csv(CANDIDATES_CSV, index=False, encoding="utf-8")

    target_index = index_df[index_df["facility_id"].isin(target_set)].copy()
    summary = {
        "target_facility_ids": target_ids,
        "target_facilities_count": len(target_ids),
        "target_mean_I_peak": (
            float(target_index["I_peak"].mean()) if not target_index.empty else None
        ),
        "exclude_transport_types": sorted(excluded_transport_types),
        "allow_cross_type_transfers": allow_cross_type_transfers,
        "transfer_compatibility": transfer_compatibility_for_run(rl_cfg),
        "local_routes_all_count": len(target_routes_all),
        "local_routes_count": len(target_routes),
        "metro_routes_count": int(sum(1 for route_id in target_routes if route_transport.get(route_id) == "metro")),
        "transfer_action_pairs": action_pairs_for_routes(target_routes),
        "affected_facilities_count": len(affected),
        "non_target_affected_count": len(non_target_affected),
        "avg_total_min": (
            float(time_components_df["avg_total_min"].mean()) if not time_components_df.empty else None
        ),
        "avg_wait_min": (
            float(time_components_df["avg_wait_min"].mean()) if not time_components_df.empty else None
        ),
        "avg_walk_min": (
            float(time_components_df["avg_walk_min"].mean()) if not time_components_df.empty else None
        ),
        "avg_transit_min": (
            float(time_components_df["avg_transit_min"].mean()) if not time_components_df.empty else None
        ),
        "wait_share_pct": (
            float(time_components_df["wait_share_pct"].mean()) if not time_components_df.empty else None
        ),
        "routes_csv": str(ROUTES_CSV),
        "affected_csv": str(AFFECTED_CSV),
        "candidates_csv": str(CANDIDATES_CSV),
        "time_components_csv": str(TIME_COMPONENTS_CSV),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"10e_group_debug: target_ids = {', '.join(target_ids)}")
    print(
        "10e_group_debug: "
        f"routes={summary['local_routes_count']} "
        f"actions={summary['transfer_action_pairs']} "
        f"affected={summary['affected_facilities_count']} "
        f"non_target={summary['non_target_affected_count']} "
        f"wait_share={float(summary['wait_share_pct'] or 0.0):.1f}%"
    )
    print(f"10e_group_debug: routes -> {ROUTES_CSV}")
    print(f"10e_group_debug: affected -> {AFFECTED_CSV}")
    print(f"10e_group_debug: candidates -> {CANDIDATES_CSV}")
    print(f"10e_group_debug: time components -> {TIME_COMPONENTS_CSV}")


if __name__ == "__main__":
    run()
