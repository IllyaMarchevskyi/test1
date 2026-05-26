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

    PROCESSED_DIR = Path("./data/processed")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ACCESSIBILITY_INDEX = PROCESSED_DIR / "accessibility_index_baseline.csv"
    CATCHMENT_BUILDINGS = PROCESSED_DIR / "catchment_buildings_baseline.parquet"
    EASYWAY_ROUTES = Path("../gtfs_static/easyway_routes.csv")
    EASYWAY_METRO = Path("../gtfs_static/easyway_metro.csv")

    SUMMARY_JSON = PROCESSED_DIR / "rl_target_group_debug_summary.json"
    ROUTES_CSV = PROCESSED_DIR / "rl_target_group_routes.csv"
    AFFECTED_CSV = PROCESSED_DIR / "rl_target_group_affected_facilities.csv"
    CANDIDATES_CSV = PROCESSED_DIR / "rl_target_group_candidates.csv"

    required = [ACCESSIBILITY_INDEX, CATCHMENT_BUILDINGS, EASYWAY_ROUTES]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Відсутні входи для 10e_group_debug: {missing}")

    rl_cfg = cfg.get("rl", {})
    target_ids_raw = rl_cfg.get("target_facility_ids", [])
    if isinstance(target_ids_raw, str):
        target_ids = [part.strip() for part in target_ids_raw.split(",") if part.strip()]
    elif isinstance(target_ids_raw, (list, tuple)):
        target_ids = [str(item).strip() for item in target_ids_raw if str(item).strip()]
    else:
        target_ids = []
    single_target = str(rl_cfg.get("target_facility_id", "")).strip()
    if not target_ids and single_target:
        target_ids = [single_target]

    if not target_ids:
        raise ValueError(
            "10e_group_debug: у config.toml не задано target_facility_id або target_facility_ids."
        )

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
        counts: dict[str, int] = {}
        for route_id in route_ids:
            transport = route_transport.get(str(route_id), "unknown")
            counts[transport] = counts.get(transport, 0) + 1
        return int(sum(count * (count - 1) for count in counts.values()))

    def affected_for_routes(route_ids: set[str]) -> set[str]:
        affected: set[str] = set()
        for route_id in route_ids:
            affected.update(route_to_facilities.get(str(route_id), set()))
        return affected

    target_set = set(target_ids)
    target_routes = set()
    for target_id in target_ids:
        target_routes.update(facility_to_routes.get(target_id, set()))
    affected = affected_for_routes(target_routes)
    affected.update(target_set)
    non_target_affected = affected - target_set

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
        routes = facility_to_routes.get(facility_id, set())
        if not routes:
            continue
        facility_affected = affected_for_routes(routes)
        pairs = action_pairs_for_routes(routes)
        transports = sorted({route_transport.get(route_id, "unknown") for route_id in routes})
        candidate_rows.append(
            {
                "facility_id": facility_id,
                "name": str(row.name),
                "I_peak": float(row.I_peak),
                "local_route_count": len(routes),
                "transport_types": ",".join(transports),
                "transfer_action_pairs": pairs,
                "affected_facilities_count": len(facility_affected),
                "non_target_affected_count": max(0, len(facility_affected) - 1),
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
        "local_routes_count": len(target_routes),
        "transfer_action_pairs": action_pairs_for_routes(target_routes),
        "affected_facilities_count": len(affected),
        "non_target_affected_count": len(non_target_affected),
        "routes_csv": str(ROUTES_CSV),
        "affected_csv": str(AFFECTED_CSV),
        "candidates_csv": str(CANDIDATES_CSV),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"10e_group_debug: target_ids = {', '.join(target_ids)}")
    print(
        "10e_group_debug: "
        f"routes={summary['local_routes_count']} "
        f"actions={summary['transfer_action_pairs']} "
        f"affected={summary['affected_facilities_count']} "
        f"non_target={summary['non_target_affected_count']}"
    )
    print(f"10e_group_debug: routes -> {ROUTES_CSV}")
    print(f"10e_group_debug: affected -> {AFFECTED_CSV}")
    print(f"10e_group_debug: candidates -> {CANDIDATES_CSV}")


if __name__ == "__main__":
    run()
