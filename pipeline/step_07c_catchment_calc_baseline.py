"""
07c Baseline Catchment Calculation.

Baseline branch for catchment without transfers.
Keeps old 07c intact and writes separate baseline outputs.
"""


def run() -> None:
    from config_loader import cfg
    import os
    import pickle
    import warnings

    import geopandas as gpd
    import networkx as nx
    import osmnx as ox
    import pandas as pd
    from tqdm.auto import tqdm

    warnings.filterwarnings("ignore")

    T_SHORT = cfg["catchment"]["threshold_short_min"]
    T_LONG = cfg["catchment"]["threshold_long_min"]
    GRP_WALK_SHORT    = f"walk_{T_SHORT}min"
    GRP_TRANSIT_SHORT = f"transit_{T_SHORT}min"
    GRP_WALK_LONG     = f"walk_{T_LONG}min"
    GRP_TRANSIT_LONG  = f"transit_{T_LONG}min"

    # Groups that are currently enabled (read from config; all groups = default)
    ACTIVE_GROUPS = set(cfg["catchment"].get("active_groups",
        [GRP_WALK_SHORT, GRP_TRANSIT_SHORT, GRP_WALK_LONG, GRP_TRANSIT_LONG]))

    # Effective max thresholds: skip computation beyond the highest active group
    _walk_thresh    = max(T_SHORT if GRP_WALK_SHORT    in ACTIVE_GROUPS else 0,
                          T_LONG  if GRP_WALK_LONG     in ACTIVE_GROUPS else 0)
    _transit_thresh = max(T_SHORT if GRP_TRANSIT_SHORT in ACTIVE_GROUPS else 0,
                          T_LONG  if GRP_TRANSIT_LONG  in ACTIVE_GROUPS else 0)
    R_WALK    = _walk_thresh    * 75   # metres — walk Dijkstra cutoff
    R_TRANSIT = _transit_thresh        # minutes — max total transit time
    PROCESSED_DIR = "./data/processed"

    BUILDINGS_PATH = "../data/processed/buildings.parquet"
    STOP_BLD_LONG_PATH = f"{PROCESSED_DIR}/stop_to_bld_long_baseline.parquet"
    STOP_FAC_EXIT_PATH = f"{PROCESSED_DIR}/stop_to_fac_exit_baseline.parquet"
    REACH_PEAK_REV_PATH = f"{PROCESSED_DIR}/stop_reachability_peak_reversed_baseline.parquet"
    REACH_OFFPEAK_REV_PATH = f"{PROCESSED_DIR}/stop_reachability_offpeak_reversed_baseline.parquet"
    WAIT_PEAK_PATH = f"{PROCESSED_DIR}/wait_times_peak_baseline.parquet"
    WAIT_OFFPEAK_PATH = f"{PROCESSED_DIR}/wait_times_offpeak_baseline.parquet"
    CATCHMENT_CACHE = f"{PROCESSED_DIR}/catchment_results_baseline.csv"
    BUILDINGS_CACHE = f"{PROCESSED_DIR}/catchment_buildings_baseline.parquet"

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    required_paths = [
        BUILDINGS_PATH,
        STOP_BLD_LONG_PATH,
        STOP_FAC_EXIT_PATH,
        REACH_PEAK_REV_PATH,
        REACH_OFFPEAK_REV_PATH,
        WAIT_PEAK_PATH,
        WAIT_OFFPEAK_PATH,
        cfg["paths"]["scores"],
        cfg["paths"]["walk_graph"],
    ]
    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Відсутні baseline-входи для 07c: {missing}")

    if os.path.exists(CATCHMENT_CACHE) and os.path.exists(BUILDINGS_CACHE):
        outputs_mtime = min(os.path.getmtime(CATCHMENT_CACHE), os.path.getmtime(BUILDINGS_CACHE))
        inputs_mtime = max(os.path.getmtime(path) for path in required_paths)
        if outputs_mtime >= inputs_mtime:
            catchment_df = pd.read_csv(CATCHMENT_CACHE)
            catchment_buildings = pd.read_parquet(BUILDINGS_CACHE)
            print(f"Baseline 07c кеш завантажено: {len(catchment_df)} закладів")
            print(f"  Buildings: {len(catchment_buildings):,} записів")
            return
        print("Baseline 07c: вхідні baseline-дані новіші за кеш, перераховуємо.")

    print("Baseline 07c: завантаження precomputed даних...")
    buildings = gpd.read_parquet(BUILDINGS_PATH).set_geometry("geometry")
    stop_bld_long = pd.read_parquet(STOP_BLD_LONG_PATH)
    stop_fac_exit = pd.read_parquet(STOP_FAC_EXIT_PATH)
    reach_peak_rev = pd.read_parquet(REACH_PEAK_REV_PATH)
    reach_offpeak_rev = pd.read_parquet(REACH_OFFPEAK_REV_PATH)
    wait_peak = pd.read_parquet(WAIT_PEAK_PATH)
    wait_offpeak = pd.read_parquet(WAIT_OFFPEAK_PATH)
    scores = pd.read_csv(cfg["paths"]["scores"])

    with open(cfg["paths"]["walk_graph"], "rb") as f:
        g_raw = pickle.load(f)
    g_proj = ox.project_graph(g_raw, to_crs="EPSG:32636")

    stop_bld_long["stop_id"] = stop_bld_long["stop_id"].astype(str)
    stop_bld_long["building_id"] = stop_bld_long["building_id"].astype(int)
    stop_fac_exit["stop_id"] = stop_fac_exit["stop_id"].astype(str)
    stop_fac_exit["facility_id"] = stop_fac_exit["facility_id"].astype(str)
    reach_peak_rev["stop_A"] = reach_peak_rev["stop_A"].astype(str)
    reach_peak_rev["stop_B"] = reach_peak_rev["stop_B"].astype(str)
    reach_offpeak_rev["stop_A"] = reach_offpeak_rev["stop_A"].astype(str)
    reach_offpeak_rev["stop_B"] = reach_offpeak_rev["stop_B"].astype(str)
    wait_peak["stop_A"] = wait_peak["stop_A"].astype(str)
    wait_peak["stop_B"] = wait_peak["stop_B"].astype(str)
    wait_offpeak["stop_A"] = wait_offpeak["stop_A"].astype(str)
    wait_offpeak["stop_B"] = wait_offpeak["stop_B"].astype(str)

    print("Будуємо словники для baseline 07c...")
    stop_bld_dict = {}
    for row in tqdm(stop_bld_long.itertuples(index=False), total=len(stop_bld_long), desc="stop -> buildings"):
        stop_bld_dict.setdefault(row.stop_id, {})[int(row.building_id)] = float(row.walk_min)

    fac_stop_dict = {}
    for row in stop_fac_exit.itertuples(index=False):
        fac_stop_dict.setdefault(str(row.facility_id), {})[row.stop_id] = float(row.walk_min)

    rev_peak_dict = {}
    for row in tqdm(reach_peak_rev.itertuples(index=False), total=len(reach_peak_rev), desc="reverse peak"):
        rev_peak_dict.setdefault(row.stop_B, {})[row.stop_A] = {
            "transit_min": float(row.transit_min),
            "route_id": getattr(row, "route_id", None),
            "route": getattr(row, "route", None),
            "transport": getattr(row, "transport", None),
            "direction": getattr(row, "direction", None),
            "route_options": getattr(row, "route_options", None),
        }

    rev_offpeak_dict = {}
    for row in tqdm(reach_offpeak_rev.itertuples(index=False), total=len(reach_offpeak_rev), desc="reverse offpeak"):
        rev_offpeak_dict.setdefault(row.stop_B, {})[row.stop_A] = {
            "transit_min": float(row.transit_min),
            "route_id": getattr(row, "route_id", None),
            "route": getattr(row, "route", None),
            "transport": getattr(row, "transport", None),
            "direction": getattr(row, "direction", None),
            "route_options": getattr(row, "route_options", None),
        }

    wait_peak_value_col = "adj_wait_min" if "adj_wait_min" in wait_peak.columns else "avg_wait_min"
    wait_offpeak_value_col = "adj_wait_min" if "adj_wait_min" in wait_offpeak.columns else "avg_wait_min"
    wait_peak_dict = {(row.stop_A, row.stop_B): float(getattr(row, wait_peak_value_col)) for row in wait_peak.itertuples(index=False)}
    wait_offpeak_dict = {
        (row.stop_A, row.stop_B): float(getattr(row, wait_offpeak_value_col))
        for row in wait_offpeak.itertuples(index=False)
    }

    print("Precompute nearest nodes для будинків...")
    bld_nodes = ox.distance.nearest_nodes(g_proj, X=buildings.geometry.x.values, Y=buildings.geometry.y.values)
    bld_by_node = {}
    for bid, node_id in zip(buildings["building_id"].values, bld_nodes):
        bld_by_node.setdefault(int(node_id), []).append(int(bid))

    facilities = gpd.GeoDataFrame(
        scores[["facility_id", "facility_type", "name", "lat", "lon"]].copy(),
        geometry=gpd.points_from_xy(scores["lon"], scores["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32636")
    facility_nodes = ox.distance.nearest_nodes(g_proj, X=facilities.geometry.x.values, Y=facilities.geometry.y.values)
    facility_rows = []
    for score_row, center_node in zip(scores.itertuples(index=False), facility_nodes):
        facility_rows.append(
            {
                "facility_id": str(score_row.facility_id),
                "facility_type": score_row.facility_type,
                "name": score_row.name,
                "center_node": int(center_node),
            }
        )

    def classify_group(time_min: float, mode: str) -> str | None:
        """Return the best active group for (time_min, mode), or None if not in any active group."""
        if mode == "walk" and time_min <= T_SHORT and GRP_WALK_SHORT in ACTIVE_GROUPS:
            return GRP_WALK_SHORT
        if mode == "transit" and time_min <= T_SHORT and GRP_TRANSIT_SHORT in ACTIVE_GROUPS:
            return GRP_TRANSIT_SHORT
        if mode == "walk" and time_min <= T_LONG and GRP_WALK_LONG in ACTIVE_GROUPS:
            return GRP_WALK_LONG
        if mode == "transit" and time_min <= T_LONG and GRP_TRANSIT_LONG in ACTIVE_GROUPS:
            return GRP_TRANSIT_LONG
        return None

    def count_transit_direct(rev_dict, wait_dict, facility_id: str) -> dict[int, dict[str, object]]:
        if R_TRANSIT == 0:
            return {}
        stops_near_fac = fac_stop_dict.get(facility_id, {})
        bld_min_time = {}

        for stop_c, walk_out in stops_near_fac.items():
            for stop_a, transit_info in rev_dict.get(stop_c, {}).items():
                transit_min = float(transit_info["transit_min"])
                wait_min = wait_dict.get((stop_a, stop_c), 999.0)
                if wait_min >= 999.0:
                    continue

                fixed_time = wait_min + transit_min + walk_out
                if fixed_time > R_TRANSIT:
                    continue

                for bid, walk_in in stop_bld_dict.get(stop_a, {}).items():
                    total_min = walk_in + fixed_time
                    current = bld_min_time.get(bid)
                    current_total = float(current["total_min"]) if current is not None else float("inf")
                    if total_min <= R_TRANSIT and total_min < current_total:
                        bld_min_time[bid] = {
                            "total_min": float(total_min),
                            "mode": "transit",
                            "walk_in_min": float(walk_in),
                            "wait_min": float(wait_min),
                            "transit_min": float(transit_min),
                            "walk_out_min": float(walk_out),
                            "route_id": transit_info.get("route_id"),
                            "route": transit_info.get("route"),
                            "transport": transit_info.get("transport"),
                            "direction": transit_info.get("direction"),
                            "route_options": transit_info.get("route_options"),
                            "source_stop": str(stop_a),
                            "dest_stop": str(stop_c),
                        }

        return bld_min_time

    def count_walk_direct(center_node: int) -> dict[int, float]:
        if R_WALK == 0:
            return {}
        dists = nx.single_source_dijkstra_path_length(g_proj, center_node, cutoff=R_WALK, weight="length")
        walk_times = {}
        for node_id, dist_m in dists.items():
            if node_id not in bld_by_node or dist_m > R_WALK:
                continue
            walk_min = float(dist_m) / 75.0
            for bid in bld_by_node[node_id]:
                if walk_min < walk_times.get(bid, float("inf")):
                    walk_times[bid] = walk_min
        return walk_times

    results = []
    building_rows = []
    diag_rows = []

    for fac_row in tqdm(facility_rows, total=len(facility_rows), desc="Baseline 07c facilities"):
        fid = fac_row["facility_id"]
        walk_times = count_walk_direct(fac_row["center_node"])
        transit_peak_times = count_transit_direct(rev_peak_dict, wait_peak_dict, fid)
        transit_offpeak_times = count_transit_direct(rev_offpeak_dict, wait_offpeak_dict, fid)

        def _walk_entry(t: float) -> dict:
            return {
                "total_min": float(t), "mode": "walk",
                "walk_in_min": None, "wait_min": None, "transit_min": None,
                "walk_out_min": None, "route_id": None, "route": None,
                "transport": None, "direction": None, "route_options": None,
                "source_stop": None, "dest_stop": None,
            }

        # Only include walk buildings that fall in an active walk group.
        # Buildings outside active groups are excluded here; transit may still claim them below.
        best_peak    = {bid: _walk_entry(t) for bid, t in walk_times.items()
                        if classify_group(float(t), "walk") is not None}
        best_offpeak = {bid: _walk_entry(t) for bid, t in walk_times.items()
                        if classify_group(float(t), "walk") is not None}

        for bid, transit_info in transit_peak_times.items():
            if float(transit_info["total_min"]) < float(best_peak.get(bid, {"total_min": float("inf")})["total_min"]):
                best_peak[bid] = transit_info

        for bid, transit_info in transit_offpeak_times.items():
            if float(transit_info["total_min"]) < float(best_offpeak.get(bid, {"total_min": float("inf")})["total_min"]):
                best_offpeak[bid] = transit_info

        peak_groups = {}
        offpeak_groups = {}

        for bid, detail in best_peak.items():
            group = classify_group(float(detail["total_min"]), str(detail["mode"]))
            if group is not None:
                peak_groups[bid] = group

        for bid, detail in best_offpeak.items():
            group = classify_group(float(detail["total_min"]), str(detail["mode"]))
            if group is not None:
                offpeak_groups[bid] = group

        peak_counts = pd.Series(list(peak_groups.values())).value_counts()
        offpeak_counts = pd.Series(list(offpeak_groups.values())).value_counts()

        results.append(
            {
                "facility_id": fid,
                "facility_type": fac_row["facility_type"],
                "name": fac_row["name"],
                f"peak_{GRP_WALK_SHORT}": int(peak_counts.get(GRP_WALK_SHORT, 0)),
                f"peak_{GRP_TRANSIT_SHORT}": int(peak_counts.get(GRP_TRANSIT_SHORT, 0)),
                f"peak_{GRP_WALK_LONG}": int(peak_counts.get(GRP_WALK_LONG, 0)),
                f"peak_{GRP_TRANSIT_LONG}": int(peak_counts.get(GRP_TRANSIT_LONG, 0)),
                f"peak_total_{T_SHORT}min": int(peak_counts.get(GRP_WALK_SHORT, 0) + peak_counts.get(GRP_TRANSIT_SHORT, 0)),
                f"peak_total_{T_LONG}min": int(peak_counts.sum()),
                f"offpeak_{GRP_WALK_SHORT}": int(offpeak_counts.get(GRP_WALK_SHORT, 0)),
                f"offpeak_{GRP_TRANSIT_SHORT}": int(offpeak_counts.get(GRP_TRANSIT_SHORT, 0)),
                f"offpeak_{GRP_WALK_LONG}": int(offpeak_counts.get(GRP_WALK_LONG, 0)),
                f"offpeak_{GRP_TRANSIT_LONG}": int(offpeak_counts.get(GRP_TRANSIT_LONG, 0)),
                f"offpeak_total_{T_SHORT}min": int(offpeak_counts.get(GRP_WALK_SHORT, 0) + offpeak_counts.get(GRP_TRANSIT_SHORT, 0)),
                f"offpeak_total_{T_LONG}min": int(offpeak_counts.sum()),
            }
        )

        all_bids = set(peak_groups) | set(offpeak_groups)
        for bid in all_bids:
            peak_detail = best_peak.get(bid, {})
            offpeak_detail = best_offpeak.get(bid, {})
            building_rows.append(
                (
                    fid,
                    bid,
                    peak_groups.get(bid),
                    offpeak_groups.get(bid),
                    peak_detail.get("mode"),
                    peak_detail.get("total_min"),
                    peak_detail.get("walk_in_min"),
                    peak_detail.get("wait_min"),
                    peak_detail.get("transit_min"),
                    peak_detail.get("walk_out_min"),
                    peak_detail.get("route_id"),
                    peak_detail.get("route"),
                    peak_detail.get("transport"),
                    peak_detail.get("direction"),
                    peak_detail.get("route_options"),
                    peak_detail.get("source_stop"),
                    peak_detail.get("dest_stop"),
                    offpeak_detail.get("mode"),
                    offpeak_detail.get("total_min"),
                    offpeak_detail.get("walk_in_min"),
                    offpeak_detail.get("wait_min"),
                    offpeak_detail.get("transit_min"),
                    offpeak_detail.get("walk_out_min"),
                    offpeak_detail.get("route_id"),
                    offpeak_detail.get("route"),
                    offpeak_detail.get("transport"),
                    offpeak_detail.get("direction"),
                    offpeak_detail.get("route_options"),
                    offpeak_detail.get("source_stop"),
                    offpeak_detail.get("dest_stop"),
                )
            )

        diag_rows.append(
            {
                "facility_id": fid,
                "walk_buildings": len(walk_times),
                "peak_transit_buildings": len(transit_peak_times),
                "offpeak_transit_buildings": len(transit_offpeak_times),
                "exit_stops": len(fac_stop_dict.get(fid, {})),
            }
        )

    catchment_df = pd.DataFrame(results)
    catchment_buildings = pd.DataFrame(
        building_rows,
        columns=[
            "facility_id",
            "building_id",
            "group_peak",
            "group_offpeak",
            "peak_mode",
            "peak_total_min",
            "peak_walk_in_min",
            "peak_wait_min",
            "peak_transit_min",
            "peak_walk_out_min",
            "peak_route_id",
            "peak_route",
            "peak_transport",
            "peak_direction",
            "peak_route_options",
            "peak_source_stop",
            "peak_dest_stop",
            "offpeak_mode",
            "offpeak_total_min",
            "offpeak_walk_in_min",
            "offpeak_wait_min",
            "offpeak_transit_min",
            "offpeak_walk_out_min",
            "offpeak_route_id",
            "offpeak_route",
            "offpeak_transport",
            "offpeak_direction",
            "offpeak_route_options",
            "offpeak_source_stop",
            "offpeak_dest_stop",
        ],
    )
    catchment_df.to_csv(CATCHMENT_CACHE, index=False, encoding="utf-8")
    catchment_buildings.to_parquet(BUILDINGS_CACHE, index=False)

    diag_df = pd.DataFrame(diag_rows)
    print(f"Baseline 07c збережено: {CATCHMENT_CACHE}")
    print(f"Baseline 07c buildings: {BUILDINGS_CACHE}")
    print(f"Середній walk_buildings: {diag_df['walk_buildings'].mean():.1f}")
    print(f"Середній peak transit buildings: {diag_df['peak_transit_buildings'].mean():.1f}")
    print(f"Середній offpeak transit buildings: {diag_df['offpeak_transit_buildings'].mean():.1f}")


if __name__ == "__main__":
    run()
