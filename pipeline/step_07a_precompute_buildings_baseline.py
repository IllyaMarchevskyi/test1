"""
07a Baseline Precompute Buildings.

Baseline branch for catchment without transfers.
Keeps old 07a intact and writes separate baseline caches.
"""


def run() -> None:
    from config_loader import cfg
    import os
    import pickle
    import warnings
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import geopandas as gpd
    import networkx as nx
    import osmnx as ox
    import pandas as pd
    from shapely import wkt
    from tqdm.auto import tqdm

    warnings.filterwarnings("ignore")

    T_SHORT = cfg["catchment"]["threshold_short_min"]
    T_LONG = cfg["catchment"]["threshold_long_min"]
    R_SHORT = T_SHORT * 75
    R_LONG = T_LONG * 75
    R_EXIT = T_LONG * 75
    WALK_SPD_M_MIN = 75.0
    PROCESSED_DIR = "./data/processed"

    BUILDINGS_PATH = "../data/processed/buildings.parquet"
    CACHE_STOP_BLD_SHORT = f"{PROCESSED_DIR}/stop_to_bld_short_baseline.parquet"
    CACHE_STOP_BLD_LONG = f"{PROCESSED_DIR}/stop_to_bld_long_baseline.parquet"
    CACHE_STOP_FAC_EXIT = f"{PROCESSED_DIR}/stop_to_fac_exit_baseline.parquet"
    OSM_EASYWAY_PATH = "../gtfs_static/osm_easyway_data.csv"
    OSM_EASYWAY_METRO_PATH = "../gtfs_static/osm_easyway_metro_data.csv"
    OSM_STOPS_CSV_PATH = "../gtfs_static/osm_stops.csv"
    GMETRO_CSV_PATH = "../gtfs_static/gmetro.csv"

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"Baseline 07a: T_SHORT={T_SHORT} хв, T_LONG={T_LONG} хв")
    print(f"Радіуси: short={R_SHORT}м, long={R_LONG}м, exit={R_EXIT}м")

    graph_path = cfg["paths"]["walk_graph"]
    with open(graph_path, "rb") as f:
        g_raw = pickle.load(f)
    g_proj = ox.project_graph(g_raw, to_crs="EPSG:32636")

    scores = pd.read_csv(cfg["paths"]["scores"])

    if os.path.exists(BUILDINGS_PATH):
        buildings = gpd.read_parquet(BUILDINGS_PATH).set_geometry("geometry")
        print(f"Будинки з кешу: {len(buildings):,}")
    else:
        print("Будуємо buildings.parquet з OSM...")
        buildings = ox.features_from_place(
            "Kyiv, Ukraine",
            tags={"building": ["residential", "apartments", "house", "dormitory", "cabin", "yes"]},
        )
        buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        buildings = buildings.set_geometry(buildings.geometry.centroid)
        buildings = buildings.to_crs("EPSG:32636").reset_index(drop=True)
        buildings["building_id"] = buildings.index.astype(int)
        buildings = gpd.GeoDataFrame(buildings[["building_id", "geometry"]], geometry="geometry", crs="EPSG:32636")
        buildings.to_parquet(BUILDINGS_PATH)
        print(f"Будинки збережені: {BUILDINGS_PATH}")

    cache_paths = [CACHE_STOP_BLD_SHORT, CACHE_STOP_BLD_LONG, CACHE_STOP_FAC_EXIT]
    all_cached = all(os.path.exists(path) for path in cache_paths)
    freshness_inputs = [OSM_EASYWAY_PATH, OSM_STOPS_CSV_PATH]
    if os.path.exists(OSM_EASYWAY_METRO_PATH) and os.path.exists(GMETRO_CSV_PATH):
        freshness_inputs.extend([OSM_EASYWAY_METRO_PATH, GMETRO_CSV_PATH])
    inputs_mtime = max(os.path.getmtime(path) for path in freshness_inputs)
    caches_fresh = all(os.path.getmtime(path) >= inputs_mtime for path in cache_paths if os.path.exists(path))
    if all_cached and caches_fresh:
        stop_bld_short = pd.read_parquet(CACHE_STOP_BLD_SHORT)
        stop_bld_long = pd.read_parquet(CACHE_STOP_BLD_LONG)
        stop_fac_exit = pd.read_parquet(CACHE_STOP_FAC_EXIT)
        print("Baseline 07a кеш завантажено:")
        print(f"  short: {len(stop_bld_short):,}")
        print(f"  long:  {len(stop_bld_long):,}")
        print(f"  exit:  {len(stop_fac_exit):,}")
        return
    if all_cached and not caches_fresh:
        print("Baseline 07a: кеш застарів відносно osm_stops/osm_easyway, перебудовуємо.")

    print(f"Завантажуємо місток OSM -> easyway: {OSM_EASYWAY_PATH}")
    osm_easyway_surface = pd.read_csv(OSM_EASYWAY_PATH, usecols=["osm_id", "stop_id"]).dropna()
    osm_easyway_surface["osm_id"] = osm_easyway_surface["osm_id"].astype(str)
    osm_easyway_surface["stop_id"] = osm_easyway_surface["stop_id"].astype(str)
    osm_easyway_surface = osm_easyway_surface.drop_duplicates(subset=["osm_id", "stop_id"]).reset_index(drop=True)

    print(f"Завантажуємо координати зупинок із: {OSM_STOPS_CSV_PATH}")
    osm_stops_raw = pd.read_csv(OSM_STOPS_CSV_PATH)
    osm_stops_raw = osm_stops_raw.dropna(subset=["geometry"]).copy()
    osm_stops_raw["geometry"] = osm_stops_raw["geometry"].map(wkt.loads)
    osm_stops_raw["osm_id"] = osm_stops_raw.index.astype(str)
    osm_stops_surface = gpd.GeoDataFrame(osm_stops_raw, geometry="geometry", crs="EPSG:4326")
    osm_stops_surface = osm_stops_surface[osm_stops_surface.geometry.geom_type == "Point"].copy()
    osm_stops_surface = osm_stops_surface.merge(osm_easyway_surface, on="osm_id", how="inner")
    osm_stops_surface = osm_stops_surface.drop_duplicates(subset=["osm_id", "stop_id"]).reset_index(drop=True)

    stop_frames = [osm_stops_surface]
    if os.path.exists(OSM_EASYWAY_METRO_PATH) and os.path.exists(GMETRO_CSV_PATH):
        print(f"Завантажуємо metro-місток: {OSM_EASYWAY_METRO_PATH}")
        osm_easyway_metro = pd.read_csv(OSM_EASYWAY_METRO_PATH, usecols=["osm_id", "stop_id"]).dropna()
        osm_easyway_metro["osm_id"] = osm_easyway_metro["osm_id"].astype(str)
        osm_easyway_metro["stop_id"] = osm_easyway_metro["stop_id"].astype(str)
        osm_easyway_metro = osm_easyway_metro.drop_duplicates(subset=["osm_id", "stop_id"]).reset_index(drop=True)

        print(f"Завантажуємо координати метро з: {GMETRO_CSV_PATH}")
        gmetro_raw = pd.read_csv(GMETRO_CSV_PATH).dropna(subset=["geometry"]).copy()
        gmetro_raw["geometry"] = gmetro_raw["geometry"].map(wkt.loads)
        gmetro_raw["osm_id"] = gmetro_raw.index.astype(str)
        gmetro = gpd.GeoDataFrame(gmetro_raw, geometry="geometry", crs="EPSG:4326")
        gmetro = gmetro[gmetro.geometry.geom_type == "Point"].copy()
        gmetro = gmetro.merge(osm_easyway_metro, on="osm_id", how="inner")
        gmetro = gmetro.drop_duplicates(subset=["osm_id", "stop_id"]).reset_index(drop=True)
        stop_frames.append(gmetro)
    else:
        print("Metro-файли не знайдено, 07a_base рахуємо без метро.")

    osm_stops = pd.concat(stop_frames, ignore_index=True)
    osm_stops = gpd.GeoDataFrame(osm_stops, geometry="geometry", crs="EPSG:4326")
    osm_stops = osm_stops.drop_duplicates(subset=["stop_id"]).to_crs("EPSG:32636").reset_index(drop=True)

    facilities = gpd.GeoDataFrame(
        scores[["facility_id", "facility_type", "name"]].copy(),
        geometry=gpd.points_from_xy(scores["lon"], scores["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32636")

    print("Precompute nearest nodes для будинків...")
    bld_nodes = ox.distance.nearest_nodes(g_proj, X=buildings.geometry.x.values, Y=buildings.geometry.y.values)
    bld_by_node = {}
    for bid, node_id in zip(buildings["building_id"].values, bld_nodes):
        bld_by_node.setdefault(int(node_id), []).append(int(bid))

    print("Precompute nearest nodes для закладів...")
    fac_nodes = ox.distance.nearest_nodes(g_proj, X=facilities.geometry.x.values, Y=facilities.geometry.y.values)
    fac_by_node = {}
    for fid, node_id in zip(facilities["facility_id"].values, fac_nodes):
        fac_by_node.setdefault(int(node_id), []).append(str(fid))

    stop_nodes = ox.distance.nearest_nodes(g_proj, X=osm_stops.geometry.x.values, Y=osm_stops.geometry.y.values)
    stop_centers = [(str(stop_id), int(node_id)) for stop_id, node_id in zip(osm_stops["stop_id"].values, stop_nodes)]

    def run_dijkstra(stop_id: str, center_node: int):
        dist_map = dict(nx.single_source_dijkstra_path_length(g_proj, center_node, cutoff=R_LONG, weight="length"))
        rows_short = []
        rows_long = []
        rows_exit = []
        for node_id, dist_m in dist_map.items():
            if node_id in bld_by_node and dist_m <= R_LONG:
                walk_min = round(dist_m / WALK_SPD_M_MIN, 3)
                for bid in bld_by_node[node_id]:
                    rows_long.append((stop_id, bid, walk_min))
                    if dist_m <= R_SHORT:
                        rows_short.append((stop_id, bid, walk_min))
            if node_id in fac_by_node and dist_m <= R_EXIT:
                walk_min = round(dist_m / WALK_SPD_M_MIN, 3)
                for fid in fac_by_node[node_id]:
                    rows_exit.append((stop_id, fid, walk_min))
        return rows_short, rows_long, rows_exit

    rows_short = []
    rows_long = []
    rows_exit = []
    progress = tqdm(total=len(stop_centers), desc="Baseline 07a Dijkstra")
    with ThreadPoolExecutor(max_workers=max(1, int(cfg.get("rl", {}).get("n_envs", os.cpu_count() or 4)))) as executor:
        futures = [executor.submit(run_dijkstra, stop_id, node_id) for stop_id, node_id in stop_centers]
        for future in as_completed(futures):
            part_short, part_long, part_exit = future.result()
            rows_short.extend(part_short)
            rows_long.extend(part_long)
            rows_exit.extend(part_exit)
            progress.update(1)
    progress.close()

    stop_bld_short = pd.DataFrame(rows_short, columns=["stop_id", "building_id", "walk_min"])
    stop_bld_long = pd.DataFrame(rows_long, columns=["stop_id", "building_id", "walk_min"])
    stop_fac_exit = pd.DataFrame(rows_exit, columns=["stop_id", "facility_id", "walk_min"])

    stop_bld_short.to_parquet(CACHE_STOP_BLD_SHORT, index=False)
    stop_bld_long.to_parquet(CACHE_STOP_BLD_LONG, index=False)
    stop_fac_exit.to_parquet(CACHE_STOP_FAC_EXIT, index=False)

    print("Baseline 07a кеш збережено:")
    print(f"  {CACHE_STOP_BLD_SHORT}")
    print(f"  {CACHE_STOP_BLD_LONG}")
    print(f"  {CACHE_STOP_FAC_EXIT}")


if __name__ == "__main__":
    run()
