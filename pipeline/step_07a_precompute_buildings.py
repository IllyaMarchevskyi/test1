"""
07a Precompute Buildings.

Generated mechanically from notebooks/07a_precompute_buildings.ipynb.
Run via main.py from the repository root.
"""


def run() -> None:
    # ---- Notebook code cell 2 ----
    from config_loader import cfg
    import os
    import pickle
    import warnings

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import networkx as nx
    import osmnx as ox
    from shapely.geometry import Point, LineString
    from shapely.ops import unary_union
    from tqdm.auto import tqdm

    warnings.filterwarnings('ignore')

    T_SHORT = cfg['catchment']['threshold_short_min']   # 10 хв
    T_LONG  = cfg['catchment']['threshold_long_min']    # 30 хв
    R_SHORT = T_SHORT * 75                              # 750 м
    R_LONG  = T_LONG  * 75                             # 2250 м
    R_EXIT  = (T_SHORT // 2) * 75                      # 375 м
    # R_EXIT  = T_LONG  * 75                   # 375 м

    GRP_WALK_SHORT    = f'walk_{T_SHORT}min'
    GRP_TRANSIT_SHORT = f'transit_{T_SHORT}min'
    GRP_WALK_LONG     = f'walk_{T_LONG}min'
    GRP_TRANSIT_LONG  = f'transit_{T_LONG}min'

    print(f'Пороги:  {T_SHORT} хв / {T_LONG} хв')
    print(f'Радіуси: R_SHORT={R_SHORT}м  R_LONG={R_LONG}м  R_EXIT={R_EXIT}м')

    GRAPH_PATH = cfg['paths']['walk_graph']
    print(f'Завантажуємо граф: {GRAPH_PATH}')
    with open(GRAPH_PATH, 'rb') as _f:
        G_raw = pickle.load(_f)
    print(f'Граф: {G_raw.number_of_nodes():,} вузлів, {G_raw.number_of_edges():,} ребер')

    osm_stops = gpd.read_file('../data/osm/osm_data.gpkg', layer='stops')
    scores    = pd.read_csv(cfg['paths']['scores'])
    print(f'Зупинок: {len(osm_stops)}')
    print(f'Закладів: {len(scores)}')

    # ---- Notebook code cell 4 ----
    BUILDINGS_PATH = '../data/processed/buildings.parquet'

    def build_buildings_cache():
        print('Завантажуємо будинки з OSM (може зайняти 5–15 хв)...')
    #     buildings = ox.features_from_place(
    #       'Kyiv, Ukraine',
    #       tags={'building': [
    #           'residential',
    #           'apartments',
    #           'house',
    #           'detached',
    #           'semidetached_house',
    #           'terrace',
    #           'dormitory',
    #           'bungalow',
    #           'cabin',
    #           'yes'
    #       ]}
    #   ) 80 тис
        buildings = ox.features_from_place(
            'Kyiv, Ukraine',
            tags={'building': [
                'residential',
                'apartments',
                'house',
                'dormitory',
                'cabin',
                'yes'
            ]}
        )


        # Залишаємо тільки полігони (деякі об'єкти — лінії або точки)
        buildings = buildings[buildings.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()

        # Центроїд кожного будинку -> одразу записуємо в active geometry
        buildings = buildings.set_geometry(buildings.geometry.centroid)

        # Проектуємо в метричну CRS для розрахунків у метрах
        buildings = buildings.to_crs('EPSG:32636')
        buildings = buildings.reset_index(drop=True)
        buildings['building_id'] = buildings.index.astype(int)

        # Залишаємо тільки потрібні колонки і явно фіксуємо geometry + CRS
        buildings = buildings[['building_id', 'geometry']].copy()
        buildings = gpd.GeoDataFrame(buildings, geometry='geometry', crs='EPSG:32636')

        buildings.to_parquet(BUILDINGS_PATH)
        print(f'Будинки збережені до кешу: {BUILDINGS_PATH}')
        print(f'Завантажено з OSM: {len(buildings):,} будинків')
        return buildings

    if os.path.exists(BUILDINGS_PATH):
        try:
            buildings = gpd.read_parquet(BUILDINGS_PATH)
            buildings = buildings.set_geometry('geometry')
            print(f'Будинки завантажені з кешу: {len(buildings):,}')
        except Exception as e:
            print(f'Пошкоджений кеш buildings.parquet: {e}')
            print('Перебудовуємо кеш заново...')
            os.remove(BUILDINGS_PATH)
            buildings = build_buildings_cache()
    else:
        buildings = build_buildings_cache()

    print(f'\nCRS: {buildings.crs}')
    print(f'Колонки: {list(buildings.columns)}')

    # ---- Notebook code cell 6 ----
    # import folium

    # buildings_wgs84 = buildings.to_crs('EPSG:4326')

    # center_lat = float(buildings_wgs84.geometry.y.mean())
    # center_lon = float(buildings_wgs84.geometry.x.mean())

    # m_buildings = folium.Map(
    #     location=[center_lat, center_lon],
    #     zoom_start=11,
    #     tiles='CartoDB positron'
    # )

    # for row in tqdm(buildings_wgs84.itertuples(index=False),
    #                 total=len(buildings_wgs84),
    #                 desc='Малюємо будинки на карті'):
    #     folium.CircleMarker(
    #         location=[row.geometry.y, row.geometry.x],
    #         radius=3,
    #         color='#2E86DE',
    #         fill=True,
    #         fill_color='#2E86DE',
    #         fill_opacity=0.7,
    #         weight=0,
    #         popup=f'building_id: {row.building_id}'
    #     ).add_to(m_buildings)

    # print(f'На карту додано {len(buildings_wgs84):,} будинків')
    # m_buildings

    # ---- Notebook code cell 8 ----
    CACHE_STOP_BLD_SHORT = '../data/processed/stop_to_bld_short.parquet'
    CACHE_STOP_BLD_LONG  = '../data/processed/stop_to_bld_long.parquet'
    CACHE_STOP_FAC_EXIT  = '../data/processed/stop_to_fac_exit.parquet'

    all_cached = all(os.path.exists(p) for p in [
        CACHE_STOP_BLD_SHORT,
        CACHE_STOP_BLD_LONG,
        CACHE_STOP_FAC_EXIT,
    ])

    if all_cached:
        stop_bld_short = pd.read_parquet(CACHE_STOP_BLD_SHORT)
        stop_bld_long  = pd.read_parquet(CACHE_STOP_BLD_LONG)
        stop_fac_exit  = pd.read_parquet(CACHE_STOP_FAC_EXIT)

        stop_bld_short['stop_id'] = stop_bld_short['stop_id'].astype(str)
        stop_bld_long['stop_id'] = stop_bld_long['stop_id'].astype(str)
        stop_fac_exit['stop_id'] = stop_fac_exit['stop_id'].astype(str)
        stop_bld_short['building_id'] = stop_bld_short['building_id'].astype(int)
        stop_bld_long['building_id'] = stop_bld_long['building_id'].astype(int)
        stop_fac_exit['facility_id'] = stop_fac_exit['facility_id'].astype(str)

        print('Завантажено з кешу:')
        print(f'  stop_to_bld_short: {len(stop_bld_short):,} записів')
        print(f'  stop_to_bld_long:  {len(stop_bld_long):,} записів')
        print(f'  stop_to_fac_exit:  {len(stop_fac_exit):,} записів')

    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print('Проектуємо граф у EPSG:32636...')
        G_proj = ox.project_graph(G_raw, to_crs='EPSG:32636')
        stops_proj = osm_stops.to_crs('EPSG:32636').copy()
        facilities_gdf = gpd.GeoDataFrame(
            scores[['facility_id', 'facility_type']].copy(),
            geometry=gpd.points_from_xy(scores['lon'], scores['lat']),
            crs='EPSG:4326'
        ).to_crs('EPSG:32636')
        facilities_gdf = facilities_gdf.reset_index(drop=True)

        OSM_EASYWAY_PATH = '../gtfs_static/osm_easyway_data.csv'
        print(f'Завантажуємо місток OSM → easyway: {OSM_EASYWAY_PATH}')
        osm_easyway = pd.read_csv(OSM_EASYWAY_PATH, usecols=['osm_id', 'stop_id'])
        osm_easyway['osm_id'] = osm_easyway['osm_id'].astype(str)
        osm_easyway['stop_id'] = osm_easyway['stop_id'].astype(str)
        osm_easyway = osm_easyway.dropna(subset=['osm_id', 'stop_id']).copy()

        osm_to_easyway = (
            osm_easyway
            .drop_duplicates(subset=['osm_id', 'stop_id'])
            .groupby('osm_id')['stop_id']
            .agg(list)
            .to_dict()
        )

        if 'osm_stop_id' in stops_proj.columns:
            stops_proj['osm_stop_id'] = stops_proj['osm_stop_id'].astype(str)
        elif 'stop_id' in stops_proj.columns:
            stops_proj['osm_stop_id'] = stops_proj['stop_id'].astype(str)
        else:
            stops_proj['osm_stop_id'] = stops_proj.index.astype(str)

        # У osm_data.gpkg stop-id зберігається як OSM_372, а в osm_easyway_data.csv як 372.
        # Тому нормалізуємо ключ перед мапінгом.
        stops_proj['osm_id_key'] = (
            stops_proj['osm_stop_id']
            .astype(str)
            .str.replace(r'^OSM_', '', regex=True)
        )

        stops_proj['easyway_stop_ids'] = stops_proj['osm_id_key'].map(
            lambda osm_id: osm_to_easyway.get(str(osm_id), [])
        )

        matched_osm = stops_proj['easyway_stop_ids'].map(len).gt(0).sum()
        total_osm = len(stops_proj)
        print(f'Зупинок з easyway-відповідністю: {matched_osm}/{total_osm} ({matched_osm/total_osm*100:.1f}%)')
        print('Незматчені OSM-зупинки буде пропущено на етапі precompute')

        stops_proj = stops_proj[stops_proj['easyway_stop_ids'].map(len) > 0].copy()
        stops_proj = stops_proj.explode('easyway_stop_ids').rename(columns={'easyway_stop_ids': 'stop_id'})
        stops_proj['stop_id'] = stops_proj['stop_id'].astype(str)
        stops_proj = stops_proj.drop_duplicates(subset=['osm_stop_id', 'stop_id']).reset_index(drop=True)
        print(f'Рядків після розгортання osm_id -> stop_id: {len(stops_proj):,}')

        WALK_SPD_M_MIN = 75.0

        print('Precompute nearest nodes для будинків...')
        xs = buildings.geometry.x.values
        ys = buildings.geometry.y.values
        nns = ox.distance.nearest_nodes(G_proj, X=xs, Y=ys)
        bld_by_node = {}
        for bid, nn in zip(buildings['building_id'].values, nns):
            bld_by_node.setdefault(int(nn), []).append(int(bid))

        print('Precompute nearest nodes для закладів...')
        fac_xs = facilities_gdf.geometry.x.values
        fac_ys = facilities_gdf.geometry.y.values
        fac_nns = ox.distance.nearest_nodes(G_proj, X=fac_xs, Y=fac_ys)
        fac_by_node = {}
        for fid, nn in zip(facilities_gdf['facility_id'].values, fac_nns):
            fac_by_node.setdefault(int(nn), []).append(str(fid))

        def _process_stop(sid, dist_map, bld_by_node, fac_by_node,
                          R_SHORT, R_LONG, R_EXIT, WALK_SPD_M_MIN):
            r_short, r_long, r_fac = [], [], []
            for nn, dist_m in dist_map.items():
                if nn in bld_by_node and dist_m <= R_LONG:
                    wm = round(dist_m / WALK_SPD_M_MIN, 3)
                    for bid in bld_by_node[nn]:
                        r_long.append((sid, bid, wm))
                        if dist_m <= R_SHORT:
                            r_short.append((sid, bid, wm))

                if nn in fac_by_node and dist_m <= R_EXIT:
                    wm = round(dist_m / WALK_SPD_M_MIN, 3)
                    for fid in fac_by_node[nn]:
                        r_fac.append((sid, fid, wm))

            return r_short, r_long, r_fac

        def _run_dijkstra(sid, center_node):
            try:
                dist_map = dict(nx.single_source_dijkstra_path_length(
                    G_proj, center_node, cutoff=R_LONG, weight='length'
                ))
                return sid, dist_map, None
            except Exception as e:
                return sid, None, str(e)

        print('Precompute center_node для зупинок...')
        stop_xs = stops_proj.geometry.x.values
        stop_ys = stops_proj.geometry.y.values
        stop_nns = ox.distance.nearest_nodes(G_proj, X=stop_xs, Y=stop_ys)
        stop_centers = [
            (str(sid), int(center_node))
            for sid, center_node in zip(stops_proj['stop_id'].values, stop_nns)
        ]

        rows_short = []
        rows_long = []
        rows_fac = []
        dijkstra_errors = []
        success_count = 0
        error_count = 0

        progress = tqdm(total=len(stop_centers), desc='Dijkstra parallel')
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = {
                executor.submit(_run_dijkstra, sid, center_node): sid
                for sid, center_node in stop_centers
            }

            for future in as_completed(futures):
                sid, dist_map, err = future.result()
                if err is not None:
                    error_count += 1
                    dijkstra_errors.append((sid, err))
                else:
                    success_count += 1
                    r_short, r_long, r_fac = _process_stop(
                        sid, dist_map, bld_by_node, fac_by_node,
                        R_SHORT, R_LONG, R_EXIT, WALK_SPD_M_MIN
                    )
                    rows_short.extend(r_short)
                    rows_long.extend(r_long)
                    rows_fac.extend(r_fac)

                progress.update(1)
                progress.set_postfix({'success': success_count, 'errors': error_count})

        progress.close()

        stop_bld_short = pd.DataFrame(rows_short, columns=['stop_id', 'building_id', 'walk_min'])
        stop_bld_long = pd.DataFrame(rows_long, columns=['stop_id', 'building_id', 'walk_min'])
        stop_fac_exit = pd.DataFrame(rows_fac, columns=['stop_id', 'facility_id', 'walk_min'])

        if not stop_bld_short.empty:
            stop_bld_short = stop_bld_short.astype({'stop_id': str, 'building_id': int, 'walk_min': float})
        if not stop_bld_long.empty:
            stop_bld_long = stop_bld_long.astype({'stop_id': str, 'building_id': int, 'walk_min': float})
        if not stop_fac_exit.empty:
            stop_fac_exit = stop_fac_exit.astype({'stop_id': str, 'facility_id': str, 'walk_min': float})

        stop_bld_short.to_parquet(CACHE_STOP_BLD_SHORT, index=False)
        stop_bld_long.to_parquet(CACHE_STOP_BLD_LONG, index=False)
        stop_fac_exit.to_parquet(CACHE_STOP_FAC_EXIT, index=False)

        print('Кеш збережено:')
        print(f'  {CACHE_STOP_BLD_SHORT}')
        print(f'  {CACHE_STOP_BLD_LONG}')
        print(f'  {CACHE_STOP_FAC_EXIT}')
        print(f'  stop_bld_short: {len(stop_bld_short):,} пар')
        print(f'  stop_bld_long:  {len(stop_bld_long):,} пар')
        print(f'  stop_fac_exit:  {len(stop_fac_exit):,} пар')
        print(f'  Dijkstra success: {success_count}')
        print(f'  Dijkstra errors:  {error_count}')
        if dijkstra_errors:
            print(f'  Перші помилки: {dijkstra_errors[:3]}')

    print(f'\n{"─" * 50}')
    print(f'Унікальних stop_id у short: {stop_bld_short["stop_id"].nunique() if len(stop_bld_short) else 0}')
    print(f'Унікальних stop_id у long:  {stop_bld_long["stop_id"].nunique() if len(stop_bld_long) else 0}')
    print(f'Унікальних stop_id у exit:  {stop_fac_exit["stop_id"].nunique() if len(stop_fac_exit) else 0}')
    print(f'Середній walk_min short: {stop_bld_short["walk_min"].mean():.2f}' if len(stop_bld_short) else 'Середній walk_min short: n/a')
    print(f'Середній walk_min long:  {stop_bld_long["walk_min"].mean():.2f}' if len(stop_bld_long) else 'Середній walk_min long: n/a')
    print(f'Середній walk_min exit:  {stop_fac_exit["walk_min"].mean():.2f}' if len(stop_fac_exit) else 'Середній walk_min exit: n/a')


if __name__ == "__main__":
    run()
