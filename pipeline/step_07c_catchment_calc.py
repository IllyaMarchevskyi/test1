"""
07c Catchment Calculation.

Generated mechanically from notebooks/07c_catchment_calc.ipynb.
Run via main.py from the repository root.
"""


def run() -> None:
    # ---- Notebook code cell 2 ----
    from config_loader import cfg
    import os
    import pickle
    import warnings
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from joblib import Parallel, delayed

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import networkx as nx
    import osmnx as ox
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from tqdm.auto import tqdm

    warnings.filterwarnings('ignore')

    T_SHORT = cfg['catchment']['threshold_short_min']   # 10 хв
    T_LONG  = cfg['catchment']['threshold_long_min']    # 30 хв
    R_SHORT = T_SHORT * 75                              # 750 м
    R_LONG  = T_LONG  * 75                             # 2250 м
    R_EXIT  = (T_SHORT // 2) * 75                      # 375 м
    # R_EXIT  = R_LONG * 75                              # 375 м

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

    scores = pd.read_csv(cfg['paths']['scores'])
    print(f'Закладів: {len(scores)}')

    # ---- Notebook code cell 4 ----
    CATCHMENT_CACHE = '../data/processed/catchment_results.csv'
    BUILDINGS_CACHE = '../data/processed/catchment_buildings.parquet'

    _both_cached = os.path.exists(CATCHMENT_CACHE) and os.path.exists(BUILDINGS_CACHE)

    if _both_cached:
        catchment_df = pd.read_csv(CATCHMENT_CACHE)
        catchment_buildings = pd.read_parquet(BUILDINGS_CACHE)
        print(f'Catchment завантажено з кешу: {len(catchment_df)} закладів')
        print(f'  Buildings: {len(catchment_buildings):,} записів')

    else:
        _required = {
            'buildings': '../data/processed/buildings.parquet',
            'reach_peak_rev': '../data/processed/stop_reachability_peak_reversed.parquet',
            'reach_offpeak_rev': '../data/processed/stop_reachability_offpeak_reversed.parquet',
            'wait_peak': '../data/processed/wait_times_peak.parquet',
            'wait_offpeak': '../data/processed/wait_times_offpeak.parquet',
            'stop_bld_long': '../data/processed/stop_to_bld_long.parquet',
            'stop_fac_exit': '../data/processed/stop_to_fac_exit.parquet',
        }
        _missing = [k for k, p in _required.items() if not os.path.exists(p)]
        if _missing:
            raise FileNotFoundError(
                f'Відсутні precomputed файли: {_missing}\n'
                'Спочатку запустіть 07a та 07b.'
            )

        print('Завантаження precomputed даних...')
        buildings = gpd.read_parquet('../data/processed/buildings.parquet')
        buildings = buildings.reset_index(drop=True)

        reach_peak_rev = pd.read_parquet('../data/processed/stop_reachability_peak_reversed.parquet')
        reach_opk_rev = pd.read_parquet('../data/processed/stop_reachability_offpeak_reversed.parquet')
        wait_peak = pd.read_parquet('../data/processed/wait_times_peak.parquet')
        wait_offpeak = pd.read_parquet('../data/processed/wait_times_offpeak.parquet')
        stop_bld_long = pd.read_parquet('../data/processed/stop_to_bld_long.parquet')
        stop_fac_exit = pd.read_parquet('../data/processed/stop_to_fac_exit.parquet')
        scores = pd.read_csv('../data/processed/accessibility_scores.csv')

        for _df in [reach_peak_rev, reach_opk_rev, wait_peak, wait_offpeak, stop_bld_long, stop_fac_exit]:
            if 'stop_id' in _df.columns:
                _df['stop_id'] = _df['stop_id'].astype(str)
            for col in ['stop_A', 'stop_B']:
                if col in _df.columns:
                    _df[col] = _df[col].astype(str)

        stop_bld_long['building_id'] = stop_bld_long['building_id'].astype(int)
        stop_fac_exit['facility_id'] = stop_fac_exit['facility_id'].astype(str)

        if 'G_proj' not in vars():
            print('Проєктуємо граф у EPSG:32636...')
            G_proj = ox.project_graph(G_raw, to_crs='EPSG:32636')

        EDGE_BUFFER_M = 25

        print('Будуємо словники зворотніх матриць...')
        rev_peak_dict = {}
        for row in tqdm(reach_peak_rev.itertuples(index=False), total=len(reach_peak_rev), desc='reverse peak'):
            rev_peak_dict.setdefault(row.stop_B, {})[row.stop_A] = row.transit_min

        rev_opk_dict = {}
        for row in tqdm(reach_opk_rev.itertuples(index=False), total=len(reach_opk_rev), desc='reverse offpeak'):
            rev_opk_dict.setdefault(row.stop_B, {})[row.stop_A] = row.transit_min

        wait_peak_dict = {(row.stop_A, row.stop_B): row.avg_wait_min for row in wait_peak.itertuples(index=False)}
        wait_offpeak_dict = {(row.stop_A, row.stop_B): row.avg_wait_min for row in wait_offpeak.itertuples(index=False)}

        print('Будуємо словник stop -> buildings...')
        stop_bld_dict = {}
        for row in tqdm(stop_bld_long.itertuples(index=False), total=len(stop_bld_long), desc='stop -> buildings'):
            stop_bld_dict.setdefault(row.stop_id, {})[int(row.building_id)] = float(row.walk_min)

        print('Будуємо словник facility -> exit stops...')
        fac_stop_dict = {}
        for row in stop_fac_exit.itertuples(index=False):
            fac_stop_dict.setdefault(str(row.facility_id), {})[row.stop_id] = float(row.walk_min)

        peak_dest_stops = set(rev_peak_dict)
        peak_source_stops = {stop_a for targets in rev_peak_dict.values() for stop_a in targets}
        offpeak_dest_stops = set(rev_opk_dict)
        offpeak_source_stops = {stop_a for targets in rev_opk_dict.values() for stop_a in targets}
        stop_bld_stops = set(stop_bld_dict)
        fac_exit_stops = {stop_id for targets in fac_stop_dict.values() for stop_id in targets}

        print('Діагностика 07c: покриття словників')
        print(f'  stop_bld_dict stop_id: {len(stop_bld_stops):,}')
        print(f'  fac_stop_dict facility_id: {len(fac_stop_dict):,}')
        print(f'  fac_exit unique stop_id: {len(fac_exit_stops):,}')
        print(f'  rev_peak destinations: {len(peak_dest_stops):,} | sources: {len(peak_source_stops):,}')
        print(f'  rev_offpeak destinations: {len(offpeak_dest_stops):,} | sources: {len(offpeak_source_stops):,}')
        print(f'  Overlap building stops × peak sources: {len(stop_bld_stops & peak_source_stops):,}')
        print(f'  Overlap building stops × offpeak sources: {len(stop_bld_stops & offpeak_source_stops):,}')
        print(f'  Overlap exit stops × peak destinations: {len(fac_exit_stops & peak_dest_stops):,}')
        print(f'  Overlap exit stops × offpeak destinations: {len(fac_exit_stops & offpeak_dest_stops):,}')
        print(f"  Valid wait_peak pairs: {(wait_peak['avg_wait_min'] < 999).sum():,} / {len(wait_peak):,}")
        print(f"  Valid wait_offpeak pairs: {(wait_offpeak['avg_wait_min'] < 999).sum():,} / {len(wait_offpeak):,}")

        print('Precompute nearest nodes для будинків...')
        bld_xs = buildings.geometry.x.values
        bld_ys = buildings.geometry.y.values
        bld_nns = ox.distance.nearest_nodes(G_proj, X=bld_xs, Y=bld_ys)
        bld_by_node = {}
        for bid, nn in zip(buildings['building_id'].values, bld_nns):
            bld_by_node.setdefault(int(nn), []).append(int(bid))

        print('Precompute center_node для закладів...')
        fac_points = gpd.GeoDataFrame(
            scores[['facility_id', 'facility_type', 'name', 'lon', 'lat']].copy(),
            geometry=gpd.points_from_xy(scores['lon'], scores['lat']),
            crs='EPSG:4326'
        ).to_crs('EPSG:32636')
        fac_center_nodes = ox.distance.nearest_nodes(
            G_proj,
            X=fac_points.geometry.x.values,
            Y=fac_points.geometry.y.values,
        )
        facility_rows = []
        for row, center_node in zip(scores.itertuples(index=False), fac_center_nodes):
            facility_rows.append({
                'facility_id': str(row.facility_id),
                'facility_type': row.facility_type,
                'name': row.name,
                'center_node': int(center_node),
            })

        def count_transit(rev_dict, wait_dict, facility_id, threshold):
            stops_near_fac = fac_stop_dict.get(str(facility_id), {})
            if not stops_near_fac:
                return {}

            bld_min_time = {}

            # Сценарій 1: прямий маршрут stop_A -> stop_C без пересадки.
            for stop_c, walk_exit in stops_near_fac.items():
                direct_sources = rev_dict.get(stop_c, {})
                for stop_a, transit_ac in direct_sources.items():
                    # avg_wait_1 = wait_dict.get((stop_a, stop_c), 999.0)
                    # if avg_wait_1 >= 999:
                    #     continue
                    avg_wait_1 = 0.0
                    
                    fixed_time = avg_wait_1 + transit_ac + walk_exit
                    if fixed_time > threshold:
                        continue
                    for bid, walk_in in stop_bld_dict.get(stop_a, {}).items():
                        t_total = walk_in + fixed_time
                        if t_total <= threshold and t_total < bld_min_time.get(bid, float('inf')):
                            bld_min_time[bid] = t_total

            # Сценарій 2: максимум одна пересадка stop_A -> stop_B -> stop_C.
            for stop_c, walk_exit in stops_near_fac.items():
                second_leg_sources = rev_dict.get(stop_c, {})
                for stop_b, transit_bc in second_leg_sources.items():
                    if stop_b == stop_c:
                        continue
                    # avg_wait_2 = wait_dict.get((stop_b, stop_c), 999.0)
                    # if avg_wait_2 >= 999:
                    #     continue
                    avg_wait_2 = 0.0

                    second_leg_fixed = avg_wait_2 + transit_bc + walk_exit
                    if second_leg_fixed > threshold:
                        continue

                    first_leg_sources = rev_dict.get(stop_b, {})
                    if not first_leg_sources:
                        continue

                    for stop_a, transit_ab in first_leg_sources.items():
                        if stop_a == stop_b or stop_a == stop_c:
                            continue
                        # avg_wait_1 = wait_dict.get((stop_a, stop_b), 999.0)
                        # if avg_wait_1 >= 999:
                        #     continue

                        avg_wait_1 = 0.0

                        fixed_time = avg_wait_1 + transit_ab + second_leg_fixed
                        if fixed_time > threshold:
                            continue

                        for bid, walk_in in stop_bld_dict.get(stop_a, {}).items():
                            t_total = walk_in + fixed_time
                            if t_total <= threshold and t_total < bld_min_time.get(bid, float('inf')):
                                bld_min_time[bid] = t_total

            return bld_min_time

        def classify_group(time_min, mode):
            if mode == 'walk' and time_min <= T_SHORT:
                return GRP_WALK_SHORT
            if mode == 'transit' and time_min <= T_SHORT:
                return GRP_TRANSIT_SHORT
            if mode == 'walk' and time_min <= T_LONG:
                return GRP_WALK_LONG
            if mode == 'transit' and time_min <= T_LONG:
                return GRP_TRANSIT_LONG
            return None

        def process_facility(fac_row):
            fid = str(fac_row['facility_id'])
            center_node = int(fac_row['center_node'])

            local_diag = {
                'fac_total': 1,
                'fac_with_exit_stops': 1 if fac_stop_dict.get(fid) else 0,
                'fac_with_walk_buildings': 0,
                'fac_with_peak_transit': 0,
                'fac_with_offpeak_transit': 0,
                'peak_transit_buildings_total': 0,
                'offpeak_transit_buildings_total': 0,
                'peak_transit_short_total': 0,
                'peak_transit_long_total': 0,
                'offpeak_transit_short_total': 0,
                'offpeak_transit_long_total': 0,
            }

            _dists = nx.single_source_dijkstra_path_length(
                G_proj, center_node, cutoff=R_LONG, weight='length'
            )

            bld_walk_time = {}
            for node, dist_m in _dists.items():
                if node not in bld_by_node or dist_m > R_LONG:
                    continue
                walk_min = float(dist_m) / 75.0
                for bid in bld_by_node[node]:
                    if walk_min < bld_walk_time.get(bid, float('inf')):
                        bld_walk_time[bid] = walk_min

            transit_peak_times = count_transit(rev_peak_dict, wait_peak_dict, fid, T_LONG)
            transit_offpeak_times = count_transit(rev_opk_dict, wait_offpeak_dict, fid, T_LONG)

            if bld_walk_time:
                local_diag['fac_with_walk_buildings'] = 1
            if transit_peak_times:
                local_diag['fac_with_peak_transit'] = 1
            if transit_offpeak_times:
                local_diag['fac_with_offpeak_transit'] = 1
            local_diag['peak_transit_buildings_total'] = len(transit_peak_times)
            local_diag['offpeak_transit_buildings_total'] = len(transit_offpeak_times)

            best_peak = {bid: (walk_t, 'walk') for bid, walk_t in bld_walk_time.items()}
            best_offpeak = {bid: (walk_t, 'walk') for bid, walk_t in bld_walk_time.items()}

            for bid, transit_t in transit_peak_times.items():
                if transit_t < best_peak.get(bid, (float('inf'), 'walk'))[0]:
                    best_peak[bid] = (transit_t, 'transit')

            for bid, transit_t in transit_offpeak_times.items():
                if transit_t < best_offpeak.get(bid, (float('inf'), 'walk'))[0]:
                    best_offpeak[bid] = (transit_t, 'transit')

            peak_groups = {}
            for bid, (time_min, mode) in best_peak.items():
                group = classify_group(time_min, mode)
                if group is not None:
                    peak_groups[bid] = group

            offpeak_groups = {}
            for bid, (time_min, mode) in best_offpeak.items():
                group = classify_group(time_min, mode)
                if group is not None:
                    offpeak_groups[bid] = group

            peak_counts = pd.Series(list(peak_groups.values())).value_counts()
            offpeak_counts = pd.Series(list(offpeak_groups.values())).value_counts()
            local_diag['peak_transit_short_total'] = int(peak_counts.get(GRP_TRANSIT_SHORT, 0))
            local_diag['peak_transit_long_total'] = int(peak_counts.get(GRP_TRANSIT_LONG, 0))
            local_diag['offpeak_transit_short_total'] = int(offpeak_counts.get(GRP_TRANSIT_SHORT, 0))
            local_diag['offpeak_transit_long_total'] = int(offpeak_counts.get(GRP_TRANSIT_LONG, 0))

            result_row = {
                'facility_id': fid,
                'facility_type': fac_row['facility_type'],
                'name': fac_row['name'],
                f'peak_{GRP_WALK_SHORT}': int(peak_counts.get(GRP_WALK_SHORT, 0)),
                f'peak_{GRP_TRANSIT_SHORT}': int(peak_counts.get(GRP_TRANSIT_SHORT, 0)),
                f'peak_{GRP_WALK_LONG}': int(peak_counts.get(GRP_WALK_LONG, 0)),
                f'peak_{GRP_TRANSIT_LONG}': int(peak_counts.get(GRP_TRANSIT_LONG, 0)),
                f'peak_total_{T_SHORT}min': int(peak_counts.get(GRP_WALK_SHORT, 0) + peak_counts.get(GRP_TRANSIT_SHORT, 0)),
                f'peak_total_{T_LONG}min': int(peak_counts.sum()),
                f'offpeak_{GRP_WALK_SHORT}': int(offpeak_counts.get(GRP_WALK_SHORT, 0)),
                f'offpeak_{GRP_TRANSIT_SHORT}': int(offpeak_counts.get(GRP_TRANSIT_SHORT, 0)),
                f'offpeak_{GRP_WALK_LONG}': int(offpeak_counts.get(GRP_WALK_LONG, 0)),
                f'offpeak_{GRP_TRANSIT_LONG}': int(offpeak_counts.get(GRP_TRANSIT_LONG, 0)),
                f'offpeak_total_{T_SHORT}min': int(offpeak_counts.get(GRP_WALK_SHORT, 0) + offpeak_counts.get(GRP_TRANSIT_SHORT, 0)),
                f'offpeak_total_{T_LONG}min': int(offpeak_counts.sum()),
            }

            building_rows = []
            all_bids = set(peak_groups) | set(offpeak_groups)
            for bid in all_bids:
                building_rows.append((fid, bid, peak_groups.get(bid), offpeak_groups.get(bid)))

            return {
                'result_row': result_row,
                'building_rows': building_rows,
                'diag': local_diag,
            }

        def merge_diag(diag_total, diag_local):
            for key, value in diag_local.items():
                diag_total[key] += value

        results = []
        all_bld_rows = []
        diag = {
            'fac_total': 0,
            'fac_with_exit_stops': 0,
            'fac_with_walk_buildings': 0,
            'fac_with_peak_transit': 0,
            'fac_with_offpeak_transit': 0,
            'peak_transit_buildings_total': 0,
            'offpeak_transit_buildings_total': 0,
            'peak_transit_short_total': 0,
            'peak_transit_long_total': 0,
            'offpeak_transit_short_total': 0,
            'offpeak_transit_long_total': 0,
        }

        import multiprocessing as mp

        # Зміни це число, якщо треба більше/менше паралельності.
        # На Windows краще починати з 2-4: кожен процес отримує великі словники.
        PARALLEL_WORKERS = 12

        if 'fork' in mp.get_all_start_methods():
            PARALLEL_BACKEND = 'processpool_fork'
            mp_ctx = mp.get_context('fork')
        elif PARALLEL_WORKERS > 1:
            PARALLEL_BACKEND = 'joblib_loky'
            mp_ctx = None
        else:
            PARALLEL_BACKEND = 'serial'
            mp_ctx = None

        print(f'Розрахунок catchment: {PARALLEL_WORKERS} процес(и), backend={PARALLEL_BACKEND}')

        def collect_payload(payload):
            results.append(payload['result_row'])
            all_bld_rows.extend(payload['building_rows'])
            merge_diag(diag, payload['diag'])

        success_count = 0
        error_count = 0
        catchment_errors = []

        def safe_process_facility(fac_row):
            fid = str(fac_row.get('facility_id', 'unknown'))
            try:
                return {'ok': True, 'facility_id': fid, 'payload': process_facility(fac_row), 'error': None}
            except Exception as e:
                return {'ok': False, 'facility_id': fid, 'payload': None, 'error': str(e)}

        def handle_result(item):
            if item['ok']:
                collect_payload(item['payload'])
                return True
            catchment_errors.append((item['facility_id'], item['error']))
            return False

        if PARALLEL_BACKEND == 'processpool_fork':
            with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS, mp_context=mp_ctx) as executor:
                futures = [executor.submit(safe_process_facility, fac_row) for fac_row in facility_rows]
                progress = tqdm(as_completed(futures), total=len(futures), desc='Catchment закладів')
                for future in progress:
                    try:
                        if handle_result(future.result()):
                            success_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        error_count += 1
                        catchment_errors.append(('executor', str(e)))
                    progress.set_postfix({'success': success_count, 'errors': error_count})
        elif PARALLEL_BACKEND == 'joblib_loky':
            payloads = Parallel(n_jobs=PARALLEL_WORKERS, backend='loky', return_as='generator_unordered')(
                delayed(safe_process_facility)(fac_row)
                for fac_row in facility_rows
            )
            progress = tqdm(payloads, total=len(facility_rows), desc='Catchment закладів')
            for item in progress:
                if handle_result(item):
                    success_count += 1
                else:
                    error_count += 1
                progress.set_postfix({'success': success_count, 'errors': error_count})
        else:
            progress = tqdm(facility_rows, total=len(facility_rows), desc='Catchment закладів')
            for fac_row in progress:
                item = safe_process_facility(fac_row)
                if handle_result(item):
                    success_count += 1
                else:
                    error_count += 1
                progress.set_postfix({'success': success_count, 'errors': error_count})

        print(f'Catchment виконано: success={success_count}, errors={error_count}')
        if catchment_errors:
            print(f'Перші помилки: {catchment_errors[:3]}')

        catchment_df = pd.DataFrame(results)
        catchment_df.to_csv(CATCHMENT_CACHE, index=False, encoding='utf-8')
        print(f'\ncatchment_results збережено: {len(catchment_df)} закладів -> {CATCHMENT_CACHE}')

        catchment_buildings = pd.DataFrame(
            all_bld_rows,
            columns=['facility_id', 'building_id', 'group_peak', 'group_offpeak']
        )
        catchment_buildings.to_parquet(BUILDINGS_CACHE, index=False)
        print(f'catchment_buildings збережено: {len(catchment_buildings):,} записів -> {BUILDINGS_CACHE}')
        print(f'Оброблено закладів: {len(catchment_df)}')
        print('\nДіагностика 07c: підсумок по закладах')
        print(f"  Закладів усього: {diag['fac_total']}")
        print(f"  Є exit-зупинки: {diag['fac_with_exit_stops']}")
        print(f"  Є walk-досяжні будинки: {diag['fac_with_walk_buildings']}")
        print(f"  Є peak transit-кандидати: {diag['fac_with_peak_transit']}")
        print(f"  Є offpeak transit-кандидати: {diag['fac_with_offpeak_transit']}")
        print(f"  Peak transit buildings total: {diag['peak_transit_buildings_total']:,}")
        print(f"  Offpeak transit buildings total: {diag['offpeak_transit_buildings_total']:,}")
        print(f"  Peak transit short assigned: {diag['peak_transit_short_total']:,}")
        print(f"  Peak transit long assigned: {diag['peak_transit_long_total']:,}")
        print(f"  Offpeak transit short assigned: {diag['offpeak_transit_short_total']:,}")
        print(f"  Offpeak transit long assigned: {diag['offpeak_transit_long_total']:,}")


if __name__ == "__main__":
    run()
