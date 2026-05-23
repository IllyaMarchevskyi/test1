"""
07b Transit Matrix.

Generated mechanically from notebooks/07b_transit_matrix.ipynb.
Run via main.py from the repository root.
"""


def run() -> None:
    # ---- Notebook code cell 2 ----
    from config_loader import cfg
    import os
    import warnings

    import pandas as pd
    from tqdm.auto import tqdm

    warnings.filterwarnings('ignore')

    T_SHORT = cfg['catchment']['threshold_short_min']   # 10 хв
    T_LONG  = cfg['catchment']['threshold_long_min']    # 30 хв
    R_SHORT = T_SHORT * 75                              # 750 м
    R_LONG  = T_LONG  * 75                             # 2250 м
    R_EXIT  = (T_SHORT // 2) * 75                      # 375 м

    GRP_WALK_SHORT    = f'walk_{T_SHORT}min'
    GRP_TRANSIT_SHORT = f'transit_{T_SHORT}min'
    GRP_WALK_LONG     = f'walk_{T_LONG}min'
    GRP_TRANSIT_LONG  = f'transit_{T_LONG}min'

    print(f'Пороги: {T_SHORT} хв / {T_LONG} хв')

    # ---- Notebook code cell 4 ----
    EASYWAY_PATH   = '../gtfs_static/kyiv-routes.csv'
    CACHE_PEAK     = '../data/processed/stop_reachability_peak.parquet'
    CACHE_OFFPEAK  = '../data/processed/stop_reachability_offpeak.parquet'
    CACHE_PEAK_REV = '../data/processed/stop_reachability_peak_reversed.parquet'
    CACHE_OPK_REV  = '../data/processed/stop_reachability_offpeak_reversed.parquet'

    # -- Завантажуємо easyway (завжди -- потрібне також у Секції 4) ----------------
    print('Завантажуємо kyivroutes...')
    easyway = pd.read_csv(EASYWAY_PATH)
    easyway = easyway[easyway['schedules'] != r'\N'].copy()

    def parse_schedules(s):
        """Парсимо рядок часів у відсортований список секунд від опівночі."""
        times = []
        for t in str(s).strip().split(','):
            t = t.strip()
            if not t or t == r'\N':
                continue
            parts = t.split(':')
            sec = int(parts[0]) * 3600 + int(parts[1]) * 60
            times.append(sec)
        return sorted(times)

    easyway['times']   = easyway['schedules'].apply(parse_schedules)
    # Нормалізуємо stop_id до str (узгодженість з OSM-зупинками)
    easyway['stop_id'] = easyway['stop_id'].astype(str)
    print(f'Рядків з розкладом: {len(easyway):,}')

    _all_cached_3 = all(os.path.exists(p) for p in
                        [CACHE_PEAK, CACHE_OFFPEAK, CACHE_PEAK_REV, CACHE_OPK_REV])

    if _all_cached_3:
        reach_peak     = pd.read_parquet(CACHE_PEAK)
        reach_offpeak  = pd.read_parquet(CACHE_OFFPEAK)
        reach_peak_rev = pd.read_parquet(CACHE_PEAK_REV)
        reach_opk_rev  = pd.read_parquet(CACHE_OPK_REV)
        print('Завантажено матриці досяжності з кешу')
        for name, df in [('peak', reach_peak), ('offpeak', reach_offpeak),
                         ('peak_rev', reach_peak_rev), ('opk_rev', reach_opk_rev)]:
            print(f'  {name}: {len(df):,} пар')

    else:
        # -- Часові вікна (секунди від опівночі) ------------------------------------------
        def hhmm_to_sec(t: str) -> int:
            h, m = map(int, t.split(':'))
            return h * 3600 + m * 60

        PEAK_WIN = [
            (hhmm_to_sec(cfg['peak_hours']['morning_start']),
             hhmm_to_sec(cfg['peak_hours']['morning_end'])),    # 07:00-09:00
            (hhmm_to_sec(cfg['peak_hours']['evening_start']),
             hhmm_to_sec(cfg['peak_hours']['evening_end'])),    # 17:00-19:00
        ]
        OP_START = hhmm_to_sec(cfg['offpeak_hours']['start'])   # 10:00
        OP_END   = hhmm_to_sec(cfg['offpeak_hours']['end'])     # 17:00

        def in_peak(sec):
            return any(s <= sec <= e for s, e in PEAK_WIN)

        def in_offpeak(sec):
            return OP_START <= sec <= OP_END

        # -- Побудова матриць досяжності з kyivroutes ----------------------------------------
        # {stop_A_id: {stop_B_id: min_transit_min}}
        stop_reachability_peak    = {}
        stop_reachability_offpeak = {}

        groups = easyway.groupby(['route_id', 'direction', 'calendar'])
        print(f'Обробляємо {len(groups)} груп (маршрут x напрям x календар)...')

        for (route_id, direction, calendar), group in tqdm(
                groups, desc='Маршрути', total=len(groups)):
            stops     = group.sort_values('index')
            stop_list = stops.to_dict('records')

            for i, stop_A in enumerate(stop_list):
                sid_A   = stop_A['stop_id']
                times_A = stop_A['times']
                if not times_A:
                    continue

                for stop_B in stop_list[i + 1:]:
                    sid_B   = stop_B['stop_id']
                    times_B = stop_B['times']
                    if not times_B:
                        continue

                    # i-й час на A і i-й час на B -- це ОДИН рейс
                    n_trips          = min(len(times_A), len(times_B))
                    peak_transits    = []
                    offpeak_transits = []

                    for k in range(n_trips):
                        depart_A    = times_A[k]
                        arrive_B    = times_B[k]
                        transit_min = (arrive_B - depart_A) / 60
                        if transit_min <= 0:
                            continue
                        # Пікові години враховуємо тільки для будніх маршрутів
                        if calendar in ('Weekdays', 'All Week') and in_peak(depart_A):
                            peak_transits.append(transit_min)
                        if calendar in ('Weekdays', 'All Week') and in_offpeak(depart_A):
                            offpeak_transits.append(transit_min)

                    # Зберігаємо мінімальний час для пари
                    if peak_transits:
                        prev  = stop_reachability_peak.get(sid_A, {}).get(sid_B, 999)
                        new_t = min(peak_transits)
                        if new_t < prev:
                            stop_reachability_peak.setdefault(sid_A, {})[sid_B] = new_t

                    if offpeak_transits:
                        prev  = stop_reachability_offpeak.get(sid_A, {}).get(sid_B, 999)
                        new_t = min(offpeak_transits)
                        if new_t < prev:
                            stop_reachability_offpeak.setdefault(sid_A, {})[sid_B] = new_t

        # -- Зворотні матриці (індексація по destination) ----------------------------------------
        reversed_peak    = {}
        reversed_offpeak = {}

        for sid_A, targets in stop_reachability_peak.items():
            for sid_B, t in targets.items():
                prev = reversed_peak.get(sid_B, {}).get(sid_A, 999)
                if t < prev:
                    reversed_peak.setdefault(sid_B, {})[sid_A] = t

        for sid_A, targets in stop_reachability_offpeak.items():
            for sid_B, t in targets.items():
                prev = reversed_offpeak.get(sid_B, {}).get(sid_A, 999)
                if t < prev:
                    reversed_offpeak.setdefault(sid_B, {})[sid_A] = t

        # -- Конвертуємо у DataFrame і зберігаємо -------------------------------------------------
        def dict_to_df(d):
            return pd.DataFrame([
                {'stop_A': a, 'stop_B': b, 'transit_min': t}
                for a, targets in d.items()
                for b, t in targets.items()
            ])

        reach_peak     = dict_to_df(stop_reachability_peak)
        reach_offpeak  = dict_to_df(stop_reachability_offpeak)
        reach_peak_rev = dict_to_df(reversed_peak)
        reach_opk_rev  = dict_to_df(reversed_offpeak)

        for path, df, label in [
            (CACHE_PEAK,     reach_peak,     'peak'),
            (CACHE_OFFPEAK,  reach_offpeak,  'offpeak'),
            (CACHE_PEAK_REV, reach_peak_rev, 'peak_rev'),
            (CACHE_OPK_REV,  reach_opk_rev,  'opk_rev'),
        ]:
            df.to_parquet(path, index=False)
            print(f'  {label}: {len(df):,} пар -> {path}')

    # -- Підсумок -----------------------------------------------------------------------
    print(f'\n{chr(8212)*50}')
    print(f'Матриця пік:    {len(reach_peak):,} пар '
          f'({reach_peak["stop_A"].nunique()} зупинок-джерел)')
    print(f'Матриця міжпік: {len(reach_offpeak):,} пар '
          f'({reach_offpeak["stop_A"].nunique()} зупинок-джерел)')
    if len(reach_peak):
        print(f'Середній час у дорозі пік:     '
              f'{reach_peak["transit_min"].mean():.1f} хв')
    if len(reach_offpeak):
        print(f'Середній час у дорозі міжпік:  '
              f'{reach_offpeak["transit_min"].mean():.1f} хв')

    # ---- Notebook code cell 6 ----
    FORCE_RECOMPUTE   = False
    CACHE_WAIT_PEAK    = '../data/processed/wait_times_peak.parquet'
    CACHE_WAIT_OFFPEAK = '../data/processed/wait_times_offpeak.parquet'

    _all_cached_4 = (not FORCE_RECOMPUTE) and all(
        os.path.exists(p) for p in [CACHE_WAIT_PEAK, CACHE_WAIT_OFFPEAK]
    )

    if _all_cached_4:
        wait_peak    = pd.read_parquet(CACHE_WAIT_PEAK)
        wait_offpeak = pd.read_parquet(CACHE_WAIT_OFFPEAK)
        print('Завантажено часи очікування з кешу')

    else:
        # -- Часові вікна ---------------------------------------------------------------
        if 'hhmm_to_sec' not in vars():
            def hhmm_to_sec(t: str) -> int:
                h, m = map(int, t.split(':'))
                return h * 3600 + m * 60

        PEAK_START_1  = hhmm_to_sec(cfg['peak_hours']['morning_start'])    # 07:00
        PEAK_END_1    = hhmm_to_sec(cfg['peak_hours']['morning_end'])      # 09:00
        PEAK_START_2  = hhmm_to_sec(cfg['peak_hours']['evening_start'])    # 17:00
        PEAK_END_2    = hhmm_to_sec(cfg['peak_hours']['evening_end'])      # 19:00
        OFFPEAK_START = hhmm_to_sec(cfg['offpeak_hours']['start'])         # 10:00
        OFFPEAK_END   = hhmm_to_sec(cfg['offpeak_hours']['end'])           # 17:00

        def in_peak(sec):
            return ((PEAK_START_1 <= sec <= PEAK_END_1) or
                    (PEAK_START_2 <= sec <= PEAK_END_2))

        def in_offpeak(sec):
            return OFFPEAK_START <= sec <= OFFPEAK_END

        # -- Відновлення easyway якщо Секція 3 завантажила з кешу -----------------------
        if 'easyway' not in vars():
            print('Відновлюємо easyway для Секції 4...')
            if 'parse_schedules' not in vars():
                def parse_schedules(s):
                    times = []
                    for t in str(s).strip().split(','):
                        t = t.strip()
                        if not t or t == r'\N':
                            continue
                        parts = t.split(':')
                        sec = int(parts[0]) * 3600 + int(parts[1]) * 60
                        times.append(sec)
                    return sorted(times)
            easyway = pd.read_csv('../gtfs_static/kyiv-routes.csv')
            easyway = easyway[easyway['schedules'] != r'\N'].copy()
            easyway['times']   = easyway['schedules'].apply(parse_schedules)
            easyway['stop_id'] = easyway['stop_id'].astype(str)

        # -- Передобробка: stop_id x (маршрут, напрям, календар) -> часи ----------------
        stop_route_times = {}
        grouped_routes = easyway.groupby(['route_id', 'direction', 'calendar'])

        for (route_id, direction, calendar), group in grouped_routes:
            route_key = (route_id, direction, calendar)
            stops = group.sort_values('index')
            for _, row in stops.iterrows():
                key = (str(row['stop_id']), route_key)
                stop_route_times[key] = row['times']

        # -- Збираємо відправлення тільки для рейсів, що реально з'єднують пару ---------
        pair_depart_times_peak = {}
        pair_depart_times_offpeak = {}
        grouped_routes = easyway.groupby(['route_id', 'direction', 'calendar'])

        for (route_id, direction, calendar), group in tqdm(
                grouped_routes, desc='Пари відправлень'):
            route_key = (route_id, direction, calendar)
            stops = group.sort_values('index')
            stop_list = stops.to_dict('records')

            for i, stop_A in enumerate(stop_list):
                sid_A = str(stop_A['stop_id'])
                times_A = stop_route_times.get((sid_A, route_key), [])
                if not times_A:
                    continue

                for stop_B in stop_list[i + 1:]:
                    sid_B = str(stop_B['stop_id'])
                    times_B = stop_route_times.get((sid_B, route_key), [])
                    if not times_B:
                        continue

                    n_trips = min(len(times_A), len(times_B))
                    for k in range(n_trips):
                        depart_A = times_A[k]
                        arrive_B = times_B[k]
                        if arrive_B <= depart_A:
                            continue
                        if calendar in ('Weekdays', 'All Week') and in_peak(depart_A):
                            pair_depart_times_peak.setdefault(
                                (sid_A, sid_B), []).append(depart_A)
                        if calendar in ('Weekdays', 'All Week') and in_offpeak(depart_A):
                            pair_depart_times_offpeak.setdefault(
                                (sid_A, sid_B), []).append(depart_A)

        # -- Середній час очікування рахуємо тільки з відправлень потрібної пари ---------
        PEAK_WINDOW_DURATION_MIN = ((PEAK_END_1 - PEAK_START_1) + (PEAK_END_2 - PEAK_START_2)) / 60
        OFFPEAK_WINDOW_DURATION_MIN = (OFFPEAK_END - OFFPEAK_START) / 60

        def calc_avg_wait_from_departures(departures, fallback_window_min):
            if not departures:
                return 999.0
            sorted_departures = sorted(set(departures))
            if len(sorted_departures) == 1:
                return fallback_window_min / 2
            intervals = [
                (sorted_departures[k + 1] - sorted_departures[k]) / 60
                for k in range(len(sorted_departures) - 1)
            ]
            return sum(intervals) / len(intervals) / 2

        wait_peak_dict = {}
        wait_offpeak_dict = {}

        print('Розраховуємо часи очікування (пік)...')
        for _, row in tqdm(reach_peak.iterrows(),
                           total=len(reach_peak), desc='Wait peak'):
            sid_A = str(row['stop_A'])
            sid_B = str(row['stop_B'])
            wait_peak_dict[(sid_A, sid_B)] = calc_avg_wait_from_departures(
                pair_depart_times_peak.get((sid_A, sid_B), []),
                PEAK_WINDOW_DURATION_MIN,
            )

        print('Розраховуємо часи очікування (міжпік)...')
        for _, row in tqdm(reach_offpeak.iterrows(),
                           total=len(reach_offpeak), desc='Wait offpeak'):
            sid_A = str(row['stop_A'])
            sid_B = str(row['stop_B'])
            wait_offpeak_dict[(sid_A, sid_B)] = calc_avg_wait_from_departures(
                pair_depart_times_offpeak.get((sid_A, sid_B), []),
                OFFPEAK_WINDOW_DURATION_MIN,
            )

        # -- Конвертуємо у DataFrame і зберігаємо ---------------------------------------
        wait_peak = pd.DataFrame([
            {'stop_A': a, 'stop_B': b, 'avg_wait_min': w}
            for (a, b), w in wait_peak_dict.items()
        ])
        wait_offpeak = pd.DataFrame([
            {'stop_A': a, 'stop_B': b, 'avg_wait_min': w}
            for (a, b), w in wait_offpeak_dict.items()
        ])

        wait_peak.to_parquet(CACHE_WAIT_PEAK, index=False)
        wait_offpeak.to_parquet(CACHE_WAIT_OFFPEAK, index=False)
        print(f'Кеш збережено: {CACHE_WAIT_PEAK}')
        print(f'Кеш збережено: {CACHE_WAIT_OFFPEAK}')

    # -- Підсумок -----------------------------------------------------------------------
    NO_SERVICE_MIN = 999
    _wp = wait_peak[wait_peak['avg_wait_min'] < NO_SERVICE_MIN]
    _wo = wait_offpeak[wait_offpeak['avg_wait_min'] < NO_SERVICE_MIN]

    print(f'\n{chr(8212)*50}')
    print(f'Матриця пік:    {len(reach_peak):,} пар  |  '
          f'Часів очікування: {len(wait_peak):,}')
    print(f'Матриця міжпік: {len(reach_offpeak):,} пар  |  '
          f'Часів очікування: {len(wait_offpeak):,}')
    print()

    peak_src = set(reach_peak['stop_A'].astype(str)) if len(reach_peak) else set()
    peak_dst = set(reach_peak['stop_B'].astype(str)) if len(reach_peak) else set()
    offpeak_src = set(reach_offpeak['stop_A'].astype(str)) if len(reach_offpeak) else set()
    offpeak_dst = set(reach_offpeak['stop_B'].astype(str)) if len(reach_offpeak) else set()

    print('Діагностика 07b:')
    print(f'  Peak stop_A: {len(peak_src):,}  | stop_B: {len(peak_dst):,}')
    print(f'  Offpeak stop_A: {len(offpeak_src):,}  | stop_B: {len(offpeak_dst):,}')
    print(f'  Valid wait peak: {_wp.shape[0]:,} / {len(wait_peak):,} пар')
    print(f'  Valid wait offpeak: {_wo.shape[0]:,} / {len(wait_offpeak):,} пар')
    if len(wait_peak):
        print(f'  Частка valid wait peak: {_wp.shape[0] / len(wait_peak) * 100:.1f}%')
    if len(wait_offpeak):
        print(f'  Частка valid wait offpeak: {_wo.shape[0] / len(wait_offpeak) * 100:.1f}%')
    if len(_wp):
        print(f"  Середній wait peak (valid): {_wp['avg_wait_min'].mean():.2f} хв")
    if len(_wo):
        print(f"  Середній wait offpeak (valid): {_wo['avg_wait_min'].mean():.2f} хв")
    if len(_wp):
        print(f'Середній час очікування (пік,    доступні пари): '
              f'{_wp["avg_wait_min"].mean():.1f} хв  '
              f'[медіана {_wp["avg_wait_min"].median():.1f} хв]')
    if len(_wo):
        print(f'Середній час очікування (міжпік, доступні пари): '
              f'{_wo["avg_wait_min"].mean():.1f} хв  '
              f'[медіана {_wo["avg_wait_min"].median():.1f} хв]')
    print(f'\nПар недоступних (999) пік:    '
          f'{(wait_peak["avg_wait_min"] >= NO_SERVICE_MIN).sum()}')
    print(f'Пар недоступних (999) міжпік: '
          f'{(wait_offpeak["avg_wait_min"] >= NO_SERVICE_MIN).sum()}')


if __name__ == "__main__":
    run()
