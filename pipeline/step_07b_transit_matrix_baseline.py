"""
07b Baseline Transit Matrix.

Baseline branch for direct transit without transfers.
Keeps old 07b intact and writes separate baseline caches.
"""


def run() -> None:
    from config_loader import cfg
    import os
    import warnings

    import pandas as pd
    from tqdm.auto import tqdm

    warnings.filterwarnings("ignore")

    MIN_TRANSIT_MIN = 1.5

    EASYWAY_PATH = "../gtfs_static/easyway_routes.csv"
    PROCESSED_DIR = "./data/processed"
    CACHE_PEAK = f"{PROCESSED_DIR}/stop_reachability_peak_baseline.parquet"
    CACHE_OFFPEAK = f"{PROCESSED_DIR}/stop_reachability_offpeak_baseline.parquet"
    CACHE_PEAK_REV = f"{PROCESSED_DIR}/stop_reachability_peak_reversed_baseline.parquet"
    CACHE_OPK_REV = f"{PROCESSED_DIR}/stop_reachability_offpeak_reversed_baseline.parquet"
    CACHE_WAIT_PEAK = f"{PROCESSED_DIR}/wait_times_peak_baseline.parquet"
    CACHE_WAIT_OFFPEAK = f"{PROCESSED_DIR}/wait_times_offpeak_baseline.parquet"

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"Baseline 07b: мінімальний transit_min = {MIN_TRANSIT_MIN} хв")

    def hhmm_to_sec(value: str) -> int:
        hour, minute = map(int, value.split(":"))
        return hour * 3600 + minute * 60

    peak_windows = [
        (hhmm_to_sec(cfg["peak_hours"]["morning_start"]), hhmm_to_sec(cfg["peak_hours"]["morning_end"])),
        (hhmm_to_sec(cfg["peak_hours"]["evening_start"]), hhmm_to_sec(cfg["peak_hours"]["evening_end"])),
    ]
    offpeak_start = hhmm_to_sec(cfg["offpeak_hours"]["start"])
    offpeak_end = hhmm_to_sec(cfg["offpeak_hours"]["end"])

    def in_peak(sec: int) -> bool:
        return any(start <= sec <= end for start, end in peak_windows)

    def in_offpeak(sec: int) -> bool:
        return offpeak_start <= sec <= offpeak_end

    def parse_schedules(value: str) -> list[int]:
        times = []
        for raw in str(value).strip().split(","):
            raw = raw.strip()
            if not raw or raw == r"\N":
                continue
            hh, mm, ss = raw.split(":")
            times.append(int(hh) * 3600 + int(mm) * 60 + int(ss))
        return sorted(times)

    def route_label(row: pd.Series | dict) -> str:
        transport = str(row["transport"]).strip()
        route = str(row["route"]).strip()
        return " ".join(part for part in [transport, route] if part)

    def dict_to_df(data: dict[str, dict[str, dict[str, object]]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"stop_A": stop_a, "stop_B": stop_b, **payload}
                for stop_a, targets in data.items()
                for stop_b, payload in targets.items()
            ]
        )

    def reversed_dict_to_df(data: dict[str, dict[str, dict[str, object]]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"stop_A": stop_src, "stop_B": stop_dest, **payload}
                for stop_dest, sources in data.items()
                for stop_src, payload in sources.items()
            ]
        )

    def cached_outputs_valid() -> bool:
        try:
            reach_peak_rev = pd.read_parquet(CACHE_PEAK_REV, columns=["stop_A", "stop_B"])
            wait_peak = pd.read_parquet(CACHE_WAIT_PEAK, columns=["stop_A", "stop_B", "avg_wait_min"])
        except Exception:
            return False

        if reach_peak_rev.empty or wait_peak.empty:
            return False

        merged = reach_peak_rev.merge(wait_peak, on=["stop_A", "stop_B"], how="left")
        valid_share = float((merged["avg_wait_min"].fillna(999.0) < 999.0).mean())
        print(f"Baseline 07b: cache validation share={valid_share:.3f}")
        return valid_share >= 0.5

    print(f"Baseline 07b: завантажуємо {EASYWAY_PATH}")
    easyway = pd.read_csv(EASYWAY_PATH)
    easyway = easyway[easyway["schedules"] != r"\N"].copy()
    easyway["stop_id"] = easyway["stop_id"].astype(str)
    easyway["times"] = easyway["schedules"].apply(parse_schedules)
    print(f"Рядків з розкладом: {len(easyway):,}")

    FORCE_RECOMPUTE = False
    caches_ready = (not FORCE_RECOMPUTE) and all(
        os.path.exists(path)
        for path in [CACHE_PEAK, CACHE_OFFPEAK, CACHE_PEAK_REV, CACHE_OPK_REV, CACHE_WAIT_PEAK, CACHE_WAIT_OFFPEAK]
    )
    if caches_ready:
        if cached_outputs_valid():
            print("Baseline 07b кеш уже існує, пропускаємо перебудову.")
            return
        print("Baseline 07b: кеш застарілий або неузгоджений, перебудовуємо.")

    stop_reachability_peak = {}
    stop_reachability_offpeak = {}
    pair_depart_times_peak_morning = {}
    pair_depart_times_peak_evening = {}
    pair_depart_times_offpeak = {}

    groups = easyway.groupby(["route_id", "direction", "calendar"])
    print(f"Груп маршрутів: {len(groups)}")

    for (_, _, calendar), group in tqdm(groups, total=len(groups), desc="Baseline 07b маршрути"):
        stops = group.sort_values("index")
        stop_list = stops.to_dict("records")
        for i, stop_a in enumerate(stop_list):
            sid_a = str(stop_a["stop_id"])
            times_a = stop_a["times"]
            if not times_a:
                continue
            for stop_b in stop_list[i + 1:]:
                sid_b = str(stop_b["stop_id"])
                times_b = stop_b["times"]
                if not times_b:
                    continue

                n_trips = min(len(times_a), len(times_b))
                for trip_idx in range(n_trips):
                    depart_a = times_a[trip_idx]
                    arrive_b = times_b[trip_idx]
                    transit_min = (arrive_b - depart_a) / 60.0
                    if transit_min <= 0:
                        continue
                    if transit_min < MIN_TRANSIT_MIN:
                        continue

                    if calendar in ("Weekdays", "All Week") and in_peak(depart_a):
                        prev_payload = stop_reachability_peak.setdefault(sid_a, {}).get(sid_b)
                        label = route_label(stop_a)
                        if prev_payload is None:
                            prev_payload = {
                                "transit_min": transit_min,
                                "route_id": str(stop_a["route_id"]),
                                "route": str(stop_a["route"]),
                                "transport": str(stop_a["transport"]),
                                "direction": str(stop_a["direction"]),
                                "route_options_set": {label},
                            }
                            stop_reachability_peak[sid_a][sid_b] = prev_payload
                        else:
                            prev_payload.setdefault("route_options_set", set()).add(label)
                        prev = float(prev_payload["transit_min"]) if prev_payload is not None else 999.0
                        if transit_min < prev:
                            prev_payload["transit_min"] = transit_min
                            prev_payload["route_id"] = str(stop_a["route_id"])
                            prev_payload["route"] = str(stop_a["route"])
                            prev_payload["transport"] = str(stop_a["transport"])
                            prev_payload["direction"] = str(stop_a["direction"])
                        if peak_windows[0][0] <= depart_a <= peak_windows[0][1]:
                            pair_depart_times_peak_morning.setdefault((sid_a, sid_b), []).append(depart_a)
                        elif peak_windows[1][0] <= depart_a <= peak_windows[1][1]:
                            pair_depart_times_peak_evening.setdefault((sid_a, sid_b), []).append(depart_a)

                    if calendar in ("Weekdays", "All Week") and in_offpeak(depart_a):
                        prev_payload = stop_reachability_offpeak.setdefault(sid_a, {}).get(sid_b)
                        label = route_label(stop_a)
                        if prev_payload is None:
                            prev_payload = {
                                "transit_min": transit_min,
                                "route_id": str(stop_a["route_id"]),
                                "route": str(stop_a["route"]),
                                "transport": str(stop_a["transport"]),
                                "direction": str(stop_a["direction"]),
                                "route_options_set": {label},
                            }
                            stop_reachability_offpeak[sid_a][sid_b] = prev_payload
                        else:
                            prev_payload.setdefault("route_options_set", set()).add(label)
                        prev = float(prev_payload["transit_min"]) if prev_payload is not None else 999.0
                        if transit_min < prev:
                            prev_payload["transit_min"] = transit_min
                            prev_payload["route_id"] = str(stop_a["route_id"])
                            prev_payload["route"] = str(stop_a["route"])
                            prev_payload["transport"] = str(stop_a["transport"])
                            prev_payload["direction"] = str(stop_a["direction"])
                        pair_depart_times_offpeak.setdefault((sid_a, sid_b), []).append(depart_a)

    for reachability in [stop_reachability_peak, stop_reachability_offpeak]:
        for targets in reachability.values():
            for payload in targets.values():
                options = sorted(payload.pop("route_options_set", set()))
                payload["route_options"] = ", ".join(options)

    reversed_peak = {}
    reversed_offpeak = {}

    for sid_a, targets in stop_reachability_peak.items():
        for sid_b, payload in targets.items():
            transit_min = float(payload["transit_min"])
            prev_payload = reversed_peak.get(sid_b, {}).get(sid_a)
            prev = float(prev_payload["transit_min"]) if prev_payload is not None else 999.0
            if transit_min < prev:
                reversed_peak.setdefault(sid_b, {})[sid_a] = payload.copy()

    for sid_a, targets in stop_reachability_offpeak.items():
        for sid_b, payload in targets.items():
            transit_min = float(payload["transit_min"])
            prev_payload = reversed_offpeak.get(sid_b, {}).get(sid_a)
            prev = float(prev_payload["transit_min"]) if prev_payload is not None else 999.0
            if transit_min < prev:
                reversed_offpeak.setdefault(sid_b, {})[sid_a] = payload.copy()

    reach_peak = dict_to_df(stop_reachability_peak)
    reach_offpeak = dict_to_df(stop_reachability_offpeak)
    reach_peak_rev = reversed_dict_to_df(reversed_peak)
    reach_opk_rev = reversed_dict_to_df(reversed_offpeak)

    morning_window_duration_min = (peak_windows[0][1] - peak_windows[0][0]) / 60.0
    evening_window_duration_min = (peak_windows[1][1] - peak_windows[1][0]) / 60.0
    offpeak_window_duration_min = (offpeak_end - offpeak_start) / 60.0

    def calc_avg_wait_window(departures: list[int], fallback_window_min: float) -> float:
        if not departures:
            return 999.0
        departures = sorted(set(departures))
        if len(departures) == 1:
            return fallback_window_min / 2.0
        intervals = [(departures[i + 1] - departures[i]) / 60.0 for i in range(len(departures) - 1)]
        return sum(intervals) / len(intervals) / 2.0

    def calc_peak_wait(stop_a: str, stop_b: str) -> float:
        morning_wait = calc_avg_wait_window(
            pair_depart_times_peak_morning.get((stop_a, stop_b), []),
            morning_window_duration_min,
        )
        evening_wait = calc_avg_wait_window(
            pair_depart_times_peak_evening.get((stop_a, stop_b), []),
            evening_window_duration_min,
        )
        valid = [value for value in [morning_wait, evening_wait] if value < 999.0]
        if not valid:
            return 999.0
        return sum(valid) / len(valid)

    wait_peak = pd.DataFrame(
        [
            {"stop_A": stop_a, "stop_B": stop_b, "avg_wait_min": calc_peak_wait(stop_a, stop_b)}
            for stop_a, targets in stop_reachability_peak.items()
            for stop_b in targets
        ]
    )
    if wait_peak.empty:
        wait_peak = pd.DataFrame(columns=["stop_A", "stop_B", "avg_wait_min"])
    wait_offpeak = pd.DataFrame(
        [
            {"stop_A": stop_a, "stop_B": stop_b, "avg_wait_min": calc_avg_wait_window(departures, offpeak_window_duration_min)}
            for (stop_a, stop_b), departures in pair_depart_times_offpeak.items()
        ]
    )

    reach_peak.to_parquet(CACHE_PEAK, index=False)
    reach_offpeak.to_parquet(CACHE_OFFPEAK, index=False)
    reach_peak_rev.to_parquet(CACHE_PEAK_REV, index=False)
    reach_opk_rev.to_parquet(CACHE_OPK_REV, index=False)
    wait_peak.to_parquet(CACHE_WAIT_PEAK, index=False)
    wait_offpeak.to_parquet(CACHE_WAIT_OFFPEAK, index=False)

    print("Baseline 07b кеш збережено:")
    print(f"  peak:     {len(reach_peak):,}")
    print(f"  offpeak:  {len(reach_offpeak):,}")
    print(f"  waitpeak: {len(wait_peak):,}")
    print(f"  waitoff:  {len(wait_offpeak):,}")


if __name__ == "__main__":
    run()
