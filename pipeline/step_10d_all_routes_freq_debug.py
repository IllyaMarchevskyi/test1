"""
10d All routes frequency debug.

Діагностичний крок, який виводить усі маршрути для будніх днів
і показує:
- route_id
- transport
- route
- current_freq (route-level рейси/год або EasyWay fallback)
- rl_initial_freq (нормалізована RL-інтенсивність)
"""


def run() -> None:
    from config_loader import cfg
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from utils.dispatch_frequency import (
        apply_dispatch_peak_frequency,
        build_easyway_route_stats,
        peak_windows_from_config,
    )

    PROCESSED_DIR = Path("./data/processed")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    EASYWAY_ROUTES = Path("../gtfs_static/easyway_routes.csv")
    EASYWAY_METRO = Path("../gtfs_static/easyway_metro.csv")
    DISPATCH_ROUTE_STATS = PROCESSED_DIR / "dispatch_route_stats.csv"
    OUTPUT_CSV = PROCESSED_DIR / "all_routes_weekdays_freq_debug.csv"
    freq_scaling = str(cfg.get("rl", {}).get("freq_scaling", "log")).strip().lower() or "log"
    peak_cfg = cfg.get("peak_hours", {})
    peak_windows = peak_windows_from_config(peak_cfg)
    total_peak_hours = float(peak_cfg.get("total_peak_hours", 4))

    required = [EASYWAY_ROUTES]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Відсутні входи для 10d_all_routes_freq_debug: {missing}")

    def parse_times(raw: str) -> list[int]:
        times: list[int] = []
        for token in str(raw).split(","):
            token = token.strip()
            if not token or token == r"\N":
                continue
            hh, mm, ss = token.split(":")
            times.append(int(hh) * 3600 + int(mm) * 60 + int(ss))
        return times

    parts = [pd.read_csv(EASYWAY_ROUTES)]
    if EASYWAY_METRO.exists():
        parts.append(pd.read_csv(EASYWAY_METRO))
    df = pd.concat(parts, ignore_index=True)

    df["route_id"] = df["route_id"].astype(str)
    df["transport"] = df["transport"].astype(str)
    df["route"] = df["route"].astype(str)
    df["direction"] = df["direction"].astype(str)
    df["calendar"] = df["calendar"].astype(str)
    df["times"] = df["schedules"].apply(parse_times)
    df["n_departures"] = df["times"].apply(len)

    weekdays = df[df["calendar"].str.strip().str.lower() == "weekdays"].copy()
    if weekdays.empty:
        raise ValueError("10d_all_routes_freq_debug: у даних не знайдено записів для weekdays.")

    summary = build_easyway_route_stats(
        weekdays,
        peak_windows,
        total_peak_hours,
        group_by_direction=False,
    )
    summary = apply_dispatch_peak_frequency(
        summary,
        DISPATCH_ROUTE_STATS,
        total_peak_hours,
    )
    summary["rl_initial_freq"] = 6.0
    for _, sub_idx in summary.groupby("transport").groups.items():
        current_freq = summary.loc[sub_idx, "current_freq"].astype(float)
        raw = np.log1p(current_freq) if freq_scaling == "log" else current_freq
        min_raw = float(raw.min())
        max_raw = float(raw.max())
        if max_raw > min_raw:
            scaled = 1.0 + ((raw - min_raw) / (max_raw - min_raw) * 11.0)
        else:
            scaled = pd.Series(6.0, index=raw.index)
        summary.loc[sub_idx, "rl_initial_freq"] = scaled.round(2)
    summary = summary.sort_values(
        ["current_freq", "transport", "route", "route_id"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    summary.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"10d_debug: weekdays routes -> {OUTPUT_CSV}")
    print(f"10d_debug: усього route-level записів = {len(summary)}")
    print("10d_debug: current_freq_per_hour -> rl_initial_freq:")
    for row in summary.itertuples(index=False):
        route_peak = getattr(row, "dispatch_peak_trips", None)
        peak_part = ""
        if route_peak is not None and not pd.isna(route_peak):
            peak_part = f", route_peak_trips={float(route_peak):.0f}"
        elif hasattr(row, "easyway_peak_departures"):
            peak_part = f", easyway_peak_departures={float(row.easyway_peak_departures):.0f}"
        print(
            f"  {row.route_id} ({row.transport} {row.route}) | "
            f"stops={row.n_stops}, directions={getattr(row, 'directions_count', 1)}, "
            f"stop_departures={row.total_departures}, "
            f"source={getattr(row, 'current_freq_source', 'unknown')}, "
            f"current_freq_per_hour={row.current_freq:.2f}{peak_part} "
            f"-> rl_initial_freq={float(row.rl_initial_freq):.2f}"
        )


if __name__ == "__main__":
    run()
