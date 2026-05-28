"""
10i Dispatch schedule parser.

Читає диспетчерські CSV з pipeline/transports і витягує дані,
які можна використовувати у фінальних рекомендаціях:
- кількість випусків;
- кількість рейсів у піковий період;
- перший/останній рейс;
- приблизний час рейсу та обороту;
- місткість за типом транспорту.
"""

from __future__ import annotations


def run() -> None:
    from config_loader import cfg
    import csv
    import json
    import re
    import unicodedata
    from pathlib import Path

    import numpy as np
    import pandas as pd

    TRANSPORTS_DIR = Path("./transports")
    PROCESSED_DIR = Path("./data/processed")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ROUTE_STATS_CSV = PROCESSED_DIR / "dispatch_route_stats.csv"
    DIRECTION_STATS_CSV = PROCESSED_DIR / "dispatch_direction_stats.csv"
    RELEASE_TRIPS_CSV = PROCESSED_DIR / "dispatch_release_trips.csv"
    REPORT_JSON = PROCESSED_DIR / "dispatch_parse_report.json"

    if not TRANSPORTS_DIR.exists():
        raise FileNotFoundError(f"10i_dispatch_parser: папку не знайдено: {TRANSPORTS_DIR}")

    peak_cfg = cfg.get("peak_hours", {})
    rl_cfg = cfg.get("rl", {})
    vehicle_capacity = {
        str(k).strip().lower(): int(v)
        for k, v in dict(rl_cfg.get("vehicle_capacity", {})).items()
    }
    default_capacity = {"bus": 80, "trol": 100, "tram": 160, "metro": 1000}
    for key, value in default_capacity.items():
        vehicle_capacity.setdefault(key, value)

    def hm_to_seconds(value: str) -> int:
        hh, mm = str(value).split(":")[:2]
        return int(hh) * 3600 + int(mm) * 60

    morning_start_s = hm_to_seconds(peak_cfg.get("morning_start", "07:00"))
    morning_end_s = hm_to_seconds(peak_cfg.get("morning_end", "09:00"))
    evening_start_s = hm_to_seconds(peak_cfg.get("evening_start", "17:00"))
    evening_end_s = hm_to_seconds(peak_cfg.get("evening_end", "19:00"))

    title_re = re.compile(r"^(Автобус|Тролейбус)\s*№\s*([^;]+)")
    file_re = re.compile(r"^(bus|trol)_([0-9]+[a-z]*)\.csv$")
    suffix_trans = str.maketrans(
        {
            "а": "a",
            "А": "a",
            "д": "d",
            "Д": "d",
            "к": "k",
            "К": "k",
            "т": "t",
            "Т": "t",
            "р": "r",
            "Р": "r",
        }
    )
    type_by_title = {"Автобус": "bus", "Тролейбус": "trol"}
    type_title_by_code = {"bus": "Автобус", "trol": "Тролейбус"}
    depot_markers = ("депо", "тред", "автопарк", "ап", "атп")

    def normalize_route(value: str) -> str:
        normalized = unicodedata.normalize("NFC", str(value))
        normalized = re.sub(r"\s+", "", normalized)
        return normalized.translate(suffix_trans).lower()

    def parse_time(value: str) -> int | None:
        raw = str(value).strip()
        if not raw or raw in {"—", "-"}:
            return None
        parts = raw.split(":")
        if len(parts) < 2:
            return None
        try:
            hh = int(parts[0])
            mm = int(parts[1])
        except ValueError:
            return None
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return None
        return hh * 3600 + mm * 60

    def fmt_time(seconds: int | float | None) -> str | None:
        if seconds is None or pd.isna(seconds):
            return None
        sec = int(seconds) % (24 * 3600)
        return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}"

    def is_peak(seconds: int) -> bool:
        return (morning_start_s <= seconds < morning_end_s) or (evening_start_s <= seconds < evening_end_s)

    def duration_min(start_s: int, end_s: int) -> float:
        diff = end_s - start_s
        if diff < 0:
            diff += 24 * 3600
        return diff / 60.0

    files = sorted(TRANSPORTS_DIR.glob("*/*.csv"))
    if not files:
        raise FileNotFoundError("10i_dispatch_parser: у pipeline/transports немає CSV-файлів.")

    route_rows: list[dict] = []
    direction_rows: list[dict] = []
    trip_rows: list[dict] = []
    problems: list[dict] = []

    for path in files:
        rel = str(path)
        file_match = file_re.match(path.name)
        if not file_match:
            problems.append({"file": rel, "problem": "bad_filename", "details": path.name})
            continue
        file_transport = file_match.group(1)
        file_route = file_match.group(2)
        route_key = f"{file_transport}_{file_route}"

        lines = [line.strip("\n\r") for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(lines) < 4:
            problems.append({"file": rel, "problem": "too_few_lines", "details": len(lines)})
            continue

        title_cells = next(csv.reader([lines[0]], delimiter=";"))
        title = title_cells[0].strip() if title_cells else ""
        date_label = title_cells[1].strip() if len(title_cells) > 1 else ""
        title_match = title_re.search(title)
        if not title_match:
            problems.append({"file": rel, "problem": "bad_title", "details": title})
            continue
        title_transport = type_by_title.get(title_match.group(1), "")
        title_route = normalize_route(title_match.group(2))
        if title_transport != file_transport or title_route != file_route:
            problems.append(
                {
                    "file": rel,
                    "problem": "route_mismatch",
                    "details": f"file={file_transport}_{file_route}, title={title_transport}_{title_route}",
                }
            )

        current_direction_name: str | None = None
        current_header: list[str] | None = None
        direction_index = 0
        direction_trip_counts: dict[int, int] = {}

        for line_no, line in enumerate(lines[1:], start=2):
            cells = [cell.strip() for cell in next(csv.reader([line], delimiter=";"))]
            if not cells or not cells[0]:
                continue
            if cells[0] == "№ випуску":
                current_header = cells
                direction_index += 1
                direction_trip_counts[direction_index] = 0
                if current_direction_name is None:
                    current_direction_name = f"direction_{direction_index}"
                continue
            if current_header is None or not cells[0].isdigit():
                current_direction_name = cells[0]
                current_header = None
                continue
            if len(cells) != len(current_header):
                problems.append(
                    {
                        "file": rel,
                        "problem": "bad_row_len",
                        "details": f"line={line_no}, got={len(cells)}, expected={len(current_header)}",
                    }
                )
                continue

            release_id = int(cells[0])
            stop_names = current_header[1:]
            times = [parse_time(cell) for cell in cells[1:]]
            non_empty = [(idx, sec) for idx, sec in enumerate(times) if sec is not None]
            if len(non_empty) < 2:
                continue

            first_idx, start_s = non_empty[0]
            last_idx, end_s = non_empty[-1]
            trip_duration = duration_min(start_s, end_s)
            if not (0 < trip_duration < 6 * 60):
                problems.append(
                    {
                        "file": rel,
                        "problem": "bad_trip_duration",
                        "details": f"line={line_no}, duration={trip_duration:.1f}",
                    }
                )
                continue

            direction_trip_counts[direction_index] += 1
            first_stop = stop_names[first_idx]
            last_stop = stop_names[last_idx]
            control_points = [name for name in stop_names if name]
            depot_points = [
                name
                for name in control_points
                if any(marker in name.lower() for marker in depot_markers)
            ]
            starts_inside_route = first_idx > 0
            ends_inside_route = last_idx < len(stop_names) - 1
            peak_flag = is_peak(start_s)

            trip_rows.append(
                {
                    "route_key": route_key,
                    "transport": file_transport,
                    "transport_title": type_title_by_code.get(file_transport, file_transport),
                    "route": file_route,
                    "source_file": rel,
                    "date_label": date_label,
                    "direction_index": direction_index,
                    "direction_name": current_direction_name,
                    "release_id": release_id,
                    "trip_seq_in_direction": direction_trip_counts[direction_index],
                    "start_stop": first_stop,
                    "end_stop": last_stop,
                    "start_time": fmt_time(start_s),
                    "end_time": fmt_time(end_s),
                    "start_seconds": start_s,
                    "end_seconds": end_s,
                    "duration_min": trip_duration,
                    "control_points_count": len(control_points),
                    "starts_inside_route": starts_inside_route,
                    "ends_inside_route": ends_inside_route,
                    "has_depot_marker": bool(depot_points),
                    "depot_markers": ", ".join(sorted(set(depot_points))),
                    "is_peak_trip": peak_flag,
                    "capacity_per_vehicle": vehicle_capacity.get(file_transport),
                    "capacity_trip": vehicle_capacity.get(file_transport, 0),
                }
            )

        route_trips = [row for row in trip_rows if row["route_key"] == route_key]
        if not route_trips:
            problems.append({"file": rel, "problem": "no_valid_trips", "details": route_key})
            continue

        route_df = pd.DataFrame(route_trips)
        direction_stats = []
        for direction_idx, group in route_df.groupby("direction_index"):
            direction_stats.append(
                {
                    "route_key": route_key,
                    "transport": file_transport,
                    "route": file_route,
                    "source_file": rel,
                    "direction_index": int(direction_idx),
                    "direction_name": str(group["direction_name"].iloc[0]),
                    "release_count": int(group["release_id"].nunique()),
                    "trip_count": int(len(group)),
                    "peak_trip_count": int(group["is_peak_trip"].sum()),
                    "first_time": fmt_time(float(group["start_seconds"].min())),
                    "last_time": fmt_time(float(group["start_seconds"].max())),
                    "median_one_way_duration_min": float(group["duration_min"].median()),
                    "avg_one_way_duration_min": float(group["duration_min"].mean()),
                    "min_one_way_duration_min": float(group["duration_min"].min()),
                    "max_one_way_duration_min": float(group["duration_min"].max()),
                    "control_points_max": int(group["control_points_count"].max()),
                    "starts_inside_route_count": int(group["starts_inside_route"].sum()),
                    "ends_inside_route_count": int(group["ends_inside_route"].sum()),
                    "depot_markers": ", ".join(
                        sorted({marker for marker in group["depot_markers"] if str(marker).strip()})
                    ),
                }
            )
        direction_rows.extend(direction_stats)

        direction_df = pd.DataFrame(direction_stats)
        if len(direction_df) >= 2:
            round_trip_duration = float(direction_df["median_one_way_duration_min"].sum())
        else:
            round_trip_duration = float(direction_df["median_one_way_duration_min"].iloc[0] * 2.0)

        peak_trips = int(route_df["is_peak_trip"].sum())
        weekday_trips = int(len(route_df))
        release_count = int(route_df["release_id"].nunique())
        capacity = int(vehicle_capacity.get(file_transport, 0))
        route_rows.append(
            {
                "route_key": route_key,
                "transport": file_transport,
                "transport_title": type_title_by_code.get(file_transport, file_transport),
                "route": file_route,
                "source_file": rel,
                "date_label": date_label,
                "directions_count": int(route_df["direction_index"].nunique()),
                "release_count": release_count,
                "release_ids": ", ".join(str(x) for x in sorted(route_df["release_id"].unique())),
                "weekday_trips": weekday_trips,
                "peak_trips": peak_trips,
                "offpeak_or_other_trips": weekday_trips - peak_trips,
                "first_trip_time": fmt_time(float(route_df["start_seconds"].min())),
                "last_trip_time": fmt_time(float(route_df["start_seconds"].max())),
                "median_one_way_duration_min": float(route_df["duration_min"].median()),
                "round_trip_duration_min": round_trip_duration,
                "capacity_per_vehicle": capacity,
                "peak_capacity_places": int(peak_trips * capacity),
                "weekday_capacity_places": int(weekday_trips * capacity),
                "depot_markers": ", ".join(
                    sorted({marker for marker in route_df["depot_markers"] if str(marker).strip()})
                ),
                "starts_inside_route_count": int(route_df["starts_inside_route"].sum()),
                "ends_inside_route_count": int(route_df["ends_inside_route"].sum()),
            }
        )

    route_stats_df = pd.DataFrame(route_rows).sort_values(["transport", "route"])
    direction_stats_df = pd.DataFrame(direction_rows).sort_values(["transport", "route", "direction_index"])
    trips_df = pd.DataFrame(trip_rows).sort_values(
        ["transport", "route", "direction_index", "trip_seq_in_direction"]
    )

    route_stats_df.to_csv(ROUTE_STATS_CSV, index=False, encoding="utf-8")
    direction_stats_df.to_csv(DIRECTION_STATS_CSV, index=False, encoding="utf-8")
    trips_df.to_csv(RELEASE_TRIPS_CSV, index=False, encoding="utf-8")

    report = {
        "files_total": len(files),
        "routes_parsed": int(len(route_stats_df)),
        "trips_parsed": int(len(trips_df)),
        "by_transport": route_stats_df["transport"].value_counts().to_dict() if not route_stats_df.empty else {},
        "vehicle_capacity": vehicle_capacity,
        "problems_count": len(problems),
        "problems": problems[:100],
        "outputs": {
            "route_stats_csv": str(ROUTE_STATS_CSV),
            "direction_stats_csv": str(DIRECTION_STATS_CSV),
            "release_trips_csv": str(RELEASE_TRIPS_CSV),
        },
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "10i_dispatch_parser: "
        f"files={len(files)} routes={len(route_stats_df)} trips={len(trips_df)} "
        f"problems={len(problems)}"
    )
    print(f"10i_dispatch_parser: route stats -> {ROUTE_STATS_CSV}")
    print(f"10i_dispatch_parser: release trips -> {RELEASE_TRIPS_CSV}")
    print(f"10i_dispatch_parser: report -> {REPORT_JSON}")


if __name__ == "__main__":
    run()
