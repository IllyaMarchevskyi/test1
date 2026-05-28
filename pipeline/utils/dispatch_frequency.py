"""Оцінка пікової інтенсивності маршрутів для RL."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def hm_to_seconds(value: object) -> int:
    hh, mm = str(value).split(":")[:2]
    return int(hh) * 3600 + int(mm) * 60


def peak_windows_from_config(peak_cfg: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        (
            hm_to_seconds(peak_cfg.get("morning_start", "07:00")),
            hm_to_seconds(peak_cfg.get("morning_end", "09:00")),
        ),
        (
            hm_to_seconds(peak_cfg.get("evening_start", "17:00")),
            hm_to_seconds(peak_cfg.get("evening_end", "19:00")),
        ),
    )


def parse_schedule_seconds(value: object) -> list[int]:
    times: list[int] = []
    for raw in str(value).strip().split(","):
        raw = raw.strip()
        if not raw or raw == r"\N":
            continue
        parts = raw.split(":")
        if len(parts) < 2:
            continue
        try:
            hh = int(parts[0])
            mm = int(parts[1])
            ss = int(parts[2]) if len(parts) > 2 else 0
        except ValueError:
            continue
        times.append(hh * 3600 + mm * 60 + ss)
    return sorted(times)


def count_peak_times(times: list[int], peak_windows: tuple[tuple[int, int], ...]) -> int:
    return sum(
        1
        for seconds in times
        if any(start <= seconds < end for start, end in peak_windows)
    )


def build_easyway_route_stats(
    easyway: pd.DataFrame,
    peak_windows: tuple[tuple[int, int], ...],
    total_peak_hours: float,
    *,
    group_by_direction: bool = False,
) -> pd.DataFrame:
    """
    Будує route-level або direction-level частоту з EasyWay без множення на кількість зупинок.

    Для fallback беремо кількість пікових відправлень на першій зупинці кожного
    напрямку і сумуємо напрямки на route-level. Це не повний диспетчерський
    розклад, але значно коректніше за старий total_stop_departures / 11.
    """
    df = easyway.copy()
    df["route_id"] = df["route_id"].astype(str)
    df["transport"] = df["transport"].astype(str)
    df["route"] = df["route"].astype(str)
    df["direction"] = df["direction"].astype(str)
    df["stop_id"] = df["stop_id"].astype(str)
    if "times" not in df.columns:
        df["times"] = df["schedules"].apply(parse_schedule_seconds)
    if "n_departures" not in df.columns:
        df["n_departures"] = df["times"].apply(len)
    df["n_peak_departures"] = df["times"].apply(lambda times: count_peak_times(times, peak_windows))
    if "index" in df.columns:
        df["_stop_order"] = pd.to_numeric(df["index"], errors="coerce")
    else:
        df["_stop_order"] = pd.NA
    df["_stop_order"] = df["_stop_order"].fillna(df.groupby(["route_id", "direction"]).cumcount())

    direction_cols = ["route_id", "transport", "route", "direction"]
    direction_totals = (
        df.groupby(direction_cols, as_index=False)
        .agg(
            direction_n_stops=("stop_id", "nunique"),
            direction_stop_departures=("n_departures", "sum"),
        )
        .reset_index(drop=True)
    )
    first_stops = (
        df.sort_values(["route_id", "direction", "_stop_order"])
        .groupby(direction_cols, as_index=False)
        .first()
        .loc[
            :,
            direction_cols + ["n_departures", "n_peak_departures"],
        ]
        .rename(
            columns={
                "n_departures": "easyway_direction_weekday_departures",
                "n_peak_departures": "easyway_direction_peak_departures",
            }
        )
    )
    direction_stats = direction_totals.merge(first_stops, on=direction_cols, how="left")

    if group_by_direction:
        stats = direction_stats.rename(
            columns={
                "direction_n_stops": "n_stops",
                "direction_stop_departures": "total_departures",
                "easyway_direction_peak_departures": "easyway_peak_departures",
                "easyway_direction_weekday_departures": "easyway_weekday_departures",
            }
        )
        stats["directions_count"] = 1
    else:
        stats = (
            direction_stats.groupby(["route_id", "transport", "route"], as_index=False)
            .agg(
                n_stops=("direction_n_stops", "max"),
                total_departures=("direction_stop_departures", "sum"),
                directions_count=("direction", "nunique"),
                easyway_peak_departures=("easyway_direction_peak_departures", "sum"),
                easyway_weekday_departures=("easyway_direction_weekday_departures", "sum"),
            )
            .reset_index(drop=True)
        )

    stats["current_freq"] = (
        pd.to_numeric(stats["easyway_peak_departures"], errors="coerce").fillna(0.0)
        / max(float(total_peak_hours), 1e-9)
    )
    stats["current_freq_source"] = "easyway_first_stop_peak_departures_per_hour"
    return stats


def normalize_route_label(value: str) -> str:
    table = str.maketrans(
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
    return str(value).replace(" ", "").translate(table).lower()


def apply_dispatch_peak_frequency(
    route_stats: pd.DataFrame,
    dispatch_stats_path: Path,
    total_peak_hours: float,
    dispatch_direction_stats_path: Path | None = None,
) -> pd.DataFrame:
    """
    Для bus/trol замінює current_freq на повні пікові рейси/год з 10i.

    Якщо у route_stats є direction і передано dispatch_direction_stats_path,
    використовуємо direction-level рейси. Інакше лишається route-level
    агрегація по двох напрямках.

    EasyWay fallback має бути вже порахований до виклику цієї функції.
    Він не має базуватись на total_stop_departures / 11, бо це множить
    рейси на кількість зупинок.
    """
    result = route_stats.copy()
    if "current_freq_source" not in result.columns:
        result["current_freq_source"] = "easyway_first_stop_peak_departures_per_hour"

    if not dispatch_stats_path.exists() or total_peak_hours <= 0:
        result["dispatch_peak_trips"] = pd.NA
        result["dispatch_weekday_trips"] = pd.NA
        result["dispatch_release_count"] = pd.NA
        result["dispatch_direction_peak_trips"] = pd.NA
        result["dispatch_direction_weekday_trips"] = pd.NA
        return result

    dispatch = pd.read_csv(dispatch_stats_path)
    if dispatch.empty:
        result["dispatch_peak_trips"] = pd.NA
        result["dispatch_weekday_trips"] = pd.NA
        result["dispatch_release_count"] = pd.NA
        result["dispatch_direction_peak_trips"] = pd.NA
        result["dispatch_direction_weekday_trips"] = pd.NA
        return result

    dispatch["transport"] = dispatch["transport"].astype(str)
    dispatch["route"] = dispatch["route"].astype(str)
    dispatch["_route_norm"] = dispatch["route"].apply(normalize_route_label)
    dispatch["_dispatch_key"] = dispatch["transport"] + "_" + dispatch["_route_norm"]

    result["_route_norm"] = result["route"].astype(str).apply(normalize_route_label)
    result["_dispatch_key"] = result["transport"].astype(str) + "_" + result["_route_norm"]
    result = result.merge(
        dispatch[
            [
                "_dispatch_key",
                "peak_trips",
                "weekday_trips",
                "release_count",
            ]
        ].rename(
            columns={
                "peak_trips": "dispatch_peak_trips",
                "weekday_trips": "dispatch_weekday_trips",
                "release_count": "dispatch_release_count",
            }
        ),
        on="_dispatch_key",
        how="left",
    )

    has_dispatch = result["dispatch_peak_trips"].notna() & (pd.to_numeric(result["dispatch_peak_trips"], errors="coerce") > 0)
    result.loc[has_dispatch, "current_freq"] = (
        pd.to_numeric(result.loc[has_dispatch, "dispatch_peak_trips"], errors="coerce") / float(total_peak_hours)
    )
    result.loc[has_dispatch, "current_freq_source"] = "dispatch_full_peak_trips_per_hour"

    if dispatch_direction_stats_path is not None and dispatch_direction_stats_path.exists() and "direction" in result.columns:
        direction_dispatch = pd.read_csv(dispatch_direction_stats_path)
        if not direction_dispatch.empty:
            direction_dispatch["transport"] = direction_dispatch["transport"].astype(str)
            direction_dispatch["route"] = direction_dispatch["route"].astype(str)
            direction_dispatch["_route_norm"] = direction_dispatch["route"].apply(normalize_route_label)
            direction_dispatch["_dispatch_key"] = (
                direction_dispatch["transport"] + "_" + direction_dispatch["_route_norm"]
            )

            def direction_to_index(value: object) -> int | None:
                direction = str(value).strip().lower()
                if direction == "forward":
                    return 1
                if direction == "backward":
                    return 2
                try:
                    return int(float(direction))
                except ValueError:
                    return None

            result["_direction_index"] = result["direction"].apply(direction_to_index)
            result = result.merge(
                direction_dispatch[
                    [
                        "_dispatch_key",
                        "direction_index",
                        "peak_trip_count",
                        "full_trip_count",
                    ]
                ].rename(
                    columns={
                        "direction_index": "_direction_index",
                        "peak_trip_count": "dispatch_direction_peak_trips",
                        "full_trip_count": "dispatch_direction_weekday_trips",
                    }
                ),
                on=["_dispatch_key", "_direction_index"],
                how="left",
            )
            has_direction_dispatch = result["dispatch_direction_peak_trips"].notna()
            result.loc[has_direction_dispatch, "current_freq"] = (
                pd.to_numeric(
                    result.loc[has_direction_dispatch, "dispatch_direction_peak_trips"],
                    errors="coerce",
                )
                / float(total_peak_hours)
            )
            result.loc[
                has_direction_dispatch,
                "current_freq_source",
            ] = "dispatch_full_peak_trips_per_hour_by_direction"
            result = result.drop(columns=["_direction_index"], errors="ignore")
    else:
        result["dispatch_direction_peak_trips"] = pd.NA
        result["dispatch_direction_weekday_trips"] = pd.NA

    result = result.drop(columns=["_route_norm", "_dispatch_key"], errors="ignore")
    return result
