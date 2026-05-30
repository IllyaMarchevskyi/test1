"""
08 Preprocessing Baseline.

Два кроки підготовки для індексу доступності:
  - Ваги будинків (w_b): площа × поверховість / 50.0  (колишній 08a)
  - Ентропія маршрутного різноманіття H(f)/H_max      (колишній 08b)
"""


def run_building_weights() -> None:
    """Обчислює ваги будинків w_b. Виходить building_weights_baseline.parquet."""
    from config_loader import cfg
    import os
    import warnings

    import geopandas as gpd
    import osmnx as ox
    import pandas as pd

    warnings.filterwarnings("ignore")

    PROCESSED_DIR = "./data/processed"
    BUILDINGS_PATH = "../data/processed/buildings.parquet"
    OUTPUT_PATH = f"{PROCESSED_DIR}/building_weights_baseline.parquet"

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.exists(BUILDINGS_PATH):
        raise FileNotFoundError(f"Не знайдено {BUILDINGS_PATH}. Спочатку запусти 07a.")

    if os.path.exists(OUTPUT_PATH) and os.path.getmtime(OUTPUT_PATH) >= os.path.getmtime(BUILDINGS_PATH):
        cached = pd.read_parquet(OUTPUT_PATH)
        print(f"08_weights: кеш building_weights завантажено: {len(cached):,} будинків")
        return

    print("08_weights: завантажуємо базові будинки...")
    base_buildings = gpd.read_parquet(BUILDINGS_PATH).set_geometry("geometry")
    base_buildings = base_buildings.to_crs(cfg["city"]["crs_metric"]).copy()

    print("08_weights: завантажуємо деталізовані будинки з OSM...")
    detailed = ox.features_from_place(
        cfg["city"]["name"],
        tags={
            "building": [
                "residential", "apartments", "house", "detached",
                "dormitory", "cabin", "yes",
            ]
        },
    )
    detailed = detailed[detailed.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    detailed = detailed.to_crs(cfg["city"]["crs_metric"])
    detailed["area_sqm"] = detailed.geometry.area
    detailed["centroid_geom"] = detailed.geometry.centroid
    detailed = detailed.set_geometry("centroid_geom").reset_index(drop=True)

    def parse_levels(value) -> float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        if not text:
            return None
        for sep in [";", ",", "-", "/"]:
            if sep in text:
                text = text.split(sep)[0].strip()
                break
        try:
            parsed = float(text)
        except ValueError:
            return None
        return parsed if parsed > 0 else None

    detailed["building_type"] = detailed.get("building", pd.Series(index=detailed.index, dtype="object")).astype("string")
    detailed["levels_raw"] = detailed.get("building:levels", pd.Series(index=detailed.index, dtype="object"))
    detailed["levels_num"] = detailed["levels_raw"].map(parse_levels)

    print("08_weights: зіставляємо building_id з деталізованими будинками...")
    detailed_pts = detailed[["building_type", "levels_raw", "levels_num", "area_sqm", "centroid_geom"]].copy()
    detailed_pts = detailed_pts.rename(columns={"centroid_geom": "geometry"})
    detailed_pts = gpd.GeoDataFrame(detailed_pts, geometry="geometry", crs=cfg["city"]["crs_metric"])
    matched = gpd.sjoin_nearest(
        base_buildings[["building_id", "geometry"]].copy(),
        detailed_pts,
        how="left",
        max_distance=1.0,
        distance_col="match_dist_m",
    )
    matched = matched.drop_duplicates(subset=["building_id"]).reset_index(drop=True)

    default_levels = {"apartments": 9.0, "residential": 5.0, "dormitory": 5.0}

    def compute_weight(row: pd.Series) -> pd.Series:
        raw_btype = row.get("building_type")
        btype = "" if pd.isna(raw_btype) else str(raw_btype).strip().lower()
        levels_num = row.get("levels_num")
        raw_area = row.get("area_sqm")
        area = 0.0 if pd.isna(raw_area) else float(raw_area)
        levels_used = None
        rule = "fallback_1"
        weight = 1.0

        if btype in {"house", "detached"}:
            rule = "house_like"
        elif btype in {"apartments", "residential", "dormitory"}:
            levels_used = float(levels_num) if pd.notna(levels_num) else default_levels[btype]
            weight = (levels_used * area) / 50.0 if area > 0 else 1.0
            rule = "multi_family_formula"
        elif btype == "yes":
            if pd.notna(levels_num) and float(levels_num) >= 4.0:
                levels_used = float(levels_num)
                weight = (levels_used * area) / 50.0 if area > 0 else 1.0
                rule = "generic_tall_formula"
            else:
                rule = "generic_low_or_missing"
        else:
            rule = "unknown_fallback"

        if weight < 1.0 or not pd.notna(weight):
            weight = 1.0
        return pd.Series({"levels_used": levels_used, "weight_wb": float(weight), "weight_rule": rule})

    weights = matched.apply(compute_weight, axis=1)
    result = pd.concat(
        [
            matched[["building_id", "building_type", "levels_raw", "levels_num", "area_sqm", "match_dist_m"]].reset_index(drop=True),
            weights.reset_index(drop=True),
        ],
        axis=1,
    )
    result["building_type"] = result["building_type"].fillna("unknown")
    result["levels_display"] = result["levels_used"].where(result["levels_used"].notna(), result["levels_num"]).round(1)
    result.to_parquet(OUTPUT_PATH, index=False)

    print(f"08_weights: збережено → {OUTPUT_PATH}")
    print(f"  Будинків: {len(result):,}  |  формула w_b: {(result['weight_wb'] > 1).sum():,}  |  fallback w=1: {(result['weight_wb'] == 1).sum():,}")


def run_entropy() -> None:
    """Обчислює ентропію H(f)/H_max для кожного закладу. Виходить facility_entropy_baseline.parquet."""
    from config_loader import cfg
    import math
    import os
    import warnings

    import pandas as pd
    from tqdm.auto import tqdm

    warnings.filterwarnings("ignore")

    PROCESSED_DIR = "./data/processed"
    STOP_FAC_EXIT_PATH = f"{PROCESSED_DIR}/stop_to_fac_exit_baseline.parquet"
    EASYWAY_PATH = "../gtfs_static/easyway_routes.csv"
    EASYWAY_TRAM_PATH = "../gtfs_static/easyway_tram_data.csv"
    SCORES_PATH = cfg["paths"]["scores"]
    OUT_PARQUET = f"{PROCESSED_DIR}/facility_entropy_baseline.parquet"
    OUT_CSV = f"{PROCESSED_DIR}/facility_entropy_baseline.csv"
    OUT_PREVIEW_CSV = f"{PROCESSED_DIR}/facility_entropy_preview_baseline.csv"

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    required = [STOP_FAC_EXIT_PATH, EASYWAY_PATH, SCORES_PATH]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Відсутні входи для 08_entropy: {missing}")

    if all(os.path.exists(p) for p in [OUT_PARQUET, OUT_CSV]):
        outputs_mtime = min(os.path.getmtime(OUT_PARQUET), os.path.getmtime(OUT_CSV))
        if outputs_mtime >= max(os.path.getmtime(p) for p in required):
            cached = pd.read_parquet(OUT_PARQUET)
            print(f"08_entropy: кеш ентропії завантажено: {len(cached):,} закладів")
            return

    def hhmm_to_sec(value: str) -> int:
        h, m = map(int, value.split(":"))
        return h * 3600 + m * 60

    peak_windows = [
        (hhmm_to_sec(cfg["peak_hours"]["morning_start"]), hhmm_to_sec(cfg["peak_hours"]["morning_end"])),
        (hhmm_to_sec(cfg["peak_hours"]["evening_start"]), hhmm_to_sec(cfg["peak_hours"]["evening_end"])),
    ]
    offpeak_start = hhmm_to_sec(cfg["offpeak_hours"]["start"])
    offpeak_end = hhmm_to_sec(cfg["offpeak_hours"]["end"])

    def in_peak(sec: int) -> bool:
        return any(s <= sec < e for s, e in peak_windows)

    def in_offpeak(sec: int) -> bool:
        return offpeak_start <= sec < offpeak_end

    def parse_schedules(value: str) -> list[int]:
        times = []
        for raw in str(value).strip().split(","):
            raw = raw.strip()
            if not raw or raw == r"\N":
                continue
            hh, mm, ss = raw.split(":")
            times.append(int(hh) * 3600 + int(mm) * 60 + int(ss))
        return sorted(times)

    def entropy_stats(route_counts: dict) -> tuple:
        positive = {r: c for r, c in route_counts.items() if c > 0}
        total = sum(positive.values())
        n = len(positive)
        if total <= 0 or n == 0:
            return 0.0, 0.0, 0.0, 0, 0
        H = -sum((c / total) * math.log2(c / total) for c in positive.values())
        Hmax = math.log2(n) if n > 1 else 0.0
        return H, Hmax, (H / Hmax if Hmax > 0 else 0.0), n, total

    print("08_entropy: завантажуємо дані...")
    stop_fac_exit = pd.read_parquet(STOP_FAC_EXIT_PATH)
    stop_fac_exit["facility_id"] = stop_fac_exit["facility_id"].astype(str)
    stop_fac_exit["stop_id"] = stop_fac_exit["stop_id"].astype(str)
    facility_meta = pd.read_csv(SCORES_PATH, usecols=["facility_id", "facility_type", "name"])
    facility_meta["facility_id"] = facility_meta["facility_id"].astype(str)

    easyway_parts = [pd.read_csv(EASYWAY_PATH)]
    if os.path.exists(EASYWAY_TRAM_PATH):
        easyway_parts.append(pd.read_csv(EASYWAY_TRAM_PATH))
    easyway = pd.concat(easyway_parts, ignore_index=True)
    easyway = easyway[easyway["schedules"] != r"\N"].copy()
    easyway["stop_id"] = easyway["stop_id"].astype(str)
    easyway["route_label"] = (easyway["transport"].astype(str).str.strip() + " " + easyway["route"].astype(str).str.strip()).str.strip()
    easyway["times"] = easyway["schedules"].apply(parse_schedules)
    easyway = easyway[easyway["calendar"].isin(["Weekdays", "All Week"])].copy()
    easyway["peak_count"] = easyway["times"].apply(lambda t: sum(1 for s in t if in_peak(s)))
    easyway["offpeak_count"] = easyway["times"].apply(lambda t: sum(1 for s in t if in_offpeak(s)))

    route_counts_by_stop = (
        easyway.groupby(["stop_id", "route_label"], as_index=False)[["peak_count", "offpeak_count"]]
        .sum()
        .reset_index(drop=True)
    )
    facility_route_reps = (
        stop_fac_exit
        .merge(route_counts_by_stop, on="stop_id", how="left")
        .dropna(subset=["route_label"])
        .sort_values(["facility_id", "route_label", "walk_min", "stop_id"])
        .drop_duplicates(subset=["facility_id", "route_label"], keep="first")
        .reset_index(drop=True)
    )

    results = []
    grouped = facility_route_reps.groupby("facility_id", sort=False)
    print(f"08_entropy: рахуємо ентропію для {stop_fac_exit['facility_id'].nunique():,} закладів...")

    for fid in tqdm(stop_fac_exit["facility_id"].drop_duplicates().tolist(), desc="08_entropy"):
        if fid in grouped.groups:
            grp = grouped.get_group(fid)
            peak_counts = {str(r.route_label): int(r.peak_count) for r in grp.itertuples(index=False) if int(r.peak_count) > 0}
            offpeak_counts = {str(r.route_label): int(r.offpeak_count) for r in grp.itertuples(index=False) if int(r.offpeak_count) > 0}
        else:
            peak_counts = offpeak_counts = {}

        H_pk, Hmax_pk, Hn_pk, n_pk, dep_pk = entropy_stats(peak_counts)
        H_op, Hmax_op, Hn_op, n_op, dep_op = entropy_stats(offpeak_counts)
        results.append({
            "facility_id": fid,
            "H_peak": H_pk, "H_offpeak": H_op,
            "Hmax_peak": Hmax_pk, "Hmax_offpeak": Hmax_op,
            "Hnorm_peak": Hn_pk, "Hnorm_offpeak": Hn_op,
            "n_routes_peak": n_pk, "n_routes_offpeak": n_op,
            "stop_departures_peak": dep_pk, "stop_departures_offpeak": dep_op,
            "departures_peak": dep_pk, "departures_offpeak": dep_op,
        })

    entropy_df = pd.DataFrame(results)
    entropy_df.to_parquet(OUT_PARQUET, index=False)
    entropy_df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    preview_df = entropy_df.merge(facility_meta, on="facility_id", how="left")
    preview_df.to_csv(OUT_PREVIEW_CSV, index=False, encoding="utf-8")

    print(f"08_entropy: збережено → {OUT_PARQUET}")
    print(f"  Закладів: {len(entropy_df):,}  |  Hnorm_peak > 0: {(entropy_df['Hnorm_peak'] > 0).sum():,}")
    print(f"  Середній Hnorm_peak: {entropy_df['Hnorm_peak'].mean():.3f}  |  Hnorm_offpeak: {entropy_df['Hnorm_offpeak'].mean():.3f}")


def run() -> None:
    run_building_weights()
    run_entropy()


if __name__ == "__main__":
    run()
