"""
08a Building Weights Baseline.

Computes residential building weights w_b and derived metadata for the
extended accessibility index.
"""


def run() -> None:
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
        raise FileNotFoundError(f"Не знайдено {BUILDINGS_PATH}. Спочатку запусти 07a_base.")

    if os.path.exists(OUTPUT_PATH) and os.path.getmtime(OUTPUT_PATH) >= os.path.getmtime(BUILDINGS_PATH):
        cached = pd.read_parquet(OUTPUT_PATH)
        print(f"08a_base: кеш building_weights завантажено: {len(cached):,} будинків")
        return

    print("08a_base: завантажуємо базові будинки...")
    base_buildings = gpd.read_parquet(BUILDINGS_PATH).set_geometry("geometry")
    base_buildings = base_buildings.to_crs(cfg["city"]["crs_metric"]).copy()

    print("08a_base: завантажуємо деталізовані будинки з OSM...")
    detailed = ox.features_from_place(
        cfg["city"]["name"],
        tags={
            "building": [
                "residential",
                "apartments",
                "house",
                "detached",
                "dormitory",
                "cabin",
                "yes",
            ]
        },
    )
    detailed = detailed[detailed.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    detailed = detailed.to_crs(cfg["city"]["crs_metric"])
    detailed["area_sqm"] = detailed.geometry.area
    detailed["centroid_geom"] = detailed.geometry.centroid
    detailed = detailed.set_geometry("centroid_geom")
    detailed = detailed.reset_index(drop=True)

    def parse_levels(value) -> float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        if not text:
            return None
        for separator in [";", ",", "-", "/"]:
            if separator in text:
                text = text.split(separator)[0].strip()
                break
        try:
            parsed = float(text)
        except ValueError:
            return None
        if parsed <= 0:
            return None
        return parsed

    detailed["building_type"] = detailed.get("building", pd.Series(index=detailed.index, dtype="object")).astype("string")
    detailed["levels_raw"] = detailed.get("building:levels", pd.Series(index=detailed.index, dtype="object"))
    detailed["levels_num"] = detailed["levels_raw"].map(parse_levels)

    print("08a_base: зіставляємо building_id з деталізованими будинками...")
    detailed_points = detailed[["building_type", "levels_raw", "levels_num", "area_sqm", "centroid_geom"]].copy()
    detailed_points = detailed_points.rename(columns={"centroid_geom": "geometry"})
    detailed_points = gpd.GeoDataFrame(detailed_points, geometry="geometry", crs=cfg["city"]["crs_metric"])
    matched = gpd.sjoin_nearest(
        base_buildings[["building_id", "geometry"]].copy(),
        detailed_points,
        how="left",
        max_distance=1.0,
        distance_col="match_dist_m",
    )
    matched = matched.drop_duplicates(subset=["building_id"]).reset_index(drop=True)

    default_levels = {
        "apartments": 9.0,
        "residential": 5.0,
        "dormitory": 5.0,
    }

    def compute_weight(row: pd.Series) -> pd.Series:
        raw_building_type = row.get("building_type")
        building_type = "" if pd.isna(raw_building_type) else str(raw_building_type).strip().lower()
        levels_num = row.get("levels_num")
        raw_area_sqm = row.get("area_sqm")
        area_sqm = 0.0 if pd.isna(raw_area_sqm) else float(raw_area_sqm)
        levels_used = None
        rule = "fallback_1"
        weight = 1.0

        if building_type in {"house", "detached"}:
            rule = "house_like"
            weight = 1.0
        elif building_type in {"apartments", "residential", "dormitory"}:
            levels_used = float(levels_num) if pd.notna(levels_num) else default_levels[building_type]
            weight = (levels_used * area_sqm) / 50.0 if area_sqm > 0 else 1.0
            rule = "multi_family_formula"
        elif building_type == "yes":
            if pd.notna(levels_num) and float(levels_num) >= 4.0:
                levels_used = float(levels_num)
                weight = (levels_used * area_sqm) / 50.0 if area_sqm > 0 else 1.0
                rule = "generic_tall_formula"
            else:
                rule = "generic_low_or_missing"
                weight = 1.0
        else:
            rule = "unknown_fallback"
            weight = 1.0

        if weight < 1.0 or not pd.notna(weight):
            weight = 1.0

        return pd.Series(
            {
                "levels_used": levels_used,
                "weight_wb": float(weight),
                "weight_rule": rule,
            }
        )

    weights = matched.apply(compute_weight, axis=1)
    result = pd.concat(
        [
            matched[["building_id", "building_type", "levels_raw", "levels_num", "area_sqm", "match_dist_m"]].reset_index(drop=True),
            weights.reset_index(drop=True),
        ],
        axis=1,
    )

    result["building_type"] = result["building_type"].fillna("unknown")
    result["levels_display"] = result["levels_used"].where(result["levels_used"].notna(), result["levels_num"])
    result["levels_display"] = result["levels_display"].round(1)

    result.to_parquet(OUTPUT_PATH, index=False)

    print(f"08a_base: building_weights збережено: {OUTPUT_PATH}")
    print(f"  Будинків: {len(result):,}")
    print(f"  Формула w_b: {(result['weight_wb'] > 1).sum():,}")
    print(f"  fallback w=1: {(result['weight_wb'] == 1).sum():,}")


if __name__ == "__main__":
    run()
