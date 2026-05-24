"""
09 Accessibility Index Baseline.

Рахує індекс доступності I*(f), глобальні метрики нерівності
та Transit Gap Index для baseline-гілки без пересадок.
"""


def run() -> None:
    from config_loader import cfg
    import json
    import os
    import pickle
    import warnings
    from pathlib import Path

    import geopandas as gpd
    import networkx as nx
    import numpy as np
    import osmnx as ox
    import pandas as pd
    from tqdm.auto import tqdm

    warnings.filterwarnings("ignore")

    PROCESSED_DIR = Path("./data/processed")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    BUILDING_WEIGHTS_PATH = PROCESSED_DIR / "building_weights_baseline.parquet"
    CATCHMENT_BUILDINGS_PATH = PROCESSED_DIR / "catchment_buildings_baseline.parquet"
    FACILITY_ENTROPY_PATH = PROCESSED_DIR / "facility_entropy_baseline.parquet"
    SCORES_PATH = Path(cfg["paths"]["scores"])
    BUILDINGS_PATH = Path("../data/processed/buildings.parquet")

    OUT_INDEX_CSV = PROCESSED_DIR / "accessibility_index_baseline.csv"
    OUT_PREVIEW_CSV = PROCESSED_DIR / "accessibility_index_preview_baseline.csv"
    OUT_GLOBAL_JSON = PROCESSED_DIR / "global_metrics_baseline.json"
    CAR_INDEX_CSV = PROCESSED_DIR / "car_accessibility_baseline.csv"
    DRIVE_GRAPH_CACHE = PROCESSED_DIR / "kyiv_drive_graph_proj_baseline.pkl"

    required = [
        BUILDING_WEIGHTS_PATH,
        CATCHMENT_BUILDINGS_PATH,
        FACILITY_ENTROPY_PATH,
        SCORES_PATH,
        BUILDINGS_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Відсутні входи для 09_index: {missing}")

    def outputs_fresh(outputs: list[Path], inputs: list[Path]) -> bool:
        if not all(path.exists() for path in outputs):
            return False
        return min(path.stat().st_mtime for path in outputs) >= max(path.stat().st_mtime for path in inputs)

    if outputs_fresh([OUT_INDEX_CSV, OUT_PREVIEW_CSV, OUT_GLOBAL_JSON], required):
        cached = pd.read_csv(OUT_INDEX_CSV)
        print(f"09_index: кеш індексу завантажено: {len(cached):,} закладів")
        print(f"Середній I*_peak: {cached['I_peak'].mean():.4f}")
        print(f"Медіана I*_peak: {cached['I_peak'].median():.4f}")
        print(f"Мін/макс: {cached['I_peak'].min():.4f} / {cached['I_peak'].max():.4f}")
        return

    print("09_index: завантажуємо baseline-дані...")
    weights = pd.read_parquet(BUILDING_WEIGHTS_PATH, columns=["building_id", "weight_wb"])
    catchment = pd.read_parquet(CATCHMENT_BUILDINGS_PATH)
    entropy = pd.read_parquet(
        FACILITY_ENTROPY_PATH,
        columns=["facility_id", "Hnorm_peak", "Hnorm_offpeak"],
    )
    scores = pd.read_csv(SCORES_PATH, usecols=["facility_id", "facility_type", "name", "lat", "lon"])

    weights["building_id"] = weights["building_id"].astype(int)
    weights["weight_wb"] = pd.to_numeric(weights["weight_wb"], errors="coerce").fillna(1.0).clip(lower=1.0)
    catchment["facility_id"] = catchment["facility_id"].astype(str)
    catchment["building_id"] = catchment["building_id"].astype(int)
    entropy["facility_id"] = entropy["facility_id"].astype(str)
    scores["facility_id"] = scores["facility_id"].astype(str)

    total_city_weight = float(weights["weight_wb"].sum())
    if total_city_weight <= 0:
        raise ValueError("Сума weight_wb по місту дорівнює нулю, індекс не можна нормалізувати.")

    print(f"09_index: сумарна вага міста Σw_b = {total_city_weight:,.2f}")
    print("09_index: приєднуємо weight_wb та H_norm...")
    catchment = catchment.merge(weights, on="building_id", how="left")
    catchment["weight_wb"] = pd.to_numeric(catchment["weight_wb"], errors="coerce").fillna(1.0).clip(lower=1.0)

    base_df = scores.merge(entropy, on="facility_id", how="left")
    base_df["Hnorm_peak"] = pd.to_numeric(base_df["Hnorm_peak"], errors="coerce").fillna(0.0)
    base_df["Hnorm_offpeak"] = pd.to_numeric(base_df["Hnorm_offpeak"], errors="coerce").fillna(0.0)

    def aggregate_period(period: str) -> pd.DataFrame:
        group_col = f"group_{period}"
        total_col = f"{period}_total_min"
        subset = catchment[catchment[group_col].notna() & catchment[total_col].notna()].copy()
        if subset.empty:
            return pd.DataFrame(columns=["facility_id", f"weighted_sum_{period}", f"n_buildings_{period}"])

        subset[total_col] = pd.to_numeric(subset[total_col], errors="coerce")
        subset = subset[subset[total_col].notna()].copy()
        subset["decay_weight"] = subset["weight_wb"] * np.exp(-0.05 * subset[total_col].astype(float))

        grouped = (
            subset.groupby("facility_id", as_index=False)
            .agg(
                **{
                    f"weighted_sum_{period}": ("decay_weight", "sum"),
                    f"n_buildings_{period}": ("building_id", "size"),
                }
            )
            .reset_index(drop=True)
        )
        return grouped

    print("09_index: агрегація peak/offpeak...")
    peak_df = aggregate_period("peak")
    offpeak_df = aggregate_period("offpeak")

    result = base_df.merge(peak_df, on="facility_id", how="left").merge(offpeak_df, on="facility_id", how="left")
    result["weighted_sum_peak"] = pd.to_numeric(result["weighted_sum_peak"], errors="coerce").fillna(0.0)
    result["weighted_sum_offpeak"] = pd.to_numeric(result["weighted_sum_offpeak"], errors="coerce").fillna(0.0)
    result["n_buildings_peak"] = pd.to_numeric(result["n_buildings_peak"], errors="coerce").fillna(0).astype(int)
    result["n_buildings_offpeak"] = pd.to_numeric(result["n_buildings_offpeak"], errors="coerce").fillna(0).astype(int)

    result["I_peak"] = (result["weighted_sum_peak"] / total_city_weight) * result["Hnorm_peak"]
    result["I_offpeak"] = (result["weighted_sum_offpeak"] / total_city_weight) * result["Hnorm_offpeak"]
    result["R"] = np.where(result["I_peak"] > 0, result["I_offpeak"] / result["I_peak"], np.nan)

    def gini(values: pd.Series) -> float | None:
        arr = pd.to_numeric(values, errors="coerce").dropna()
        arr = arr[arr >= 0].to_numpy(dtype=float, copy=True)
        if len(arr) == 0:
            return None
        total = arr.sum()
        if total <= 0:
            return 0.0
        arr.sort()
        n = len(arr)
        idx = np.arange(1, n + 1, dtype=float)
        cumsum = np.sum((2 * idx - n - 1) * arr)
        return float(cumsum / (n * total))

    def compute_moran(index_df: pd.DataFrame) -> tuple[float | None, float | None]:
        try:
            from esda.moran import Moran
            import libpysal
        except ModuleNotFoundError:
            print("09_index: esda/libpysal не встановлені, пропускаємо індекс Морана.")
            return None, None

        moran_df = index_df[["facility_id", "I_peak"]].merge(
            scores[["facility_id", "lat", "lon"]],
            on="facility_id",
            how="left",
        )
        moran_df = moran_df.dropna(subset=["I_peak", "lat", "lon"]).copy()
        if len(moran_df) < 6:
            print("09_index: недостатньо точок для Moran's I.")
            return None, None

        facilities_gdf = gpd.GeoDataFrame(
            moran_df,
            geometry=gpd.points_from_xy(moran_df["lon"], moran_df["lat"]),
            crs="EPSG:4326",
        )
        weights = libpysal.weights.KNN.from_dataframe(facilities_gdf, k=5)
        weights.transform = "r"
        moran = Moran(moran_df["I_peak"].to_numpy(dtype=float), weights)
        return float(moran.I), float(moran.p_sim)

    def load_drive_graph():
        if DRIVE_GRAPH_CACHE.exists():
            with open(DRIVE_GRAPH_CACHE, "rb") as fh:
                return pickle.load(fh)

        print("09_index: будуємо автомобільний граф Києва...")
        drive_graph = ox.graph_from_place(cfg["city"]["name"], network_type="drive")
        drive_graph = ox.project_graph(drive_graph, to_crs=cfg["city"]["crs_metric"])
        with open(DRIVE_GRAPH_CACHE, "wb") as fh:
            pickle.dump(drive_graph, fh)
        return drive_graph

    def compute_car_accessibility() -> pd.DataFrame:
        inputs = [BUILDING_WEIGHTS_PATH, BUILDINGS_PATH, SCORES_PATH]
        if outputs_fresh([CAR_INDEX_CSV], inputs):
            print("09_index: кеш автомобільної доступності завантажено.")
            return pd.read_csv(CAR_INDEX_CSV)

        print("09_index: рахуємо I*_car для всіх закладів...")
        buildings = gpd.read_parquet(BUILDINGS_PATH).set_geometry("geometry")
        buildings["building_id"] = buildings["building_id"].astype(int)
        building_geo = buildings.merge(weights, on="building_id", how="left")
        building_geo["weight_wb"] = pd.to_numeric(building_geo["weight_wb"], errors="coerce").fillna(1.0).clip(lower=1.0)

        drive_graph = load_drive_graph()

        bld_nodes = ox.distance.nearest_nodes(
            drive_graph,
            X=building_geo.geometry.x.to_numpy(),
            Y=building_geo.geometry.y.to_numpy(),
        )
        buildings_by_node: dict[int, list[tuple[int, float]]] = {}
        for bid, node, weight in zip(
            building_geo["building_id"].to_numpy(),
            bld_nodes,
            building_geo["weight_wb"].to_numpy(),
        ):
            buildings_by_node.setdefault(int(node), []).append((int(bid), float(weight)))

        facilities_gdf = gpd.GeoDataFrame(
            scores[["facility_id", "lat", "lon"]].copy(),
            geometry=gpd.points_from_xy(scores["lon"], scores["lat"]),
            crs="EPSG:4326",
        ).to_crs(cfg["city"]["crs_metric"])
        facility_nodes = ox.distance.nearest_nodes(
            drive_graph,
            X=facilities_gdf.geometry.x.to_numpy(),
            Y=facilities_gdf.geometry.y.to_numpy(),
        )

        car_rows = []
        cutoff_m = 500 * 60
        for facility_row, center_node in tqdm(
            zip(scores.itertuples(index=False), facility_nodes),
            total=len(scores),
            desc="09_index I_car",
        ):
            dists = nx.single_source_dijkstra_path_length(
                drive_graph,
                int(center_node),
                cutoff=cutoff_m,
                weight="length",
            )
            weighted_sum = 0.0
            n_buildings = 0
            for node_id, dist_m in dists.items():
                if node_id not in buildings_by_node:
                    continue
                car_min = float(dist_m) / 500.0
                decay = float(np.exp(-0.05 * car_min))
                for _, weight_wb in buildings_by_node[node_id]:
                    weighted_sum += weight_wb * decay
                    n_buildings += 1

            car_rows.append(
                {
                    "facility_id": str(facility_row.facility_id),
                    "I_car": weighted_sum / total_city_weight,
                    "n_buildings_car": n_buildings,
                }
            )

        car_df = pd.DataFrame(car_rows)
        car_df.to_csv(CAR_INDEX_CSV, index=False, encoding="utf-8")
        return car_df

    try:
        car_df = compute_car_accessibility()
        result = result.merge(car_df, on="facility_id", how="left")
        result["TGI"] = np.where(
            pd.to_numeric(result["I_car"], errors="coerce").fillna(0.0) > 0,
            1.0 - (result["I_peak"] / result["I_car"]),
            np.nan,
        )
    except Exception as exc:
        print(f"09_index: не вдалося порахувати TGI, пропускаємо цей блок. Причина: {exc}")
        result["I_car"] = np.nan
        result["TGI"] = np.nan

    gini_peak = gini(result["I_peak"])
    gini_offpeak = gini(result["I_offpeak"])
    moran_peak, moran_p_value = compute_moran(result)

    output_df = result[
        [
            "facility_id",
            "I_peak",
            "I_offpeak",
            "R",
            "Hnorm_peak",
            "Hnorm_offpeak",
            "n_buildings_peak",
            "n_buildings_offpeak",
            "TGI",
        ]
    ].rename(
        columns={
            "Hnorm_peak": "H_norm_peak",
            "Hnorm_offpeak": "H_norm_offpeak",
        }
    )
    output_df.to_csv(OUT_INDEX_CSV, index=False, encoding="utf-8")

    preview_df = result[
        [
            "facility_id",
            "facility_type",
            "name",
            "I_peak",
            "I_offpeak",
            "R",
            "TGI",
        ]
    ].copy()
    preview_df.to_csv(OUT_PREVIEW_CSV, index=False, encoding="utf-8")

    global_metrics = {
        "gini_peak": gini_peak,
        "gini_offpeak": gini_offpeak,
        "moran_peak": moran_peak,
        "moran_p_value": moran_p_value,
        "mean_tgi": None if output_df["TGI"].dropna().empty else float(output_df["TGI"].dropna().mean()),
        "mean_I_peak": None if output_df["I_peak"].dropna().empty else float(output_df["I_peak"].dropna().mean()),
        "mean_I_offpeak": None if output_df["I_offpeak"].dropna().empty else float(output_df["I_offpeak"].dropna().mean()),
        "mean_R": None if output_df["R"].dropna().empty else float(output_df["R"].dropna().mean()),
    }
    OUT_GLOBAL_JSON.write_text(json.dumps(global_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    top5 = preview_df.sort_values("I_peak", ascending=False).head(5)
    bottom5 = preview_df.sort_values("I_peak", ascending=True).head(5)

    print(f"09_index: індекс збережено в {OUT_INDEX_CSV}")
    print(f"09_index: preview збережено в {OUT_PREVIEW_CSV}")
    print(f"09_index: глобальні метрики збережено в {OUT_GLOBAL_JSON}")
    print(f"Середній I*_peak: {output_df['I_peak'].mean():.4f}")
    print(f"Медіана I*_peak: {output_df['I_peak'].median():.4f}")
    print(f"Мін/макс: {output_df['I_peak'].min():.4f} / {output_df['I_peak'].max():.4f}")
    print("Топ-5 найдоступніших закладів:")
    for row in top5.itertuples(index=False):
        print(f"  {str(row.name)[:55]:<55} {row.facility_type:<8} I*={row.I_peak:.4f} R={row.R if pd.notna(row.R) else None}")
    print("Топ-5 найменш доступних закладів:")
    for row in bottom5.itertuples(index=False):
        print(f"  {str(row.name)[:55]:<55} {row.facility_type:<8} I*={row.I_peak:.4f} R={row.R if pd.notna(row.R) else None}")
    print(f"Джині (пік): {gini_peak:.4f}" if gini_peak is not None else "Джині (пік): None")
    print(f"Джині (міжпік): {gini_offpeak:.4f}" if gini_offpeak is not None else "Джині (міжпік): None")
    if moran_peak is not None:
        print(f"Індекс Морана (пік): {moran_peak:.4f}")
        print(f"p-value: {moran_p_value:.4f}")


if __name__ == "__main__":
    run()
