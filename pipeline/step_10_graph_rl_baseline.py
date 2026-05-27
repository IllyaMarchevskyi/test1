"""
10 Graph RL Baseline.

Базовий RL-крок для оптимізації частот маршрутів на графі маршрутів.
Реалізація робоча, але залежить від зовнішніх бібліотек:
torch, torch_geometric, stable-baselines3, gymnasium/gym.
"""


def run() -> None:
    from config_loader import cfg
    import json
    import os
    import warnings
    from pathlib import Path

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    from shapely import wkt
    from tqdm.auto import tqdm

    warnings.filterwarnings("ignore")

    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        try:
            import gym  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Для 10_rl потрібні gymnasium або gym. "
                "Також знадобляться torch, torch_geometric і stable-baselines3."
            ) from exc

    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Для 10_rl бракує залежностей. Встановіть: "
            "torch, torch_geometric, stable-baselines3, gymnasium."
        ) from exc

    PROCESSED_DIR = Path("./data/processed")
    OUTPUTS_DIR = Path("./data/outputs")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    ACCESSIBILITY_INDEX = PROCESSED_DIR / "accessibility_index_baseline.csv"
    CATCHMENT_BUILDINGS = PROCESSED_DIR / "catchment_buildings_baseline.parquet"
    FACILITY_ENTROPY = PROCESSED_DIR / "facility_entropy_baseline.parquet"
    BUILDING_WEIGHTS = PROCESSED_DIR / "building_weights_baseline.parquet"
    EASYWAY_ROUTES = Path("../gtfs_static/easyway_routes.csv")
    EASYWAY_METRO = Path("../gtfs_static/easyway_metro.csv")
    SCORES_PATH = Path("../data/processed/accessibility_scores.csv")
    OSM_BRIDGE_PATH = Path("../gtfs_static/osm_easyway_data.csv")
    OSM_STOPS_PATH = Path("../gtfs_static/osm_stops.csv")
    OSM_BRIDGE_METRO_PATH = Path("../gtfs_static/osm_easyway_metro_data.csv")
    GMETRO_PATH = Path("../gtfs_static/gmetro.csv")

    RL_RESULTS_JSON = PROCESSED_DIR / "rl_results.json"
    OPT_FREQ_CSV = PROCESSED_DIR / "optimal_frequencies.csv"
    TARGET_BEFORE_AFTER_JSON = PROCESSED_DIR / "target_facility_before_after.json"
    TARGETS_BEFORE_AFTER_JSON = PROCESSED_DIR / "target_facilities_before_after.json"
    AFFECTED_BEFORE_AFTER_CSV = PROCESSED_DIR / "rl_affected_facilities_before_after.csv"
    GLOBAL_BEFORE_AFTER_CSV = PROCESSED_DIR / "rl_global_facilities_before_after.csv"
    OPT_FREQ_TARGET_CSV = PROCESSED_DIR / "optimal_frequencies_H327.csv"
    OPT_FREQ_TARGETS_CSV = PROCESSED_DIR / "optimal_frequencies_targets.csv"
    MODEL_PATH = PROCESSED_DIR / "rl_model"
    CHECKPOINT_DIR = PROCESSED_DIR / "rl_checkpoints"
    LEARNING_CURVE_PNG = OUTPUTS_DIR / "rl_learning_curve.png"
    TOP_CHANGES_PNG = OUTPUTS_DIR / "rl_top_route_changes.png"
    SCATTER_PNG = OUTPUTS_DIR / "rl_before_after_scatter.png"
    HIST_PNG = OUTPUTS_DIR / "rl_i_peak_hist.png"
    TARGET_LEARNING_CURVE_PNG = OUTPUTS_DIR / "rl_H327_training_curve.png"
    TARGET_ROUTE_CHANGES_PNG = OUTPUTS_DIR / "rl_H327_route_changes.png"
    TARGET_WAIT_SCATTER_PNG = OUTPUTS_DIR / "rl_H327_wait_before_after_scatter.png"
    TARGETS_LEARNING_CURVE_PNG = OUTPUTS_DIR / "rl_targets_training_curve.png"
    TARGETS_ROUTE_CHANGES_PNG = OUTPUTS_DIR / "rl_targets_route_changes.png"
    TARGETS_WAIT_SCATTER_PNG = OUTPUTS_DIR / "rl_targets_wait_before_after_scatter.png"
    RL_CFG = cfg.get("rl", {})
    USE_SUBPROC = bool(RL_CFG.get("use_subproc", False))
    N_ENVS = max(1, int(RL_CFG.get("n_envs", 1)))
    MAX_STEPS = max(1, int(RL_CFG.get("max_steps", 50)))
    TOTAL_TIMESTEPS = max(1, int(RL_CFG.get("total_timesteps", 50000)))
    LEARNING_RATE = float(RL_CFG.get("learning_rate", 3e-4))
    PPO_N_STEPS = max(1, int(RL_CFG.get("n_steps", 50)))
    LOG_EVERY = max(1, int(RL_CFG.get("log_every", 100)))
    CHECKPOINT_EVERY = max(1, int(RL_CFG.get("checkpoint_every", 500)))
    MAX_ROUTE_DELTA = max(0.0, float(RL_CFG.get("max_route_delta", 3.0)))
    ACTION_STEP = max(0.01, float(RL_CFG.get("action_step", 1.0)))
    NON_TARGET_HARM_WEIGHT = float(RL_CFG.get("non_target_harm_weight", 1.0))
    NON_TARGET_HARM_TOLERANCE = max(0.0, float(RL_CFG.get("non_target_harm_tolerance", 0.0)))
    METRIC_EPS = max(0.0, float(RL_CFG.get("metric_epsilon", 1e-9)))
    TARGET_WAIT_REWARD_WEIGHT = float(RL_CFG.get("target_wait_reward_weight", 0.0))

    def parse_config_list(value) -> list[str]:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    EXCLUDED_TRANSPORT_TYPES = set(parse_config_list(RL_CFG.get("exclude_transport_types", [])))
    TARGET_FACILITY_ID_RAW = RL_CFG.get("target_facility_id", "")
    TARGET_FACILITY_ID = str(TARGET_FACILITY_ID_RAW).strip() or None
    TARGET_FACILITY_IDS_RAW = RL_CFG.get("target_facility_ids", [])
    TARGET_FACILITY_IDS = parse_config_list(TARGET_FACILITY_IDS_RAW)
    if not TARGET_FACILITY_IDS and TARGET_FACILITY_ID:
        TARGET_FACILITY_IDS = [TARGET_FACILITY_ID]
    TARGET_MODE = bool(TARGET_FACILITY_IDS)
    PRIMARY_TARGET_ID = TARGET_FACILITY_IDS[0] if TARGET_FACILITY_IDS else None
    TARGET_LABEL = (
        PRIMARY_TARGET_ID
        if len(TARGET_FACILITY_IDS) == 1
        else f"{len(TARGET_FACILITY_IDS)} targets"
    )

    required = [
        ACCESSIBILITY_INDEX,
        CATCHMENT_BUILDINGS,
        FACILITY_ENTROPY,
        BUILDING_WEIGHTS,
        EASYWAY_ROUTES,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Відсутні входи для 10_rl: {missing}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model_zip_path = Path(f"{MODEL_PATH}.zip")
    final_outputs = [
        RL_RESULTS_JSON,
        OPT_FREQ_CSV,
        model_zip_path,
        LEARNING_CURVE_PNG,
        TOP_CHANGES_PNG,
        SCATTER_PNG,
        HIST_PNG,
    ]
    if TARGET_MODE:
        final_outputs.extend(
            [
                OPT_FREQ_TARGETS_CSV,
                TARGETS_BEFORE_AFTER_JSON,
                AFFECTED_BEFORE_AFTER_CSV,
                TARGETS_LEARNING_CURVE_PNG,
                TARGETS_ROUTE_CHANGES_PNG,
                TARGETS_WAIT_SCATTER_PNG,
            ]
        )
        if len(TARGET_FACILITY_IDS) == 1:
            final_outputs.extend(
                [
                    OPT_FREQ_TARGET_CSV,
                    TARGET_BEFORE_AFTER_JSON,
                    TARGET_LEARNING_CURVE_PNG,
                    TARGET_ROUTE_CHANGES_PNG,
                    TARGET_WAIT_SCATTER_PNG,
                ]
            )
    else:
        final_outputs.append(GLOBAL_BEFORE_AFTER_CSV)
    cached_target_ids: list[str] = []
    cached_action_mode = ""
    cached_run = {}
    if RL_RESULTS_JSON.exists():
        try:
            cached_run = json.loads(RL_RESULTS_JSON.read_text(encoding="utf-8")).get("run_config", {})
            cached_target_ids = cached_run.get("target_facility_ids") or []
            if not cached_target_ids:
                cached_single = str(cached_run.get("target_facility_id") or "").strip()
                cached_target_ids = [cached_single] if cached_single else []
            cached_action_mode = str(cached_run.get("action_mode") or "")
        except Exception:
            cached_target_ids = []
            cached_action_mode = ""
            cached_run = {}

    def cache_config_matches(cached_run: dict) -> bool:
        cached_excluded = set(parse_config_list(cached_run.get("exclude_transport_types", [])))
        return (
            cached_target_ids == TARGET_FACILITY_IDS
            and cached_action_mode == "redistribution_pair"
            and int(cached_run.get("max_steps", -1)) == MAX_STEPS
            and int(cached_run.get("total_timesteps", -1)) == TOTAL_TIMESTEPS
            and float(cached_run.get("max_route_delta", -1.0)) == MAX_ROUTE_DELTA
            and float(cached_run.get("action_step", -1.0)) == ACTION_STEP
            and float(cached_run.get("non_target_harm_weight", -1.0)) == NON_TARGET_HARM_WEIGHT
            and float(cached_run.get("non_target_harm_tolerance", -1.0)) == NON_TARGET_HARM_TOLERANCE
            and float(cached_run.get("target_wait_reward_weight", -1.0)) == TARGET_WAIT_REWARD_WEIGHT
            and float(cached_run.get("metric_epsilon", -1.0)) == METRIC_EPS
            and cached_excluded == EXCLUDED_TRANSPORT_TYPES
        )

    if all(path.exists() for path in final_outputs):
        outputs_mtime = min(path.stat().st_mtime for path in final_outputs)
        inputs_mtime = max(path.stat().st_mtime for path in required)
        if (
            outputs_mtime >= inputs_mtime
            and cache_config_matches(cached_run)
        ):
            print("10_rl: кеш RL-результатів уже актуальний, пропускаємо повторне навчання.")
            print(f"  model:   {model_zip_path}")
            print(f"  results: {RL_RESULTS_JSON}")
            print(f"  optimal: {OPT_FREQ_CSV}")
            return

    print("10_rl: завантажуємо baseline-артефакти...")
    print(
        "10_rl: конфіг "
        f"use_subproc={USE_SUBPROC} n_envs={N_ENVS} max_steps={MAX_STEPS} "
        f"total_timesteps={TOTAL_TIMESTEPS} max_route_delta={MAX_ROUTE_DELTA} "
        f"action_step={ACTION_STEP} "
        f"non_target_harm_weight={NON_TARGET_HARM_WEIGHT} "
        f"non_target_harm_tolerance={NON_TARGET_HARM_TOLERANCE} "
        f"target_wait_reward_weight={TARGET_WAIT_REWARD_WEIGHT} "
        f"exclude_transport_types={sorted(EXCLUDED_TRANSPORT_TYPES)}"
    )
    index_df = pd.read_csv(ACCESSIBILITY_INDEX)
    global_index_df = index_df.copy()
    catchment = pd.read_parquet(CATCHMENT_BUILDINGS)
    entropy = pd.read_parquet(FACILITY_ENTROPY)
    weights = pd.read_parquet(BUILDING_WEIGHTS, columns=["building_id", "weight_wb"])
    easyway_parts = [pd.read_csv(EASYWAY_ROUTES)]
    if EASYWAY_METRO.exists():
        easyway_parts.append(pd.read_csv(EASYWAY_METRO))
    easyway = pd.concat(easyway_parts, ignore_index=True)
    print(
        f"10_rl: index={len(index_df):,} catchment={len(catchment):,} "
        f"entropy={len(entropy):,} weights={len(weights):,} easyway={len(easyway):,}"
    )
    if TARGET_MODE:
        if len(TARGET_FACILITY_IDS) == 1:
            print(f"10_rl: локальний режим для закладу {PRIMARY_TARGET_ID}.")
        else:
            print(
                "10_rl: локальний режим для групи закладів: "
                f"{', '.join(TARGET_FACILITY_IDS)}."
            )
    else:
        print("10_rl: глобальний режим по всіх закладах.")

    index_df["facility_id"] = index_df["facility_id"].astype(str)
    catchment["facility_id"] = catchment["facility_id"].astype(str)
    weights["building_id"] = weights["building_id"].astype(int)
    weights["weight_wb"] = pd.to_numeric(weights["weight_wb"], errors="coerce").fillna(1.0).clip(lower=1.0)

    total_city_weight = float(weights["weight_wb"].sum())
    route_to_int = {"bus": 0, "trol": 1, "tram": 2, "metro": 3}

    def parse_schedules(value: str) -> list[int]:
        times = []
        for raw in str(value).strip().split(","):
            raw = raw.strip()
            if not raw or raw == r"\N":
                continue
            hh, mm, ss = raw.split(":")
            times.append(int(hh) * 3600 + int(mm) * 60 + int(ss))
        return sorted(times)

    def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius = 6371000.0
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
        return float(2.0 * radius * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))

    def load_stop_coords_map() -> dict[str, tuple[float, float]]:
        frames: list[pd.DataFrame] = []

        if OSM_BRIDGE_PATH.exists() and OSM_STOPS_PATH.exists():
            bridge = pd.read_csv(OSM_BRIDGE_PATH, usecols=["osm_id", "stop_id"]).dropna()
            bridge["osm_id"] = bridge["osm_id"].astype(str)
            bridge["stop_id"] = bridge["stop_id"].astype(str)
            osm_stops_raw = pd.read_csv(OSM_STOPS_PATH).dropna(subset=["geometry"]).copy()
            osm_stops_raw["geometry"] = osm_stops_raw["geometry"].map(wkt.loads)
            osm_stops_raw["osm_id"] = osm_stops_raw.index.astype(str)
            osm_stops_raw["lon"] = osm_stops_raw["geometry"].map(lambda geom: getattr(geom, "x", np.nan))
            osm_stops_raw["lat"] = osm_stops_raw["geometry"].map(lambda geom: getattr(geom, "y", np.nan))
            frames.append(
                bridge.merge(
                    osm_stops_raw[["osm_id", "lon", "lat"]],
                    on="osm_id",
                    how="left",
                )[["stop_id", "lon", "lat"]]
            )

        if OSM_BRIDGE_METRO_PATH.exists() and GMETRO_PATH.exists():
            bridge_metro = pd.read_csv(OSM_BRIDGE_METRO_PATH, usecols=["osm_id", "stop_id"]).dropna()
            bridge_metro["osm_id"] = bridge_metro["osm_id"].astype(str)
            bridge_metro["stop_id"] = bridge_metro["stop_id"].astype(str)
            gmetro_raw = pd.read_csv(GMETRO_PATH).dropna(subset=["geometry"]).copy()
            gmetro_raw["geometry"] = gmetro_raw["geometry"].map(wkt.loads)
            gmetro_raw["osm_id"] = gmetro_raw.index.astype(str)
            gmetro_raw["lon"] = gmetro_raw["geometry"].map(lambda geom: getattr(geom, "x", np.nan))
            gmetro_raw["lat"] = gmetro_raw["geometry"].map(lambda geom: getattr(geom, "y", np.nan))
            frames.append(
                bridge_metro.merge(
                    gmetro_raw[["osm_id", "lon", "lat"]],
                    on="osm_id",
                    how="left",
                )[["stop_id", "lon", "lat"]]
            )

        if not frames:
            return {}

        coords_df = pd.concat(frames, ignore_index=True)
        coords_df = coords_df.dropna(subset=["lon", "lat"]).drop_duplicates(subset=["stop_id"])
        return {
            str(row.stop_id): (float(row.lat), float(row.lon))
            for row in coords_df.itertuples(index=False)
        }

    easyway = easyway[easyway["schedules"] != r"\N"].copy()
    easyway["stop_id"] = easyway["stop_id"].astype(str)
    easyway["route_id"] = easyway["route_id"].astype(str)
    easyway["transport"] = easyway["transport"].astype(str)
    easyway["route"] = easyway["route"].astype(str)
    easyway["times"] = easyway["schedules"].apply(parse_schedules)
    easyway["n_departures"] = easyway["times"].apply(len)
    easyway_all = easyway.copy()

    catchment["facility_id"] = catchment["facility_id"].astype(str)
    catchment["peak_route_id"] = catchment["peak_route_id"].astype(str)
    catchment["peak_mode"] = catchment["peak_mode"].astype(str)

    local_route_ids: list[str] | None = None
    if TARGET_MODE:
        target_rows = catchment[catchment["facility_id"].isin(TARGET_FACILITY_IDS)].copy()
        if target_rows.empty:
            raise ValueError(
                "10_rl: не знайдено записів catchment для target_facility_ids="
                f"{TARGET_FACILITY_IDS}"
            )
        route_mask = (
            target_rows["peak_mode"].eq("transit")
            & target_rows["peak_route_id"].notna()
            & target_rows["peak_route_id"].ne("nan")
            & target_rows["peak_route_id"].ne("")
        )
        local_route_ids = sorted(target_rows.loc[route_mask, "peak_route_id"].astype(str).unique().tolist())
        if not local_route_ids:
            raise ValueError(
                "10_rl: для target_facility_ids="
                f"{TARGET_FACILITY_IDS} не знайдено transit-маршрутів у catchment_buildings."
            )
        easyway = easyway[easyway["route_id"].astype(str).isin(local_route_ids)].copy()
        if EXCLUDED_TRANSPORT_TYPES:
            before_routes = easyway["route_id"].nunique()
            easyway = easyway[~easyway["transport"].isin(EXCLUDED_TRANSPORT_TYPES)].copy()
            after_routes = easyway["route_id"].nunique()
            print(
                "10_rl: виключено типи транспорту "
                f"{sorted(EXCLUDED_TRANSPORT_TYPES)}: маршрутів {before_routes} -> {after_routes}."
            )
            if easyway.empty:
                raise ValueError(
                    "10_rl: після exclude_transport_types не лишилось маршрутів для RL."
                )
            local_route_ids = sorted(easyway["route_id"].astype(str).unique().tolist())
        print(
            f"10_rl: локальна підмережа = {len(local_route_ids)} маршрут(ів), "
            f"цільових закладів={len(TARGET_FACILITY_IDS)}, "
            f"рядків target-catchment={len(target_rows):,}."
        )
    else:
        catchment = catchment.copy()
        if EXCLUDED_TRANSPORT_TYPES:
            before_routes = easyway["route_id"].nunique()
            easyway = easyway[~easyway["transport"].isin(EXCLUDED_TRANSPORT_TYPES)].copy()
            after_routes = easyway["route_id"].nunique()
            print(
                "10_rl: глобально виключено типи транспорту "
                f"{sorted(EXCLUDED_TRANSPORT_TYPES)}: маршрутів {before_routes} -> {after_routes}."
            )

    def build_rl_initial_freq(df: pd.DataFrame) -> pd.DataFrame:
        route_stats_full = (
            df.groupby("route_id", as_index=False)
            .agg(
                transport=("transport", "first"),
                route=("route", "first"),
                n_stops=("stop_id", "nunique"),
                total_departures=("n_departures", "sum"),
            )
            .reset_index(drop=True)
        )
        route_stats_full["current_freq"] = (route_stats_full["total_departures"] / 11.0).clip(lower=0.0)
        route_stats_full["transport_type"] = route_stats_full["transport"].map(route_to_int).fillna(0).astype(int)
        route_stats_full["active"] = 1
        route_stats_full["rl_initial_freq"] = 6.0

        # Нормалізуємо частоти окремо по кожному типу транспорту.
        # Шкала відносна в межах типу: bus/trol/tram/metro не
        # порівнюються напряму між собою за абсолютною частотою.
        for transport_name, sub_idx in route_stats_full.groupby("transport").groups.items():
            raw = np.log1p(route_stats_full.loc[sub_idx, "current_freq"].astype(float))
            min_raw = float(raw.min())
            max_raw = float(raw.max())
            if max_raw > min_raw:
                scaled = 1.0 + ((raw - min_raw) / (max_raw - min_raw) * 11.0)
            else:
                scaled = pd.Series(6.0, index=raw.index)
            route_stats_full.loc[sub_idx, "rl_initial_freq"] = scaled.round(2)

        print("\n=== Нормалізовані частоти маршрутів по типах транспорту ===")
        print(f"{'Route':>8} | {'raw_freq':>10} | {'rl_freq':>8} | {'type':>6}")
        print("-" * 44)
        top5 = route_stats_full.nlargest(5, "rl_initial_freq")
        for row in top5.itertuples(index=False):
            print(
                f"{str(row.route_id):>8} | "
                f"{float(row.current_freq):>10.2f} | "
                f"{float(row.rl_initial_freq):>8.2f} | "
                f"{str(row.transport):>6}"
            )
        print("  ...")
        bottom5 = route_stats_full.nsmallest(5, "rl_initial_freq")
        for row in bottom5.itertuples(index=False):
            print(
                f"{str(row.route_id):>8} | "
                f"{float(row.current_freq):>10.2f} | "
                f"{float(row.rl_initial_freq):>8.2f} | "
                f"{str(row.transport):>6}"
            )

        return route_stats_full

    route_stats_full = build_rl_initial_freq(easyway_all)
    route_stats = (
        easyway.groupby("route_id", as_index=False)
        .agg(
            transport=("transport", "first"),
            route=("route", "first"),
            n_stops=("stop_id", "nunique"),
            total_departures=("n_departures", "sum"),
        )
        .reset_index(drop=True)
    )
    route_stats = route_stats.merge(
        route_stats_full[["route_id", "current_freq", "transport_type", "active", "rl_initial_freq"]],
        on="route_id",
        how="left",
    )

    route_peak_mean = (
        catchment[catchment["peak_route_id"].notna()]
        .merge(index_df[["facility_id", "I_peak"]], on="facility_id", how="left")
        .groupby("peak_route_id", as_index=False)["I_peak"]
        .mean()
        .rename(columns={"peak_route_id": "route_id", "I_peak": "mean_I_peak"})
    )
    route_stats = route_stats.merge(route_peak_mean, on="route_id", how="left")
    route_stats["mean_I_peak"] = pd.to_numeric(route_stats["mean_I_peak"], errors="coerce").fillna(0.0)
    if route_stats.empty:
        raise ValueError("10_rl: після фільтрації не лишилось маршрутів для RL-середовища.")
    print(f"10_rl: маршрутів у графі = {len(route_stats):,}")

    route_stops = easyway.groupby("route_id")["stop_id"].apply(set).to_dict()
    route_ids = route_stats["route_id"].tolist()
    route_index = {route_id: idx for idx, route_id in enumerate(route_ids)}
    route_type_to_indices: dict[int, list[int]] = {}
    for idx, transport_type in enumerate(route_stats["transport_type"].to_numpy(dtype=int)):
        route_type_to_indices.setdefault(int(transport_type), []).append(idx)
    transfer_actions = [
        (donor_idx, receiver_idx)
        for same_type_indices in route_type_to_indices.values()
        for donor_idx in same_type_indices
        for receiver_idx in same_type_indices
        if donor_idx != receiver_idx
    ]
    if not transfer_actions:
        raise ValueError(
            "10_rl: не вдалося побудувати action space перерозподілу. "
            "Потрібно щонайменше два маршрути одного типу транспорту."
        )
    print(f"10_rl: action space перерозподілу = {len(transfer_actions):,} пар donor→receiver.")

    graph = nx.Graph()
    for route_id in route_ids:
        graph.add_node(route_index[route_id])

    stop_to_routes: dict[str, set[str]] = {}
    for route_id, stops in route_stops.items():
        for stop_id in stops:
            stop_to_routes.setdefault(stop_id, set()).add(route_id)

    for routes in tqdm(stop_to_routes.values(), desc="10_rl edges"):
        routes_list = [route_index[rid] for rid in routes if rid in route_index]
        for i in range(len(routes_list)):
            for j in range(i + 1, len(routes_list)):
                graph.add_edge(routes_list[i], routes_list[j])

    edge_pairs = list(graph.edges())
    if edge_pairs:
        edge_index = torch.tensor(
            np.array(edge_pairs + [(b, a) for a, b in edge_pairs]).T,
            dtype=torch.long,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    print(f"10_rl: ребер у графі = {len(edge_pairs):,}")

    catchment = catchment.merge(weights, on="building_id", how="left")
    catchment["weight_wb"] = pd.to_numeric(catchment["weight_wb"], errors="coerce").fillna(1.0).clip(lower=1.0)
    catchment = catchment.merge(entropy[["facility_id", "Hnorm_peak"]], on="facility_id", how="left")
    catchment["Hnorm_peak"] = pd.to_numeric(catchment["Hnorm_peak"], errors="coerce").fillna(0.0)

    initial_i_peak = dict(zip(index_df["facility_id"], pd.to_numeric(index_df["I_peak"], errors="coerce").fillna(0.0)))
    hnorm_by_facility = dict(zip(entropy["facility_id"].astype(str), pd.to_numeric(entropy["Hnorm_peak"], errors="coerce").fillna(0.0)))
    base_freq_by_route = dict(zip(route_stats["route_id"], pd.to_numeric(route_stats["rl_initial_freq"], errors="coerce").fillna(6.0)))
    base_freq_by_idx = np.array(
        [
            max(float(base_freq_by_route.get(route_id, 1.0)), 1.0)
            for route_id in route_ids
        ],
        dtype=np.float32,
    )

    catchment["_route_idx"] = catchment["peak_route_id"].astype(str).map(route_index)
    catchment["_is_transit_route"] = (
        catchment["peak_mode"].eq("transit")
        & catchment["_route_idx"].notna()
        & catchment["peak_wait_min"].notna()
    )
    numeric_cols = [
        "peak_total_min",
        "peak_wait_min",
        "peak_walk_in_min",
        "peak_transit_min",
        "peak_walk_out_min",
        "weight_wb",
    ]
    for col in numeric_cols:
        catchment[col] = pd.to_numeric(catchment[col], errors="coerce")

    facility_arrays: dict[str, dict[str, np.ndarray | float]] = {}
    facilities_by_route: dict[int, set[str]] = {}
    for fid, group in catchment.groupby("facility_id", sort=False):
        valid = group["peak_total_min"].notna().to_numpy()
        if not valid.any():
            continue

        group_valid = group.loc[valid]
        route_idx = group_valid["_route_idx"].fillna(-1).to_numpy(dtype=np.int32)
        is_transit = group_valid["_is_transit_route"].to_numpy(dtype=bool)
        for idx in np.unique(route_idx[route_idx >= 0]):
            facilities_by_route.setdefault(int(idx), set()).add(str(fid))

        facility_arrays[str(fid)] = {
            "total": group_valid["peak_total_min"].to_numpy(dtype=np.float32),
            "wait": group_valid["peak_wait_min"].fillna(0.0).to_numpy(dtype=np.float32),
            "walk_in": group_valid["peak_walk_in_min"].fillna(0.0).to_numpy(dtype=np.float32),
            "transit": group_valid["peak_transit_min"].fillna(0.0).to_numpy(dtype=np.float32),
            "walk_out": group_valid["peak_walk_out_min"].fillna(0.0).to_numpy(dtype=np.float32),
            "weight": group_valid["weight_wb"].fillna(1.0).to_numpy(dtype=np.float32),
            "route_idx": route_idx,
            "is_transit": is_transit,
            "hnorm": float(hnorm_by_facility.get(str(fid), 0.0)),
        }

    target_facility_set = set(TARGET_FACILITY_IDS)
    transfer_action_affected = [
        facilities_by_route.get(donor_idx, set())
        | facilities_by_route.get(receiver_idx, set())
        for donor_idx, receiver_idx in transfer_actions
    ]
    target_initial_by_facility = {
        fid: float(initial_i_peak.get(fid, 0.0))
        for fid in TARGET_FACILITY_IDS
    }
    target_initial_i_peak = (
        float(np.mean(list(target_initial_by_facility.values())))
        if TARGET_MODE
        else None
    )
    global_initial_i_peak = dict(
        zip(
            global_index_df["facility_id"].astype(str),
            pd.to_numeric(global_index_df["I_peak"], errors="coerce").fillna(0.0),
        )
    )
    global_initial_mean = float(pd.to_numeric(global_index_df["I_peak"], errors="coerce").fillna(0.0).mean())
    global_n_facilities = max(len(global_initial_i_peak), 1)

    class TransitGAT(torch.nn.Module):
        def __init__(self, in_channels: int = 6, hidden: int = 64, out: int = 32):
            super().__init__()
            self.gat1 = GATConv(in_channels, hidden, heads=4)
            self.gat2 = GATConv(hidden * 4, out, heads=1)

        def forward(self, x, edge_index):
            x = F.elu(self.gat1(x, edge_index))
            x = self.gat2(x, edge_index)
            global_state = x.mean(dim=0)
            return x, global_state

    class KyivTransitEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.route_ids = route_ids
            self.route_types = route_stats["transport_type"].to_numpy(dtype=int)
            self.initial_freq = route_stats["rl_initial_freq"].to_numpy(dtype=float)
            self.max_steps = MAX_STEPS

            self.initial_budget = {
                int(tt): float(self.initial_freq[self.route_types == tt].sum())
                for tt in np.unique(self.route_types)
            }
            self.action_space = gym.spaces.Discrete(len(transfer_actions))
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self.route_ids) * 6,),
                dtype=np.float32,
            )
            self.reset()

        def _get_obs(self):
            rows = []
            max_budget = max(self.initial_budget.values()) if self.initial_budget else 1
            max_freq = max(float(self.initial_freq.max()), 1.0)
            max_stops = max(int(route_stats["n_stops"].max()), 1)
            max_i = max(float(route_stats["mean_I_peak"].max()), 1e-6)
            for idx in range(len(self.route_ids)):
                ttype = float(self.route_types[idx]) / 3.0
                current_freq = float(self.current_freq[idx]) / max_freq
                n_stops = float(route_stats.iloc[idx]["n_stops"]) / max_stops
                mean_i_peak = float(route_stats.iloc[idx]["mean_I_peak"]) / max_i if max_i > 0 else 0.0
                active = float(self.active[idx])
                budget_norm = float(self.budget[int(self.route_types[idx])]) / max_budget if max_budget > 0 else 0.0
                rows.extend([ttype, current_freq, n_stops, mean_i_peak, active, budget_norm])
            return np.array(rows, dtype=np.float32)

        def _recalc_I(self, facility_id: str) -> float:
            data = facility_arrays.get(facility_id)
            if data is None:
                return 0.0

            total = data["total"]
            adjusted_total = np.array(total, dtype=np.float32, copy=True)
            route_idx = data["route_idx"]
            transit_mask = data["is_transit"]

            if bool(np.any(transit_mask)):
                idxs = route_idx[transit_mask]
                valid = (
                    (idxs >= 0)
                    & (self.current_freq[idxs] > 0)
                    & (self.active[idxs] > 0)
                )
                if bool(np.any(valid)):
                    transit_positions = np.flatnonzero(transit_mask)[valid]
                    valid_route_idxs = idxs[valid]
                    # Спрощення для локального RL:
                    # середній wait масштабуємо обернено до частоти,
                    # але sigma лишаємо сталою. Тобто міняємо лише
                    # середню компоненту очікування, а не повністю
                    # перебудовуємо розподіл інтервалів.
                    scaled_wait = (
                        data["wait"][transit_positions]
                        * (base_freq_by_idx[valid_route_idxs] / self.current_freq[valid_route_idxs])
                    )
                    adjusted_total[transit_positions] = (
                        data["walk_in"][transit_positions]
                        + scaled_wait
                        + data["transit"][transit_positions]
                        + data["walk_out"][transit_positions]
                    )

                invalid_positions = np.flatnonzero(transit_mask)[~valid] if "valid" in locals() else []
                if len(invalid_positions):
                    keep_mask = np.ones(len(adjusted_total), dtype=bool)
                    keep_mask[invalid_positions] = False
                    adjusted_total = adjusted_total[keep_mask]
                    weights_arr = data["weight"][keep_mask]
                else:
                    weights_arr = data["weight"]
            else:
                weights_arr = data["weight"]

            if len(adjusted_total) == 0:
                return 0.0

            weighted_sum = float(np.sum(weights_arr * np.exp(-0.05 * adjusted_total)))
            return (weighted_sum / total_city_weight) * float(data["hnorm"])

        def _mean_i_peak(self, facility_ids: set[str] | list[str]) -> float:
            if not facility_ids:
                return 0.0
            return float(np.mean([self.I_peak.get(str(fid), 0.0) for fid in facility_ids]))

        def _target_mean(self) -> float:
            return self._mean_i_peak(TARGET_FACILITY_IDS) if TARGET_MODE else self.prev_mean

        def _facility_wait_saving(self, facility_id: str) -> float:
            data = facility_arrays.get(facility_id)
            if data is None:
                return 0.0

            route_idx = data["route_idx"]
            transit_mask = data["is_transit"]
            if not bool(np.any(transit_mask)):
                return 0.0

            idxs = route_idx[transit_mask]
            valid = (
                (idxs >= 0)
                & (self.current_freq[idxs] > 0)
                & (self.active[idxs] > 0)
            )
            if not bool(np.any(valid)):
                return 0.0

            transit_positions = np.flatnonzero(transit_mask)[valid]
            valid_route_idxs = idxs[valid]
            base_wait = data["wait"][transit_positions]
            scaled_wait = base_wait * (base_freq_by_idx[valid_route_idxs] / self.current_freq[valid_route_idxs])
            weights_arr = data["weight"][transit_positions]
            if float(np.sum(weights_arr)) <= 0.0:
                return float(np.mean(base_wait - scaled_wait))
            return float(np.average(base_wait - scaled_wait, weights=weights_arr))

        def _target_wait_saving(self) -> float:
            if not TARGET_MODE:
                return 0.0
            return float(np.mean([self._facility_wait_saving(fid) for fid in TARGET_FACILITY_IDS]))

        def _non_target_harm(self) -> float:
            non_target_ids = self.affected_facility_ids - target_facility_set
            if not non_target_ids:
                return 0.0
            before = float(np.mean([initial_i_peak.get(fid, 0.0) for fid in non_target_ids]))
            after = self._mean_i_peak(non_target_ids)
            # Толеранс відсікає числовий шум: штрафуємо тільки реальне
            # погіршення нецільових закладів, а не похибку float.
            return max(0.0, before - after - NON_TARGET_HARM_TOLERANCE)

        def _objective_value(self) -> float:
            if TARGET_MODE:
                return self._target_mean() - (NON_TARGET_HARM_WEIGHT * self._non_target_harm())
            return self.prev_mean

        def step(self, action):
            action_idx = int(action)
            donor_idx, receiver_idx = transfer_actions[action_idx]
            transport_type = int(self.route_types[donor_idx])

            invalid_action = False

            if int(self.route_types[receiver_idx]) != transport_type:
                invalid_action = True
            elif self.current_freq[donor_idx] - ACTION_STEP < 1.0:
                invalid_action = True
            elif self.current_freq[donor_idx] - ACTION_STEP < max(1.0, self.initial_freq[donor_idx] - MAX_ROUTE_DELTA):
                invalid_action = True
            elif self.current_freq[receiver_idx] + ACTION_STEP > 12.0:
                invalid_action = True
            elif self.current_freq[receiver_idx] + ACTION_STEP > min(12.0, self.initial_freq[receiver_idx] + MAX_ROUTE_DELTA):
                invalid_action = True
            else:
                self.current_freq[donor_idx] -= ACTION_STEP
                self.current_freq[receiver_idx] += ACTION_STEP

            if not invalid_action:
                affected = transfer_action_affected[action_idx]
                self.affected_facility_ids.update(affected)
                for fid in affected:
                    self.I_peak[fid] = self._recalc_I(fid)

            if TARGET_MODE:
                new_value = self._target_mean()
                new_wait_saving = self._target_wait_saving()
                wait_reward = new_wait_saving - self.prev_target_wait_saving
                non_target_harm = self._non_target_harm()
                reward = -1.0 if invalid_action else (
                    (new_value - self.prev_value)
                    + (TARGET_WAIT_REWARD_WEIGHT * wait_reward)
                    - (NON_TARGET_HARM_WEIGHT * non_target_harm)
                )
                self.prev_value = new_value
                self.prev_target_wait_saving = new_wait_saving
                self.current_target_i_peak = new_value
                self.current_target_wait_saving = new_wait_saving
                self.current_objective_value = self._objective_value()
                self.current_non_target_harm = non_target_harm
            else:
                new_mean = float(np.mean(list(self.I_peak.values()))) if self.I_peak else 0.0
                reward = -1.0 if invalid_action else (new_mean - self.prev_mean)
                self.prev_mean = new_mean
                self.current_objective_value = new_mean

            self.step_count += 1
            terminated = self.step_count >= self.max_steps
            truncated = False
            return self._get_obs(), reward, terminated, truncated, {}

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.current_freq = self.initial_freq.copy()
            self.active = np.ones(len(self.route_ids), dtype=int)
            self.budget = self.initial_budget.copy()
            self.step_count = 0
            self.I_peak = initial_i_peak.copy()
            self.prev_mean = float(np.mean(list(self.I_peak.values()))) if self.I_peak else 0.0
            self.prev_value = float(target_initial_i_peak or 0.0)
            self.current_target_i_peak = float(target_initial_i_peak or 0.0)
            self.prev_target_wait_saving = 0.0
            self.current_target_wait_saving = 0.0
            self.affected_facility_ids: set[str] = set()
            self.current_non_target_harm = 0.0
            self.current_objective_value = self._objective_value()
            return self._get_obs(), {}

    class ProgressCallback(BaseCallback):
        def __init__(self, log_every: int = 100, checkpoint_every: int = 500):
            super().__init__()
            self.log_every = log_every
            self.checkpoint_every = checkpoint_every
            self.history = []
            self.target_i_history = []
            self.target_wait_history = []

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards")
            if rewards is not None and len(rewards):
                self.history.append(float(np.mean(rewards)))

            if TARGET_MODE:
                try:
                    current_i = self.training_env.get_attr("current_target_i_peak")[0]
                    self.target_i_history.append(float(current_i))
                    current_wait = self.training_env.get_attr("current_target_wait_saving")[0]
                    self.target_wait_history.append(float(current_wait))
                except Exception:
                    pass

            if self.n_calls % self.log_every == 0 and self.history:
                recent = self.history[-min(len(self.history), self.log_every):]
                progress_pct = (100.0 * self.num_timesteps / TOTAL_TIMESTEPS) if TOTAL_TIMESTEPS > 0 else 0.0
                if TARGET_MODE and self.target_i_history:
                    current_i = self.target_i_history[-1]
                    print(
                        f"10_rl: learn progress={progress_pct:.1f}% "
                        f"calls={self.n_calls} timesteps={self.num_timesteps} "
                        f"mean_reward={np.mean(recent):.6f} "
                        f"I_peak({TARGET_LABEL})={current_i:.6f} "
                        f"wait_saving={float(self.target_wait_history[-1] if self.target_wait_history else 0.0):.3f}хв"
                    )
                else:
                    print(
                        f"10_rl: learn progress={progress_pct:.1f}% "
                        f"calls={self.n_calls} timesteps={self.num_timesteps} "
                        f"mean_reward={np.mean(recent):.6f}"
                    )

            if self.n_calls % self.checkpoint_every == 0:
                checkpoint_path = CHECKPOINT_DIR / f"ppo_checkpoint_{self.n_calls}"
                self.model.save(str(checkpoint_path))
            return True

    def make_env():
        return KyivTransitEnv()

    print("10_rl: створюємо середовище...")
    if USE_SUBPROC:
        n_envs = min(N_ENVS, os.cpu_count() or 1)
        print(f"10_rl: запускаємо {n_envs} паралельних середовищ через SubprocVecEnv.")
        try:
            env = SubprocVecEnv([make_env for _ in range(n_envs)])
        except PermissionError as exc:
            print(
                "10_rl: SubprocVecEnv недоступний у цьому середовищі "
                f"({exc}); переходимо на DummyVecEnv."
            )
            n_envs = N_ENVS
            env = DummyVecEnv([make_env for _ in range(n_envs)])
    else:
        n_envs = N_ENVS
        print(f"10_rl: запускаємо {n_envs} середовище(ищ) через DummyVecEnv.")
        env = DummyVecEnv([make_env for _ in range(n_envs)])
    print("10_rl: середовище створено успішно.")

    model_fresh = (
        model_zip_path.exists()
        and model_zip_path.stat().st_mtime >= max(path.stat().st_mtime for path in required)
        and cache_config_matches(cached_run)
    )

    if model_fresh:
        print(f"10_rl: знайдено актуальну модель, підвантажуємо {model_zip_path} і пропускаємо learn().")
        model = PPO.load(str(MODEL_PATH), env=env)
        callback = None
    else:
        print("10_rl: створюємо PPO модель...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=LEARNING_RATE,
            n_steps=PPO_N_STEPS,
            verbose=1,
        )
        print("10_rl: PPO модель створено.")
        callback = ProgressCallback(log_every=LOG_EVERY, checkpoint_every=CHECKPOINT_EVERY)
        print(f"10_rl: стартуємо model.learn(total_timesteps={TOTAL_TIMESTEPS})...")
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        print("10_rl: навчання завершено.")
        model.save(str(MODEL_PATH))

    print("10_rl: запускаємо детерміновану оцінку навченої політики...")
    eval_env = KyivTransitEnv()
    obs, _ = eval_env.reset()
    done = False
    route_deltas = []
    best_eval_step = 0
    best_eval_value = (
        float(eval_env.current_objective_value)
        if TARGET_MODE
        else float(eval_env.prev_mean)
    )
    best_eval_target_value = float(eval_env.current_target_i_peak) if TARGET_MODE else float(eval_env.prev_mean)
    best_freq_snapshot = eval_env.current_freq.copy()
    eval_progress = tqdm(total=eval_env.max_steps, desc="10_rl eval", leave=True)
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        current_eval_value = (
            float(eval_env.current_objective_value)
            if TARGET_MODE
            else float(eval_env.prev_mean)
        )
        if current_eval_value > best_eval_value:
            best_eval_value = current_eval_value
            best_eval_target_value = float(eval_env.current_target_i_peak) if TARGET_MODE else current_eval_value
            best_eval_step = int(eval_env.step_count)
            best_freq_snapshot = eval_env.current_freq.copy()
        eval_progress.update(1)
        eval_progress.set_postfix(
            step=eval_env.step_count,
            mean_I_peak=f"{(eval_env.current_target_i_peak if TARGET_MODE else eval_env.prev_mean):.6f}",
        )
    eval_progress.close()
    if TARGET_MODE:
        print(
            f"10_rl: найкращий стан для {TARGET_LABEL} знайдено на кроці {best_eval_step} "
            f"з objective={best_eval_value:.6f}, I_peak={best_eval_target_value:.6f}"
        )

    optimal_freq = best_freq_snapshot if TARGET_MODE else eval_env.current_freq
    if TARGET_MODE:
        initial_freq_equal = bool(np.allclose(best_freq_snapshot, eval_env.initial_freq))
        print(
            "10_rl debug: best-state summary | "
            f"step={best_eval_step} "
            f"best_objective={best_eval_value:.6f} "
            f"same_as_initial={initial_freq_equal}"
        )
        changed_count = int(np.sum(~np.isclose(best_freq_snapshot, eval_env.initial_freq)))
        print(f"10_rl debug: best snapshot changed routes = {changed_count}")

    for idx, route_id in enumerate(route_ids):
        initial = float(eval_env.initial_freq[idx])
        optimal = float(optimal_freq[idx])
        route_deltas.append(
            {
                "route_id": route_id,
                "initial_freq": round(initial, 2),
                "optimal_freq": round(optimal, 2),
                "delta": round(optimal - initial, 2),
                "transport_type": int(eval_env.route_types[idx]),
                "transport": route_stats.iloc[idx]["transport"],
                "route": route_stats.iloc[idx]["route"],
            }
        )

    optimal_freq_df = pd.DataFrame(route_deltas).sort_values("delta", ascending=False).reset_index(drop=True)
    optimal_freq_df.to_csv(OPT_FREQ_CSV, index=False, encoding="utf-8")
    if TARGET_MODE:
        optimal_freq_df.to_csv(OPT_FREQ_TARGETS_CSV, index=False, encoding="utf-8")
    if len(TARGET_FACILITY_IDS) == 1:
        optimal_freq_df.to_csv(OPT_FREQ_TARGET_CSV, index=False, encoding="utf-8")

    target_changed_routes = optimal_freq_df[optimal_freq_df["delta"] != 0].copy()
    if TARGET_MODE and SCORES_PATH.exists():
        scores = pd.read_csv(SCORES_PATH, usecols=["facility_id", "lat", "lon", "name"])
        scores["facility_id"] = scores["facility_id"].astype(str)
        target_rows = scores[scores["facility_id"].isin(TARGET_FACILITY_IDS)].copy()
        stop_coords_map = load_stop_coords_map()

        if not target_rows.empty and stop_coords_map and not target_changed_routes.empty:
            facility_coords = {
                str(row.facility_id): (float(row.lat), float(row.lon), str(getattr(row, "name", row.facility_id)))
                for row in target_rows.itertuples(index=False)
            }
            target_label_text = (
                f"{PRIMARY_TARGET_ID} ({facility_coords.get(PRIMARY_TARGET_ID, (None, None, PRIMARY_TARGET_ID))[2]})"
                if len(TARGET_FACILITY_IDS) == 1 and PRIMARY_TARGET_ID
                else ", ".join(TARGET_FACILITY_IDS)
            )
            print(f"10_rl: маршрути, що змінились для {target_label_text}:")
            for row in target_changed_routes.itertuples(index=False):
                route_stop_ids = route_stops.get(str(row.route_id), set())
                distances = []
                for stop_id in route_stop_ids:
                    coords = stop_coords_map.get(str(stop_id))
                    if coords is None:
                        continue
                    stop_lat, stop_lon = coords
                    for target_id, (facility_lat, facility_lon, facility_name) in facility_coords.items():
                        distances.append(
                            (
                                target_id,
                                facility_name,
                                haversine_m(stop_lat, stop_lon, facility_lat, facility_lon),
                            )
                        )
                if distances:
                    nearest_target_id, nearest_name, min_dist_m = min(distances, key=lambda item: item[2])
                    print(
                        f"  {row.route_id} ({row.transport} {row.route}): "
                        f"delta={float(row.delta):+.2f}, найближча зупинка до {nearest_target_id} "
                        f"({nearest_name}): {min_dist_m:.0f}м"
                    )
                else:
                    print(
                        f"  {row.route_id} ({row.transport} {row.route}): "
                        f"delta={float(row.delta):+.2f}, найближча зупинка до target-group: н/д"
                    )

    before_df = index_df[["facility_id", "I_peak"]].rename(columns={"I_peak": "I_peak_before"})
    if TARGET_MODE:
        best_eval_env = KyivTransitEnv()
        best_eval_env.reset()
        best_eval_env.current_freq = best_freq_snapshot.copy()
        target_wait_saving_after = float(best_eval_env._target_wait_saving())
        changed_route_indices = [
            route_index[str(row.route_id)]
            for row in target_changed_routes.itertuples(index=False)
            if str(row.route_id) in route_index
        ]
        best_affected_facility_ids = set()
        for route_idx in changed_route_indices:
            best_affected_facility_ids.update(facilities_by_route.get(route_idx, set()))
        best_affected_facility_ids.update(TARGET_FACILITY_IDS)
        best_target_i_by_facility = {
            fid: float(best_eval_env._recalc_I(fid))
            for fid in TARGET_FACILITY_IDS
        }
        best_affected_i_by_facility = {
            fid: float(best_eval_env._recalc_I(fid))
            for fid in sorted(best_affected_facility_ids)
        }
        print("10_rl debug: recomputed target values at best snapshot:")
        for target_id in TARGET_FACILITY_IDS:
            print(
                f"  {target_id}: "
                f"baseline={float(target_initial_by_facility.get(target_id, 0.0)):.6f} "
                f"recomputed={float(best_target_i_by_facility.get(target_id, 0.0)):.6f}"
            )
        after_df = pd.DataFrame(
            [
                {"facility_id": fid, "I_peak_after": val}
                for fid, val in best_affected_i_by_facility.items()
            ]
        )
    else:
        target_wait_saving_after = None
        after_df = pd.DataFrame(
            [{"facility_id": fid, "I_peak_after": val} for fid, val in eval_env.I_peak.items()]
        )
    compare_df = before_df.merge(after_df, on="facility_id", how="left")
    compare_df["I_peak_after"] = pd.to_numeric(compare_df["I_peak_after"], errors="coerce").fillna(compare_df["I_peak_before"])

    facility_names = {
        str(row.facility_id): str(row.name)
        for row in index_df[["facility_id", "name"]].dropna(subset=["name"]).itertuples(index=False)
    } if "name" in index_df.columns else {}
    if SCORES_PATH.exists():
        try:
            scores_names = pd.read_csv(SCORES_PATH, usecols=["facility_id", "name"])
            scores_names["facility_id"] = scores_names["facility_id"].astype(str)
            for row in scores_names.dropna(subset=["name"]).itertuples(index=False):
                facility_names.setdefault(str(row.facility_id), str(row.name))
        except Exception:
            pass

    def gini(values):
        arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float, copy=True)
        if len(arr) == 0 or arr.sum() <= 0:
            return 0.0
        arr.sort()
        n = len(arr)
        idx = np.arange(1, n + 1, dtype=float)
        return float(np.sum((2 * idx - n - 1) * arr) / (n * arr.sum()))

    before_mean = float(compare_df["I_peak_before"].mean())
    after_mean = float(compare_df["I_peak_after"].mean())
    target_after_i_peak = (
        float(np.mean(list(best_target_i_by_facility.values())))
        if TARGET_MODE
        else None
    )
    if TARGET_MODE:
        print(
            "10_rl debug: target-group mean | "
            f"baseline={float(target_initial_i_peak or 0.0):.6f} "
            f"after={float(target_after_i_peak or 0.0):.6f}"
        )
    global_after_mean = None
    affected_summary = None
    global_facilities_summary = None
    affected_compare = pd.DataFrame()
    if TARGET_MODE:
        global_after_mean = after_mean
        affected_compare = compare_df[compare_df["facility_id"].isin(best_affected_facility_ids)].copy()
        non_target_affected_compare = affected_compare[
            ~affected_compare["facility_id"].isin(TARGET_FACILITY_IDS)
        ].copy()
        if not affected_compare.empty:
            affected_compare["delta"] = affected_compare["I_peak_after"] - affected_compare["I_peak_before"]
            affected_compare["delta_clean"] = affected_compare["delta"].where(affected_compare["delta"].abs() > METRIC_EPS, 0.0)
            affected_compare["delta_pct"] = np.where(
                affected_compare["I_peak_before"].abs() > 0,
                affected_compare["delta"] / affected_compare["I_peak_before"] * 100.0,
                np.nan,
            )
            affected_compare["is_target"] = affected_compare["facility_id"].isin(TARGET_FACILITY_IDS)
            affected_compare["name"] = affected_compare["facility_id"].map(facility_names).fillna(affected_compare["facility_id"])
            affected_compare = affected_compare[
                [
                    "facility_id",
                    "name",
                    "is_target",
                    "I_peak_before",
                    "I_peak_after",
                    "delta",
                    "delta_clean",
                    "delta_pct",
                ]
            ].sort_values(["is_target", "delta"], ascending=[False, True])
            affected_compare.to_csv(AFFECTED_BEFORE_AFTER_CSV, index=False, encoding="utf-8")

            non_target_affected_compare["delta"] = (
                non_target_affected_compare["I_peak_after"]
                - non_target_affected_compare["I_peak_before"]
            )
            non_target_affected_compare["delta_clean"] = non_target_affected_compare["delta"].where(
                non_target_affected_compare["delta"].abs() > METRIC_EPS,
                0.0,
            )
            affected_summary = {
                "affected_facilities_count": int(len(affected_compare)),
                "non_target_affected_count": int(len(non_target_affected_compare)),
                "improved_count": int((affected_compare["delta_clean"] > 0).sum()),
                "worsened_count": int((affected_compare["delta_clean"] < 0).sum()),
                "non_target_mean_before": (
                    float(non_target_affected_compare["I_peak_before"].mean())
                    if not non_target_affected_compare.empty
                    else None
                ),
                "non_target_mean_after": (
                    float(non_target_affected_compare["I_peak_after"].mean())
                    if not non_target_affected_compare.empty
                    else None
                ),
                "non_target_mean_delta": (
                    float(non_target_affected_compare["delta_clean"].mean())
                    if not non_target_affected_compare.empty
                    else None
                ),
                "max_drop": float(affected_compare["delta_clean"].min()),
                "metric_epsilon": METRIC_EPS,
                "affected_before_after_csv": str(AFFECTED_BEFORE_AFTER_CSV),
            }
    else:
        global_compare = compare_df.copy()
        global_compare["delta"] = global_compare["I_peak_after"] - global_compare["I_peak_before"]
        global_compare["delta_clean"] = global_compare["delta"].where(global_compare["delta"].abs() > METRIC_EPS, 0.0)
        global_compare["delta_pct"] = np.where(
            global_compare["I_peak_before"].abs() > 0,
            global_compare["delta"] / global_compare["I_peak_before"] * 100.0,
            np.nan,
        )
        global_compare["name"] = global_compare["facility_id"].map(facility_names).fillna(global_compare["facility_id"])
        global_compare["is_improved"] = global_compare["delta_clean"] > 0
        global_compare["is_worsened"] = global_compare["delta_clean"] < 0
        global_compare = global_compare[
            [
                "facility_id",
                "name",
                "I_peak_before",
                "I_peak_after",
                "delta",
                "delta_clean",
                "delta_pct",
                "is_improved",
                "is_worsened",
            ]
        ].sort_values("delta_clean", ascending=False)
        global_compare.to_csv(GLOBAL_BEFORE_AFTER_CSV, index=False, encoding="utf-8")
        global_facilities_summary = {
            "facilities_count": int(len(global_compare)),
            "improved_count": int(global_compare["is_improved"].sum()),
            "worsened_count": int(global_compare["is_worsened"].sum()),
            "unchanged_count": int((global_compare["delta_clean"] == 0).sum()),
            "max_gain": float(global_compare["delta_clean"].max()) if not global_compare.empty else 0.0,
            "max_drop": float(global_compare["delta_clean"].min()) if not global_compare.empty else 0.0,
            "mean_delta": float(global_compare["delta_clean"].mean()) if not global_compare.empty else 0.0,
            "median_delta": float(global_compare["delta_clean"].median()) if not global_compare.empty else 0.0,
            "metric_epsilon": METRIC_EPS,
            "global_before_after_csv": str(GLOBAL_BEFORE_AFTER_CSV),
        }

    results = {
        "target_facility_id": PRIMARY_TARGET_ID,
        "target_facility_ids": TARGET_FACILITY_IDS,
        "before": {
            "I_peak_target_mean": float(target_initial_i_peak) if TARGET_MODE else None,
            "target_wait_saving_min": 0.0 if TARGET_MODE else None,
            "mean_I_peak": global_initial_mean if TARGET_MODE else before_mean,
            "gini": gini(compare_df["I_peak_before"]),
            "moran": None,
        },
        "after": {
            "I_peak_target_mean": target_after_i_peak if TARGET_MODE else None,
            "target_wait_saving_min": target_wait_saving_after if TARGET_MODE else None,
            "mean_I_peak": global_after_mean if TARGET_MODE else after_mean,
            "gini": gini(compare_df["I_peak_after"]),
            "moran": None,
        },
        "route_changes": {
            "increased": optimal_freq_df[optimal_freq_df["delta"] > 0][["route_id", "delta"]].to_dict("records"),
            "decreased": optimal_freq_df[optimal_freq_df["delta"] < 0][["route_id", "delta"]].to_dict("records"),
            "disabled": optimal_freq_df[optimal_freq_df["optimal_freq"] == 0][["route_id", "delta"]].to_dict("records"),
        },
        "affected_facilities": affected_summary,
        "global_facilities": global_facilities_summary,
        "run_config": {
            "target_facility_id": PRIMARY_TARGET_ID,
            "target_facility_ids": TARGET_FACILITY_IDS,
            "use_subproc": USE_SUBPROC,
            "n_envs": n_envs,
            "max_steps": MAX_STEPS,
            "total_timesteps": TOTAL_TIMESTEPS,
            "learning_rate": LEARNING_RATE,
            "n_steps": PPO_N_STEPS,
            "max_route_delta": MAX_ROUTE_DELTA,
            "action_step": ACTION_STEP,
            "non_target_harm_weight": NON_TARGET_HARM_WEIGHT,
            "non_target_harm_tolerance": NON_TARGET_HARM_TOLERANCE,
            "target_wait_reward_weight": TARGET_WAIT_REWARD_WEIGHT,
            "metric_epsilon": METRIC_EPS,
            "exclude_transport_types": sorted(EXCLUDED_TRANSPORT_TYPES),
            "action_mode": "redistribution_pair",
        },
    }
    RL_RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if TARGET_MODE:
        per_target_rows = []
        for target_id in TARGET_FACILITY_IDS:
            facility_name = facility_names.get(target_id, target_id)
            before_val = float(target_initial_by_facility.get(target_id, 0.0))
            after_val = float(best_target_i_by_facility.get(target_id, 0.0))
            per_target_rows.append(
                {
                    "facility_id": target_id,
                    "name": facility_name,
                    "I_peak_before": before_val,
                    "I_peak_after": after_val,
                    "delta": after_val - before_val,
                    "delta_pct": (((after_val - before_val) / before_val) * 100.0) if before_val != 0.0 else None,
                }
            )

        targets_before_after = {
            "target_facility_ids": TARGET_FACILITY_IDS,
            "I_peak_before_mean": float(target_initial_i_peak or 0.0),
            "I_peak_after_mean": float(target_after_i_peak or 0.0),
            "target_wait_saving_min": target_wait_saving_after,
            "delta": float((target_after_i_peak or 0.0) - (target_initial_i_peak or 0.0)),
            "delta_pct": (
                float((((target_after_i_peak or 0.0) - (target_initial_i_peak or 0.0)) / target_initial_i_peak) * 100.0)
                if float(target_initial_i_peak or 0.0) != 0.0
                else None
            ),
            "facilities": per_target_rows,
            "routes_considered": route_ids,
            "routes_changed": {
                str(row.route_id): round(float(row.delta), 2)
                for row in target_changed_routes.itertuples(index=False)
            },
        }
        TARGETS_BEFORE_AFTER_JSON.write_text(
            json.dumps(targets_before_after, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if len(TARGET_FACILITY_IDS) == 1:
            single_row = per_target_rows[0]
            target_before_after = {
                **single_row,
                "target_wait_saving_min": target_wait_saving_after,
                "routes_considered": route_ids,
                "routes_changed": {
                    str(row.route_id): round(float(row.delta), 2)
                    for row in target_changed_routes.itertuples(index=False)
                },
            }
            TARGET_BEFORE_AFTER_JSON.write_text(
                json.dumps(target_before_after, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # ── графік 1: крива навчання (reward) ────────────────────────────────────
    history = callback.history if (callback is not None and callback.history) else [0.0]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(history, color="#BBDEFB", alpha=0.45, linewidth=1, label="Сирий reward")
    smooth_w = max(1, len(history) // 50)
    smoothed = pd.Series(history).rolling(smooth_w, min_periods=1).mean()
    ax.plot(smoothed, color="#1565C0", linewidth=2.2, label=f"Ковзне середнє (вікно={smooth_w})")
    ax.axhline(0, color="#555", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_title("Крива навчання PPO: середній reward по кроках", fontsize=14, fontweight="bold")
    ax.set_xlabel("Крок callback", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(LEARNING_CURVE_PNG, dpi=150)
    plt.close(fig)

    # ── графік 2: крива I*_peak target-group під час навчання ────────────────
    if TARGET_MODE:
        target_i_history = (
            callback.target_i_history
            if (callback is not None and callback.target_i_history)
            else [float(target_initial_i_peak or 0.0)]
        )
        baseline_val = float(target_initial_i_peak or 0.0)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(target_i_history, color="#A5D6A7", alpha=0.45, linewidth=1, label=f"I*_peak({TARGET_LABEL})")
        smooth_w2 = max(1, len(target_i_history) // 50)
        smoothed2 = pd.Series(target_i_history).rolling(smooth_w2, min_periods=1).mean()
        ax.plot(smoothed2, color="#2E7D32", linewidth=2.2, label="Ковзне середнє")
        ax.axhline(
            baseline_val,
            color="#C62828", linestyle="--", linewidth=1.5,
            label=f"Базовий рівень: {baseline_val:.6f}",
        )
        final_val = float(target_i_history[-1])
        delta_pct = ((final_val - baseline_val) / baseline_val * 100.0) if baseline_val != 0 else 0.0
        ylabel = (
            f"I*_peak({PRIMARY_TARGET_ID})"
            if len(TARGET_FACILITY_IDS) == 1 and PRIMARY_TARGET_ID
            else "Середній I*_peak(target-group)"
        )
        ax.set_title(
            f"Навчання PPO: {ylabel} по кроках  "
            f"(Δ = {delta_pct:+.2f}%)",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Крок callback", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(TARGETS_LEARNING_CURVE_PNG, dpi=150)
        if len(TARGET_FACILITY_IDS) == 1:
            fig.savefig(TARGET_LEARNING_CURVE_PNG, dpi=150)
        plt.close(fig)

    # ── графік 3: топ-10 маршрутів за зміною частоти ─────────────────────────
    top_changes = (
        optimal_freq_df
        .reindex(optimal_freq_df["delta"].abs().sort_values(ascending=False).index)
        .head(10)
        .copy()
    )
    top_changes["label"] = top_changes.apply(
        lambda r: f"{r['transport']} {r['route']}", axis=1
    )
    bar_colors_global = ["#2E7D32" if d > 0 else "#C62828" for d in top_changes["delta"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(top_changes)), top_changes["delta"], color=bar_colors_global)
    ax.set_xticks(range(len(top_changes)))
    ax.set_xticklabels(top_changes["label"], rotation=40, ha="right", fontsize=10)
    ax.axhline(0, color="#333", linewidth=0.9)
    for bar, val in zip(bars, top_changes["delta"]):
        offset = 0.05 if val >= 0 else -0.18
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{float(val):+.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_title("Топ-10 маршрутів за зміною частоти рейсів/год", fontsize=14, fontweight="bold")
    ax.set_xlabel("Маршрут (вид транспорту + номер)", fontsize=12)
    ax.set_ylabel("Δ частоти (рейс/год)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(TOP_CHANGES_PNG, dpi=150)
    plt.close(fig)

    if TARGET_MODE:
        bar_colors_local = ["#2E7D32" if d > 0 else "#C62828" for d in top_changes["delta"]]
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(top_changes)), top_changes["delta"], color=bar_colors_local)
        ax.set_xticks(range(len(top_changes)))
        ax.set_xticklabels(top_changes["label"], rotation=40, ha="right", fontsize=10)
        ax.axhline(0, color="#333", linewidth=0.9)
        for bar, val in zip(bars, top_changes["delta"]):
            offset = 0.05 if val >= 0 else -0.18
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{float(val):+.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
        ax.set_title(
            f"Зміни частот маршрутів локальної підмережі {TARGET_LABEL}",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Маршрут (вид транспорту + номер)", fontsize=12)
        ax.set_ylabel("Δ частоти (рейс/год)", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(TARGETS_ROUTE_CHANGES_PNG, dpi=150)
        if len(TARGET_FACILITY_IDS) == 1:
            fig.savefig(TARGET_ROUTE_CHANGES_PNG, dpi=150)
        plt.close(fig)

    # ── графік 4: scatter до/після по всіх закладах ───────────────────────────
    max_val = max(float(compare_df["I_peak_before"].max()), float(compare_df["I_peak_after"].max()))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        compare_df["I_peak_before"], compare_df["I_peak_after"],
        s=18, alpha=0.5, color="#1565C0", edgecolors="none", label="Заклади",
    )
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="#555", linewidth=1.2, label="Без змін")
    if TARGET_MODE:
        target_rows = compare_df[compare_df["facility_id"].isin(TARGET_FACILITY_IDS)]
        if not target_rows.empty:
            ax.scatter(
                target_rows["I_peak_before"], target_rows["I_peak_after"],
                s=90, color="#E65100", zorder=5, label="Target-group"
            )
            for row in target_rows.itertuples(index=False):
                ax.annotate(
                    str(row.facility_id),
                    xy=(float(row.I_peak_before), float(row.I_peak_after)),
                    xytext=(8, 4), textcoords="offset points",
                    fontsize=10, color="#E65100", fontweight="bold",
                )
    ax.set_xlabel("I*_peak до оптимізації", fontsize=12)
    ax.set_ylabel("I*_peak після оптимізації", fontsize=12)
    ax.set_title("Зміна доступності закладів: до vs після", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SCATTER_PNG, dpi=150)
    plt.close(fig)

    # ── графік 5: гістограма розподілу I*_peak до/після ──────────────────────
    med_before = float(compare_df["I_peak_before"].median())
    med_after = float(compare_df["I_peak_after"].median())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(compare_df["I_peak_before"], bins=30, alpha=0.65, label="До", color="#1565C0", edgecolor="white")
    ax.hist(compare_df["I_peak_after"], bins=30, alpha=0.65, label="Після", color="#E65100", edgecolor="white")
    ax.axvline(med_before, color="#0D47A1", linestyle="--", linewidth=1.8,
               label=f"Медіана до: {med_before:.4f}")
    ax.axvline(med_after, color="#BF360C", linestyle="--", linewidth=1.8,
               label=f"Медіана після: {med_after:.4f}")
    ax.set_title(
        "Розподіл індексу доступності I*_peak до та після оптимізації",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("I*_peak", fontsize=12)
    ax.set_ylabel("Кількість закладів", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HIST_PNG, dpi=150)
    plt.close(fig)

    # ── графік 6: scatter очікування до/після для змінених маршрутів target-group ──
    if TARGET_MODE:
        wait_plot_rows = []
        changed_route_set = set(target_changed_routes["route_id"].astype(str))
        for row in catchment.itertuples(index=False):
            route_id = str(getattr(row, "peak_route_id", ""))
            if route_id not in changed_route_set:
                continue
            old_wait = getattr(row, "peak_wait_min", np.nan)
            if pd.isna(old_wait):
                continue
            route_idx = route_index.get(route_id)
            if route_idx is None:
                continue
            base_freq = max(float(base_freq_by_route.get(route_id, 1.0)), 1.0)
            curr_freq = max(float(optimal_freq[route_idx]), 1.0)
            new_wait = float(old_wait) * (base_freq / curr_freq)
            route_label = f"{route_stats[route_stats['route_id'] == route_id]['transport'].values[0]} " \
                          f"{route_stats[route_stats['route_id'] == route_id]['route'].values[0]}" \
                          if route_id in route_stats["route_id"].values else route_id
            wait_plot_rows.append({
                "route_id": route_id,
                "route_label": route_label,
                "wait_before": float(old_wait),
                "wait_after": new_wait,
            })
        if wait_plot_rows:
            wait_df = pd.DataFrame(wait_plot_rows)
            max_wait = max(float(wait_df["wait_before"].max()), float(wait_df["wait_after"].max()))
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(
                wait_df["wait_before"], wait_df["wait_after"],
                s=20, alpha=0.6, color="#1565C0", edgecolors="none",
            )
            ax.plot([0, max_wait], [0, max_wait], linestyle="--", color="#555",
                    linewidth=1.2, label="Без змін")
            improved = (wait_df["wait_after"] < wait_df["wait_before"]).sum()
            ax.text(
                0.03, 0.96,
                f"Покращено будівель: {improved} / {len(wait_df)}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", alpha=0.8),
            )
            ax.set_xlabel("Час очікування до (хв)", fontsize=12)
            ax.set_ylabel("Час очікування після (хв)", fontsize=12)
            ax.set_title(
                f"Час очікування до/після зміни частот\n(маршрути локальної підмережі {TARGET_LABEL})",
                fontsize=13, fontweight="bold",
            )
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(TARGETS_WAIT_SCATTER_PNG, dpi=150)
            if len(TARGET_FACILITY_IDS) == 1:
                fig.savefig(TARGET_WAIT_SCATTER_PNG, dpi=150)
            plt.close(fig)

    gat_model = TransitGAT()
    feature_tensor = torch.tensor(
        route_stats[["transport_type", "current_freq", "n_stops", "mean_I_peak", "active"]]
        .assign(budget_remaining=1.0)
        .to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    _node_embeddings, global_state = gat_model(feature_tensor, edge_index)
    print(f"10_rl: GAT глобальний embedding shape = {tuple(global_state.shape)}")
    print(f"10_rl: модель збережено в {MODEL_PATH}.zip")
    print(f"10_rl: результати збережено в {RL_RESULTS_JSON}")


if __name__ == "__main__":
    run()
