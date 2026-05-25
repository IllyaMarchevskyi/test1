"""
10 Graph RL Baseline.

Базовий RL-крок для оптимізації частот маршрутів на графі маршрутів.
Реалізація робоча, але залежить від зовнішніх бібліотек:
torch, torch_geometric, stable-baselines3, gymnasium/gym.
"""


def run() -> None:
    import json
    import os
    import warnings
    from pathlib import Path

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
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
        from stable_baselines3.common.vec_env import SubprocVecEnv
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

    RL_RESULTS_JSON = PROCESSED_DIR / "rl_results.json"
    OPT_FREQ_CSV = PROCESSED_DIR / "optimal_frequencies.csv"
    MODEL_PATH = PROCESSED_DIR / "rl_model"
    CHECKPOINT_DIR = PROCESSED_DIR / "rl_checkpoints"
    LEARNING_CURVE_PNG = OUTPUTS_DIR / "rl_learning_curve.png"
    TOP_CHANGES_PNG = OUTPUTS_DIR / "rl_top_route_changes.png"
    SCATTER_PNG = OUTPUTS_DIR / "rl_before_after_scatter.png"
    HIST_PNG = OUTPUTS_DIR / "rl_i_peak_hist.png"

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

    print("10_rl: завантажуємо baseline-артефакти...")
    index_df = pd.read_csv(ACCESSIBILITY_INDEX)
    catchment = pd.read_parquet(CATCHMENT_BUILDINGS)
    entropy = pd.read_parquet(FACILITY_ENTROPY)
    weights = pd.read_parquet(BUILDING_WEIGHTS, columns=["building_id", "weight_wb"])
    easyway = pd.read_csv(EASYWAY_ROUTES)

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

    easyway = easyway[easyway["schedules"] != r"\N"].copy()
    easyway["stop_id"] = easyway["stop_id"].astype(str)
    easyway["route_id"] = easyway["route_id"].astype(str)
    easyway["transport"] = easyway["transport"].astype(str)
    easyway["route"] = easyway["route"].astype(str)
    easyway["times"] = easyway["schedules"].apply(parse_schedules)
    easyway["n_departures"] = easyway["times"].apply(len)

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
    route_stats["current_freq"] = (route_stats["total_departures"] / 11.0).clip(lower=0.0)
    route_stats["transport_type"] = route_stats["transport"].map(route_to_int).fillna(0).astype(int)
    route_stats["active"] = 1

    route_peak_mean = (
        catchment[catchment["peak_route_id"].notna()]
        .merge(index_df[["facility_id", "I_peak"]], on="facility_id", how="left")
        .groupby("peak_route_id", as_index=False)["I_peak"]
        .mean()
        .rename(columns={"peak_route_id": "route_id", "I_peak": "mean_I_peak"})
    )
    route_stats = route_stats.merge(route_peak_mean, on="route_id", how="left")
    route_stats["mean_I_peak"] = pd.to_numeric(route_stats["mean_I_peak"], errors="coerce").fillna(0.0)

    route_stops = easyway.groupby("route_id")["stop_id"].apply(set).to_dict()
    route_ids = route_stats["route_id"].tolist()
    route_index = {route_id: idx for idx, route_id in enumerate(route_ids)}

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

    catchment = catchment.merge(weights, on="building_id", how="left")
    catchment["weight_wb"] = pd.to_numeric(catchment["weight_wb"], errors="coerce").fillna(1.0).clip(lower=1.0)
    catchment = catchment.merge(entropy[["facility_id", "Hnorm_peak"]], on="facility_id", how="left")
    catchment["Hnorm_peak"] = pd.to_numeric(catchment["Hnorm_peak"], errors="coerce").fillna(0.0)
    catchment["peak_route_id"] = catchment["peak_route_id"].astype(str)

    facility_rows = {}
    facilities_by_route: dict[int, set[str]] = {}
    for row in catchment.itertuples(index=False):
        fid = str(row.facility_id)
        facility_rows.setdefault(fid, []).append(row)
        rid = str(row.peak_route_id)
        if rid and rid != "nan" and rid in route_index:
            facilities_by_route.setdefault(route_index[rid], set()).add(fid)

    initial_i_peak = dict(zip(index_df["facility_id"], pd.to_numeric(index_df["I_peak"], errors="coerce").fillna(0.0)))
    hnorm_by_facility = dict(zip(entropy["facility_id"].astype(str), pd.to_numeric(entropy["Hnorm_peak"], errors="coerce").fillna(0.0)))
    base_freq_by_route = dict(zip(route_stats["route_id"], route_stats["current_freq"]))

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
            self.initial_freq = route_stats["current_freq"].round().clip(lower=0, upper=12).to_numpy(dtype=int)
            self.max_steps = 50

            self.initial_budget = {
                int(tt): int(self.initial_freq[self.route_types == tt].sum())
                for tt in np.unique(self.route_types)
            }
            self.action_space = gym.spaces.Discrete(len(self.route_ids) * 2)
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
            max_freq = max(int(self.initial_freq.max()), 1)
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
            rows = facility_rows.get(facility_id, [])
            if not rows:
                return 0.0

            weighted_sum = 0.0
            for row in rows:
                total_min = getattr(row, "peak_total_min", np.nan)
                if pd.isna(total_min):
                    continue

                route_id = str(getattr(row, "peak_route_id", ""))
                wait_min = getattr(row, "peak_wait_min", np.nan)
                walk_in = getattr(row, "peak_walk_in_min", np.nan)
                transit_min = getattr(row, "peak_transit_min", np.nan)
                walk_out = getattr(row, "peak_walk_out_min", np.nan)
                mode = getattr(row, "peak_mode", None)
                weight_wb = float(getattr(row, "weight_wb", 1.0))

                adjusted_total = float(total_min)
                if mode == "transit" and route_id in route_index and pd.notna(wait_min):
                    idx = route_index[route_id]
                    base_freq = max(float(base_freq_by_route.get(route_id, 1.0)), 1.0)
                    current_freq = max(float(self.current_freq[idx]), 0.0)
                    if current_freq <= 0 or not self.active[idx]:
                        continue
                    scaled_wait = float(wait_min) * (base_freq / current_freq)
                    adjusted_total = (
                        float(walk_in or 0.0)
                        + scaled_wait
                        + float(transit_min or 0.0)
                        + float(walk_out or 0.0)
                    )

                weighted_sum += weight_wb * float(np.exp(-0.05 * adjusted_total))

            return (weighted_sum / total_city_weight) * float(hnorm_by_facility.get(facility_id, 0.0))

        def step(self, action):
            route_idx = int(action // 2)
            action_type = int(action % 2)
            transport_type = int(self.route_types[route_idx])

            if action_type == 0:
                if self.budget.get(transport_type, 0) <= 0:
                    return self._get_obs(), -1.0, False, False, {}
                if self.current_freq[route_idx] >= 12:
                    return self._get_obs(), -1.0, False, False, {}
                if not self.active[route_idx]:
                    self.active[route_idx] = 1
                self.current_freq[route_idx] += 1
                self.budget[transport_type] -= 1
            else:
                if self.current_freq[route_idx] <= 0:
                    return self._get_obs(), -1.0, False, False, {}
                self.current_freq[route_idx] -= 1
                self.budget[transport_type] += 1
                if self.current_freq[route_idx] == 0:
                    self.active[route_idx] = 0

            affected = facilities_by_route.get(route_idx, set())
            for fid in affected:
                self.I_peak[fid] = self._recalc_I(fid)

            new_mean = float(np.mean(list(self.I_peak.values()))) if self.I_peak else 0.0
            reward = new_mean - self.prev_mean
            self.prev_mean = new_mean

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
            return self._get_obs(), {}

    class ProgressCallback(BaseCallback):
        def __init__(self, log_every: int = 100, checkpoint_every: int = 500):
            super().__init__()
            self.log_every = log_every
            self.checkpoint_every = checkpoint_every
            self.history = []

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards")
            if rewards is not None and len(rewards):
                self.history.append(float(np.mean(rewards)))

            if self.n_calls % self.log_every == 0 and self.history:
                print(f"10_rl: step={self.n_calls} mean_reward={np.mean(self.history[-self.log_every:]):.6f}")

            if self.n_calls % self.checkpoint_every == 0:
                checkpoint_path = CHECKPOINT_DIR / f"ppo_checkpoint_{self.n_calls}"
                self.model.save(str(checkpoint_path))
            return True

    def make_env():
        return KyivTransitEnv()

    n_envs = min(16, os.cpu_count() or 1)
    print(f"10_rl: запускаємо {n_envs} паралельних середовищ.")
    env = SubprocVecEnv([make_env for _ in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=50,
        verbose=1,
    )
    callback = ProgressCallback(log_every=100, checkpoint_every=500)
    total_timesteps = 5000 * 50
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(str(MODEL_PATH))

    eval_env = KyivTransitEnv()
    obs, _ = eval_env.reset()
    done = False
    route_deltas = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated

    optimal_freq = eval_env.current_freq
    for idx, route_id in enumerate(route_ids):
        initial = int(eval_env.initial_freq[idx])
        optimal = int(optimal_freq[idx])
        route_deltas.append(
            {
                "route_id": route_id,
                "initial_freq": initial,
                "optimal_freq": optimal,
                "delta": optimal - initial,
                "transport_type": int(eval_env.route_types[idx]),
                "transport": route_stats.iloc[idx]["transport"],
                "route": route_stats.iloc[idx]["route"],
            }
        )

    optimal_freq_df = pd.DataFrame(route_deltas).sort_values("delta", ascending=False).reset_index(drop=True)
    optimal_freq_df.to_csv(OPT_FREQ_CSV, index=False, encoding="utf-8")

    before_df = index_df[["facility_id", "I_peak"]].rename(columns={"I_peak": "I_peak_before"})
    after_df = pd.DataFrame(
        [{"facility_id": fid, "I_peak_after": val} for fid, val in eval_env.I_peak.items()]
    )
    compare_df = before_df.merge(after_df, on="facility_id", how="left")
    compare_df["I_peak_after"] = pd.to_numeric(compare_df["I_peak_after"], errors="coerce").fillna(compare_df["I_peak_before"])

    def gini(values):
        arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        if len(arr) == 0 or arr.sum() <= 0:
            return 0.0
        arr.sort()
        n = len(arr)
        idx = np.arange(1, n + 1, dtype=float)
        return float(np.sum((2 * idx - n - 1) * arr) / (n * arr.sum()))

    before_mean = float(compare_df["I_peak_before"].mean())
    after_mean = float(compare_df["I_peak_after"].mean())

    results = {
        "before": {
            "mean_I_peak": before_mean,
            "gini": gini(compare_df["I_peak_before"]),
            "moran": None,
        },
        "after": {
            "mean_I_peak": after_mean,
            "gini": gini(compare_df["I_peak_after"]),
            "moran": None,
        },
        "route_changes": {
            "increased": optimal_freq_df[optimal_freq_df["delta"] > 0][["route_id", "delta"]].to_dict("records"),
            "decreased": optimal_freq_df[optimal_freq_df["delta"] < 0][["route_id", "delta"]].to_dict("records"),
            "disabled": optimal_freq_df[optimal_freq_df["optimal_freq"] == 0][["route_id", "delta"]].to_dict("records"),
        },
    }
    RL_RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    history = callback.history if callback.history else [0.0]
    plt.figure(figsize=(10, 4))
    plt.plot(history, color="#1B6B23")
    plt.title("Навчання PPO: середній reward по кроках")
    plt.xlabel("Крок callback")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(LEARNING_CURVE_PNG, dpi=150)
    plt.close()

    top_changes = optimal_freq_df.reindex(optimal_freq_df["delta"].abs().sort_values(ascending=False).index).head(10)
    plt.figure(figsize=(10, 5))
    plt.bar(top_changes["route_id"], top_changes["delta"], color="#EB9328")
    plt.title("Топ-10 маршрутів за зміною частоти")
    plt.xlabel("route_id")
    plt.ylabel("Δ частоти")
    plt.tight_layout()
    plt.savefig(TOP_CHANGES_PNG, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(compare_df["I_peak_before"], compare_df["I_peak_after"], s=12, alpha=0.6, color="#2980B9")
    max_val = max(compare_df["I_peak_before"].max(), compare_df["I_peak_after"].max())
    plt.plot([0, max_val], [0, max_val], linestyle="--", color="#333333")
    plt.xlabel("I*_peak до")
    plt.ylabel("I*_peak після")
    plt.title("До vs після оптимізації")
    plt.tight_layout()
    plt.savefig(SCATTER_PNG, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(compare_df["I_peak_before"], bins=30, alpha=0.6, label="До", color="#1FFF2E")
    plt.hist(compare_df["I_peak_after"], bins=30, alpha=0.6, label="Після", color="#FF0000")
    plt.title("Розподіл I*_peak до/після")
    plt.xlabel("I*_peak")
    plt.ylabel("Кількість закладів")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HIST_PNG, dpi=150)
    plt.close()

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
