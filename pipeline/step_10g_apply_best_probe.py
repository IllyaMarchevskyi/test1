"""
10g Greedy RL probe baseline.

Multi-step heuristic baseline for the target RL setup. The step repeatedly
chooses the best currently valid donor->receiver transfer and applies it
until no positive objective improvement remains.
"""

from __future__ import annotations


def run() -> None:
    from config_loader import cfg
    import json
    from pathlib import Path

    import folium
    import numpy as np
    import pandas as pd
    from branca.element import Element

    PROCESSED_DIR = Path("./data/processed")
    OUTPUTS_DIR = Path("./data/outputs")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    ACCESSIBILITY_INDEX = PROCESSED_DIR / "accessibility_index_baseline.csv"
    CATCHMENT_BUILDINGS = PROCESSED_DIR / "catchment_buildings_baseline.parquet"
    FACILITY_ENTROPY = PROCESSED_DIR / "facility_entropy_baseline.parquet"
    BUILDING_WEIGHTS = PROCESSED_DIR / "building_weights_baseline.parquet"
    MAP_DATA_JSON = PROCESSED_DIR / "map_data_baseline.json"
    RL_RESULTS_JSON = PROCESSED_DIR / "rl_results.json"
    EASYWAY_ROUTES = Path("../gtfs_static/easyway_routes.csv")
    EASYWAY_METRO = Path("../gtfs_static/easyway_metro.csv")
    OSM_EASYWAY_DATA = Path("../gtfs_static/osm_easyway_data.csv")
    OSM_EASYWAY_METRO_DATA = Path("../gtfs_static/osm_easyway_metro_data.csv")

    RESULTS_JSON = PROCESSED_DIR / "rl_best_probe_results.json"
    TARGET_BEFORE_AFTER_JSON = PROCESSED_DIR / "target_facilities_best_probe_before_after.json"
    TARGET_BEFORE_AFTER_CSV = PROCESSED_DIR / "rl_best_probe_target_before_after.csv"
    OPTIMAL_FREQ_CSV = PROCESSED_DIR / "optimal_frequencies_best_probe.csv"
    STEPS_CSV = PROCESSED_DIR / "rl_best_probe_steps.csv"
    COMPARISON_JSON = PROCESSED_DIR / "rl_greedy_vs_ppo_comparison.json"
    TARGET_MAP_HTML = OUTPUTS_DIR / "map_catchment_interactive_best_probe_targets.html"

    required = [
        ACCESSIBILITY_INDEX,
        CATCHMENT_BUILDINGS,
        FACILITY_ENTROPY,
        BUILDING_WEIGHTS,
        EASYWAY_ROUTES,
        MAP_DATA_JSON,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Відсутні входи для 10g_apply_best_probe: {missing}")

    rl_cfg = cfg.get("rl", {})

    def parse_config_list(value) -> list[str]:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    target_ids = parse_config_list(rl_cfg.get("target_facility_ids", []))
    single_target = str(rl_cfg.get("target_facility_id", "")).strip()
    if not target_ids and single_target:
        target_ids = [single_target]
    target_selection = str(rl_cfg.get("target_selection", "bottom_n")).strip().lower()
    target_auto_count = max(1, int(rl_cfg.get("target_auto_count", 10)))
    target_auto_min_actions = max(1, int(rl_cfg.get("target_auto_min_actions", 6)))
    target_auto_max_candidates = max(target_auto_count, int(rl_cfg.get("target_auto_max_candidates", 250)))
    target_auto_min_i_peak = max(0.0, float(rl_cfg.get("target_auto_min_i_peak", 1e-6)))
    require_osm_mapping = bool(rl_cfg.get("require_osm_mapping", False))
    freq_scaling = str(rl_cfg.get("freq_scaling", "log")).strip().lower() or "log"
    allow_cross_type_transfers = bool(rl_cfg.get("allow_cross_type_transfers", False))

    excluded_transport_types = set(parse_config_list(rl_cfg.get("exclude_transport_types", [])))
    max_steps = max(1, int(rl_cfg.get("max_steps", 50)))
    max_route_delta = max(0.0, float(rl_cfg.get("max_route_delta", 3.0)))
    action_step = max(0.01, float(rl_cfg.get("action_step", 1.0)))
    non_target_harm_weight = float(rl_cfg.get("non_target_harm_weight", 1.0))
    non_target_harm_tolerance = max(0.0, float(rl_cfg.get("non_target_harm_tolerance", 0.0)))
    target_wait_reward_weight = float(rl_cfg.get("target_wait_reward_weight", 0.0))
    metric_epsilon = max(0.0, float(rl_cfg.get("metric_epsilon", 1e-8)))

    route_to_int = {"bus": 0, "trol": 1, "tram": 2, "metro": 3}

    def parse_schedules(value: str) -> list[int]:
        times: list[int] = []
        for raw in str(value).strip().split(","):
            raw = raw.strip()
            if not raw or raw == r"\N":
                continue
            hh, mm, ss = raw.split(":")
            times.append(int(hh) * 3600 + int(mm) * 60 + int(ss))
        return sorted(times)

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
        route_stats_full["rl_initial_freq"] = 6.0

        # Та сама нормалізація, що в 10_rl/10f: log або linear + min-max у межах типу транспорту.
        for _, sub_idx in route_stats_full.groupby("transport").groups.items():
            current_freq = route_stats_full.loc[sub_idx, "current_freq"].astype(float)
            raw = np.log1p(current_freq) if freq_scaling == "log" else current_freq
            min_raw = float(raw.min())
            max_raw = float(raw.max())
            if max_raw > min_raw:
                scaled = 1.0 + ((raw - min_raw) / (max_raw - min_raw) * 11.0)
            else:
                scaled = pd.Series(6.0, index=raw.index)
            route_stats_full.loc[sub_idx, "rl_initial_freq"] = scaled.round(2)
        return route_stats_full

    index_df = pd.read_csv(ACCESSIBILITY_INDEX)
    index_df["facility_id"] = index_df["facility_id"].astype(str)
    index_df["I_peak"] = pd.to_numeric(index_df["I_peak"], errors="coerce").fillna(0.0)
    initial_i_peak = dict(zip(index_df["facility_id"], index_df["I_peak"]))

    catchment = pd.read_parquet(CATCHMENT_BUILDINGS)
    catchment["facility_id"] = catchment["facility_id"].astype(str)
    catchment["peak_route_id"] = catchment["peak_route_id"].astype(str)
    catchment["peak_mode"] = catchment["peak_mode"].astype(str)

    entropy = pd.read_parquet(FACILITY_ENTROPY)
    entropy["facility_id"] = entropy["facility_id"].astype(str)
    entropy["Hnorm_peak"] = pd.to_numeric(entropy["Hnorm_peak"], errors="coerce").fillna(0.0)
    hnorm_by_facility = dict(zip(entropy["facility_id"], entropy["Hnorm_peak"]))

    weights = pd.read_parquet(BUILDING_WEIGHTS, columns=["building_id", "weight_wb"])
    weights["building_id"] = weights["building_id"].astype(int)
    weights["weight_wb"] = pd.to_numeric(weights["weight_wb"], errors="coerce").fillna(1.0).clip(lower=1.0)
    total_city_weight = float(weights["weight_wb"].sum())

    easyway_parts = [pd.read_csv(EASYWAY_ROUTES)]
    if EASYWAY_METRO.exists():
        easyway_parts.append(pd.read_csv(EASYWAY_METRO))
    easyway = pd.concat(easyway_parts, ignore_index=True)
    allowed_route_ids: set[str] | None = None
    if require_osm_mapping:
        osm_parts = []
        if OSM_EASYWAY_DATA.exists():
            osm_parts.append(pd.read_csv(OSM_EASYWAY_DATA, usecols=["route_id"]))
        if OSM_EASYWAY_METRO_DATA.exists():
            osm_parts.append(pd.read_csv(OSM_EASYWAY_METRO_DATA, usecols=["route_id"]))
        if not osm_parts:
            raise FileNotFoundError(
                "10g_apply_best_probe: require_osm_mapping=true, але osm_easyway_data.csv не знайдено."
            )
        allowed_route_ids = set(pd.concat(osm_parts, ignore_index=True)["route_id"].astype(str).unique())
        before_routes = easyway["route_id"].astype(str).nunique()
        easyway = easyway[easyway["route_id"].astype(str).isin(allowed_route_ids)].copy()
        after_routes = easyway["route_id"].astype(str).nunique()
        print(
            "10g_apply_best_probe: OSM route filter "
            f"routes={before_routes}->{after_routes}"
        )
    easyway = easyway[easyway["schedules"] != r"\N"].copy()
    easyway["stop_id"] = easyway["stop_id"].astype(str)
    easyway["route_id"] = easyway["route_id"].astype(str)
    easyway["transport"] = easyway["transport"].astype(str)
    easyway["route"] = easyway["route"].astype(str)
    easyway["times"] = easyway["schedules"].apply(parse_schedules)
    easyway["n_departures"] = easyway["times"].apply(len)

    route_stats_full = build_rl_initial_freq(easyway)
    route_transport_by_id = dict(zip(route_stats_full["route_id"].astype(str), route_stats_full["transport"].astype(str)))

    def eligible_facility_routes(facility_id: str) -> set[str]:
        records = catchment[
            catchment["facility_id"].eq(str(facility_id))
            & catchment["peak_mode"].eq("transit")
            & catchment["peak_route_id"].notna()
            & catchment["peak_route_id"].ne("nan")
            & catchment["peak_route_id"].ne("")
        ]
        routes = set(records["peak_route_id"].astype(str).unique().tolist())
        if allowed_route_ids is not None:
            routes = {route_id for route_id in routes if route_id in allowed_route_ids}
        if excluded_transport_types:
            routes = {
                route_id
                for route_id in routes
                if route_transport_by_id.get(route_id, "unknown") not in excluded_transport_types
            }
        return routes

    def action_pairs_count(route_ids: set[str]) -> int:
        counts: dict[str, int] = {}
        for route_id in route_ids:
            transport = route_transport_by_id.get(str(route_id), "unknown")
            counts[transport] = counts.get(transport, 0) + 1
        return int(sum(count * (count - 1) for count in counts.values()))

    if not target_ids:
        if target_selection not in {"bottom_n", "worst", "auto"}:
            raise ValueError(
                "10g_apply_best_probe: target_selection має бути bottom_n/worst/auto "
                "або потрібно явно задати target_facility_id(s)."
            )
        selected_ids: list[str] = []
        selected_routes: set[str] = set()
        candidates = (
            index_df[index_df["I_peak"] > target_auto_min_i_peak]
            .sort_values("I_peak", ascending=True)
            .head(target_auto_max_candidates)
        )
        for row in candidates.itertuples(index=False):
            facility_id = str(row.facility_id)
            routes = eligible_facility_routes(facility_id)
            if not routes:
                continue
            selected_ids.append(facility_id)
            selected_routes.update(routes)
            if len(selected_ids) >= target_auto_count and action_pairs_count(selected_routes) >= target_auto_min_actions:
                break
        if not selected_ids:
            raise ValueError("10g_apply_best_probe: auto target selection не знайшов закладів з transit-маршрутами.")
        target_ids = selected_ids
        print(
            "10g_apply_best_probe: auto target selection "
            f"mode={target_selection} count={len(target_ids)} "
            f"routes={len(selected_routes)} actions={action_pairs_count(selected_routes)} "
            f"targets={', '.join(target_ids)}"
        )

    target_rows = catchment[catchment["facility_id"].isin(target_ids)].copy()
    route_mask = (
        target_rows["peak_mode"].eq("transit")
        & target_rows["peak_route_id"].notna()
        & target_rows["peak_route_id"].ne("nan")
        & target_rows["peak_route_id"].ne("")
    )
    local_route_ids = sorted(target_rows.loc[route_mask, "peak_route_id"].astype(str).unique().tolist())
    if allowed_route_ids is not None:
        local_route_ids = [route_id for route_id in local_route_ids if route_id in allowed_route_ids]
    if not local_route_ids:
        raise ValueError(f"10g_apply_best_probe: для target_ids={target_ids} немає transit-маршрутів.")

    route_stats = route_stats_full[route_stats_full["route_id"].isin(local_route_ids)].copy()
    if excluded_transport_types:
        route_stats = route_stats[~route_stats["transport"].isin(excluded_transport_types)].copy()
    if route_stats.empty:
        raise ValueError("10g_apply_best_probe: після exclude_transport_types не лишилось маршрутів.")

    route_stats = route_stats.sort_values(["transport", "route", "route_id"]).reset_index(drop=True)
    route_ids = route_stats["route_id"].astype(str).tolist()
    route_index = {route_id: idx for idx, route_id in enumerate(route_ids)}
    initial_freq = route_stats["rl_initial_freq"].to_numpy(dtype=float)
    base_freq_by_idx = np.maximum(initial_freq.astype(np.float32), 1.0)
    route_types = route_stats["transport_type"].to_numpy(dtype=int)

    if allow_cross_type_transfers:
        transfer_actions = [
            (donor_idx, receiver_idx)
            for donor_idx in range(len(route_stats))
            for receiver_idx in range(len(route_stats))
            if donor_idx != receiver_idx
        ]
    else:
        route_type_to_indices: dict[int, list[int]] = {}
        for idx, transport_type in enumerate(route_types):
            route_type_to_indices.setdefault(int(transport_type), []).append(idx)
        transfer_actions = [
            (donor_idx, receiver_idx)
            for same_type_indices in route_type_to_indices.values()
            for donor_idx in same_type_indices
            for receiver_idx in same_type_indices
            if donor_idx != receiver_idx
        ]
    if not transfer_actions:
        raise ValueError("10g_apply_best_probe: action space порожній.")

    catchment = catchment.merge(weights, on="building_id", how="left")
    catchment["weight_wb"] = pd.to_numeric(catchment["weight_wb"], errors="coerce").fillna(1.0).clip(lower=1.0)
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
    for facility_id, group in catchment.groupby("facility_id", sort=False):
        valid = group["peak_total_min"].notna().to_numpy()
        if not valid.any():
            continue
        group_valid = group.loc[valid]
        route_idx = group_valid["_route_idx"].fillna(-1).to_numpy(dtype=np.int32)
        is_transit = group_valid["_is_transit_route"].to_numpy(dtype=bool)
        for idx in np.unique(route_idx[route_idx >= 0]):
            facilities_by_route.setdefault(int(idx), set()).add(str(facility_id))
        facility_arrays[str(facility_id)] = {
            "total": group_valid["peak_total_min"].to_numpy(dtype=np.float32),
            "wait": group_valid["peak_wait_min"].fillna(0.0).to_numpy(dtype=np.float32),
            "walk_in": group_valid["peak_walk_in_min"].fillna(0.0).to_numpy(dtype=np.float32),
            "transit": group_valid["peak_transit_min"].fillna(0.0).to_numpy(dtype=np.float32),
            "walk_out": group_valid["peak_walk_out_min"].fillna(0.0).to_numpy(dtype=np.float32),
            "weight": group_valid["weight_wb"].fillna(1.0).to_numpy(dtype=np.float32),
            "route_idx": route_idx,
            "is_transit": is_transit,
            "hnorm": float(hnorm_by_facility.get(str(facility_id), 0.0)),
        }

    target_set = set(target_ids)

    def recalc_i(facility_id: str, current_freq: np.ndarray) -> float:
        data = facility_arrays.get(str(facility_id))
        if data is None:
            return 0.0
        adjusted_total = np.array(data["total"], dtype=np.float32, copy=True)
        route_idx = data["route_idx"]
        transit_mask = data["is_transit"]
        weights_arr = data["weight"]
        if bool(np.any(transit_mask)):
            idxs = route_idx[transit_mask]
            valid = idxs >= 0
            if bool(np.any(valid)):
                transit_positions = np.flatnonzero(transit_mask)[valid]
                valid_route_idxs = idxs[valid]
                scaled_wait = (
                    data["wait"][transit_positions]
                    * (base_freq_by_idx[valid_route_idxs] / current_freq[valid_route_idxs])
                )
                adjusted_total[transit_positions] = (
                    data["walk_in"][transit_positions]
                    + scaled_wait
                    + data["transit"][transit_positions]
                    + data["walk_out"][transit_positions]
                )
        weighted_sum = float(np.sum(weights_arr * np.exp(-0.05 * adjusted_total)))
        return (weighted_sum / total_city_weight) * float(data["hnorm"])

    def facility_wait_saving(facility_id: str, current_freq: np.ndarray) -> float:
        data = facility_arrays.get(str(facility_id))
        if data is None:
            return 0.0
        route_idx = data["route_idx"]
        transit_mask = data["is_transit"]
        if not bool(np.any(transit_mask)):
            return 0.0
        idxs = route_idx[transit_mask]
        valid = idxs >= 0
        if not bool(np.any(valid)):
            return 0.0
        transit_positions = np.flatnonzero(transit_mask)[valid]
        valid_route_idxs = idxs[valid]
        base_wait = data["wait"][transit_positions]
        scaled_wait = base_wait * (base_freq_by_idx[valid_route_idxs] / current_freq[valid_route_idxs])
        weights_arr = data["weight"][transit_positions]
        if float(np.sum(weights_arr)) <= 0.0:
            return float(np.mean(base_wait - scaled_wait))
        return float(np.average(base_wait - scaled_wait, weights=weights_arr))

    def target_mean_i(current_freq: np.ndarray) -> float:
        return float(np.mean([recalc_i(fid, current_freq) for fid in target_ids]))

    def target_wait_saving(current_freq: np.ndarray) -> float:
        return float(np.mean([facility_wait_saving(fid, current_freq) for fid in target_ids]))

    def non_target_harm(current_freq: np.ndarray, affected_facility_ids: set[str]) -> float:
        non_target_ids = affected_facility_ids - target_set
        if not non_target_ids:
            return 0.0
        before = float(np.mean([initial_i_peak.get(fid, 0.0) for fid in non_target_ids]))
        after = float(np.mean([recalc_i(fid, current_freq) for fid in non_target_ids]))
        return max(0.0, before - after - non_target_harm_tolerance)

    def objective_value(current_freq: np.ndarray, affected_facility_ids: set[str]) -> dict[str, float]:
        mean_i = target_mean_i(current_freq)
        wait_saving = target_wait_saving(current_freq)
        harm = non_target_harm(current_freq, affected_facility_ids)
        objective = mean_i + (target_wait_reward_weight * wait_saving) - (non_target_harm_weight * harm)
        return {
            "target_i_mean": mean_i,
            "target_wait_saving_min": wait_saving,
            "non_target_harm": harm,
            "objective": objective,
        }

    def evaluate_action(
        action_id: int,
        donor_idx: int,
        receiver_idx: int,
        current_freq: np.ndarray,
        affected_facility_ids: set[str],
        current_metrics: dict[str, float],
    ) -> dict:
        donor_after = float(current_freq[donor_idx] - action_step)
        receiver_after = float(current_freq[receiver_idx] + action_step)
        invalid_reason = ""
        if donor_after < 1.0:
            invalid_reason = "donor_below_1"
        elif donor_after < max(1.0, float(initial_freq[donor_idx]) - max_route_delta):
            invalid_reason = "donor_below_delta_limit"
        elif receiver_after > 12.0:
            invalid_reason = "receiver_above_12"
        elif receiver_after > min(12.0, float(initial_freq[receiver_idx]) + max_route_delta):
            invalid_reason = "receiver_above_delta_limit"

        new_freq = current_freq.copy()
        action_affected = facilities_by_route.get(donor_idx, set()) | facilities_by_route.get(receiver_idx, set())
        new_affected = set(affected_facility_ids) | action_affected
        if not invalid_reason:
            new_freq[donor_idx] = donor_after
            new_freq[receiver_idx] = receiver_after
            new_metrics = objective_value(new_freq, new_affected)
        else:
            new_metrics = current_metrics.copy()

        donor = route_stats.iloc[donor_idx]
        receiver = route_stats.iloc[receiver_idx]
        return {
            "action_id": int(action_id),
            "valid": not bool(invalid_reason),
            "invalid_reason": invalid_reason,
            "donor_idx": int(donor_idx),
            "receiver_idx": int(receiver_idx),
            "donor_route_id": str(donor.route_id),
            "donor_transport": str(donor.transport),
            "donor_route": str(donor.route),
            "donor_initial_freq": float(initial_freq[donor_idx]),
            "donor_before_freq": float(current_freq[donor_idx]),
            "donor_after_freq": donor_after,
            "receiver_route_id": str(receiver.route_id),
            "receiver_transport": str(receiver.transport),
            "receiver_route": str(receiver.route),
            "receiver_initial_freq": float(initial_freq[receiver_idx]),
            "receiver_before_freq": float(current_freq[receiver_idx]),
            "receiver_after_freq": receiver_after,
            "target_i_before": float(current_metrics["target_i_mean"]),
            "target_i_after": float(new_metrics["target_i_mean"]),
            "delta_target_i": float(new_metrics["target_i_mean"] - current_metrics["target_i_mean"]),
            "target_wait_saving_before": float(current_metrics["target_wait_saving_min"]),
            "target_wait_saving_after": float(new_metrics["target_wait_saving_min"]),
            "target_wait_saving_delta": float(
                new_metrics["target_wait_saving_min"] - current_metrics["target_wait_saving_min"]
            ),
            "non_target_harm_before": float(current_metrics["non_target_harm"]),
            "non_target_harm_after": float(new_metrics["non_target_harm"]),
            "non_target_harm_delta": float(new_metrics["non_target_harm"] - current_metrics["non_target_harm"]),
            "objective_before": float(current_metrics["objective"]),
            "objective_after": float(new_metrics["objective"]),
            "objective_delta": float(new_metrics["objective"] - current_metrics["objective"]),
            "affected_count": len(new_affected),
            "non_target_affected_count": len(new_affected - target_set),
            "_new_freq": new_freq,
            "_new_affected": new_affected,
            "_new_metrics": new_metrics,
        }

    current_freq = initial_freq.copy()
    affected_facility_ids: set[str] = set()
    current_metrics = objective_value(current_freq, affected_facility_ids)
    baseline_metrics = current_metrics.copy()
    step_rows: list[dict] = []

    for step_idx in range(1, max_steps + 1):
        candidates = [
            evaluate_action(action_id, donor_idx, receiver_idx, current_freq, affected_facility_ids, current_metrics)
            for action_id, (donor_idx, receiver_idx) in enumerate(transfer_actions)
        ]
        valid_candidates = [
            row
            for row in candidates
            if row["valid"] and float(row["delta_target_i"]) >= -metric_epsilon
        ]
        if not valid_candidates:
            break
        best = max(
            valid_candidates,
            key=lambda row: (
                row["objective_delta"],
                row["delta_target_i"],
                row["target_wait_saving_delta"],
            ),
        )
        if float(best["objective_delta"]) <= metric_epsilon:
            break

        current_freq = best.pop("_new_freq")
        affected_facility_ids = best.pop("_new_affected")
        current_metrics = best.pop("_new_metrics")
        best["step"] = step_idx
        step_rows.append(best)

    if step_rows:
        steps_df = pd.DataFrame(step_rows)
        steps_df.to_csv(STEPS_CSV, index=False, encoding="utf-8")
    else:
        steps_df = pd.DataFrame()
        pd.DataFrame(columns=["step", "objective_delta", "delta_target_i"]).to_csv(
            STEPS_CSV,
            index=False,
            encoding="utf-8",
        )

    route_changes = []
    for idx, route_id in enumerate(route_ids):
        delta = float(current_freq[idx] - initial_freq[idx])
        if abs(delta) <= metric_epsilon:
            continue
        row = route_stats.iloc[idx]
        route_changes.append(
            {
                "route_id": str(route_id),
                "transport": str(row.transport),
                "route": str(row.route),
                "initial_freq": float(initial_freq[idx]),
                "after_freq": float(current_freq[idx]),
                "delta": delta,
            }
        )
    route_changes = sorted(route_changes, key=lambda item: abs(item["delta"]), reverse=True)
    pd.DataFrame(route_changes).to_csv(OPTIMAL_FREQ_CSV, index=False, encoding="utf-8")

    map_data = json.loads(MAP_DATA_JSON.read_text(encoding="utf-8"))
    facility_meta = {
        str(item.get("id")): {
            "name": item.get("name", str(item.get("id"))),
            "type": item.get("type", ""),
            "lat": item.get("lat"),
            "lon": item.get("lon"),
            "stats": item.get("stats", {}),
            "buildings_geojson": item.get("buildings_geojson"),
            "n_buildings": item.get("n_buildings"),
        }
        for item in map_data.get("facilities", [])
    }

    target_rows_output = []
    for facility_id in target_ids:
        before = float(initial_i_peak.get(facility_id, 0.0))
        after = float(recalc_i(facility_id, current_freq))
        delta = after - before
        delta_pct = (delta / before * 100.0) if before else 0.0
        meta = facility_meta.get(facility_id, {})
        target_rows_output.append(
            {
                "facility_id": facility_id,
                "name": meta.get("name", facility_id),
                "type": meta.get("type", ""),
                "I_peak_before": before,
                "I_peak_after": after,
                "delta": delta,
                "delta_pct": delta_pct,
            }
        )
    target_df = pd.DataFrame(target_rows_output)
    target_df.to_csv(TARGET_BEFORE_AFTER_CSV, index=False, encoding="utf-8")

    target_before_mean = float(target_df["I_peak_before"].mean())
    target_after_mean = float(target_df["I_peak_after"].mean())
    target_delta_mean = target_after_mean - target_before_mean

    ppo_summary = None
    if RL_RESULTS_JSON.exists():
        try:
            ppo_results = json.loads(RL_RESULTS_JSON.read_text(encoding="utf-8"))
            raw_before = ppo_results.get("before", {}).get("I_peak_target_mean")
            raw_after = ppo_results.get("after", {}).get("I_peak_target_mean")
            if raw_before is None or raw_after is None:
                ppo_summary = {
                    "available": False,
                    "reason": "rl_results.json is not a target-mode PPO run",
                    "source": str(RL_RESULTS_JSON),
                }
            else:
                ppo_before = float(raw_before)
                ppo_after = float(raw_after)
                ppo_summary = {
                    "available": True,
                    "before": ppo_before,
                    "after": ppo_after,
                    "delta": ppo_after - ppo_before,
                    "changed_routes_count": sum(
                        len(ppo_results.get("route_changes", {}).get(key, []))
                        for key in ["increased", "decreased", "disabled"]
                    ),
                    "source": str(RL_RESULTS_JSON),
                }
        except Exception as exc:
            ppo_summary = {"available": False, "error": str(exc), "source": str(RL_RESULTS_JSON)}

    comparison = {
        "target_facility_ids": target_ids,
        "greedy": {
            "steps_applied": len(step_rows),
            "before": target_before_mean,
            "after": target_after_mean,
            "delta": target_delta_mean,
            "delta_pct": (target_delta_mean / target_before_mean * 100.0) if target_before_mean else 0.0,
            "changed_routes_count": len(route_changes),
        },
        "ppo": ppo_summary,
        "recommendation": "greedy_baseline" if len(step_rows) > 0 and target_delta_mean > 0.0 else "ppo_or_baseline",
    }
    COMPARISON_JSON.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    target_before_after = {
        "method": "greedy_probe_heuristic",
        "target_facility_ids": target_ids,
        "I_peak_before_mean": target_before_mean,
        "I_peak_after_mean": target_after_mean,
        "delta_mean": target_delta_mean,
        "delta_pct_mean": (target_delta_mean / target_before_mean * 100.0) if target_before_mean else 0.0,
        "facilities": target_rows_output,
        "steps_applied": len(step_rows),
        "route_changes": route_changes,
        "objective": {
            "before": baseline_metrics,
            "after": current_metrics,
            "delta": float(current_metrics["objective"] - baseline_metrics["objective"]),
        },
    }
    TARGET_BEFORE_AFTER_JSON.write_text(
        json.dumps(target_before_after, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results = {
        "mode": "greedy_probe_heuristic",
        "target_facility_ids": target_ids,
        "before": {
            "I_peak_target_mean": target_before_mean,
            "target_wait_saving_min": float(baseline_metrics["target_wait_saving_min"]),
            "objective": float(baseline_metrics["objective"]),
        },
        "after": {
            "I_peak_target_mean": target_after_mean,
            "target_wait_saving_min": float(current_metrics["target_wait_saving_min"]),
            "objective": float(current_metrics["objective"]),
        },
        "delta": {
            "I_peak_target_mean": target_delta_mean,
            "I_peak_target_mean_pct": (target_delta_mean / target_before_mean * 100.0)
            if target_before_mean
            else 0.0,
            "objective": float(current_metrics["objective"] - baseline_metrics["objective"]),
        },
        "steps_applied": len(step_rows),
        "route_changes": {
            "increased": [row for row in route_changes if row["delta"] > 0],
            "decreased": [row for row in route_changes if row["delta"] < 0],
        },
        "comparison": comparison,
        "run_config": {
            "max_steps": max_steps,
            "max_route_delta": max_route_delta,
            "action_step": action_step,
            "freq_scaling": freq_scaling,
            "allow_cross_type_transfers": allow_cross_type_transfers,
            "non_target_harm_weight": non_target_harm_weight,
            "non_target_harm_tolerance": non_target_harm_tolerance,
            "target_wait_reward_weight": target_wait_reward_weight,
            "metric_epsilon": metric_epsilon,
            "exclude_transport_types": sorted(excluded_transport_types),
        },
        "outputs": {
            "target_before_after_json": str(TARGET_BEFORE_AFTER_JSON),
            "target_before_after_csv": str(TARGET_BEFORE_AFTER_CSV),
            "optimal_frequencies_csv": str(OPTIMAL_FREQ_CSV),
            "steps_csv": str(STEPS_CSV),
            "comparison_json": str(COMPARISON_JSON),
            "target_map_html": str(TARGET_MAP_HTML),
        },
    }
    RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    def format_pct(value: float) -> str:
        return f"{value:+.4f}%"

    map_targets = [facility_meta[fid] | {"facility_id": fid} for fid in target_ids if fid in facility_meta]
    if map_targets:
        center_lat = float(np.mean([float(item["lat"]) for item in map_targets if item.get("lat") is not None]))
        center_lon = float(np.mean([float(item["lon"]) for item in map_targets if item.get("lon") is not None]))
    else:
        center_lat = float(cfg["city"]["center_lat"])
        center_lon = float(cfg["city"]["center_lon"])

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )
    layer_targets = folium.FeatureGroup(name="Target-заклади", show=True)
    before_after_by_id = {row["facility_id"]: row for row in target_rows_output}

    for item in map_targets:
        facility_id = str(item["facility_id"])
        row = before_after_by_id[facility_id]
        color = "#147a3f" if row["delta"] > metric_epsilon else "#b33a3a" if row["delta"] < -metric_epsilon else "#666666"
        radius = max(7, min(18, 9 + abs(float(row["delta_pct"])) * 25))
        stats = item.get("stats") or {}
        popup_html = (
            "<div style='width:270px;font-family:Arial,sans-serif;font-size:13px'>"
            f"<b style='font-size:14px'>{row['name']}</b><br>"
            f"<span style='color:#666'>ID: {facility_id}</span><br>"
            f"<span style='color:#666'>{row['type']}</span>"
            "<hr style='margin:6px 0'>"
            f"<b>I*_peak до:</b> {row['I_peak_before']:.9f}<br>"
            f"<b>I*_peak після:</b> {row['I_peak_after']:.9f}<br>"
            f"<b>Delta:</b> {row['delta']:+.9f} ({format_pct(row['delta_pct'])})"
            "<hr style='margin:6px 0'>"
            f"<b>Пік:</b><br>"
            f"&nbsp;Пішки 10 хв: <b>{int(stats.get('peak_walk_short', 0)):,}</b><br>"
            f"&nbsp;Транспорт 10 хв: <b>{int(stats.get('peak_transit_short', 0)):,}</b><br>"
            f"&nbsp;Пішки 30 хв: <b>{int(stats.get('peak_walk_long', 0)):,}</b><br>"
            f"&nbsp;Транспорт 30 хв: <b>{int(stats.get('peak_transit_long', 0)):,}</b>"
            "</div>"
        )
        folium.CircleMarker(
            location=[float(item["lat"]), float(item["lon"])],
            radius=radius,
            color="white",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.88,
            popup=folium.Popup(popup_html, max_width=310),
            tooltip=f"{facility_id}: {row['name']} ({format_pct(row['delta_pct'])})",
        ).add_to(layer_targets)

    layer_targets.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    top_changes_html = "".join(
        f"<li>{item['transport']} {item['route']}: {item['initial_freq']:.2f} -> "
        f"{item['after_freq']:.2f} ({item['delta']:+.2f})</li>"
        for item in route_changes[:6]
    )
    if not top_changes_html:
        top_changes_html = "<li>Змін маршрутів немає</li>"

    panel_html = f"""
    <div style="
      position: fixed; top: 16px; right: 16px; z-index: 9999;
      width: 330px; background: rgba(255,255,255,.94);
      border: 1px solid #d9d9d9; border-radius: 6px;
      padding: 12px 14px; font-family: Arial, sans-serif;
      box-shadow: 0 2px 12px rgba(0,0,0,.18); font-size: 13px;">
      <div style="font-weight:700;font-size:15px;margin-bottom:6px;">Greedy RL результат</div>
      <div>Target: {", ".join(target_ids)}</div>
      <div>Кроків: <b>{len(step_rows)}</b></div>
      <div>I*_peak mean: <b>{target_before_mean:.9f}</b> -> <b>{target_after_mean:.9f}</b></div>
      <div>Delta: <b>{target_delta_mean:+.9f}</b>
        ({format_pct((target_delta_mean / target_before_mean * 100.0) if target_before_mean else 0.0)})</div>
      <hr style="margin:8px 0">
      <div style="font-weight:700;margin-bottom:4px;">Зміни маршрутів</div>
      <ul style="margin:0 0 0 18px;padding:0;">{top_changes_html}</ul>
    </div>
    """
    m.get_root().html.add_child(Element(panel_html))
    m.save(str(TARGET_MAP_HTML))

    print(
        "10g_apply_best_probe: greedy "
        f"steps={len(step_rows)} target_mean={target_before_mean:.6f}->{target_after_mean:.6f} "
        f"delta={target_delta_mean:+.10f}"
    )
    if ppo_summary and "delta" in ppo_summary:
        print(
            "10g_apply_best_probe: PPO comparison "
            f"delta={ppo_summary['delta']:+.10f}, greedy delta={target_delta_mean:+.10f}"
        )
    print(f"10g_apply_best_probe: steps -> {STEPS_CSV}")
    print(f"10g_apply_best_probe: comparison -> {COMPARISON_JSON}")
    print(f"10g_apply_best_probe: target map -> {TARGET_MAP_HTML}")


if __name__ == "__main__":
    run()
