"""
10f RL action probe.

Brute-force перевірка всіх одно-крокових donor->receiver дій для
поточного target-mode. Крок не навчає PPO, а напряму рахує:
- delta I*_peak target-групи;
- target wait saving;
- non-target harm;
- objective_delta за тією ж логікою, що й reward у 10_rl.
"""


def run() -> None:
    from config_loader import cfg
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    PROCESSED_DIR = Path("./data/processed")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ACCESSIBILITY_INDEX = PROCESSED_DIR / "accessibility_index_baseline.csv"
    CATCHMENT_BUILDINGS = PROCESSED_DIR / "catchment_buildings_baseline.parquet"
    FACILITY_ENTROPY = PROCESSED_DIR / "facility_entropy_baseline.parquet"
    BUILDING_WEIGHTS = PROCESSED_DIR / "building_weights_baseline.parquet"
    EASYWAY_ROUTES = Path("../gtfs_static/easyway_routes.csv")
    EASYWAY_METRO = Path("../gtfs_static/easyway_metro.csv")

    OUTPUT_CSV = PROCESSED_DIR / "rl_action_probe.csv"
    SUMMARY_JSON = PROCESSED_DIR / "rl_action_probe_summary.json"

    required = [
        ACCESSIBILITY_INDEX,
        CATCHMENT_BUILDINGS,
        FACILITY_ENTROPY,
        BUILDING_WEIGHTS,
        EASYWAY_ROUTES,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Відсутні входи для 10f_action_probe: {missing}")

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
    if not target_ids:
        raise ValueError("10f_action_probe: потрібно задати target_facility_id або target_facility_ids.")

    excluded_transport_types = set(parse_config_list(rl_cfg.get("exclude_transport_types", [])))
    max_route_delta = max(1, int(rl_cfg.get("max_route_delta", 3)))
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

        # Та сама нормалізація, що в 10_rl: log1p + min-max у межах типу транспорту.
        for _, sub_idx in route_stats_full.groupby("transport").groups.items():
            raw = np.log1p(route_stats_full.loc[sub_idx, "current_freq"].astype(float))
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
    easyway = easyway[easyway["schedules"] != r"\N"].copy()
    easyway["stop_id"] = easyway["stop_id"].astype(str)
    easyway["route_id"] = easyway["route_id"].astype(str)
    easyway["transport"] = easyway["transport"].astype(str)
    easyway["route"] = easyway["route"].astype(str)
    easyway["times"] = easyway["schedules"].apply(parse_schedules)
    easyway["n_departures"] = easyway["times"].apply(len)

    route_stats_full = build_rl_initial_freq(easyway)

    target_rows = catchment[catchment["facility_id"].isin(target_ids)].copy()
    route_mask = (
        target_rows["peak_mode"].eq("transit")
        & target_rows["peak_route_id"].notna()
        & target_rows["peak_route_id"].ne("nan")
        & target_rows["peak_route_id"].ne("")
    )
    local_route_ids = sorted(target_rows.loc[route_mask, "peak_route_id"].astype(str).unique().tolist())
    if not local_route_ids:
        raise ValueError(f"10f_action_probe: для target_ids={target_ids} немає transit-маршрутів.")

    route_stats = route_stats_full[route_stats_full["route_id"].isin(local_route_ids)].copy()
    if excluded_transport_types:
        route_stats = route_stats[~route_stats["transport"].isin(excluded_transport_types)].copy()
    if route_stats.empty:
        raise ValueError("10f_action_probe: після exclude_transport_types не лишилось маршрутів.")

    route_stats = route_stats.sort_values(["transport", "route", "route_id"]).reset_index(drop=True)
    route_ids = route_stats["route_id"].astype(str).tolist()
    route_index = {route_id: idx for idx, route_id in enumerate(route_ids)}
    initial_freq = route_stats["rl_initial_freq"].to_numpy(dtype=float)
    base_freq_by_idx = np.maximum(initial_freq.astype(np.float32), 1.0)
    route_types = route_stats["transport_type"].to_numpy(dtype=int)

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
        raise ValueError("10f_action_probe: action space порожній.")

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

    target_set = set(target_ids)
    initial_target_mean = float(np.mean([initial_i_peak.get(fid, 0.0) for fid in target_ids]))
    rows = []
    for action_id, (donor_idx, receiver_idx) in enumerate(transfer_actions):
        donor_after = initial_freq[donor_idx] - 1.0
        receiver_after = initial_freq[receiver_idx] + 1.0
        invalid_reason = ""
        if donor_after < 1.0:
            invalid_reason = "donor_below_1"
        elif donor_after < max(1.0, initial_freq[donor_idx] - float(max_route_delta)):
            invalid_reason = "donor_below_delta_limit"
        elif receiver_after > 12.0:
            invalid_reason = "receiver_above_12"
        elif receiver_after > min(12.0, initial_freq[receiver_idx] + float(max_route_delta)):
            invalid_reason = "receiver_above_delta_limit"

        current_freq = initial_freq.copy()
        if not invalid_reason:
            current_freq[donor_idx] = donor_after
            current_freq[receiver_idx] = receiver_after

        affected = set(facilities_by_route.get(donor_idx, set())) | set(facilities_by_route.get(receiver_idx, set()))
        non_target_affected = affected - target_set

        target_after = target_mean_i(current_freq) if not invalid_reason else initial_target_mean
        wait_saving = target_wait_saving(current_freq) if not invalid_reason else 0.0

        non_target_harm = 0.0
        if non_target_affected and not invalid_reason:
            before = float(np.mean([initial_i_peak.get(fid, 0.0) for fid in non_target_affected]))
            after = float(np.mean([recalc_i(fid, current_freq) for fid in non_target_affected]))
            non_target_harm = max(0.0, before - after - non_target_harm_tolerance)

        delta_target = target_after - initial_target_mean
        objective_delta = (
            delta_target
            + (target_wait_reward_weight * wait_saving)
            - (non_target_harm_weight * non_target_harm)
        )

        donor = route_stats.iloc[donor_idx]
        receiver = route_stats.iloc[receiver_idx]
        row = {
            "action_id": action_id,
            "valid": not bool(invalid_reason),
            "invalid_reason": invalid_reason,
            "donor_route_id": str(donor.route_id),
            "donor_transport": str(donor.transport),
            "donor_route": str(donor.route),
            "donor_initial_freq": float(initial_freq[donor_idx]),
            "donor_after_freq": float(donor_after),
            "receiver_route_id": str(receiver.route_id),
            "receiver_transport": str(receiver.transport),
            "receiver_route": str(receiver.route),
            "receiver_initial_freq": float(initial_freq[receiver_idx]),
            "receiver_after_freq": float(receiver_after),
            "target_i_before": initial_target_mean,
            "target_i_after": target_after,
            "delta_target_i": delta_target,
            "target_wait_saving_min": wait_saving,
            "non_target_harm": non_target_harm,
            "objective_delta": objective_delta,
            "affected_count": len(affected),
            "non_target_affected_count": len(non_target_affected),
        }
        for target_id in target_ids:
            target_after_one = recalc_i(target_id, current_freq) if not invalid_reason else initial_i_peak.get(target_id, 0.0)
            row[f"delta_{target_id}"] = target_after_one - initial_i_peak.get(target_id, 0.0)
        rows.append(row)

    probe_df = pd.DataFrame(rows)
    probe_df["delta_target_clean"] = probe_df["delta_target_i"].where(
        probe_df["delta_target_i"].abs() > metric_epsilon,
        0.0,
    )
    probe_df["objective_clean"] = probe_df["objective_delta"].where(
        probe_df["objective_delta"].abs() > metric_epsilon,
        0.0,
    )
    probe_df = probe_df.sort_values(
        ["valid", "objective_delta", "delta_target_i", "target_wait_saving_min"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    probe_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    valid_df = probe_df[probe_df["valid"]].copy()
    best = valid_df.iloc[0].to_dict() if not valid_df.empty else None
    summary = {
        "target_facility_ids": target_ids,
        "exclude_transport_types": sorted(excluded_transport_types),
        "routes_count": len(route_ids),
        "actions_count": len(probe_df),
        "valid_actions_count": int(probe_df["valid"].sum()),
        "positive_objective_actions_count": int((valid_df["objective_clean"] > 0).sum()) if not valid_df.empty else 0,
        "positive_target_actions_count": int((valid_df["delta_target_clean"] > 0).sum()) if not valid_df.empty else 0,
        "target_i_before": initial_target_mean,
        "best_action": best,
        "output_csv": str(OUTPUT_CSV),
        "run_config": {
            "max_route_delta": max_route_delta,
            "non_target_harm_weight": non_target_harm_weight,
            "non_target_harm_tolerance": non_target_harm_tolerance,
            "target_wait_reward_weight": target_wait_reward_weight,
            "metric_epsilon": metric_epsilon,
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "10f_action_probe: "
        f"routes={len(route_ids)} actions={len(probe_df)} valid={summary['valid_actions_count']} "
        f"positive_objective={summary['positive_objective_actions_count']} "
        f"positive_target={summary['positive_target_actions_count']}"
    )
    if best:
        print(
            "10f_action_probe: best "
            f"{best['donor_transport']} {best['donor_route']} -> "
            f"{best['receiver_transport']} {best['receiver_route']} | "
            f"objective_delta={best['objective_delta']:.10f} "
            f"delta_target={best['delta_target_i']:.10f} "
            f"wait_saving={best['target_wait_saving_min']:.4f}хв "
            f"non_target_harm={best['non_target_harm']:.10f}"
        )
    print(f"10f_action_probe: table -> {OUTPUT_CSV}")
    print(f"10f_action_probe: summary -> {SUMMARY_JSON}")


if __name__ == "__main__":
    run()
