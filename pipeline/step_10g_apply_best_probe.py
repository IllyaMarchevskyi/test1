"""
10g Apply best RL probe action.

Крок бере найкращу валідну дію з 10f_action_probe і формує heuristic
after-result без PPO. Це контрольний baseline: якщо brute-force знаходить
позитивну дію, а PPO її не бере, проблема в навчанні/сигналі, а не в
самій моделі перерахунку доступності.
"""

from __future__ import annotations


def run() -> None:
    import json
    from pathlib import Path

    import pandas as pd

    PROCESSED_DIR = Path("./data/processed")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    PROBE_CSV = PROCESSED_DIR / "rl_action_probe.csv"
    PROBE_SUMMARY_JSON = PROCESSED_DIR / "rl_action_probe_summary.json"
    ACCESSIBILITY_INDEX = PROCESSED_DIR / "accessibility_index_baseline.csv"
    MAP_DATA_JSON = PROCESSED_DIR / "map_data_baseline.json"

    RESULTS_JSON = PROCESSED_DIR / "rl_best_probe_results.json"
    TARGET_BEFORE_AFTER_JSON = PROCESSED_DIR / "target_facilities_best_probe_before_after.json"
    TARGET_BEFORE_AFTER_CSV = PROCESSED_DIR / "rl_best_probe_target_before_after.csv"
    OPTIMAL_FREQ_CSV = PROCESSED_DIR / "optimal_frequencies_best_probe.csv"

    missing = [str(path) for path in [PROBE_CSV, ACCESSIBILITY_INDEX] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Відсутні входи для 10g_apply_best_probe: "
            f"{missing}. Спочатку запусти 10f_action_probe."
        )

    probe_df = pd.read_csv(PROBE_CSV)
    if probe_df.empty:
        raise ValueError("10g_apply_best_probe: rl_action_probe.csv порожній.")

    required_cols = {
        "valid",
        "objective_delta",
        "delta_target_i",
        "target_i_before",
        "target_i_after",
        "donor_route_id",
        "donor_transport",
        "donor_route",
        "donor_initial_freq",
        "donor_after_freq",
        "receiver_route_id",
        "receiver_transport",
        "receiver_route",
        "receiver_initial_freq",
        "receiver_after_freq",
    }
    missing_cols = sorted(required_cols - set(probe_df.columns))
    if missing_cols:
        raise ValueError(f"10g_apply_best_probe: у rl_action_probe.csv бракує колонок: {missing_cols}")

    valid_mask = probe_df["valid"].astype(str).str.lower().isin({"true", "1", "yes"})
    valid_df = probe_df[valid_mask].copy()
    if valid_df.empty:
        raise ValueError("10g_apply_best_probe: немає валідних дій у rl_action_probe.csv.")

    valid_df["objective_delta"] = pd.to_numeric(valid_df["objective_delta"], errors="coerce").fillna(0.0)
    valid_df["delta_target_i"] = pd.to_numeric(valid_df["delta_target_i"], errors="coerce").fillna(0.0)
    valid_df["target_wait_saving_min"] = pd.to_numeric(
        valid_df.get("target_wait_saving_min", 0.0),
        errors="coerce",
    ).fillna(0.0)
    best = valid_df.sort_values(
        ["objective_delta", "delta_target_i", "target_wait_saving_min"],
        ascending=[False, False, False],
    ).iloc[0]

    summary = {}
    if PROBE_SUMMARY_JSON.exists():
        summary = json.loads(PROBE_SUMMARY_JSON.read_text(encoding="utf-8"))

    target_ids = [str(fid) for fid in summary.get("target_facility_ids", []) if str(fid).strip()]
    if not target_ids:
        ignored_delta_cols = {"delta_target_i", "delta_target_clean"}
        target_ids = [
            col.removeprefix("delta_")
            for col in probe_df.columns
            if col.startswith("delta_") and col not in ignored_delta_cols
        ]
    if not target_ids:
        raise ValueError("10g_apply_best_probe: не вдалося визначити target_facility_ids.")

    index_df = pd.read_csv(ACCESSIBILITY_INDEX)
    index_df["facility_id"] = index_df["facility_id"].astype(str)
    index_df["I_peak"] = pd.to_numeric(index_df["I_peak"], errors="coerce").fillna(0.0)
    baseline_i = dict(zip(index_df["facility_id"], index_df["I_peak"]))

    facility_meta = {}
    if MAP_DATA_JSON.exists():
        map_data = json.loads(MAP_DATA_JSON.read_text(encoding="utf-8"))
        facility_meta = {
            str(item.get("id")): {
                "name": item.get("name", str(item.get("id"))),
                "type": item.get("type", ""),
            }
            for item in map_data.get("facilities", [])
        }

    target_rows = []
    for facility_id in target_ids:
        before = float(baseline_i.get(facility_id, 0.0))
        delta_col = f"delta_{facility_id}"
        delta = float(pd.to_numeric(pd.Series([best.get(delta_col, 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        after = before + delta
        delta_pct = (delta / before * 100.0) if before else 0.0
        meta = facility_meta.get(facility_id, {})
        target_rows.append(
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

    target_df = pd.DataFrame(target_rows)
    target_df.to_csv(TARGET_BEFORE_AFTER_CSV, index=False, encoding="utf-8")

    donor_delta = float(best["donor_after_freq"]) - float(best["donor_initial_freq"])
    receiver_delta = float(best["receiver_after_freq"]) - float(best["receiver_initial_freq"])
    route_changes = [
        {
            "route_id": str(best["donor_route_id"]),
            "transport": str(best["donor_transport"]),
            "route": str(best["donor_route"]),
            "role": "donor",
            "initial_freq": float(best["donor_initial_freq"]),
            "after_freq": float(best["donor_after_freq"]),
            "delta": donor_delta,
        },
        {
            "route_id": str(best["receiver_route_id"]),
            "transport": str(best["receiver_transport"]),
            "route": str(best["receiver_route"]),
            "role": "receiver",
            "initial_freq": float(best["receiver_initial_freq"]),
            "after_freq": float(best["receiver_after_freq"]),
            "delta": receiver_delta,
        },
    ]
    pd.DataFrame(route_changes).to_csv(OPTIMAL_FREQ_CSV, index=False, encoding="utf-8")

    target_before_mean = float(pd.to_numeric(target_df["I_peak_before"], errors="coerce").mean())
    target_after_mean = float(pd.to_numeric(target_df["I_peak_after"], errors="coerce").mean())
    target_delta_mean = target_after_mean - target_before_mean

    selected_action = {
        "action_id": int(best["action_id"]) if "action_id" in best and pd.notna(best["action_id"]) else None,
        "donor": route_changes[0],
        "receiver": route_changes[1],
    }
    objective = {
        "objective_delta": float(best["objective_delta"]),
        "delta_target_i": float(best["delta_target_i"]),
        "target_wait_saving_min": float(best.get("target_wait_saving_min", 0.0)),
        "non_target_harm": float(best.get("non_target_harm", 0.0)),
        "affected_count": int(best.get("affected_count", 0)),
        "non_target_affected_count": int(best.get("non_target_affected_count", 0)),
        "is_positive_objective": bool(float(best["objective_delta"]) > 0.0),
        "is_positive_target": bool(float(best["delta_target_i"]) > 0.0),
    }

    target_before_after = {
        "method": "best_probe_heuristic",
        "target_facility_ids": target_ids,
        "I_peak_before_mean": target_before_mean,
        "I_peak_after_mean": target_after_mean,
        "delta_mean": target_delta_mean,
        "delta_pct_mean": (target_delta_mean / target_before_mean * 100.0) if target_before_mean else 0.0,
        "facilities": target_rows,
        "selected_action": selected_action,
        "objective": objective,
    }
    TARGET_BEFORE_AFTER_JSON.write_text(
        json.dumps(target_before_after, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results = {
        "mode": "best_probe_heuristic",
        "source_probe_csv": str(PROBE_CSV),
        "source_probe_summary_json": str(PROBE_SUMMARY_JSON) if PROBE_SUMMARY_JSON.exists() else None,
        "target_facility_ids": target_ids,
        "before": {
            "I_peak_target_mean": target_before_mean,
        },
        "after": {
            "I_peak_target_mean": target_after_mean,
        },
        "delta": {
            "I_peak_target_mean": target_delta_mean,
            "I_peak_target_mean_pct": (target_delta_mean / target_before_mean * 100.0)
            if target_before_mean
            else 0.0,
        },
        "selected_action": selected_action,
        "objective": objective,
        "route_changes": {
            "decreased": [route_changes[0]],
            "increased": [route_changes[1]],
        },
        "outputs": {
            "target_before_after_json": str(TARGET_BEFORE_AFTER_JSON),
            "target_before_after_csv": str(TARGET_BEFORE_AFTER_CSV),
            "optimal_frequencies_csv": str(OPTIMAL_FREQ_CSV),
        },
    }
    RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "10g_apply_best_probe: best "
        f"{best['donor_transport']} {best['donor_route']} -> "
        f"{best['receiver_transport']} {best['receiver_route']} | "
        f"objective_delta={objective['objective_delta']:.10f} "
        f"delta_target={objective['delta_target_i']:.10f} "
        f"wait_saving={objective['target_wait_saving_min']:.4f}хв "
        f"non_target_harm={objective['non_target_harm']:.10f}"
    )
    print(
        "10g_apply_best_probe: target mean "
        f"{target_before_mean:.6f} -> {target_after_mean:.6f} "
        f"(delta={target_delta_mean:+.10f})"
    )
    print(f"10g_apply_best_probe: results -> {RESULTS_JSON}")
    print(f"10g_apply_best_probe: target before/after -> {TARGET_BEFORE_AFTER_JSON}")
    print(f"10g_apply_best_probe: route frequencies -> {OPTIMAL_FREQ_CSV}")


if __name__ == "__main__":
    run()
