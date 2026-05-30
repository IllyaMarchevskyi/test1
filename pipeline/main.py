from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PIPELINE_DIR.parent

STEPS = [
    # ── Підготовка даних ─────────────────────────────────────────────────────
    ("07a",   "pipeline.step_07a_precompute_buildings_baseline", "Будинки, зупинки → кеш Dijkstra"),
    ("07b",   "pipeline.step_07b_transit_matrix_baseline",       "Матриця досяжності зупинок (пік/міжпік)"),
    ("07bt",  "pipeline.step_07b_transfer",                      "Матриця досяжності з 1 пересадкою"),
    ("07c",   "pipeline.step_07c_catchment_calc_baseline",       "Catchment: скільки людей дістається до кожного закладу"),
    ("07d",   "pipeline.step_07d_results_map_baseline",          "Інтерактивна карта catchment"),
    # ── Preprocessing ────────────────────────────────────────────────────────
    ("08",    "pipeline.step_08_preprocessing_baseline",         "Ваги будинків w_b + ентропія H(f)/H_max"),
    ("08c",   "pipeline.step_08c_missing_names_map",             "Карта закладів без назв"),
    # ── Індекс доступності ───────────────────────────────────────────────────
    ("09a",   "pipeline.step_09_accessibility_index_baseline",   "Індекс I*_peak + top/bottom-5 валідація"),
    # ── RL-оптимізація ───────────────────────────────────────────────────────
    ("10i",   "pipeline.step_10i_dispatch_parser",               "Парсинг диспетчерських розкладів"),
    ("10rl",  "pipeline.step_10_graph_rl_baseline",              "Навчання PPO-агента"),
    ("10diag","pipeline.step_10_diagnostics",                    "RL-діагностика (розклади, recompute, частоти, target, probe)"),
    ("10g",   "pipeline.step_10g_apply_best_probe",              "Greedy-оптимізація маршрутних частот"),
    ("10h",   "pipeline.step_10h_practical_recommendations",     "Практичні рекомендації → звіт"),
    # ── Регресія + SHAP + Транспортні пустелі ────────────────────────────────
    ("11",    "pipeline.step_11_regression_shap",              "Регресія I*_peak + SHAP + транспортні пустелі"),
]

STEP_OUTPUTS = {
    "07a": [
        PIPELINE_DIR / "data/processed/stop_to_bld_short_baseline.parquet",
        PIPELINE_DIR / "data/processed/stop_to_bld_long_baseline.parquet",
        PIPELINE_DIR / "data/processed/stop_to_fac_exit_baseline.parquet",
    ],
    "07b": [
        PIPELINE_DIR / "data/processed/stop_reachability_peak_baseline.parquet",
        PIPELINE_DIR / "data/processed/stop_reachability_offpeak_baseline.parquet",
        PIPELINE_DIR / "data/processed/stop_reachability_peak_reversed_baseline.parquet",
        PIPELINE_DIR / "data/processed/stop_reachability_offpeak_reversed_baseline.parquet",
        PIPELINE_DIR / "data/processed/wait_times_peak_baseline.parquet",
        PIPELINE_DIR / "data/processed/wait_times_offpeak_baseline.parquet",
    ],
    "07bt": [
        PIPELINE_DIR / "data/processed/stop_reachability_transfer_peak_baseline.parquet",
        PIPELINE_DIR / "data/processed/stop_reachability_transfer_offpeak_baseline.parquet",
    ],
    "07c": [
        PIPELINE_DIR / "data/processed/catchment_results_baseline.csv",
        PIPELINE_DIR / "data/processed/catchment_buildings_baseline.parquet",
    ],
    "07d": [
        PIPELINE_DIR / "data/processed/map_data_baseline.json",
        PIPELINE_DIR / "data/outputs/map_catchment_interactive_baseline.html",
        PIPELINE_DIR / "data/outputs/output.png",
        PIPELINE_DIR / "data/outputs/map_buildings_baseline",
    ],
    "08": [
        PIPELINE_DIR / "data/processed/building_weights_baseline.parquet",
        PIPELINE_DIR / "data/processed/facility_entropy_baseline.parquet",
        PIPELINE_DIR / "data/processed/facility_entropy_baseline.csv",
        PIPELINE_DIR / "data/processed/facility_entropy_preview_baseline.csv",
    ],
    "08c": [
        PIPELINE_DIR / "data/processed/facilities_missing_names.csv",
        PIPELINE_DIR / "data/outputs/facilities_missing_names_map.html",
    ],
    "09a": [
        PIPELINE_DIR / "data/processed/accessibility_index_baseline.csv",
        PIPELINE_DIR / "data/processed/accessibility_index_preview_baseline.csv",
        PIPELINE_DIR / "data/processed/global_metrics_baseline.json",
        PIPELINE_DIR / "data/processed/car_accessibility_baseline.csv",
        PIPELINE_DIR / "data/processed/kyiv_drive_graph_proj_baseline.pkl",
        PIPELINE_DIR / "data/processed/accessibility_index_top5_baseline.csv",
        PIPELINE_DIR / "data/processed/accessibility_index_bottom5_baseline.csv",
        PIPELINE_DIR / "data/outputs/accessibility_index_extremes_baseline.html",
    ],
    "10rl": [
        PIPELINE_DIR / "data/processed/rl_results.json",
        PIPELINE_DIR / "data/processed/optimal_frequencies.csv",
        PIPELINE_DIR / "data/processed/optimal_frequencies_H327.csv",
        PIPELINE_DIR / "data/processed/optimal_frequencies_targets.csv",
        PIPELINE_DIR / "data/processed/target_facility_before_after.json",
        PIPELINE_DIR / "data/processed/target_facilities_before_after.json",
        PIPELINE_DIR / "data/processed/rl_affected_facilities_before_after.csv",
        PIPELINE_DIR / "data/processed/rl_global_facilities_before_after.csv",
        PIPELINE_DIR / "data/processed/rl_model.zip",
        PIPELINE_DIR / "data/processed/rl_checkpoints",
        PIPELINE_DIR / "data/outputs/rl_learning_curve.png",
        PIPELINE_DIR / "data/outputs/rl_top_route_changes.png",
        PIPELINE_DIR / "data/outputs/rl_before_after_scatter.png",
        PIPELINE_DIR / "data/outputs/rl_i_peak_hist.png",
        PIPELINE_DIR / "data/outputs/rl_H327_training_curve.png",
        PIPELINE_DIR / "data/outputs/rl_H327_route_changes.png",
        PIPELINE_DIR / "data/outputs/rl_H327_wait_before_after_scatter.png",
        PIPELINE_DIR / "data/outputs/rl_targets_training_curve.png",
        PIPELINE_DIR / "data/outputs/rl_targets_route_changes.png",
        PIPELINE_DIR / "data/outputs/rl_targets_wait_before_after_scatter.png",
    ],
    "10diag": [
        PIPELINE_DIR / "data/processed/h327_route_schedule_debug.csv",
        PIPELINE_DIR / "data/processed/h327_route_schedule_stops_debug.csv",
        PIPELINE_DIR / "data/processed/rl_recompute_check_targets.csv",
        PIPELINE_DIR / "data/processed/rl_recompute_check_summary.json",
        PIPELINE_DIR / "data/processed/all_routes_weekdays_freq_debug.csv",
        PIPELINE_DIR / "data/processed/rl_target_group_debug_summary.json",
        PIPELINE_DIR / "data/processed/rl_target_group_routes.csv",
        PIPELINE_DIR / "data/processed/rl_target_group_affected_facilities.csv",
        PIPELINE_DIR / "data/processed/rl_target_group_candidates.csv",
        PIPELINE_DIR / "data/processed/rl_target_group_time_components.csv",
        PIPELINE_DIR / "data/processed/rl_action_probe.csv",
        PIPELINE_DIR / "data/processed/rl_action_probe_summary.json",
    ],
    "10g": [
        PIPELINE_DIR / "data/processed/rl_best_probe_results.json",
        PIPELINE_DIR / "data/processed/target_facilities_best_probe_before_after.json",
        PIPELINE_DIR / "data/processed/rl_best_probe_target_before_after.csv",
        PIPELINE_DIR / "data/processed/optimal_frequencies_best_probe.csv",
        PIPELINE_DIR / "data/processed/rl_best_probe_steps.csv",
        PIPELINE_DIR / "data/processed/rl_greedy_vs_ppo_comparison.json",
        PIPELINE_DIR / "data/outputs/map_catchment_interactive_best_probe_targets.html",
    ],
    "10i": [
        PIPELINE_DIR / "data/processed/dispatch_route_stats.csv",
        PIPELINE_DIR / "data/processed/dispatch_direction_stats.csv",
        PIPELINE_DIR / "data/processed/dispatch_release_trips.csv",
        PIPELINE_DIR / "data/processed/dispatch_parse_report.json",
    ],
    "10h": [
        PIPELINE_DIR / "data/processed/rl_practical_recommendations.csv",
        PIPELINE_DIR / "data/processed/rl_practical_recommendations.json",
        PIPELINE_DIR / "data/outputs/rl_recommendations_report.md",
    ],
    "11": [
        PIPELINE_DIR / "data/processed/regression_features_baseline.csv",
        PIPELINE_DIR / "data/processed/regression_metrics_baseline.json",
        PIPELINE_DIR / "data/processed/regression_predictions_baseline.csv",
        PIPELINE_DIR / "data/processed/regression_feature_importance_baseline.csv",
        PIPELINE_DIR / "data/processed/regression_shap_values_baseline.csv",
        PIPELINE_DIR / "data/processed/transport_deserts_baseline.csv",
        PIPELINE_DIR / "data/outputs/regression_feature_importance_baseline.png",
        PIPELINE_DIR / "data/outputs/regression_shap_summary_baseline.png",
        PIPELINE_DIR / "data/outputs/regression_residuals_baseline.png",
        PIPELINE_DIR / "data/outputs/transport_deserts_map_baseline.html",
    ],
}


def _select_steps(requested: list[str] | None) -> list[tuple[str, str, str]]:
    if not requested:
        return STEPS
    wanted = {step.lower() for step in requested}
    selected = [item for item in STEPS if item[0].lower() in wanted]
    missing = wanted - {item[0].lower() for item in selected}
    if missing:
        valid = ", ".join(step for step, _, _ in STEPS)
        raise SystemExit(f"Unknown step(s): {', '.join(sorted(missing))}. Valid: {valid}")
    return selected


def _clean_step_outputs(selected_steps: list[tuple[str, str, str]]) -> None:
    seen: set[Path] = set()
    removed = 0

    for step_id, _, _ in selected_steps:
        for path in STEP_OUTPUTS.get(step_id, []):
            if path in seen:
                continue
            seen.add(path)
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                print(f"Cleaned directory: {path}", flush=True)
                removed += 1
            elif path.exists():
                path.unlink()
                print(f"Cleaned file: {path}", flush=True)
                removed += 1

    if removed == 0:
        print("Nothing to clean for selected steps.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exported 07a-07d pipeline.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[step for step, _, _ in STEPS],
        help="Run only selected steps, e.g. --steps 07c 07d. Default: all steps.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete outputs of selected steps before running them.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(PIPELINE_DIR))
    sys.path.insert(0, str(ROOT_DIR))

    selected_steps = _select_steps(args.steps)
    original_cwd = Path.cwd()

    try:
        # Existing config.toml paths and notebook code use ../data and ../outputs.
        # Running from pipeline/ preserves those paths without editing the notebooks' logic.
        os.chdir(PIPELINE_DIR)

        if args.clean:
            _clean_step_outputs(selected_steps)

        for step_id, module_name, label in selected_steps:
            print(f"\n=== Running {step_id}: {label} ===", flush=True)
            module = __import__(module_name, fromlist=["run"])
            module.run()
            print(f"=== Finished {step_id}: {label} ===", flush=True)
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
