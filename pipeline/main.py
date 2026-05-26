from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PIPELINE_DIR.parent

STEPS = [
    ("07a", "pipeline.step_07a_precompute_buildings", "Precompute buildings"),
    ("07b", "pipeline.step_07b_transit_matrix", "Transit matrix"),
    ("07c", "pipeline.step_07c_catchment_calc", "Catchment calculation"),
    ("07d", "pipeline.step_07d_results_map", "Results map"),
    ("07a_base", "pipeline.step_07a_precompute_buildings_baseline", "Baseline precompute buildings"),
    ("07b_base", "pipeline.step_07b_transit_matrix_baseline", "Baseline transit matrix"),
    ("07c_base", "pipeline.step_07c_catchment_calc_baseline", "Baseline catchment calculation"),
    ("07d_base", "pipeline.step_07d_results_map_baseline", "Baseline results map"),
    ("08a_base", "pipeline.step_08a_building_weights_baseline", "Baseline building weights"),
    ("08b_base", "pipeline.step_08b_facility_entropy_baseline", "Baseline facility entropy"),
    ("08c_missing_names", "pipeline.step_08c_missing_names_map", "Facilities with missing names map"),
    ("09_index", "pipeline.step_09_accessibility_index_baseline", "Baseline accessibility index"),
    ("09_validate", "pipeline.step_09b_validate_index_baseline", "Baseline accessibility validation"),
    ("10_rl", "pipeline.step_10_graph_rl_baseline", "Baseline graph RL"),
    ("10b_debug", "pipeline.step_10b_h327_route_debug", "H327 local route schedule debug"),
    ("10c_check", "pipeline.step_10c_rl_recompute_check", "RL baseline vs recompute check"),
    ("10d_debug", "pipeline.step_10d_all_routes_freq_debug", "All routes weekdays frequency debug"),
    ("10e_group_debug", "pipeline.step_10e_target_group_debug", "RL target group debug"),
    ("10f_action_probe", "pipeline.step_10f_rl_action_probe", "RL one-step action probe"),
]

STEP_OUTPUTS = {
    "07a_base": [
        PIPELINE_DIR / "data/processed/stop_to_bld_short_baseline.parquet",
        PIPELINE_DIR / "data/processed/stop_to_bld_long_baseline.parquet",
        PIPELINE_DIR / "data/processed/stop_to_fac_exit_baseline.parquet",
    ],
    "07b_base": [
        PIPELINE_DIR / "data/processed/stop_reachability_peak_baseline.parquet",
        PIPELINE_DIR / "data/processed/stop_reachability_offpeak_baseline.parquet",
        PIPELINE_DIR / "data/processed/stop_reachability_peak_reversed_baseline.parquet",
        PIPELINE_DIR / "data/processed/stop_reachability_offpeak_reversed_baseline.parquet",
        PIPELINE_DIR / "data/processed/wait_times_peak_baseline.parquet",
        PIPELINE_DIR / "data/processed/wait_times_offpeak_baseline.parquet",
    ],
    "07c_base": [
        PIPELINE_DIR / "data/processed/catchment_results_baseline.csv",
        PIPELINE_DIR / "data/processed/catchment_buildings_baseline.parquet",
    ],
    "07d_base": [
        PIPELINE_DIR / "data/processed/map_data_baseline.json",
        PIPELINE_DIR / "data/outputs/map_catchment_interactive_baseline.html",
        PIPELINE_DIR / "data/outputs/output.png",
        PIPELINE_DIR / "data/outputs/map_buildings_baseline",
    ],
    "08a_base": [
        PIPELINE_DIR / "data/processed/building_weights_baseline.parquet",
    ],
    "08b_base": [
        PIPELINE_DIR / "data/processed/facility_entropy_baseline.parquet",
        PIPELINE_DIR / "data/processed/facility_entropy_baseline.csv",
        PIPELINE_DIR / "data/processed/facility_entropy_preview_baseline.csv",
    ],
    "08c_missing_names": [
        PIPELINE_DIR / "data/processed/facilities_missing_names.csv",
        PIPELINE_DIR / "data/outputs/facilities_missing_names_map.html",
    ],
    "09_index": [
        PIPELINE_DIR / "data/processed/accessibility_index_baseline.csv",
        PIPELINE_DIR / "data/processed/accessibility_index_preview_baseline.csv",
        PIPELINE_DIR / "data/processed/global_metrics_baseline.json",
        PIPELINE_DIR / "data/processed/car_accessibility_baseline.csv",
        PIPELINE_DIR / "data/processed/kyiv_drive_graph_proj_baseline.pkl",
    ],
    "09_validate": [
        PIPELINE_DIR / "data/processed/accessibility_index_top5_baseline.csv",
        PIPELINE_DIR / "data/processed/accessibility_index_bottom5_baseline.csv",
        PIPELINE_DIR / "data/outputs/accessibility_index_extremes_baseline.html",
    ],
    "10_rl": [
        PIPELINE_DIR / "data/processed/rl_results.json",
        PIPELINE_DIR / "data/processed/optimal_frequencies.csv",
        PIPELINE_DIR / "data/processed/optimal_frequencies_H327.csv",
        PIPELINE_DIR / "data/processed/optimal_frequencies_targets.csv",
        PIPELINE_DIR / "data/processed/target_facility_before_after.json",
        PIPELINE_DIR / "data/processed/target_facilities_before_after.json",
        PIPELINE_DIR / "data/processed/rl_affected_facilities_before_after.csv",
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
    "10b_debug": [
        PIPELINE_DIR / "data/processed/h327_route_schedule_debug.csv",
        PIPELINE_DIR / "data/processed/h327_route_schedule_stops_debug.csv",
    ],
    "10c_check": [
        PIPELINE_DIR / "data/processed/rl_recompute_check_targets.csv",
        PIPELINE_DIR / "data/processed/rl_recompute_check_summary.json",
    ],
    "10d_debug": [
        PIPELINE_DIR / "data/processed/all_routes_weekdays_freq_debug.csv",
    ],
    "10e_group_debug": [
        PIPELINE_DIR / "data/processed/rl_target_group_debug_summary.json",
        PIPELINE_DIR / "data/processed/rl_target_group_routes.csv",
        PIPELINE_DIR / "data/processed/rl_target_group_affected_facilities.csv",
        PIPELINE_DIR / "data/processed/rl_target_group_candidates.csv",
        PIPELINE_DIR / "data/processed/rl_target_group_time_components.csv",
    ],
    "10f_action_probe": [
        PIPELINE_DIR / "data/processed/rl_action_probe.csv",
        PIPELINE_DIR / "data/processed/rl_action_probe_summary.json",
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
