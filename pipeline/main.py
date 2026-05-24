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
