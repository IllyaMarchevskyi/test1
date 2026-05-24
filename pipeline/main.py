from __future__ import annotations

import argparse
import os
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
]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exported 07a-07d pipeline.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[step for step, _, _ in STEPS],
        help="Run only selected steps, e.g. --steps 07c 07d. Default: all steps.",
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

        for step_id, module_name, label in selected_steps:
            print(f"\n=== Running {step_id}: {label} ===", flush=True)
            module = __import__(module_name, fromlist=["run"])
            module.run()
            print(f"=== Finished {step_id}: {label} ===", flush=True)
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
