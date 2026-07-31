"""Compare a world fixed grid with the three ways of moving a grid with the vehicle.

The vehicle advances 4.7 cells per sweep, so an ego window centred exactly on it never
lands on a whole cell offset and has to be resampled every frame. The snap policy gives
up on centring exactly and gets a lossless whole cell translation instead. This script
measures what each choice costs.

Usage:
    uv run python examples/compare_grid_frames.py
    uv run python examples/compare_grid_frames.py --steps 12
"""

from __future__ import annotations

import argparse

from freespace_grid.analysis.metrics import score_grid
from freespace_grid.analysis.report import render_table
from freespace_grid.analysis.sharpness import measure_sharpness
from freespace_grid.pipeline.runner import RunConfig, run_mapping
from freespace_grid.pipeline.scenarios import urban_block

CONFIGS: tuple[tuple[str, RunConfig], ...] = (
    ("world fixed", RunConfig(frame="world")),
    ("ego snap", RunConfig(frame="ego", shift_policy="snap")),
    ("ego bilinear", RunConfig(frame="ego", shift_policy="bilinear")),
    ("ego nearest", RunConfig(frame="ego", shift_policy="nearest")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None, help="cap on the number of sweeps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = urban_block()
    rows: list[list[str]] = []

    for label, base in CONFIGS:
        config = RunConfig(
            frame=base.frame,
            shift_policy=base.shift_policy,
            ego_rows=base.ego_rows,
            ego_cols=base.ego_cols,
            max_steps=args.steps,
        )
        trace = run_mapping(scenario, config)
        observed = trace.observed_mask
        agreement = score_grid(trace.grid, trace.truth, scenario.model, region=observed)
        sharpness = measure_sharpness(
            trace.grid, trace.truth, scenario.model, region=observed
        )
        rows.append(
            [
                label,
                f"{trace.resamples}",
                f"{trace.lossless_resamples}",
                f"{agreement.decided_fraction:.4f}",
                f"{agreement.free_agreement:.4f}",
                f"{agreement.occupied_agreement:.4f}",
                f"{sharpness.clamped_fraction:.4f}",
                f"{sharpness.edge_contrast:.3f}",
            ]
        )

    print(f"scenario: {scenario.name}")
    print(
        "ego window: "
        f"{CONFIGS[1][1].ego_rows} by {CONFIGS[1][1].ego_cols} cells "
        f"at {scenario.grid.resolution} m"
    )
    print()
    print(
        render_table(
            [
                "policy",
                "shifts",
                "lossless",
                "decided",
                "free agr",
                "occ agr",
                "at clamp",
                "edge contrast",
            ],
            rows,
        )
    )


if __name__ == "__main__":
    main()
