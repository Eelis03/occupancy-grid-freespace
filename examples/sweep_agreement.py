"""Sweep the decision threshold and the spatial tolerance against ground truth.

A single agreement figure can be set to almost any value by moving the decision
threshold, so this script reports the whole curve instead.

Usage:
    uv run python examples/sweep_agreement.py
    uv run python examples/sweep_agreement.py --steps 12 --no-figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

from freespace_grid.analysis.figures import plot_threshold_sweep
from freespace_grid.analysis.metrics import threshold_sweep, tolerance_sweep
from freespace_grid.analysis.report import agreement_table
from freespace_grid.pipeline.runner import RunConfig, run_mapping
from freespace_grid.pipeline.scenarios import urban_block

# The free clamp at p = 0.12 puts a hard ceiling of 0.88 on any threshold that can ever
# call a cell free, so the sweep stops below it. Values that coincide exactly with an
# attainable sum of increments are also avoided: a cell sitting on the threshold to
# within one unit in the last place would be classified differently on two machines and
# the sweep would stop being reproducible.
THRESHOLDS: tuple[float, ...] = (0.55, 0.62, 0.65, 0.68, 0.72, 0.78, 0.82, 0.86)
TOLERANCES: tuple[int, ...] = (0, 1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None, help="cap on the number of sweeps")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="figure directory")
    parser.add_argument("--no-figure", action="store_true", help="skip writing the figure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = urban_block()
    trace = run_mapping(scenario, RunConfig(frame="world", max_steps=args.steps))
    observed = trace.observed_mask

    thresholds = threshold_sweep(
        trace.grid, trace.truth, scenario.model, THRESHOLDS, region=observed
    )
    tolerances = tolerance_sweep(
        trace.grid, trace.truth, scenario.model, TOLERANCES, region=observed
    )

    print(f"scenario: {trace.scenario}, sweeps: {len(trace.steps)}")
    print(f"scored over the {int(observed.sum())} cells at least one beam reached\n")
    print("decision threshold sweep")
    print(agreement_table(thresholds, by="decision_prob"))
    print()
    print("spatial tolerance sweep on the occupied class, threshold held at "
          f"{scenario.model.decision_prob:.2f}")
    print(agreement_table(tolerances, by="spatial_tolerance"))

    if not args.no_figure:
        path = plot_threshold_sweep(
            thresholds,
            args.outdir / "threshold_sweep.png",
            title="Urban block: decided fraction and agreement against threshold",
        )
        print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()
