"""Measure what a static world assumption does to a moving obstacle.

Three motions are run against a parked control with identical geometry at the final
instant, and the free clamp is then swept to show that the same parameter controls
both failures: a filter loose enough to track the obstacle leaves a trail behind it,
and a filter tight enough not to leave a trail reports free space where the obstacle
stands.

Usage:
    uv run python examples/dynamic_smear.py
    uv run python examples/dynamic_smear.py --steps 10 --no-figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

from freespace_grid.analysis.figures import Panel, plot_state_panels
from freespace_grid.analysis.report import render_table, smear_table
from freespace_grid.analysis.smear import run_smear_case
from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.pipeline.scenarios import DYNAMIC_DIRECTIONS

# The decision threshold of 0.65 puts a ceiling of 0.35 on the free clamp, since a
# clamp above it would make the free decision unreachable. The sweep runs up to 0.34.
CLAMPS: tuple[float, ...] = (0.05, 0.12, 0.20, 0.28, 0.34)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None, help="cap on the number of sweeps")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="figure directory")
    parser.add_argument("--no-figure", action="store_true", help="skip writing the figure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default = LogOddsModel()

    cases = [run_smear_case(name, max_steps=args.steps) for name in DYNAMIC_DIRECTIONS]
    print(
        f"default model: p_free {default.p_free}, p_occupied {default.p_occupied}, "
        f"clamp {default.clamp_free_prob} to {default.clamp_occupied_prob}, "
        f"decision {default.decision_prob}"
    )
    print(
        f"one occupied observation is undone by {default.forget_ratio:.2f} free "
        f"observations; a fully clamped free cell needs "
        f"{default.observations_to_occupied():.2f} occupied observations to be called occupied"
    )
    print(f"sweeps per run: {len(cases[0].moving_trace.steps)}\n")
    print("motion relative to the stationary sensor, default model")
    print(smear_table([case.report for case in cases]))

    print()
    print("free clamp sweep, approaching case")
    sweep = [
        run_smear_case(
            "approaching",
            model=LogOddsModel(clamp_free_prob=value),
            max_steps=args.steps,
            label=f"{value:.2f}",
        )
        for value in CLAMPS
    ]
    print(
        render_table(
            [
                "clamp",
                "occ obs needed",
                "smear (m)",
                "stale (m2)",
                "found",
                "unknown",
                "called free",
                "parked finds",
            ],
            [
                [
                    case.report.label,
                    f"{LogOddsModel(clamp_free_prob=value).observations_to_occupied():.2f}",
                    f"{case.report.smear_length:.2f}",
                    f"{case.report.moving.stale_area:.2f}",
                    f"{case.report.moving.detected_cells:d}",
                    f"{case.report.moving.unknown_footprint_cells:d}",
                    f"{case.report.moving.false_free_cells:d}",
                    f"{case.report.parked.detected_cells:d}",
                ]
                for value, case in zip(CLAMPS, sweep, strict=True)
            ],
        )
    )

    if not args.no_figure:
        loose = sweep[CLAMPS.index(0.28)]
        panels: tuple[Panel, ...] = (
            (
                "approaching, parked control: the obstacle and its shadow, correctly placed",
                cases[0].parked_trace,
                cases[0].region,
            ),
            (
                "approaching, moving, free clamp 0.12: obstacle largely missed, little trail",
                cases[0].moving_trace,
                cases[0].region,
            ),
            (
                "approaching, moving, free clamp 0.28: obstacle found, trail along the whole path",
                loose.moving_trace,
                loose.region,
            ),
            (
                "crossing, moving, free clamp 0.12: most of the footprint still called free",
                cases[2].moving_trace,
                cases[2].region,
            ),
        )
        path = plot_state_panels(panels, default, args.outdir / "dynamic_smear.png")
        print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()
