"""Measure what a drifting odometry does to the map, and what correcting it recovers.

Every other script in this repository hands the mapper the pose the simulator used. This
one hands it a dead reckoned pose instead, corrupted increment by increment, and then
reruns each level with scan to map matching switched on. The two halves of the table
answer two different questions: how much of the map a pose error destroys, and how much
of that a correction gets back and at what price.

Usage:
    uv run python examples/pose_drift.py
    uv run python examples/pose_drift.py --steps 10 --no-figure
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from freespace_grid.analysis.figures import Panel, plot_state_panels
from freespace_grid.analysis.metrics import score_grid
from freespace_grid.analysis.report import render_table
from freespace_grid.analysis.sharpness import measure_sharpness
from freespace_grid.pipeline.runner import MappingTrace, RunConfig, run_mapping
from freespace_grid.pipeline.scenarios import SCANNER_ODOMETRY, urban_block

# The scale column of the table multiplies all three coefficients of SCANNER_ODOMETRY,
# and because the same variates are drawn at every level the drift at scale two is
# exactly twice the drift at scale one rather than another draw of the same size.
BASE = SCANNER_ODOMETRY
SCALES: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0)
FIGURE_SCALE = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None, help="cap on the number of sweeps")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="figure directory")
    parser.add_argument("--no-figure", action="store_true", help="skip writing the figure")
    return parser.parse_args()


def row(label: str, poses: str, trace: MappingTrace) -> list[str]:
    scenario = urban_block()
    observed = trace.observed_mask
    agreement = score_grid(trace.grid, trace.truth, scenario.model, region=observed)
    sharpness = measure_sharpness(trace.grid, trace.truth, scenario.model, region=observed)
    return [
        label,
        poses,
        f"{trace.final_position_error:.3f}",
        f"{trace.peak_position_error:.3f}",
        f"{math.degrees(trace.final_heading_error):.2f}",
        f"{agreement.decided_fraction:.4f}",
        f"{agreement.free_agreement:.4f}",
        f"{agreement.occupied_agreement:.4f}",
        f"{sharpness.edge_contrast:.3f}",
    ]


def main() -> None:
    args = parse_args()
    scenario = urban_block()
    rows: list[list[str]] = []

    exact = run_mapping(scenario, RunConfig(frame="world", max_steps=args.steps))
    rows.append(row("exact", "given", exact))

    figure_panels: dict[str, MappingTrace] = {}
    for scale in SCALES:
        noise = BASE.scaled(scale)
        reckoned = run_mapping(
            scenario, RunConfig(frame="world", odometry=noise, max_steps=args.steps)
        )
        matched = run_mapping(
            scenario,
            RunConfig(
                frame="world", odometry=noise, pose_correction=True, max_steps=args.steps
            ),
        )
        rows.append(row(f"{scale:.1f}", "dead reckoned", reckoned))
        rows.append(row(f"{scale:.1f}", "matched", matched))
        if scale == FIGURE_SCALE:
            figure_panels = {"reckoned": reckoned, "matched": matched}

    print(f"scenario: {scenario.name}, sweeps: {len(exact.steps)}")
    print(
        "odometry at scale 1.0: "
        f"{BASE.translation_std_per_m * 100:.0f} percent of distance in translation, "
        f"{math.degrees(BASE.heading_std_per_m):.2f} degrees per metre in heading, "
        f"{math.degrees(BASE.heading_std_per_rad):.2f} degrees per radian turned"
    )
    print(f"pose searches performed per matched run: {figure_panels['matched'].pose_corrections}")
    print()
    print(
        render_table(
            [
                "scale",
                "poses",
                "final err (m)",
                "peak err (m)",
                "heading err (deg)",
                "decided",
                "free agr",
                "occ agr",
                "edge contrast",
            ],
            rows,
        )
    )

    if not args.no_figure:
        panels: tuple[Panel, ...] = (
            (
                "dead reckoned: every surface doubled along the drift",
                figure_panels["reckoned"],
                None,
            ),
            (
                "scan matched: surfaces back on the ground truth outline",
                figure_panels["matched"],
                None,
            ),
        )
        path = plot_state_panels(
            panels,
            scenario.model,
            args.outdir / "pose_drift.png",
            title=f"Odometry drift at scale {FIGURE_SCALE:.1f}, with and without correction",
            layout="columns",
            width=11.0,
            panel_height=4.0,
        )
        print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()
