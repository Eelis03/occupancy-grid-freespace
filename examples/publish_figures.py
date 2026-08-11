"""Regenerate the three figures tracked in docs/figures.

These are the only figures in version control and they exist because three findings in
this repository are spatial and cannot be read off a table: an obstacle that is absent
from the map, a scene whose unmapped part is larger than its mapped part, and a wall
drawn twice by a drifting pose. Everything else the analysis measures is a number and
belongs in a table.

The figures are deliberately small. Each is a flat colour decision map, so a modest
resolution loses nothing, and the whole directory is held under a quarter of a megabyte
without any compression step beyond what matplotlib does on its own.

Usage:
    uv run python examples/publish_figures.py
    uv run python examples/publish_figures.py --outdir outputs --steps 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

from freespace_grid.analysis.figures import Panel, plot_decision_map, plot_state_panels
from freespace_grid.analysis.smear import run_smear_case
from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.pipeline.runner import RunConfig, run_mapping
from freespace_grid.pipeline.scenarios import SCANNER_ODOMETRY, urban_block

DEFAULT_OUTDIR = Path("docs") / "figures"

# The corridor around the disc's path. Cropping is what makes a 24 by 40 metre grid
# readable at the width of a README column.
CORRIDOR_CROP = (7.0, 30.0, 13.0, 27.0)
# A section of the street block wide enough to hold two buildings and the corridor
# between them, which is where a doubled wall is easiest to see.
BLOCK_CROP = (18.0, 50.0, 8.0, 34.0)

# Scale two of the drift study, the middle row of the table in examples/pose_drift.py.
DRIFT = SCANNER_ODOMETRY.scaled(2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", type=Path, default=DEFAULT_OUTDIR, help="where to write the figures"
    )
    parser.add_argument("--steps", type=int, default=None, help="cap on the number of sweeps")
    return parser.parse_args()


def dynamic_obstacle_figure(outdir: Path, steps: int | None) -> Path:
    """The moving obstacle beside its parked control, at the published clamp."""
    case = run_smear_case("approaching", max_steps=steps)
    panels: tuple[Panel, ...] = (
        (
            "parked control: the disc is found and shadows the ground behind it",
            case.parked_trace,
            case.region,
        ),
        (
            "same disc, moving: its footprint is called free space",
            case.moving_trace,
            case.region,
        ),
    )
    return plot_state_panels(
        panels,
        LogOddsModel(),
        outdir / "dynamic_obstacle.png",
        title="A moving obstacle under a filter that assumes a static world",
        layout="columns",
        width=9.0,
        panel_height=2.6,
        dpi=100,
        crop=CORRIDOR_CROP,
        fontsize=8.0,
    )


def static_map_figure(outdir: Path, steps: int | None) -> Path:
    """The street block map, showing how much of the scene no beam ever reaches."""
    scenario = urban_block()
    trace = run_mapping(scenario, RunConfig(frame="world", max_steps=steps))
    return plot_decision_map(
        trace,
        scenario.model,
        outdir / "urban_block_map.png",
        title="Urban block: free in white, occupied in black, unknown in grey",
        width=7.5,
        dpi=100,
        fontsize=8.0,
    )


def pose_drift_figure(outdir: Path, steps: int | None) -> Path:
    """A drifting pose against the same drift corrected by scan to map matching."""
    scenario = urban_block()
    reckoned = run_mapping(scenario, RunConfig(frame="world", odometry=DRIFT, max_steps=steps))
    matched = run_mapping(
        scenario,
        RunConfig(frame="world", odometry=DRIFT, pose_correction=True, max_steps=steps),
    )
    panels: tuple[Panel, ...] = (
        ("dead reckoned: surfaces doubled and displaced", reckoned, None),
        ("scan matched: surfaces back on the outline", matched, None),
    )
    return plot_state_panels(
        panels,
        scenario.model,
        outdir / "pose_drift.png",
        title=(
            f"Odometry drift of {reckoned.final_position_error:.2f} metres, "
            f"uncorrected and corrected to {matched.final_position_error:.2f} metres"
        ),
        layout="columns",
        width=9.0,
        panel_height=3.6,
        dpi=100,
        crop=BLOCK_CROP,
        fontsize=8.0,
    )


def main() -> None:
    args = parse_args()
    total = 0
    for build in (dynamic_obstacle_figure, static_map_figure, pose_drift_figure):
        path = build(args.outdir, args.steps)
        size = path.stat().st_size
        total += size
        print(f"{path} {size / 1024:.1f} KiB")
    print(f"total {total / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
