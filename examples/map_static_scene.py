"""Build a free space map of the static street block and score it.

Usage:
    uv run python examples/map_static_scene.py
    uv run python examples/map_static_scene.py --steps 12 --outdir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from freespace_grid.analysis.figures import plot_map
from freespace_grid.analysis.metrics import score_grid
from freespace_grid.analysis.report import render_table
from freespace_grid.pipeline.runner import RunConfig, run_mapping
from freespace_grid.pipeline.scenarios import urban_block


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

    totals = trace.totals()
    observed = trace.observed_mask
    on_observed = score_grid(trace.grid, trace.truth, scenario.model, region=observed)
    on_grid = score_grid(trace.grid, trace.truth, scenario.model)

    print(f"scenario: {trace.scenario}")
    print(f"grid: {trace.grid.spec.rows} by {trace.grid.spec.cols} cells "
          f"at {trace.grid.spec.resolution} m, extent {trace.grid.spec.extent}")
    print(
        render_table(
            ["quantity", "value"],
            [
                ["sweeps", f"{totals['scans']}"],
                ["beams retained", f"{totals['beams']}"],
                ["range returns", f"{totals['hits']}"],
                ["range limit returns", f"{totals['max_range_beams']}"],
                ["beams dropped", f"{totals['dropped']}"],
                ["cell visits", f"{totals['cell_visits']}"],
                ["cells observed", f"{int(observed.sum())} of {trace.grid.spec.size}"],
            ],
        )
    )
    print()
    print(
        render_table(
            ["region", "cells", "decided", "free agr", "occ agr", "balanced"],
            [
                [
                    "observed",
                    f"{on_observed.scored_cells}",
                    f"{on_observed.decided_fraction:.4f}",
                    f"{on_observed.free_agreement:.4f}",
                    f"{on_observed.occupied_agreement:.4f}",
                    f"{on_observed.balanced_agreement:.4f}",
                ],
                [
                    "whole grid",
                    f"{on_grid.scored_cells}",
                    f"{on_grid.decided_fraction:.4f}",
                    f"{on_grid.free_agreement:.4f}",
                    f"{on_grid.occupied_agreement:.4f}",
                    f"{on_grid.balanced_agreement:.4f}",
                ],
            ],
        )
    )

    if not args.no_figure:
        path = plot_map(trace, scenario.model, args.outdir / "static_scene_map.png",
                        title="Urban block, world fixed grid")
        print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()
