"""The trajectory runner and the structured trace it produces.

One call drives the whole loop: for each pose, simulate a sweep, project it into the
map, and, when the map follows the vehicle, re-anchor the window. Everything the
analysis layer or a regression test needs is recorded in the returned trace rather
than printed, so the same run backs the example script, the figure, and the pinned
reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from freespace_grid.algorithm.accumulation import Accumulator, ShiftPolicy
from freespace_grid.model.grid import GridSpec
from freespace_grid.model.occupancy import CellState, OccupancyGrid
from freespace_grid.model.typing import BoolArray, IntArray
from freespace_grid.pipeline.lidar import simulate_scan
from freespace_grid.pipeline.scenarios import Scenario
from freespace_grid.pipeline.scene import occupancy_truth

__all__ = ["MappingTrace", "RunConfig", "StepRecord", "run_mapping"]

MapFrame = Literal["world", "ego"]


@dataclass(frozen=True, slots=True)
class RunConfig:
    """How the map is maintained during a run.

    Args:
        frame: ``world`` keeps one grid fixed in the world frame for the whole run.
            ``ego`` keeps a window of fixed size centred on the vehicle.
        shift_policy: How the window is re-anchored when ``frame`` is ``ego``.
        ego_rows: Row count of the ego window.
        ego_cols: Column count of the ego window.
        max_steps: Optional cap on the number of poses visited, applied by even
            subsampling so the route and elapsed time are preserved.
    """

    frame: MapFrame = "world"
    shift_policy: ShiftPolicy = "snap"
    ego_rows: int = 200
    ego_cols: int = 200
    max_steps: int | None = None


@dataclass(frozen=True, slots=True)
class StepRecord:
    """What one sweep contributed."""

    index: int
    time: float
    x: float
    y: float
    theta: float
    beams: int
    hits: int
    max_range_beams: int
    dropped: int
    placed_returns: int
    cell_visits: int
    touched_cells: int
    free_cells: int
    occupied_cells: int
    unknown_cells: int


@dataclass(frozen=True, slots=True)
class MappingTrace:
    """The complete record of one run."""

    scenario: str
    config: dict[str, Any]
    steps: tuple[StepRecord, ...]
    grid: OccupancyGrid
    observed: IntArray
    occupied_observations: IntArray
    truth: BoolArray
    resamples: int = 0
    lossless_resamples: int = 0
    mover_positions: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    @property
    def observed_mask(self) -> BoolArray:
        """Cells that at least one sweep touched."""
        return np.asarray(self.observed > 0, dtype=np.bool_)

    @property
    def final(self) -> StepRecord:
        """The last step record."""
        return self.steps[-1]

    def totals(self) -> dict[str, int]:
        """Summed beam counts over the whole run."""
        return {
            "scans": len(self.steps),
            "beams": sum(s.beams for s in self.steps),
            "hits": sum(s.hits for s in self.steps),
            "max_range_beams": sum(s.max_range_beams for s in self.steps),
            "dropped": sum(s.dropped for s in self.steps),
            "placed_returns": sum(s.placed_returns for s in self.steps),
            "cell_visits": sum(s.cell_visits for s in self.steps),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialisable summary. Arrays are reduced to counts, not embedded."""
        return {
            "scenario": self.scenario,
            "config": dict(self.config),
            "totals": self.totals(),
            "resamples": self.resamples,
            "lossless_resamples": self.lossless_resamples,
            "observed_cells": int(np.count_nonzero(self.observed)),
            "grid_cells": int(self.grid.spec.size),
            "steps": [asdict(step) for step in self.steps],
        }


def run_mapping(scenario: Scenario, config: RunConfig | None = None) -> MappingTrace:
    """Drive ``scenario`` along its trajectory and return the trace.

    Args:
        scenario: Geometry, sensor, trajectory, grid and seed.
        config: Map maintenance policy. Defaults to a world-fixed grid.

    Returns:
        A :class:`MappingTrace` holding the final map, the observation counter, the
        ground truth on the final grid, and one record per sweep.
    """
    run = config if config is not None else RunConfig()
    trajectory = scenario.trajectory
    if run.max_steps is not None:
        trajectory = trajectory.subsample(run.max_steps)

    if run.frame == "world":
        spec = scenario.grid
    else:
        first = trajectory.poses[0]
        spec = GridSpec(
            resolution=scenario.grid.resolution,
            rows=run.ego_rows,
            cols=run.ego_cols,
        ).recentered(first.x, first.y)

    accumulator = Accumulator.from_prior(spec, scenario.model)
    rng = np.random.default_rng(scenario.seed)
    records: list[StepRecord] = []
    mover_positions: list[tuple[float, float]] = []

    for index, (pose, time) in enumerate(
        zip(trajectory.poses, trajectory.times, strict=True)
    ):
        if run.frame == "ego":
            accumulator.reanchor(pose.x, pose.y, policy=run.shift_policy)

        scene = scenario.scene_at(time)
        scan = simulate_scan(scene, pose, scenario.lidar, rng)
        update = accumulator.integrate(scan.origin, scan.endpoints(), scan.is_hit)

        states = accumulator.grid.classify(scenario.model)
        records.append(
            StepRecord(
                index=index,
                time=time,
                x=pose.x,
                y=pose.y,
                theta=pose.theta,
                beams=int(scan.angles.size),
                hits=scan.hit_count,
                max_range_beams=scan.max_range_count,
                dropped=scan.dropped,
                placed_returns=update.placed_returns,
                cell_visits=update.cell_visits,
                touched_cells=update.touched_cells,
                free_cells=int(np.count_nonzero(states == int(CellState.FREE))),
                occupied_cells=int(np.count_nonzero(states == int(CellState.OCCUPIED))),
                unknown_cells=int(np.count_nonzero(states == int(CellState.UNKNOWN))),
            )
        )
        for mover in scenario.movers:
            disc = mover.at(time)
            mover_positions.append((disc.center_x, disc.center_y))

    final_time = trajectory.times[-1]
    truth = occupancy_truth(scenario.scene_at(final_time), accumulator.grid.spec)

    return MappingTrace(
        scenario=scenario.name,
        config={
            "frame": run.frame,
            "shift_policy": run.shift_policy,
            "resolution": scenario.grid.resolution,
            "rows": accumulator.grid.spec.rows,
            "cols": accumulator.grid.spec.cols,
            "seed": scenario.seed,
            "max_range": scenario.lidar.max_range,
            "angular_resolution_deg": scenario.lidar.angular_resolution_deg,
            "range_noise_std": scenario.lidar.range_noise_std,
            "dropout_prob": scenario.lidar.dropout_prob,
            "p_free": scenario.model.p_free,
            "p_occupied": scenario.model.p_occupied,
            "clamp_free_prob": scenario.model.clamp_free_prob,
            "clamp_occupied_prob": scenario.model.clamp_occupied_prob,
            "decision_prob": scenario.model.decision_prob,
            "steps": len(trajectory),
        },
        steps=tuple(records),
        grid=accumulator.grid,
        observed=accumulator.observed,
        occupied_observations=accumulator.occupied_observations,
        truth=truth,
        resamples=accumulator.resamples,
        lossless_resamples=accumulator.lossless_resamples,
        mover_positions=tuple(mover_positions),
    )
