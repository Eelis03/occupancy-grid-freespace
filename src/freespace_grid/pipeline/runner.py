"""The trajectory runner and the structured trace it produces.

One call drives the whole loop: for each pose, simulate a sweep, project it into the
map, and, when the map follows the vehicle, re-anchor the window. Everything the
analysis layer or a regression test needs is recorded in the returned trace rather
than printed, so the same run backs the example script, the figure, and the pinned
reference.

Two poses are involved once odometry is modelled and they must not be confused. The
sweep is simulated from the pose the vehicle really has, because that is where the
sensor is; the sweep is written into the map at the pose the vehicle believes it has,
because that is all the mapper knows. When ``RunConfig.odometry`` is left unset the two
are the same object and the odometry machinery is skipped entirely, so every run that
predates it is unchanged bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import numpy as np

from freespace_grid.algorithm.accumulation import Accumulator, ShiftPolicy
from freespace_grid.algorithm.scan_match import SearchWindow, match_scan, scan_body_points
from freespace_grid.model.grid import GridSpec
from freespace_grid.model.occupancy import CellState, OccupancyGrid
from freespace_grid.model.transform import Pose2D, compose
from freespace_grid.model.typing import BoolArray, IntArray
from freespace_grid.pipeline.lidar import simulate_scan
from freespace_grid.pipeline.odometry import OdometryNoise, noisy_increments, pose_error
from freespace_grid.pipeline.scenarios import Scenario
from freespace_grid.pipeline.scene import occupancy_truth
from freespace_grid.pipeline.trajectory import Trajectory

__all__ = ["MappingTrace", "RunConfig", "StepRecord", "run_mapping"]

MapFrame = Literal["world", "ego"]

# Odometry error is drawn from its own generator, seeded from the scenario seed as a
# sequence rather than as a scalar. A separate stream is what keeps the beam dropout and
# range noise of every sweep identical across noise levels, so a comparison between two
# odometry settings is a comparison of the odometry alone.
_ODOMETRY_STREAM = 1


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
        odometry: Odometry error model. ``None``, the default, means the mapper is
            handed the exact poses and no dead reckoning is performed. An
            :class:`OdometryNoise` with all coefficients zero runs the dead reckoning
            path with no error, which is how the two paths are checked against each
            other.
        pose_correction: Match each sweep against the map built so far and place it at
            the matched pose rather than at the dead reckoned one. Requires
            ``odometry`` to be set, since there is nothing to correct otherwise.
        search: Extent and resolution of the pose search. Defaults to a window of four
            cells and two degrees, stepped one cell and half a degree, refined twice.
        match_stride: Keep every ``match_stride``-th range return for matching. The
            cost of a match is linear in the point count and three degrees of freedom
            do not need several hundred points to be determined.
    """

    frame: MapFrame = "world"
    shift_policy: ShiftPolicy = "snap"
    ego_rows: int = 200
    ego_cols: int = 200
    max_steps: int | None = None
    odometry: OdometryNoise | None = None
    pose_correction: bool = False
    search: SearchWindow | None = None
    match_stride: int = 4


@dataclass(frozen=True, slots=True)
class StepRecord:
    """What one sweep contributed.

    ``x``, ``y`` and ``theta`` are the true pose. ``est_x``, ``est_y`` and ``est_theta``
    are the pose the sweep was actually written into the map at, which is the same thing
    unless an odometry model is in use. ``position_error`` and ``heading_error`` are the
    difference, in metres and radians.
    """

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
    est_x: float = 0.0
    est_y: float = 0.0
    est_theta: float = 0.0
    position_error: float = 0.0
    heading_error: float = 0.0
    match_evaluations: int = 0
    translation_correction: float = 0.0


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
    pose_corrections: int = 0

    @property
    def observed_mask(self) -> BoolArray:
        """Cells that at least one sweep touched."""
        return np.asarray(self.observed > 0, dtype=np.bool_)

    @property
    def final(self) -> StepRecord:
        """The last step record."""
        return self.steps[-1]

    @property
    def final_position_error(self) -> float:
        """Distance between the true and believed pose at the last sweep, in metres."""
        return self.steps[-1].position_error

    @property
    def peak_position_error(self) -> float:
        """Largest position error reached at any sweep, in metres."""
        return max(step.position_error for step in self.steps)

    @property
    def final_heading_error(self) -> float:
        """Heading error at the last sweep, in radians."""
        return self.steps[-1].heading_error

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
    if run.pose_correction and run.odometry is None:
        raise ValueError("pose_correction requires an odometry model to correct")
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
    increments = _odometry_increments(scenario, trajectory, run)
    search = run.search if run.search is not None else _default_search(scenario.grid.resolution)
    records: list[StepRecord] = []
    mover_positions: list[tuple[float, float]] = []
    estimate = trajectory.poses[0]
    corrections = 0

    for index, (pose, time) in enumerate(
        zip(trajectory.poses, trajectory.times, strict=True)
    ):
        if increments is None:
            estimate = pose
        elif index > 0:
            estimate = compose(estimate, increments[index - 1]).normalized()

        if run.frame == "ego":
            accumulator.reanchor(estimate.x, estimate.y, policy=run.shift_policy)

        scene = scenario.scene_at(time)
        scan = simulate_scan(scene, pose, scenario.lidar, rng)

        evaluations = 0
        correction = 0.0
        if run.pose_correction and index > 0:
            points = scan_body_points(
                scan.angles, scan.ranges, scan.is_hit, stride=run.match_stride
            )
            result = match_scan(
                accumulator.grid, scenario.model, points, estimate, window=search
            )
            estimate = result.pose
            evaluations = result.evaluations
            correction = result.translation_correction
            corrections += 1

        placed = scan if increments is None else replace(scan, pose=estimate)
        update = accumulator.integrate(placed.origin, placed.endpoints(), placed.is_hit)
        position_error, heading_error = pose_error(estimate, pose)

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
                est_x=estimate.x,
                est_y=estimate.y,
                est_theta=estimate.theta,
                position_error=position_error,
                heading_error=heading_error,
                match_evaluations=evaluations,
                translation_correction=correction,
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
            "pose_correction": run.pose_correction,
        },
        steps=tuple(records),
        grid=accumulator.grid,
        observed=accumulator.observed,
        occupied_observations=accumulator.occupied_observations,
        truth=truth,
        resamples=accumulator.resamples,
        lossless_resamples=accumulator.lossless_resamples,
        mover_positions=tuple(mover_positions),
        pose_corrections=corrections,
    )


def _odometry_increments(
    scenario: Scenario, trajectory: Trajectory, run: RunConfig
) -> tuple[Pose2D, ...] | None:
    """Draw one corrupted motion increment per transition, or ``None`` for exact poses."""
    if run.odometry is None:
        return None
    odometry_rng = np.random.default_rng([scenario.seed, _ODOMETRY_STREAM])
    return noisy_increments(trajectory, run.odometry, odometry_rng)


def _default_search(resolution: float) -> SearchWindow:
    """Search four cells and two degrees, stepped one cell and half a degree."""
    return SearchWindow(
        translation_radius=4.0 * resolution,
        heading_radius=math.radians(2.0),
        translation_step=resolution,
        heading_step=math.radians(0.5),
        refinements=2,
    )
