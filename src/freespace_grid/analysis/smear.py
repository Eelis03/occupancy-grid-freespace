"""Measuring how far a static world assumption smears a moving obstacle.

The log odds filter of this repository assumes the world does not change. When it
does, the map keeps the evidence deposited by an object that has since moved on, and
the object appears in the map as a streak rather than as a disc. This module measures
the streak, so the failure has a number attached to it rather than a caption under a
picture.

The measurement is a difference of two runs that are identical except for the motion.
The control run parks the obstacle at the position it reaches at the end of the moving
run, and both runs are scored inside the same region of interest, a band around the
path the obstacle travels. Reporting the moving run alone would confound the smear
with the resolution of the grid and the thickness of the obstacle; the difference
removes both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from freespace_grid.model.grid import GridSpec, cell_centers
from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.model.occupancy import CellState, OccupancyGrid
from freespace_grid.model.transform import Pose2D, inverse, transform_points
from freespace_grid.model.typing import BoolArray, FloatArray, IntArray
from freespace_grid.pipeline.runner import MappingTrace, RunConfig, run_mapping
from freespace_grid.pipeline.scenarios import dynamic_corridor
from freespace_grid.pipeline.scene import MovingCircle

__all__ = [
    "SmearCase",
    "SmearMetrics",
    "SmearReport",
    "compare_smear",
    "measure_smear",
    "path_region",
    "run_smear_case",
]


def path_region(
    spec: GridSpec,
    start: tuple[float, float],
    end: tuple[float, float],
    radius: float,
) -> BoolArray:
    """Return the cells whose centre lies within ``radius`` of the segment start to end."""
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    grid_x, grid_y = cell_centers(spec)
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        distance_sq = (grid_x - ax) ** 2 + (grid_y - ay) ** 2
    else:
        t = ((grid_x - ax) * dx + (grid_y - ay) * dy) / length_sq
        t = np.clip(t, 0.0, 1.0)
        distance_sq = (grid_x - (ax + t * dx)) ** 2 + (grid_y - (ay + t * dy)) ** 2
    return np.asarray(distance_sq <= radius * radius, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class SmearMetrics:
    """What one run left in the region of interest."""

    label: str
    roi_cells: int
    occupied_cells: int
    occupied_area: float
    extent_along: float
    extent_across: float
    truth_cells: int
    truth_area: float
    stale_cells: int
    stale_area: float
    false_free_cells: int
    detected_cells: int
    peak_returns_per_cell: int

    @property
    def area_ratio(self) -> float:
        """Occupied area recovered divided by the true footprint area in the region."""
        return self.occupied_area / self.truth_area if self.truth_area else 0.0

    @property
    def unknown_footprint_cells(self) -> int:
        """Footprint cells the map neither found nor wrongly cleared.

        A cell here is one the obstacle occupies and the map reports unknown. That is
        the safe failure: a planner treating unknown as impassable is not endangered by
        it, while a cell counted in ``false_free_cells`` is one the planner would drive
        into.
        """
        return self.truth_cells - self.detected_cells - self.false_free_cells


def measure_smear(
    grid: OccupancyGrid,
    truth: BoolArray,
    model: LogOddsModel,
    region: BoolArray,
    direction: FloatArray,
    *,
    label: str,
    occupied_observations: IntArray | None = None,
) -> SmearMetrics:
    """Summarise the occupied evidence inside ``region``.

    Args:
        grid: Final map of the run.
        truth: Ground truth occupancy at the final instant.
        model: Supplies the decision band.
        region: Cells to measure over, normally a band around the obstacle path.
        direction: Unit vector along the obstacle motion. Extents are reported along
            this axis and across it.
        label: Name recorded in the result.
        occupied_observations: Optional per-cell count of range returns placed during
            the run, used to report how much evidence the best served cell received.

    Returns:
        The :class:`SmearMetrics` for this run.
    """
    states = grid.classify(model)
    occupied = region & (states == int(CellState.OCCUPIED))
    called_free = region & (states == int(CellState.FREE))
    truth_in_region = region & np.asarray(truth, dtype=np.bool_)

    unit = np.asarray(direction, dtype=np.float64).reshape(2)
    if float(np.hypot(unit[0], unit[1])) == 0.0:
        raise ValueError("direction must be a non-zero vector")

    # Express the occupied cell centres in a frame whose x axis points along the motion.
    # The extents are then the coordinate ranges, with no separate normal to derive.
    grid_x, grid_y = cell_centers(grid.spec)
    centres = np.stack((grid_x[occupied], grid_y[occupied]), axis=1)
    heading = math.atan2(float(unit[1]), float(unit[0]))
    aligned = transform_points(inverse(Pose2D(0.0, 0.0, heading)), centres)
    cell = grid.spec.resolution

    return SmearMetrics(
        label=label,
        roi_cells=int(np.count_nonzero(region)),
        occupied_cells=int(np.count_nonzero(occupied)),
        occupied_area=float(np.count_nonzero(occupied)) * grid.spec.cell_area,
        extent_along=_extent(aligned[:, 0], cell),
        extent_across=_extent(aligned[:, 1], cell),
        truth_cells=int(np.count_nonzero(truth_in_region)),
        truth_area=float(np.count_nonzero(truth_in_region)) * grid.spec.cell_area,
        stale_cells=int(np.count_nonzero(occupied & ~truth_in_region)),
        stale_area=float(np.count_nonzero(occupied & ~truth_in_region)) * grid.spec.cell_area,
        false_free_cells=int(np.count_nonzero(truth_in_region & called_free)),
        detected_cells=int(np.count_nonzero(truth_in_region & occupied)),
        peak_returns_per_cell=(
            0 if occupied_observations is None else int(occupied_observations[region].max())
        ),
    )


def _extent(values: FloatArray, cell: float) -> float:
    """Span of a set of cell centre coordinates on one axis, in metres.

    One cell width is added because the centres mark cells, not points: a single cell
    has an extent of one resolution, not zero.
    """
    if values.size == 0:
        return 0.0
    return float(values.max() - values.min()) + cell


@dataclass(frozen=True, slots=True)
class SmearReport:
    """The difference between a moving run and its parked control."""

    label: str
    moving: SmearMetrics
    parked: SmearMetrics
    swept_distance: float

    @property
    def smear_length(self) -> float:
        """Extra extent along the motion axis, in metres, attributable to the motion."""
        return self.moving.extent_along - self.parked.extent_along

    @property
    def smear_fraction_of_sweep(self) -> float:
        """Smear length as a fraction of the distance the obstacle actually travelled."""
        return self.smear_length / self.swept_distance if self.swept_distance else 0.0

    @property
    def area_inflation(self) -> float:
        """Occupied area of the moving run divided by that of the parked control."""
        return (
            self.moving.occupied_area / self.parked.occupied_area
            if self.parked.occupied_area
            else 0.0
        )

    @property
    def missed_footprint_fraction(self) -> float:
        """Fraction of the obstacle's true footprint the moving run reports as free.

        This is the second failure of the static world assumption and it points the
        other way from the smear. A filter stubborn enough not to leave a trail is also
        stubborn enough to keep calling a cell free while a car stands in it.
        """
        return (
            float(self.moving.false_free_cells) / float(self.moving.truth_cells)
            if self.moving.truth_cells
            else 0.0
        )

    def as_row(self) -> dict[str, float]:
        """Flat mapping suitable for a table or a JSON record."""
        return {
            "swept_distance": self.swept_distance,
            "parked_extent_along": self.parked.extent_along,
            "moving_extent_along": self.moving.extent_along,
            "smear_length": self.smear_length,
            "smear_fraction_of_sweep": self.smear_fraction_of_sweep,
            "parked_occupied_area": self.parked.occupied_area,
            "moving_occupied_area": self.moving.occupied_area,
            "area_inflation": self.area_inflation,
            "moving_stale_cells": float(self.moving.stale_cells),
            "parked_stale_cells": float(self.parked.stale_cells),
            "moving_false_free_cells": float(self.moving.false_free_cells),
            "moving_detected_cells": float(self.moving.detected_cells),
            "moving_unknown_footprint_cells": float(self.moving.unknown_footprint_cells),
            "footprint_cells": float(self.moving.truth_cells),
            "missed_footprint_fraction": self.missed_footprint_fraction,
        }


def compare_smear(
    label: str,
    moving: SmearMetrics,
    parked: SmearMetrics,
    mover: MovingCircle,
    duration: float,
) -> SmearReport:
    """Pair a moving run with its parked control and record the swept distance."""
    return SmearReport(
        label=label,
        moving=moving,
        parked=parked,
        swept_distance=mover.speed * duration,
    )


@dataclass(frozen=True, slots=True)
class SmearCase:
    """One moving run, its parked control, and the comparison between them."""

    report: SmearReport
    moving_trace: MappingTrace
    parked_trace: MappingTrace
    region: BoolArray


def run_smear_case(
    direction: str,
    *,
    model: LogOddsModel | None = None,
    max_steps: int | None = None,
    label: str | None = None,
    margin: float = 1.0,
) -> SmearCase:
    """Run one dynamic corridor case against its parked control and measure the difference.

    Args:
        direction: One of the directions accepted by
            :func:`freespace_grid.pipeline.scenarios.dynamic_corridor`.
        model: Optional replacement log odds parameters, used by the clamp sweep.
        max_steps: Optional cap on the number of sweeps, applied to both runs.
        label: Name recorded in the report. Defaults to ``direction``.
        margin: Extra width of the region of interest beyond the obstacle radius, in
            metres.

    Returns:
        The :class:`SmearCase` holding both traces, the region of interest, and the
        comparison.
    """
    scenario = dynamic_corridor(direction, model)
    mover = scenario.movers[0]
    duration = scenario.trajectory.duration
    final = mover.at(duration)
    region = path_region(
        scenario.grid,
        (mover.center_x, mover.center_y),
        (final.center_x, final.center_y),
        mover.radius + margin,
    )
    config = RunConfig(max_steps=max_steps)
    moving_trace = run_mapping(scenario, config)
    parked_trace = run_mapping(scenario.frozen_at(duration), config)
    direction_vector = mover.direction()
    return SmearCase(
        report=compare_smear(
            label if label is not None else direction,
            measure_smear(
                moving_trace.grid,
                moving_trace.truth,
                scenario.model,
                region,
                direction_vector,
                label="moving",
                occupied_observations=moving_trace.occupied_observations,
            ),
            measure_smear(
                parked_trace.grid,
                parked_trace.truth,
                scenario.model,
                region,
                direction_vector,
                label="parked",
                occupied_observations=parked_trace.occupied_observations,
            ),
            mover,
            duration,
        ),
        moving_trace=moving_trace,
        parked_trace=parked_trace,
        region=region,
    )
