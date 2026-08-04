"""The part of the free space a vehicle can actually drive in.

Agreement is a per-cell figure and treats every free cell alike. A vehicle does not:
it can only use free space joined to where it stands, and it can only be hurt by an
error inside that space. A cell wrongly called free behind a wall the map did find is
as wrong as one in the middle of the corridor and is not the same problem, and no
number in :mod:`freespace_grid.analysis.metrics` separates the two.

This module separates them by taking the map's own decision at face value and flooding
the free cells outward from the vehicle, with occupied and unknown cells alike treated
as impassable. That is the rule a planner applies to this map, and it makes unknown the
safe failure the rest of the repository claims it is: an unknown cell costs coverage
here and never appears as a hazard.

Three quantities come out of the flood.

``reachable_fraction``
    How much of the free space the map reports is joined to the vehicle. Free space it
    cannot get to is coverage the map has been paid for and the vehicle cannot spend.

``frontier_cells``
    Reachable free cells with an unknown cell beside them, in the sense of Yamauchi's
    exploration frontier. This is where a planner has to stop, so it measures the edge
    of the useful map rather than its area.

``reachable_false_free_cells``
    Truly occupied cells the map called free and the vehicle could reach. These are the
    ones a planner drives into.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from freespace_grid.model.grid import in_bounds, world_to_cell
from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.model.occupancy import CellState, OccupancyGrid
from freespace_grid.model.typing import BoolArray, IntArray

__all__ = ["Reachability", "measure_reachability", "reachable_free"]

# Four connectivity, not eight. Two occupied cells meeting at a corner leave a diagonal
# gap that an eight connected flood walks straight through, and a vehicle does not fit
# through it. The ray traversal refuses to cut the same corner for the same reason, so
# the two halves of the package agree about what a wall one cell thick is.
_NEIGHBOURHOOD = np.array(
    [[False, True, False], [True, True, True], [False, True, False]], dtype=bool
)


@dataclass(frozen=True, slots=True)
class Reachability:
    """What one map's free space is worth to a planner starting from one place."""

    free_cells: int
    reachable_cells: int
    reachable_area: float
    components: int
    frontier_cells: int
    false_free_cells: int
    reachable_false_free_cells: int

    @property
    def reachable_fraction(self) -> float:
        """Fraction of the free cells the map reports that are joined to the vehicle."""
        return float(self.reachable_cells) / float(self.free_cells) if self.free_cells else 0.0

    @property
    def stranded_cells(self) -> int:
        """Free cells no path of free cells reaches, which a planner cannot use."""
        return self.free_cells - self.reachable_cells

    @property
    def exposed_fraction(self) -> float:
        """Fraction of the cells wrongly called free that the vehicle could drive into.

        The rest are wrong in exactly the same way and are shut behind a surface the map
        did find. Separating them is the whole point of the flood: a per-cell agreement
        counts both against the map and only one of them can cause a collision.
        """
        if not self.false_free_cells:
            return 0.0
        return float(self.reachable_false_free_cells) / float(self.false_free_cells)


def reachable_free(
    grid: OccupancyGrid,
    model: LogOddsModel,
    start: tuple[float, float],
) -> BoolArray:
    """Return the free cells joined to ``start`` by a path of free cells.

    Args:
        grid: The map to search.
        model: Supplies the decision band.
        start: World ``(x, y)`` position the vehicle plans from.

    Returns:
        Mask of the free cells in the four connected component holding the start cell.
        Occupied and unknown cells are both impassable, so this is the free space a
        planner that refuses to enter either can reach.

    Raises:
        ValueError: If ``start`` lies outside the grid, or in a cell the map does not
            call free. Flooding from an occupied or unknown cell would return an empty
            mask and make every count zero, which reads like a map that found nothing
            rather than like a start position that was wrong.
    """
    free = np.asarray(grid.classify(model) == int(CellState.FREE), dtype=np.bool_)
    return _flood(free, grid, start)[0]


def measure_reachability(
    grid: OccupancyGrid,
    truth: BoolArray,
    model: LogOddsModel,
    start: tuple[float, float],
) -> Reachability:
    """Measure the free space of ``grid`` a planner starting at ``start`` could use.

    Args:
        grid: The map to measure.
        truth: Boolean ground truth occupancy of the same shape, true where occupied.
        model: Supplies the decision band.
        start: World ``(x, y)`` position the vehicle plans from, normally its pose at
            the last sweep.

    Returns:
        The :class:`Reachability` summary.
    """
    truth_occupied = np.asarray(truth, dtype=np.bool_)
    if truth_occupied.shape != grid.spec.shape:
        raise ValueError(
            f"truth shape {truth_occupied.shape} does not match grid {grid.spec.shape}"
        )

    states = grid.classify(model)
    free = np.asarray(states == int(CellState.FREE), dtype=np.bool_)
    unknown = np.asarray(states == int(CellState.UNKNOWN), dtype=np.bool_)
    reachable, components = _flood(free, grid, start)

    beside_unknown = np.asarray(
        ndimage.binary_dilation(unknown, structure=_NEIGHBOURHOOD), dtype=np.bool_
    )
    false_free = free & truth_occupied

    return Reachability(
        free_cells=int(np.count_nonzero(free)),
        reachable_cells=int(np.count_nonzero(reachable)),
        reachable_area=float(np.count_nonzero(reachable)) * grid.spec.cell_area,
        components=components,
        frontier_cells=int(np.count_nonzero(reachable & beside_unknown)),
        false_free_cells=int(np.count_nonzero(false_free)),
        reachable_false_free_cells=int(np.count_nonzero(false_free & reachable)),
    )


def _flood(
    free: BoolArray, grid: OccupancyGrid, start: tuple[float, float]
) -> tuple[BoolArray, int]:
    """Return the component of ``free`` holding ``start``, and the component count.

    Both come from one labelling, because the number of separate pockets of free space
    is only meaningful beside the size of the one the vehicle is in.
    """
    row, col = _start_cell(grid, start)
    if not bool(free[row, col]):
        raise ValueError(f"start cell {(row, col)} is not free, so no path can begin there")
    labelled, count = ndimage.label(free, structure=_NEIGHBOURHOOD)
    labels: IntArray = np.asarray(labelled, dtype=np.int64)
    return np.asarray(labels == labels[row, col], dtype=np.bool_), int(count)


def _start_cell(grid: OccupancyGrid, start: tuple[float, float]) -> tuple[int, int]:
    """Locate ``start`` on the grid, rejecting a position the grid does not cover."""
    point = np.array([[float(start[0]), float(start[1])]], dtype=np.float64)
    cell = world_to_cell(grid.spec, point)
    if not bool(in_bounds(grid.spec, cell)[0]):
        raise ValueError(f"start {start} lies outside the grid extent {grid.spec.extent}")
    return int(cell[0, 0]), int(cell[0, 1])
