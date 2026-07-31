"""The inverse sensor model: one scan turned into a log odds increment per cell.

The forward sensor model gives ``p(z | m)``, the distribution of a range reading
given a map. Occupancy grid mapping needs the inverse, ``p(m | z)``, and the standard
construction of Moravec and Elfes supplies it directly rather than by inversion: the
cells a beam passed through are assigned a probability below the prior, and the cell
in which the beam terminated is assigned a probability above it. In log odds the
update is an addition, so the scan reduces to a sparse array of increments.

The maximum range case
----------------------

A beam that returns no echo is not a beam that measured nothing. It says that the
sensor saw no surface anywhere between its origin and its range limit, which is free
space evidence along the whole beam, and it says nothing at all about the cell at the
range limit. Treating the range limit as a range return puts a phantom wall on an arc
at the maximum range, and every such arc drawn from a different pose deposits another
phantom. Treating the beam as carrying no information instead throws away the
strongest free space evidence in the scan, because unreturned beams are exactly the
ones that crossed open ground.

This module therefore takes ``is_hit`` per beam and applies the occupied increment
only where it is true, while the free increment is applied along the full traversal
of every beam, including the terminal cell of a maximum range beam.

Within one scan a cell receives at most one increment. Many beams cross the cells
nearest the sensor, and adding one increment per beam would treat a single scan as
dozens of independent observations of those cells, saturating them in one frame. When
a cell is both crossed by one beam and terminates another, the occupied increment
wins, because a surface detected in that cell is the more specific claim.

References: H. Moravec and A. Elfes, "High resolution maps from wide angle sonar",
ICRA 1985; S. Thrun, W. Burgard and D. Fox, "Probabilistic Robotics", MIT Press,
2005, chapter 9.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from freespace_grid.algorithm.raycast import traverse_rays
from freespace_grid.model.grid import GridSpec
from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.model.occupancy import OccupancyGrid
from freespace_grid.model.typing import BoolArray, FloatArray, IntArray

__all__ = ["ScanUpdate", "apply_scan", "scan_update"]


@dataclass(frozen=True, slots=True)
class ScanUpdate:
    """The cells one scan touches, split by the increment they receive.

    Attributes:
        free_cells: ``(m, 2)`` array of ``(row, col)`` indices receiving the free
            increment. Disjoint from ``occupied_cells``.
        occupied_cells: ``(k, 2)`` array of ``(row, col)`` indices receiving the
            occupied increment.
        beams: Number of beams in the scan.
        hit_beams: Number of beams carrying a range return.
        max_range_beams: Number of beams that reached the range limit without a return.
        placed_returns: Number of range returns that landed inside the grid.
        cell_visits: Total traversal records, counting a cell once per beam crossing it.
    """

    free_cells: IntArray
    occupied_cells: IntArray
    beams: int
    hit_beams: int
    max_range_beams: int
    placed_returns: int
    cell_visits: int

    @property
    def touched_cells(self) -> int:
        """Number of distinct cells that receive an increment."""
        return int(self.free_cells.shape[0] + self.occupied_cells.shape[0])


def scan_update(
    spec: GridSpec,
    origin: FloatArray,
    endpoints: FloatArray,
    is_hit: BoolArray,
) -> ScanUpdate:
    """Reduce one scan to the disjoint sets of free and occupied cells.

    Args:
        spec: Grid geometry the scan is projected onto.
        origin: Sensor position in world coordinates, shape ``(2,)``.
        endpoints: Beam endpoints in world coordinates, shape ``(n, 2)``. For a beam
            with a range return this is the measured surface point; for a beam at the
            range limit it is the point at the range limit.
        is_hit: Shape ``(n,)``. True where the beam carries a range return.

    Returns:
        A :class:`ScanUpdate` whose two cell sets are disjoint.
    """
    ends = np.asarray(endpoints, dtype=np.float64)
    hits = np.asarray(is_hit, dtype=np.bool_)
    if ends.ndim != 2 or ends.shape[1] != 2:
        raise ValueError(f"endpoints must have shape (n, 2), got {ends.shape}")
    if hits.shape != (ends.shape[0],):
        raise ValueError(f"is_hit must have shape ({ends.shape[0]},), got {hits.shape}")

    traversal = traverse_rays(spec, np.asarray(origin, dtype=np.float64), ends)

    # A traversal record is the terminal cell of a range return only when the beam
    # carries a return and reached its endpoint without leaving the grid. Maximum
    # range beams are excluded here, and that exclusion is the whole point.
    terminal = traversal.terminal_mask()
    occupied_records = terminal & hits[traversal.ray_index]
    free_records = ~occupied_records

    occupied_flat = _unique_flat(
        traversal.rows[occupied_records], traversal.cols[occupied_records], spec
    )
    free_flat = _unique_flat(traversal.rows[free_records], traversal.cols[free_records], spec)
    free_flat = free_flat[~np.isin(free_flat, occupied_flat, assume_unique=True)]

    return ScanUpdate(
        free_cells=_unflatten(free_flat, spec),
        occupied_cells=_unflatten(occupied_flat, spec),
        beams=int(hits.size),
        hit_beams=int(np.count_nonzero(hits)),
        max_range_beams=int(np.count_nonzero(~hits)),
        placed_returns=int(np.count_nonzero(occupied_records)),
        cell_visits=traversal.total_cells,
    )


def apply_scan(
    grid: OccupancyGrid,
    model: LogOddsModel,
    origin: FloatArray,
    endpoints: FloatArray,
    is_hit: BoolArray,
    *,
    observed: IntArray | None = None,
) -> ScanUpdate:
    """Apply one scan to ``grid`` in place and return what it did.

    Args:
        grid: The map to update. Its ``log_odds`` array is modified.
        model: Increment sizes and clamp bounds.
        origin: Sensor position in world coordinates, shape ``(2,)``.
        endpoints: Beam endpoints in world coordinates, shape ``(n, 2)``.
        is_hit: True where the beam carries a range return.
        observed: Optional counter array of the grid shape. When given, each touched
            cell has its counter incremented by one, which records the cells the
            sensor was able to observe at all.

    Returns:
        The :class:`ScanUpdate` that was applied.
    """
    update = scan_update(grid.spec, origin, endpoints, is_hit)
    log_odds = grid.log_odds
    if update.free_cells.size:
        rows, cols = update.free_cells[:, 0], update.free_cells[:, 1]
        log_odds[rows, cols] += model.l_free
        if observed is not None:
            observed[rows, cols] += 1
    if update.occupied_cells.size:
        rows, cols = update.occupied_cells[:, 0], update.occupied_cells[:, 1]
        log_odds[rows, cols] += model.l_occupied
        if observed is not None:
            observed[rows, cols] += 1
    np.clip(log_odds, model.l_min, model.l_max, out=log_odds)
    return update


def _unique_flat(rows: IntArray, cols: IntArray, spec: GridSpec) -> IntArray:
    """Return the sorted unique flat indices of the given cells."""
    if rows.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(rows * spec.cols + cols)


def _unflatten(flat: IntArray, spec: GridSpec) -> IntArray:
    """Turn flat indices back into an ``(n, 2)`` array of ``(row, col)`` pairs."""
    return np.stack((flat // spec.cols, flat % spec.cols), axis=1).astype(np.int64)
