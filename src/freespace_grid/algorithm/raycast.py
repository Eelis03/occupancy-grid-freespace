"""Vectorised grid traversal along a bundle of rays.

The traversal is the digital differential analyser of Amanatides and Woo. For each
ray the algorithm keeps the parameter value at which the next vertical grid line is
crossed and the parameter value at which the next horizontal grid line is crossed,
then repeatedly steps whichever comes first. It visits every cell the segment
actually passes through, which is what an occupancy update needs: a Bresenham line
skips cells at shallow crossings and would leave holes in the free space.

The loop runs over steps rather than over rays. Every array operation inside it acts
on all rays of the scan at once, so a 720 beam scan costs a few hundred numpy calls
rather than 720 Python loops.

Reference: J. Amanatides and A. Woo, "A fast voxel traversal algorithm for ray
tracing", Eurographics 1987, pages 3 to 10.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from freespace_grid.model.grid import GridSpec
from freespace_grid.model.typing import BoolArray, FloatArray, IntArray

__all__ = ["Traversal", "traverse_rays"]


@dataclass(frozen=True, slots=True)
class Traversal:
    """Cells visited by a bundle of rays, stored ray major.

    Attributes:
        rows: Row index of every visited cell, concatenated over rays.
        cols: Column index of every visited cell, concatenated over rays.
        ray_index: Index of the ray that visited each cell.
        counts: Number of cells visited by each ray, in ray order.
        offsets: Start position of each ray inside the flat arrays, length ``n + 1``.
        reached: For each ray, whether traversal ended at the endpoint cell rather
            than by leaving the grid. When true the last cell of that ray is the cell
            containing the endpoint.
    """

    rows: IntArray
    cols: IntArray
    ray_index: IntArray
    counts: IntArray
    offsets: IntArray
    reached: BoolArray

    @property
    def total_cells(self) -> int:
        """Number of cell visits recorded, counting repeats across rays."""
        return int(self.rows.size)

    def terminal_mask(self) -> BoolArray:
        """Mask selecting the last visited cell of every ray that reached its endpoint."""
        mask = np.zeros(self.rows.shape, dtype=np.bool_)
        selected = (self.counts > 0) & self.reached
        last = self.offsets[:-1][selected] + self.counts[selected] - 1
        mask[last] = True
        return mask


def _as_ray_array(values: FloatArray, count: int, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1 and array.shape == (2,):
        array = np.broadcast_to(array, (count, 2))
    if array.ndim != 2 or array.shape != (count, 2):
        raise ValueError(f"{name} must have shape ({count}, 2) or (2,), got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float64)


def traverse_rays(spec: GridSpec, origins: FloatArray, endpoints: FloatArray) -> Traversal:
    """Walk every ray from its origin to its endpoint and record the cells crossed.

    A ray whose origin lies outside the grid contributes no cells. A ray that leaves
    the grid before reaching its endpoint contributes the cells it crossed while
    inside and is reported with ``reached`` false, so the caller knows not to place a
    range return in the last cell.

    Where the ray passes exactly through a grid corner the traversal steps the x axis
    first and then the y axis, which visits the horizontally adjacent cell. The
    alternative, cutting the corner, would let free space leak diagonally through a
    wall one cell thick.

    Args:
        spec: Grid geometry.
        origins: Ray origins in world coordinates, shape ``(n, 2)`` or ``(2,)``.
        endpoints: Ray endpoints in world coordinates, shape ``(n, 2)``.

    Returns:
        A :class:`Traversal` holding the visited cells in ray major order.
    """
    ends = np.asarray(endpoints, dtype=np.float64)
    if ends.ndim != 2 or ends.shape[1] != 2:
        raise ValueError(f"endpoints must have shape (n, 2), got {ends.shape}")
    n_rays = int(ends.shape[0])
    starts = _as_ray_array(origins, n_rays, "origins")

    empty_i = np.zeros(0, dtype=np.int64)
    if n_rays == 0:
        return Traversal(
            rows=empty_i,
            cols=empty_i,
            ray_index=empty_i,
            counts=empty_i,
            offsets=np.zeros(1, dtype=np.int64),
            reached=np.zeros(0, dtype=np.bool_),
        )

    res = spec.resolution
    delta = ends - starts

    col = np.floor((starts[:, 0] - spec.origin_x) / res).astype(np.int64)
    row = np.floor((starts[:, 1] - spec.origin_y) / res).astype(np.int64)
    inside_start = (row >= 0) & (row < spec.rows) & (col >= 0) & (col < spec.cols)
    active = inside_start & np.all(np.isfinite(delta), axis=1)

    step_col = np.sign(delta[:, 0]).astype(np.int64)
    step_row = np.sign(delta[:, 1]).astype(np.int64)

    infinity = np.inf
    moves_x = delta[:, 0] != 0.0
    moves_y = delta[:, 1] != 0.0

    t_delta_x = np.where(moves_x, res / np.where(moves_x, np.abs(delta[:, 0]), 1.0), infinity)
    t_delta_y = np.where(moves_y, res / np.where(moves_y, np.abs(delta[:, 1]), 1.0), infinity)

    boundary_x = spec.origin_x + (col + (step_col > 0).astype(np.int64)).astype(np.float64) * res
    boundary_y = spec.origin_y + (row + (step_row > 0).astype(np.int64)).astype(np.float64) * res
    t_max_x = np.where(
        moves_x, (boundary_x - starts[:, 0]) / np.where(moves_x, delta[:, 0], 1.0), infinity
    )
    t_max_y = np.where(
        moves_y, (boundary_y - starts[:, 1]) / np.where(moves_y, delta[:, 1], 1.0), infinity
    )

    span = np.ceil(np.abs(delta[:, 0]) / res) + np.ceil(np.abs(delta[:, 1]) / res)
    bound = int(np.max(span)) + 3
    max_steps = min(bound, spec.rows + spec.cols + 2)

    visited_rows = np.zeros((max_steps, n_rays), dtype=np.int64)
    visited_cols = np.zeros((max_steps, n_rays), dtype=np.int64)
    visited_ok = np.zeros((max_steps, n_rays), dtype=np.bool_)
    reached = np.zeros(n_rays, dtype=np.bool_)

    for step in range(max_steps):
        if not bool(active.any()):
            break
        visited_ok[step] = active
        visited_rows[step] = row
        visited_cols[step] = col

        finished = active & (np.minimum(t_max_x, t_max_y) > 1.0)
        reached |= finished
        active = active & ~finished

        advance_x = active & (t_max_x <= t_max_y)
        advance_y = active & ~advance_x

        col = col + np.where(advance_x, step_col, 0)
        row = row + np.where(advance_y, step_row, 0)
        t_max_x = np.where(advance_x, t_max_x + t_delta_x, t_max_x)
        t_max_y = np.where(advance_y, t_max_y + t_delta_y, t_max_y)

        inside = (row >= 0) & (row < spec.rows) & (col >= 0) & (col < spec.cols)
        active = active & inside

    ok = visited_ok.T
    counts = ok.sum(axis=1).astype(np.int64)
    offsets = np.zeros(n_rays + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    ray_index = np.repeat(np.arange(n_rays, dtype=np.int64), counts)

    return Traversal(
        rows=np.ascontiguousarray(visited_rows.T[ok], dtype=np.int64),
        cols=np.ascontiguousarray(visited_cols.T[ok], dtype=np.int64),
        ray_index=ray_index,
        counts=counts,
        offsets=offsets,
        reached=reached,
    )
