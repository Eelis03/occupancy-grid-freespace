"""Grid geometry for a planar occupancy map.

A :class:`GridSpec` fixes the resolution, the cell counts, and the world position of
the lower-left corner of cell ``(0, 0)``. Nothing here stores occupancy values; the
grid specification is a coordinate system and is treated as such.

Index convention. A cell is addressed by ``(row, col)``. ``col`` indexes the x axis
and ``row`` indexes the y axis, so an occupancy array has shape ``(rows, cols)`` and
``array[row, col]`` is the cell whose centre is at
``origin + ((col + 0.5) * resolution, (row + 0.5) * resolution)``. This is the layout
matplotlib expects from ``imshow`` with ``origin="lower"``, which keeps the analysis
layer free of transposes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from freespace_grid.model.typing import BoolArray, FloatArray, IntArray

__all__ = ["GridSpec", "cell_centers", "cell_to_world", "in_bounds", "world_to_cell"]


@dataclass(frozen=True, slots=True)
class GridSpec:
    """A regular axis-aligned grid in a planar world frame.

    Args:
        resolution: Edge length of one square cell, in metres.
        rows: Number of cells along the y axis.
        cols: Number of cells along the x axis.
        origin_x: World x coordinate of the lower-left corner of cell ``(0, 0)``.
        origin_y: World y coordinate of the lower-left corner of cell ``(0, 0)``.
    """

    resolution: float
    rows: int
    cols: int
    origin_x: float = 0.0
    origin_y: float = 0.0

    def __post_init__(self) -> None:
        if self.resolution <= 0.0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError(f"rows and cols must be positive, got {self.rows} and {self.cols}")

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape ``(rows, cols)`` of an occupancy array on this grid."""
        return (self.rows, self.cols)

    @property
    def size(self) -> int:
        """Total number of cells."""
        return self.rows * self.cols

    @property
    def cell_area(self) -> float:
        """Area of one cell, in square metres."""
        return self.resolution * self.resolution

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Bounds ``(x_min, x_max, y_min, y_max)`` of the mapped region, in metres."""
        return (
            self.origin_x,
            self.origin_x + self.cols * self.resolution,
            self.origin_y,
            self.origin_y + self.rows * self.resolution,
        )

    def recentered(self, center_x: float, center_y: float) -> GridSpec:
        """Return the same grid shape with its centre moved to ``(center_x, center_y)``.

        The resolution and the cell counts are preserved, so the returned
        specification differs only in its origin.
        """
        return GridSpec(
            resolution=self.resolution,
            rows=self.rows,
            cols=self.cols,
            origin_x=center_x - 0.5 * self.cols * self.resolution,
            origin_y=center_y - 0.5 * self.rows * self.resolution,
        )


def world_to_cell(spec: GridSpec, points: FloatArray) -> IntArray:
    """Map world points to the cells that contain them.

    Args:
        spec: Grid geometry.
        points: Array of shape ``(n, 2)`` holding world ``(x, y)`` coordinates.

    Returns:
        Integer array of shape ``(n, 2)`` holding ``(row, col)`` indices. Indices are
        not clipped: points outside the grid produce out-of-range indices, which
        :func:`in_bounds` can filter.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (n, 2), got {pts.shape}")
    col = np.floor((pts[:, 0] - spec.origin_x) / spec.resolution)
    row = np.floor((pts[:, 1] - spec.origin_y) / spec.resolution)
    return np.stack((row, col), axis=1).astype(np.int64)


def cell_to_world(spec: GridSpec, cells: IntArray) -> FloatArray:
    """Map ``(row, col)`` indices to the world coordinates of the cell centres.

    Args:
        spec: Grid geometry.
        cells: Integer array of shape ``(n, 2)`` holding ``(row, col)`` indices.

    Returns:
        Float array of shape ``(n, 2)`` holding world ``(x, y)`` cell centres.
    """
    idx = np.asarray(cells, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 2:
        raise ValueError(f"cells must have shape (n, 2), got {idx.shape}")
    x = spec.origin_x + (idx[:, 1].astype(np.float64) + 0.5) * spec.resolution
    y = spec.origin_y + (idx[:, 0].astype(np.float64) + 0.5) * spec.resolution
    return np.stack((x, y), axis=1)


def cell_centers(spec: GridSpec) -> tuple[FloatArray, FloatArray]:
    """Return the ``(x, y)`` centre coordinates of every cell as two ``(rows, cols)`` grids."""
    xs = spec.origin_x + (np.arange(spec.cols, dtype=np.float64) + 0.5) * spec.resolution
    ys = spec.origin_y + (np.arange(spec.rows, dtype=np.float64) + 0.5) * spec.resolution
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    return grid_x, grid_y


def in_bounds(spec: GridSpec, cells: IntArray) -> BoolArray:
    """Return a mask selecting the ``(row, col)`` pairs that address a cell of ``spec``."""
    idx = np.asarray(cells, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 2:
        raise ValueError(f"cells must have shape (n, 2), got {idx.shape}")
    row_ok = (idx[:, 0] >= 0) & (idx[:, 0] < spec.rows)
    col_ok = (idx[:, 1] >= 0) & (idx[:, 1] < spec.cols)
    return np.asarray(row_ok & col_ok, dtype=np.bool_)
