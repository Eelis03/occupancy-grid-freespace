"""Temporal accumulation of scans, and re-anchoring the map under vehicle motion.

Two policies are supported for a map that follows the vehicle.

``snap``
    The window origin is moved by a whole number of cells, chosen as the nearest
    whole-cell approximation of the requested centre. Every shift is then a pure
    array translation: values are copied, not combined, so the operation is lossless
    and idempotent up to the cells that fall off the edge. The cost is that the window
    centre lags the vehicle by up to half a cell.

``bilinear``
    The window is centred exactly on the vehicle and the log odds field is resampled
    by bilinear interpolation. The centre never lags, but each resampling step is a
    low pass filter applied to the whole map. Applied once per frame over a long run
    the filters compose, evidence at the clamp is pulled back towards the prior, and
    obstacle edges spread over several cells. The effect is measured in
    ``examples/compare_grid_frames.py``.

``nearest``
    The window is centred exactly on the vehicle and resampled by nearest neighbour.
    No blur is introduced, but every cell is displaced by up to half a cell per frame
    and the accumulated jitter thickens thin structures instead of blurring them.

Interpolating log odds rather than probability is deliberate: log odds is the additive
coordinate of the filter, so a linear blend of log odds is the geometric mean of the
odds, which is the combination rule that leaves the prior fixed. Blending
probabilities would drag the field towards one half at a different rate on each side
of the prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

from freespace_grid.algorithm.inverse_sensor import ScanUpdate, apply_scan
from freespace_grid.model.grid import GridSpec
from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.model.occupancy import OccupancyGrid
from freespace_grid.model.typing import BoolArray, FloatArray, IntArray

__all__ = ["Accumulator", "ShiftPolicy", "is_whole_cell_shift", "resample_grid"]

ShiftPolicy = Literal["snap", "bilinear", "nearest"]

_WHOLE_CELL_TOLERANCE = 1e-9


def is_whole_cell_shift(source: GridSpec, target: GridSpec) -> bool:
    """True when ``target`` is ``source`` translated by a whole number of cells."""
    if abs(source.resolution - target.resolution) > _WHOLE_CELL_TOLERANCE:
        return False
    d_col = (target.origin_x - source.origin_x) / source.resolution
    d_row = (target.origin_y - source.origin_y) / source.resolution
    return bool(
        abs(d_col - round(d_col)) <= _WHOLE_CELL_TOLERANCE
        and abs(d_row - round(d_row)) <= _WHOLE_CELL_TOLERANCE
    )


def resample_grid(
    grid: OccupancyGrid,
    target: GridSpec,
    model: LogOddsModel,
    *,
    interpolation: ShiftPolicy = "bilinear",
) -> OccupancyGrid:
    """Move the contents of ``grid`` onto the geometry of ``target``.

    Cells of ``target`` that no source cell covers are filled with the prior, which is
    the correct answer: the vehicle has moved into territory it has not observed.

    When the offset is a whole number of cells the copy is exact regardless of the
    requested interpolation, because the interpolation weights degenerate to one and
    zero. That path is taken explicitly so the result is bit exact rather than exact
    to within the interpolator's arithmetic.

    Args:
        grid: Source map.
        target: Destination geometry. Must share the resolution of the source.
        model: Supplies the prior used as fill value.
        interpolation: ``snap`` and ``nearest`` both round to the nearest source cell;
            ``bilinear`` blends the four surrounding cells.

    Returns:
        A new :class:`OccupancyGrid` on ``target``.
    """
    if abs(grid.spec.resolution - target.resolution) > _WHOLE_CELL_TOLERANCE:
        raise ValueError(
            "resample requires equal resolution, got "
            f"{grid.spec.resolution} and {target.resolution}"
        )

    d_col = (target.origin_x - grid.spec.origin_x) / grid.spec.resolution
    d_row = (target.origin_y - grid.spec.origin_y) / grid.spec.resolution

    if is_whole_cell_shift(grid.spec, target):
        return OccupancyGrid(
            spec=target,
            log_odds=_translate(grid.log_odds, round(d_row), round(d_col), target, model),
        )

    order = 1 if interpolation == "bilinear" else 0
    rows = np.arange(target.rows, dtype=np.float64) + d_row
    cols = np.arange(target.cols, dtype=np.float64) + d_col
    row_coords, col_coords = np.meshgrid(rows, cols, indexing="ij")
    resampled = ndimage.map_coordinates(
        grid.log_odds,
        np.stack((row_coords, col_coords), axis=0),
        order=order,
        mode="constant",
        cval=model.l_prior,
        prefilter=False,
    )
    values = np.ascontiguousarray(resampled, dtype=np.float64)
    return OccupancyGrid(spec=target, log_odds=model.clip(values))


def _translate(
    source: FloatArray,
    d_row: int,
    d_col: int,
    target: GridSpec,
    model: LogOddsModel,
) -> FloatArray:
    """Copy ``source`` into a target-shaped array offset by whole cells, filling the prior."""
    out = np.full(target.shape, model.l_prior, dtype=np.float64)
    src_rows, dst_rows = _overlap(source.shape[0], target.rows, d_row)
    src_cols, dst_cols = _overlap(source.shape[1], target.cols, d_col)
    if src_rows.stop > src_rows.start and src_cols.stop > src_cols.start:
        out[dst_rows, dst_cols] = source[src_rows, src_cols]
    return out


def _overlap(source_len: int, target_len: int, offset: int) -> tuple[slice, slice]:
    """Return the source and destination slices of ``target[i] = source[i + offset]``."""
    start = max(0, offset)
    stop = min(source_len, target_len + offset)
    stop = max(stop, start)
    return slice(start, stop), slice(start - offset, stop - offset)


@dataclass(slots=True)
class Accumulator:
    """A map that integrates scans over time and optionally follows the vehicle.

    Attributes:
        grid: The current map. Replaced, not mutated, when the window is re-anchored.
        model: Log odds parameters.
        observed: Counter of how many scans touched each cell of the current window.
            Re-anchoring carries it with nearest neighbour resampling, since a count
            is not a quantity to interpolate.
        occupied_observations: Counter of how many scans placed a range return in each
            cell. This is the diagnostic that explains a missed obstacle: a cell needs
            ``model.observations_to_occupied()`` of these in a row to escape the free
            clamp, and a moving obstacle rarely provides them.
        resamples: Number of re-anchor operations performed.
        lossless_resamples: How many of those were whole-cell translations.
    """

    grid: OccupancyGrid
    model: LogOddsModel
    observed: IntArray
    occupied_observations: IntArray
    resamples: int = 0
    lossless_resamples: int = 0

    @classmethod
    def from_prior(cls, spec: GridSpec, model: LogOddsModel) -> Accumulator:
        """Return an accumulator whose every cell holds exactly the prior."""
        return cls(
            grid=OccupancyGrid.from_prior(spec, model),
            model=model,
            observed=np.zeros(spec.shape, dtype=np.int64),
            occupied_observations=np.zeros(spec.shape, dtype=np.int64),
        )

    def integrate(
        self, origin: FloatArray, endpoints: FloatArray, is_hit: BoolArray
    ) -> ScanUpdate:
        """Apply one scan to the current map."""
        update = apply_scan(
            self.grid,
            self.model,
            origin,
            endpoints,
            is_hit,
            observed=self.observed,
        )
        if update.occupied_cells.size:
            rows, cols = update.occupied_cells[:, 0], update.occupied_cells[:, 1]
            self.occupied_observations[rows, cols] += 1
        return update

    def reanchor(
        self, center_x: float, center_y: float, *, policy: ShiftPolicy = "bilinear"
    ) -> None:
        """Move the window so that it is centred on the given world point.

        Under ``snap`` the requested centre is rounded to the nearest whole-cell
        offset from the current window, so no interpolation is ever performed.
        """
        target = self.grid.spec.recentered(center_x, center_y)
        if policy == "snap":
            res = self.grid.spec.resolution
            d_col = round((target.origin_x - self.grid.spec.origin_x) / res)
            d_row = round((target.origin_y - self.grid.spec.origin_y) / res)
            target = GridSpec(
                resolution=res,
                rows=target.rows,
                cols=target.cols,
                origin_x=self.grid.spec.origin_x + d_col * res,
                origin_y=self.grid.spec.origin_y + d_row * res,
            )
        if target == self.grid.spec:
            return
        lossless = is_whole_cell_shift(self.grid.spec, target)
        counts = _resample_counts(self.observed, self.grid.spec, target)
        returns = _resample_counts(self.occupied_observations, self.grid.spec, target)
        self.grid = resample_grid(self.grid, target, self.model, interpolation=policy)
        self.observed = counts
        self.occupied_observations = returns
        self.resamples += 1
        self.lossless_resamples += int(lossless)

    def observed_mask(self) -> BoolArray:
        """Mask of cells that at least one scan touched."""
        return np.asarray(self.observed > 0, dtype=np.bool_)


def _resample_counts(counts: IntArray, source: GridSpec, target: GridSpec) -> IntArray:
    """Carry the observation counter onto ``target`` by nearest neighbour."""
    d_col = (target.origin_x - source.origin_x) / source.resolution
    d_row = (target.origin_y - source.origin_y) / source.resolution
    rows = np.arange(target.rows, dtype=np.float64) + d_row
    cols = np.arange(target.cols, dtype=np.float64) + d_col
    row_coords, col_coords = np.meshgrid(rows, cols, indexing="ij")
    moved = ndimage.map_coordinates(
        counts.astype(np.float64),
        np.stack((row_coords, col_coords), axis=0),
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return np.ascontiguousarray(np.rint(moved), dtype=np.int64)
