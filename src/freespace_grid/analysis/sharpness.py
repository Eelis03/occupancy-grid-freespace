"""Measures of how much a map has been blurred by repeated resampling.

Agreement figures alone do not distinguish a map that puts a wall in the right place
from a map that spreads the same wall over four cells at half the confidence, because
both can classify most cells correctly. These two measures do distinguish them.

``clamped_fraction``
    The proportion of observed cells whose evidence has reached the clamp. Bilinear
    resampling averages neighbouring cells, and an average of a clamped value with
    anything below it is below the clamp, so repeated resampling drains this figure.

``edge_contrast``
    The mean magnitude of the log odds gradient in a narrow band around the true
    obstacle boundaries. A sharp map has a large step there; a blurred map spreads the
    same total change over more cells and the mean gradient falls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.model.occupancy import OccupancyGrid
from freespace_grid.model.typing import BoolArray

__all__ = ["Sharpness", "boundary_band", "measure_sharpness"]


def boundary_band(truth: BoolArray, width: int = 2) -> BoolArray:
    """Return the cells within ``width`` cells of a ground truth occupancy boundary."""
    if width < 1:
        raise ValueError(f"width must be at least one, got {width}")
    occupied = np.asarray(truth, dtype=np.bool_)
    structure = np.ones((3, 3), dtype=bool)
    grown = ndimage.binary_dilation(occupied, structure=structure, iterations=width)
    shrunk = ndimage.binary_erosion(occupied, structure=structure, iterations=width)
    return np.asarray(grown & ~shrunk, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class Sharpness:
    """How much evidence survived, and how sharply it is localised."""

    observed_cells: int
    clamped_cells: int
    band_cells: int
    edge_contrast: float
    mean_abs_evidence: float

    @property
    def clamped_fraction(self) -> float:
        """Fraction of observed cells whose evidence has reached the clamp."""
        if not self.observed_cells:
            return 0.0
        return float(self.clamped_cells) / float(self.observed_cells)


def measure_sharpness(
    grid: OccupancyGrid,
    truth: BoolArray,
    model: LogOddsModel,
    *,
    region: BoolArray | None = None,
    band_width: int = 2,
) -> Sharpness:
    """Measure clamp saturation and edge contrast of ``grid``.

    Args:
        grid: The map to measure.
        truth: Ground truth occupancy, used only to locate the boundary band.
        model: Supplies the clamp bounds.
        region: Cells to measure over. Defaults to every cell.
        band_width: Half width of the boundary band, in cells.

    Returns:
        The :class:`Sharpness` summary.
    """
    scoring = (
        np.ones(grid.spec.shape, dtype=np.bool_)
        if region is None
        else np.asarray(region, dtype=np.bool_)
    )
    evidence = grid.log_odds - model.l_prior
    at_clamp = np.asarray(model.at_clamp(grid.log_odds), dtype=np.bool_)

    band = boundary_band(truth, width=band_width) & scoring
    grad_row, grad_col = np.gradient(grid.log_odds)
    magnitude = np.hypot(grad_row, grad_col)
    contrast = float(magnitude[band].mean()) if bool(band.any()) else 0.0

    return Sharpness(
        observed_cells=int(np.count_nonzero(scoring)),
        clamped_cells=int(np.count_nonzero(scoring & at_clamp)),
        band_cells=int(np.count_nonzero(band)),
        edge_contrast=contrast,
        mean_abs_evidence=float(np.abs(evidence[scoring]).mean()) if bool(scoring.any()) else 0.0,
    )
