"""Scoring a map against ground truth.

Three numbers describe an occupancy map and no one of them is sufficient alone.

Decided fraction
    What proportion of the scored cells the map is willing to call. A mapper can push
    its agreement to one by refusing to decide anything but a handful of cells, so the
    agreement figures are meaningless without this alongside them.

Free agreement
    Of the truly free cells the map decided, the fraction it called free. This is the
    number a free space consumer cares about: a planner that trusts free space will
    drive into whatever is wrong here.

Occupied agreement
    Of the truly occupied cells the map decided, the fraction it called occupied. It
    is systematically lower than free agreement in any lidar built map, because a beam
    terminates on the near surface of an obstacle and cannot see through it, so the
    interior of a solid object is never observed.

The scoring region is an argument rather than a default. Scoring over the whole grid
mixes the map's errors with the sensor's coverage; scoring over the cells the sensor
actually reached separates them. Both are reported by the example scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.model.occupancy import CellState, OccupancyGrid, classify
from freespace_grid.model.typing import BoolArray, IntArray

__all__ = ["Agreement", "score_grid", "threshold_sweep", "tolerance_sweep"]


@dataclass(frozen=True, slots=True)
class Agreement:
    """Agreement between a classified map and ground truth over one scoring region."""

    decision_prob: float
    spatial_tolerance: int
    scored_cells: int
    decided_cells: int
    free_truth: int
    free_correct: int
    occupied_truth: int
    occupied_correct: int
    free_called_occupied: int
    occupied_called_free: int

    @property
    def decided_fraction(self) -> float:
        """Fraction of scored cells the map was willing to classify."""
        return _ratio(self.decided_cells, self.scored_cells)

    @property
    def free_agreement(self) -> float:
        """Fraction of decided, truly free cells that the map called free."""
        return _ratio(self.free_correct, self.free_truth)

    @property
    def occupied_agreement(self) -> float:
        """Fraction of decided, truly occupied cells that the map called occupied."""
        return _ratio(self.occupied_correct, self.occupied_truth)

    @property
    def balanced_agreement(self) -> float:
        """Unweighted mean of the two class agreements.

        Free cells outnumber occupied cells by more than an order of magnitude in an
        open scene, so an overall accuracy would be almost exactly the free agreement
        and would say nothing about obstacles.
        """
        return 0.5 * (self.free_agreement + self.occupied_agreement)

    def as_row(self) -> dict[str, float]:
        """Flat mapping suitable for a table or a JSON record."""
        return {
            "decision_prob": self.decision_prob,
            "spatial_tolerance": float(self.spatial_tolerance),
            "scored_cells": float(self.scored_cells),
            "decided_cells": float(self.decided_cells),
            "decided_fraction": self.decided_fraction,
            "free_agreement": self.free_agreement,
            "occupied_agreement": self.occupied_agreement,
            "balanced_agreement": self.balanced_agreement,
            "free_called_occupied": float(self.free_called_occupied),
            "occupied_called_free": float(self.occupied_called_free),
        }


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def score_grid(
    grid: OccupancyGrid,
    truth: BoolArray,
    model: LogOddsModel,
    *,
    region: BoolArray | None = None,
    decision_prob: float | None = None,
    spatial_tolerance: int = 0,
) -> Agreement:
    """Score ``grid`` against ``truth`` over ``region``.

    Args:
        grid: The map to score.
        truth: Boolean ground truth occupancy of the same shape, true where occupied.
        model: Supplies the log odds decision band.
        region: Cells to score. Defaults to every cell of the grid.
        decision_prob: Overrides ``model.decision_prob`` for this call, which is how
            the threshold sweep is produced without rebuilding the map.
        spatial_tolerance: Cells of slack allowed on occupied predictions. With a
            tolerance of ``k``, a truly occupied cell counts as agreeing when the map
            called some cell within ``k`` cells of it occupied. The tolerance applies
            only to the occupied class: it exists to separate a map that puts the
            surface one cell out from a map that misses the surface, and free space
            regions are thick enough that the same slack would make the free figure
            meaningless.

    Returns:
        The :class:`Agreement` for these settings.
    """
    truth_occupied = np.asarray(truth, dtype=np.bool_)
    if truth_occupied.shape != grid.spec.shape:
        raise ValueError(
            f"truth shape {truth_occupied.shape} does not match grid {grid.spec.shape}"
        )
    if spatial_tolerance < 0:
        raise ValueError(f"spatial_tolerance must not be negative, got {spatial_tolerance}")

    scoring = (
        np.ones(grid.spec.shape, dtype=np.bool_)
        if region is None
        else np.asarray(region, dtype=np.bool_)
    )
    if scoring.shape != grid.spec.shape:
        raise ValueError(f"region shape {scoring.shape} does not match grid {grid.spec.shape}")

    effective = model if decision_prob is None else _with_decision(model, decision_prob)
    states: IntArray = classify(grid.log_odds, effective)

    called_free = states == int(CellState.FREE)
    called_occupied = states == int(CellState.OCCUPIED)
    decided = called_free | called_occupied

    matched_occupied = called_occupied
    if spatial_tolerance > 0:
        matched_occupied = _dilate(called_occupied, spatial_tolerance)

    truth_free = ~truth_occupied
    free_pool = scoring & truth_free & decided
    occupied_pool = scoring & truth_occupied & decided

    return Agreement(
        decision_prob=effective.decision_prob,
        spatial_tolerance=spatial_tolerance,
        scored_cells=int(np.count_nonzero(scoring)),
        decided_cells=int(np.count_nonzero(scoring & decided)),
        free_truth=int(np.count_nonzero(free_pool)),
        free_correct=int(np.count_nonzero(free_pool & called_free)),
        occupied_truth=int(np.count_nonzero(occupied_pool)),
        occupied_correct=int(np.count_nonzero(occupied_pool & matched_occupied)),
        free_called_occupied=int(np.count_nonzero(free_pool & called_occupied)),
        occupied_called_free=int(np.count_nonzero(occupied_pool & called_free)),
    )


def threshold_sweep(
    grid: OccupancyGrid,
    truth: BoolArray,
    model: LogOddsModel,
    thresholds: tuple[float, ...],
    *,
    region: BoolArray | None = None,
    spatial_tolerance: int = 0,
) -> tuple[Agreement, ...]:
    """Score the same map at a series of decision thresholds.

    Raising the threshold narrows the set of cells the map is prepared to decide, so
    the decided fraction falls and the agreement on what remains rises. The pair of
    curves is the honest description of the map; either number alone can be moved at
    will by moving the threshold.
    """
    return tuple(
        score_grid(
            grid,
            truth,
            model,
            region=region,
            decision_prob=value,
            spatial_tolerance=spatial_tolerance,
        )
        for value in thresholds
    )


def tolerance_sweep(
    grid: OccupancyGrid,
    truth: BoolArray,
    model: LogOddsModel,
    tolerances: tuple[int, ...],
    *,
    region: BoolArray | None = None,
) -> tuple[Agreement, ...]:
    """Score the same map at a series of spatial tolerances on the occupied class."""
    return tuple(
        score_grid(grid, truth, model, region=region, spatial_tolerance=value)
        for value in tolerances
    )


def _with_decision(model: LogOddsModel, decision_prob: float) -> LogOddsModel:
    return LogOddsModel(
        prior=model.prior,
        p_free=model.p_free,
        p_occupied=model.p_occupied,
        clamp_free_prob=model.clamp_free_prob,
        clamp_occupied_prob=model.clamp_occupied_prob,
        decision_prob=decision_prob,
    )


def _dilate(mask: BoolArray, radius: int) -> BoolArray:
    """Grow ``mask`` by ``radius`` cells under the Chebyshev metric."""
    structure = np.ones((3, 3), dtype=bool)
    grown = ndimage.binary_dilation(mask, structure=structure, iterations=radius)
    return np.asarray(grown, dtype=np.bool_)
