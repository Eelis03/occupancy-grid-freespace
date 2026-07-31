"""The occupancy grid container and its three way decision rule."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from freespace_grid.model.grid import GridSpec
from freespace_grid.model.logodds import LogOddsModel, log_odds_to_prob
from freespace_grid.model.typing import FloatArray, IntArray

__all__ = ["CellState", "OccupancyGrid", "classify"]


class CellState(IntEnum):
    """The three outcomes a cell can be assigned.

    ``UNKNOWN`` is a first class answer, not a failure. A cell is unknown when the
    accumulated evidence has not moved its posterior far enough from the prior, which
    happens both where no beam has reached and where beams disagree.
    """

    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2


def classify(log_odds: FloatArray, model: LogOddsModel) -> IntArray:
    """Apply the three way decision rule to an array of log odds.

    A cell is ``FREE`` when its log odds fall at least ``model.l_decision`` below the
    prior, ``OCCUPIED`` when they rise at least that far above it, and ``UNKNOWN``
    otherwise. The band is symmetric in log odds, so it is symmetric in odds ratio but
    not in probability difference.
    """
    values = np.asarray(log_odds, dtype=np.float64)
    states = np.full(values.shape, int(CellState.UNKNOWN), dtype=np.int64)
    states[values <= model.l_prior - model.l_decision] = int(CellState.FREE)
    states[values >= model.l_prior + model.l_decision] = int(CellState.OCCUPIED)
    return states


@dataclass(slots=True)
class OccupancyGrid:
    """Accumulated occupancy evidence over a :class:`GridSpec`, stored as log odds.

    The array is mutated in place by the accumulation layer. ``spec`` may be replaced
    when the map is re-anchored to a moving vehicle, which is why this container is not
    frozen.
    """

    spec: GridSpec
    log_odds: FloatArray

    def __post_init__(self) -> None:
        if self.log_odds.shape != self.spec.shape:
            raise ValueError(
                f"log_odds shape {self.log_odds.shape} does not match grid {self.spec.shape}"
            )
        if self.log_odds.dtype != np.float64:
            raise ValueError(f"log_odds must be float64, got {self.log_odds.dtype}")

    @classmethod
    def from_prior(cls, spec: GridSpec, model: LogOddsModel) -> OccupancyGrid:
        """Return a grid in which every cell holds exactly the prior."""
        return cls(spec=spec, log_odds=np.full(spec.shape, model.l_prior, dtype=np.float64))

    def copy(self) -> OccupancyGrid:
        """Return a deep copy sharing no array storage."""
        return OccupancyGrid(spec=self.spec, log_odds=self.log_odds.copy())

    def probability(self) -> FloatArray:
        """Return the posterior occupancy probability of every cell."""
        return log_odds_to_prob(self.log_odds)

    def classify(self, model: LogOddsModel) -> IntArray:
        """Return the :class:`CellState` code of every cell."""
        return classify(self.log_odds, model)
