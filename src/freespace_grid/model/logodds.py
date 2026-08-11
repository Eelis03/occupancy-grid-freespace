"""Log odds representation of cell occupancy.

The occupancy of one cell is a Bernoulli variable. Storing its probability directly
forces a multiplication for every update and needs a renormalisation; storing the log
odds ``l = log(p / (1 - p))`` turns the Bayes update into an addition, which is the
form used by Elfes and by Thrun, Burgard and Fox. This module holds the conversions
and the clamped parameter set, and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from freespace_grid.model.typing import FloatArray

__all__ = ["LogOddsModel", "log_odds_to_prob", "prob_to_log_odds"]


def prob_to_log_odds(probability: FloatArray | float) -> FloatArray:
    """Convert an occupancy probability to log odds.

    Args:
        probability: Values in the open interval ``(0, 1)``.

    Returns:
        ``log(p / (1 - p))`` with the same shape as the input.
    """
    p = np.asarray(probability, dtype=np.float64)
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("probability must lie strictly inside (0, 1)")
    return np.asarray(np.log(p / (1.0 - p)), dtype=np.float64)


def log_odds_to_prob(log_odds: FloatArray | float) -> FloatArray:
    """Convert log odds to an occupancy probability.

    Uses the numerically stable branch of the logistic function so that large
    magnitudes do not overflow ``exp``.
    """
    values = np.asarray(log_odds, dtype=np.float64)
    positive = values >= 0.0
    result = np.empty(values.shape, dtype=np.float64)
    exp_neg = np.exp(-np.abs(values))
    result[positive] = 1.0 / (1.0 + exp_neg[positive])
    result[~positive] = exp_neg[~positive] / (1.0 + exp_neg[~positive])
    return result


@dataclass(frozen=True, slots=True)
class LogOddsModel:
    """Parameters of the clamped log odds occupancy filter.

    The clamp is asymmetric, following the defaults published with octomap. Without a
    clamp a cell observed free a thousand times needs a thousand contrary observations
    before it can be believed occupied, and the map stops responding to the world. The
    two bounds are not equal because the two errors are not equally expensive: a cell
    wrongly held free is a cell a planner will drive into, so the free bound is set
    closer to the prior than the occupied bound, and free evidence is therefore cheaper
    to overturn than occupied evidence.

    The ratio of the two increments sets how fast the map forgets. One occupied
    observation is undone by ``l_occupied / -l_free`` free observations, which for the
    defaults here is 2.09. That number reappears in the dynamic obstacle study as the
    rate at which the trail of a moving object is erased.

    Args:
        prior: Prior occupancy probability of an unobserved cell.
        p_free: Inverse sensor model probability assigned to a cell the beam passed
            through. Must be below ``prior``.
        p_occupied: Inverse sensor model probability assigned to the cell holding a
            range return. Must be above ``prior``.
        clamp_free_prob: Lower bound on the stored occupancy probability.
        clamp_occupied_prob: Upper bound on the stored occupancy probability.
        decision_prob: Probability distance from the prior at which a cell is called
            free or occupied. A cell whose posterior stays inside the band
            ``[1 - decision_prob, decision_prob]`` is reported unknown.
    """

    prior: float = 0.5
    p_free: float = 0.4
    p_occupied: float = 0.7
    clamp_free_prob: float = 0.12
    clamp_occupied_prob: float = 0.97
    decision_prob: float = 0.65

    def __post_init__(self) -> None:
        for name, value in (
            ("prior", self.prior),
            ("p_free", self.p_free),
            ("p_occupied", self.p_occupied),
            ("clamp_free_prob", self.clamp_free_prob),
            ("clamp_occupied_prob", self.clamp_occupied_prob),
            ("decision_prob", self.decision_prob),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly inside (0, 1), got {value}")
        if not self.p_free < self.prior < self.p_occupied:
            raise ValueError(
                f"require p_free < prior < p_occupied, got "
                f"{self.p_free}, {self.prior}, {self.p_occupied}"
            )
        if not self.clamp_free_prob <= self.p_free:
            raise ValueError(
                f"clamp_free_prob must not exceed p_free, got "
                f"{self.clamp_free_prob} and {self.p_free}"
            )
        if not self.clamp_occupied_prob >= self.p_occupied:
            raise ValueError(
                f"clamp_occupied_prob must not be below p_occupied, got "
                f"{self.clamp_occupied_prob} and {self.p_occupied}"
            )
        if not self.prior < self.decision_prob < self.clamp_occupied_prob:
            raise ValueError(
                f"require prior < decision_prob < clamp_occupied_prob, got "
                f"{self.prior}, {self.decision_prob}, {self.clamp_occupied_prob}"
            )
        # The decision band has to sit strictly inside the clamp band on both sides. A
        # threshold outside it can never be met, so the map would report every cell
        # unknown on that side no matter how much evidence it collected.
        if not self.clamp_free_prob < 1.0 - self.decision_prob:
            raise ValueError(
                f"require clamp_free_prob < 1 - decision_prob, got "
                f"{self.clamp_free_prob} and {1.0 - self.decision_prob}"
            )

    @property
    def l_prior(self) -> float:
        """Log odds of the prior."""
        return float(prob_to_log_odds(self.prior))

    @property
    def l_free(self) -> float:
        """Log odds increment applied to a cell the beam passed through. Negative."""
        return float(prob_to_log_odds(self.p_free)) - self.l_prior

    @property
    def l_occupied(self) -> float:
        """Log odds increment applied to the cell holding a range return. Positive."""
        return float(prob_to_log_odds(self.p_occupied)) - self.l_prior

    @property
    def l_min(self) -> float:
        """Lower clamp on the stored log odds."""
        return float(prob_to_log_odds(self.clamp_free_prob))

    @property
    def l_max(self) -> float:
        """Upper clamp on the stored log odds."""
        return float(prob_to_log_odds(self.clamp_occupied_prob))

    @property
    def l_decision(self) -> float:
        """Log odds half-width of the undecided band, measured from the prior."""
        return float(prob_to_log_odds(self.decision_prob)) - self.l_prior

    @property
    def forget_ratio(self) -> float:
        """Free observations needed to undo one occupied observation."""
        return self.l_occupied / -self.l_free

    def observations_to_free(self) -> float:
        """Free observations that take a cell from the prior to the decision threshold."""
        return self.l_decision / -self.l_free

    def observations_to_occupied(self) -> float:
        """Occupied observations that take a fully clamped free cell to the threshold."""
        return (self.l_prior + self.l_decision - self.l_min) / self.l_occupied

    def clip(self, log_odds: FloatArray) -> FloatArray:
        """Clamp ``log_odds`` into ``[l_min, l_max]``."""
        return np.clip(log_odds, self.l_min, self.l_max)

    def at_clamp(self, log_odds: FloatArray, tolerance: float = 1e-9) -> FloatArray:
        """Mask of the entries that have reached either clamp."""
        values = np.asarray(log_odds, dtype=np.float64)
        return np.asarray((values <= self.l_min + tolerance) | (values >= self.l_max - tolerance))
