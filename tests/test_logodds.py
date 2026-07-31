"""Tier one: the log odds representation and the three way decision rule."""

from __future__ import annotations

import numpy as np
import pytest

from freespace_grid.model.grid import GridSpec
from freespace_grid.model.logodds import LogOddsModel, log_odds_to_prob, prob_to_log_odds
from freespace_grid.model.occupancy import CellState, OccupancyGrid, classify


def test_probability_and_log_odds_round_trip() -> None:
    """The two conversions invert each other to within double precision."""
    probabilities = np.linspace(1e-9, 1.0 - 1e-9, 20001)
    recovered = log_odds_to_prob(prob_to_log_odds(probabilities))
    assert np.allclose(recovered, probabilities, rtol=0.0, atol=1e-12)


def test_log_odds_round_trip_over_the_useful_range() -> None:
    """The inverse direction round trips wherever the probability is representable.

    Beyond about fifteen in log odds the probability is within a few times ten to the
    minus seven of one, and the subtraction ``1 - p`` loses the information the reverse
    conversion would need. The filter clamps far inside that range, so the limit is
    stated rather than worked around.
    """
    values = np.linspace(-15.0, 15.0, 4001)
    recovered = prob_to_log_odds(log_odds_to_prob(values))
    assert np.allclose(recovered, values, rtol=0.0, atol=1e-8)


def test_log_odds_to_prob_does_not_overflow() -> None:
    """The stable branch keeps very large magnitudes finite and inside the unit interval."""
    extreme = np.array([-1e4, -800.0, 0.0, 800.0, 1e4])
    values = log_odds_to_prob(extreme)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)
    assert values[2] == pytest.approx(0.5)


def test_prob_to_log_odds_rejects_the_closed_endpoints() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        prob_to_log_odds(np.array([0.0]))
    with pytest.raises(ValueError, match="strictly inside"):
        prob_to_log_odds(np.array([1.0]))


def test_model_increments_have_the_expected_signs_and_sizes() -> None:
    model = LogOddsModel()
    assert model.l_free < 0.0 < model.l_occupied
    assert model.l_min < model.l_prior < model.l_max
    assert model.forget_ratio == pytest.approx(model.l_occupied / -model.l_free)
    assert model.l_free == pytest.approx(np.log(0.4 / 0.6))
    assert model.l_occupied == pytest.approx(np.log(0.7 / 0.3))


def test_model_rejects_inconsistent_parameters() -> None:
    with pytest.raises(ValueError, match="p_free < prior < p_occupied"):
        LogOddsModel(p_free=0.6, p_occupied=0.7)
    with pytest.raises(ValueError, match="clamp_free_prob must not exceed"):
        LogOddsModel(clamp_free_prob=0.45)
    with pytest.raises(ValueError, match="clamp_free_prob < 1 - decision_prob"):
        LogOddsModel(clamp_free_prob=0.34, decision_prob=0.70)


def test_repeated_identical_observations_saturate_at_the_clamp() -> None:
    """Evidence stops accumulating at the clamp and never passes it.

    Both directions are checked. Without the clamp the free branch would run away to
    minus infinity and the map would stop being able to change its mind.
    """
    model = LogOddsModel()
    free = np.array([model.l_prior])
    occupied = np.array([model.l_prior])
    for _ in range(400):
        free = model.clip(free + model.l_free)
        occupied = model.clip(occupied + model.l_occupied)
        assert free[0] >= model.l_min
        assert occupied[0] <= model.l_max
    assert free[0] == model.l_min
    assert occupied[0] == model.l_max


def test_classification_band_is_symmetric_in_log_odds() -> None:
    model = LogOddsModel()
    values = np.array(
        [
            model.l_prior,
            model.l_prior - model.l_decision + 1e-12,
            model.l_prior - model.l_decision,
            model.l_prior + model.l_decision,
            model.l_prior + model.l_decision - 1e-12,
            model.l_min,
            model.l_max,
        ]
    )
    states = classify(values, model)
    expected = [
        CellState.UNKNOWN,
        CellState.UNKNOWN,
        CellState.FREE,
        CellState.OCCUPIED,
        CellState.UNKNOWN,
        CellState.FREE,
        CellState.OCCUPIED,
    ]
    assert list(states) == [int(state) for state in expected]


def test_a_grid_built_from_the_prior_is_entirely_unknown() -> None:
    model = LogOddsModel()
    spec = GridSpec(resolution=0.5, rows=7, cols=11)
    grid = OccupancyGrid.from_prior(spec, model)
    assert np.all(grid.log_odds == model.l_prior)
    assert np.all(grid.classify(model) == int(CellState.UNKNOWN))
    assert np.allclose(grid.probability(), model.prior)


def test_occupancy_grid_rejects_a_mismatched_array() -> None:
    spec = GridSpec(resolution=1.0, rows=3, cols=4)
    with pytest.raises(ValueError, match="does not match grid"):
        OccupancyGrid(spec=spec, log_odds=np.zeros((4, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="float64"):
        OccupancyGrid(spec=spec, log_odds=np.zeros((3, 4), dtype=np.float32))


def test_observation_counts_match_the_closed_form_predictions() -> None:
    """The helper counts agree with stepping the filter one observation at a time."""
    model = LogOddsModel()
    value = model.l_prior
    steps = 0
    while value > model.l_prior - model.l_decision:
        value = max(value + model.l_free, model.l_min)
        steps += 1
    assert steps == int(np.ceil(model.observations_to_free()))

    value = model.l_min
    steps = 0
    while value < model.l_prior + model.l_decision:
        value = min(value + model.l_occupied, model.l_max)
        steps += 1
    assert steps == int(np.ceil(model.observations_to_occupied()))
