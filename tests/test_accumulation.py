"""Tier one: re-anchoring a grid under vehicle motion."""

from __future__ import annotations

import numpy as np
import pytest

from freespace_grid.algorithm.accumulation import (
    Accumulator,
    is_whole_cell_shift,
    resample_grid,
)
from freespace_grid.model.grid import GridSpec
from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.model.occupancy import OccupancyGrid

MODEL = LogOddsModel()


def random_grid(spec: GridSpec, seed: int) -> OccupancyGrid:
    rng = np.random.default_rng(seed)
    values = rng.uniform(MODEL.l_min, MODEL.l_max, size=spec.shape)
    return OccupancyGrid(spec=spec, log_odds=values)


@pytest.mark.parametrize("shift", [(0, 0), (1, 0), (0, 1), (3, -7), (-11, 5), (40, 40)])
@pytest.mark.parametrize("interpolation", ["snap", "bilinear", "nearest"])
def test_whole_cell_translation_is_exact_and_lossless(
    shift: tuple[int, int], interpolation: str
) -> None:
    """A shift by a whole number of cells copies values, it does not combine them.

    Every cell of the target that a source cell covers must hold the source value bit
    for bit, whichever interpolation was requested, and every cell it does not cover
    must hold the prior.
    """
    spec = GridSpec(resolution=0.25, rows=31, cols=37, origin_x=-2.0, origin_y=5.0)
    grid = random_grid(spec, seed=7)
    d_row, d_col = shift
    target = GridSpec(
        resolution=spec.resolution,
        rows=spec.rows,
        cols=spec.cols,
        origin_x=spec.origin_x + d_col * spec.resolution,
        origin_y=spec.origin_y + d_row * spec.resolution,
    )
    assert is_whole_cell_shift(spec, target)
    moved = resample_grid(grid, target, MODEL, interpolation=interpolation)  # type: ignore[arg-type]

    for row in range(target.rows):
        for col in range(target.cols):
            source_row, source_col = row + d_row, col + d_col
            inside = 0 <= source_row < spec.rows and 0 <= source_col < spec.cols
            expected = grid.log_odds[source_row, source_col] if inside else MODEL.l_prior
            assert moved.log_odds[row, col] == expected


def test_translating_there_and_back_restores_the_overlap_exactly() -> None:
    """Two opposite whole cell shifts leave the surviving region unchanged."""
    spec = GridSpec(resolution=0.2, rows=24, cols=24)
    grid = random_grid(spec, seed=13)
    forward = GridSpec(
        resolution=0.2, rows=24, cols=24, origin_x=0.2 * 5, origin_y=0.2 * 3
    )
    moved = resample_grid(grid, forward, MODEL, interpolation="snap")
    back = resample_grid(moved, spec, MODEL, interpolation="snap")
    # The window moved up three rows and right five columns and then came back, so the
    # first three rows and first five columns fell off the edge and returned as prior.
    assert np.array_equal(back.log_odds[3:, 5:], grid.log_odds[3:, 5:])
    assert np.all(back.log_odds[:3, :] == MODEL.l_prior)
    assert np.all(back.log_odds[:, :5] == MODEL.l_prior)


def test_fractional_bilinear_shift_blurs_and_stays_inside_the_clamp() -> None:
    """Bilinear resampling is a low pass filter: the field flattens, it never overshoots."""
    spec = GridSpec(resolution=0.5, rows=41, cols=41)
    sharp = np.full(spec.shape, MODEL.l_min, dtype=np.float64)
    sharp[20, 20] = MODEL.l_max
    grid = OccupancyGrid(spec=spec, log_odds=sharp)
    target = GridSpec(resolution=0.5, rows=41, cols=41, origin_x=0.23, origin_y=0.17)
    assert not is_whole_cell_shift(spec, target)

    moved = resample_grid(grid, target, MODEL, interpolation="bilinear")
    assert moved.log_odds.max() < grid.log_odds.max()
    assert moved.log_odds.min() >= MODEL.l_min
    assert moved.log_odds.max() <= MODEL.l_max
    # One clamped cell spreads over the four cells that surround its new position. The
    # last row and column are excluded because they blend with the prior fill outside
    # the source window, which is intended and is a separate effect.
    interior = moved.log_odds[:-1, :-1]
    assert int(np.count_nonzero(interior > MODEL.l_min + 1e-9)) == 4
    assert np.all(moved.log_odds[-1, :] == MODEL.l_prior)
    assert np.all(moved.log_odds[:, -1] == MODEL.l_prior)


def test_fractional_nearest_shift_preserves_the_value_set() -> None:
    """Nearest neighbour moves values without averaging, so no new value appears."""
    spec = GridSpec(resolution=0.5, rows=21, cols=21)
    sharp = np.full(spec.shape, MODEL.l_min, dtype=np.float64)
    sharp[10, 10] = MODEL.l_max
    grid = OccupancyGrid(spec=spec, log_odds=sharp)
    target = GridSpec(resolution=0.5, rows=21, cols=21, origin_x=0.23, origin_y=0.17)
    moved = resample_grid(grid, target, MODEL, interpolation="nearest")
    assert set(np.unique(moved.log_odds)) <= {MODEL.l_min, MODEL.l_max, MODEL.l_prior}
    assert int(np.count_nonzero(moved.log_odds == MODEL.l_max)) == 1


def test_resample_requires_matching_resolution() -> None:
    spec = GridSpec(resolution=0.5, rows=4, cols=4)
    grid = OccupancyGrid.from_prior(spec, MODEL)
    with pytest.raises(ValueError, match="equal resolution"):
        resample_grid(grid, GridSpec(resolution=0.25, rows=4, cols=4), MODEL)


def test_snap_policy_never_interpolates() -> None:
    """Under snap, every re-anchor is reported lossless however the vehicle moves."""
    spec = GridSpec(resolution=0.2, rows=40, cols=40)
    accumulator = Accumulator.from_prior(spec, MODEL)
    accumulator.grid.log_odds[:] = np.linspace(
        MODEL.l_min, MODEL.l_max, spec.size
    ).reshape(spec.shape)
    rng = np.random.default_rng(2)
    for _ in range(12):
        target = rng.uniform(2.0, 6.0, size=2)
        accumulator.reanchor(float(target[0]), float(target[1]), policy="snap")
    assert accumulator.resamples == accumulator.lossless_resamples
    assert accumulator.resamples > 0
    assert np.all(accumulator.grid.log_odds >= MODEL.l_min)
    assert np.all(accumulator.grid.log_odds <= MODEL.l_max)


def test_reanchor_carries_the_observation_counter() -> None:
    spec = GridSpec(resolution=1.0, rows=10, cols=10)
    accumulator = Accumulator.from_prior(spec, MODEL)
    accumulator.observed[4, 6] = 3
    accumulator.reanchor(7.0, 5.0, policy="snap")
    assert int(accumulator.observed.sum()) == 3
    assert accumulator.observed.dtype == np.int64


def test_reanchor_to_the_same_centre_is_a_no_operation() -> None:
    spec = GridSpec(resolution=0.5, rows=8, cols=8)
    accumulator = Accumulator.from_prior(spec, MODEL)
    centre_x = spec.origin_x + 0.5 * spec.cols * spec.resolution
    centre_y = spec.origin_y + 0.5 * spec.rows * spec.resolution
    accumulator.reanchor(centre_x, centre_y, policy="bilinear")
    assert accumulator.resamples == 0
