"""Tier one: the inverse sensor model, including the maximum range case.

The maximum range case has its own tests because getting it wrong is the classic
occupancy grid bug and because both wrong answers are plausible. Marking the range
limit as occupied paints a phantom arc at the sensor horizon; discarding the beam
throws away the free space evidence it does carry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from freespace_grid.algorithm.inverse_sensor import apply_scan, scan_update
from freespace_grid.algorithm.raycast import traverse_rays
from freespace_grid.model.grid import GridSpec, world_to_cell
from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.model.occupancy import CellState, OccupancyGrid

SPEC = GridSpec(resolution=1.0, rows=21, cols=21, origin_x=-10.5, origin_y=-10.5)
MODEL = LogOddsModel()
ORIGIN = np.array([0.0, 0.0])


def as_set(cells: np.ndarray) -> set[tuple[int, int]]:
    return {(int(row), int(col)) for row, col in cells}


def test_a_range_return_marks_free_along_the_beam_and_occupied_at_the_end() -> None:
    endpoints = np.array([[6.2, 0.0]])
    update = scan_update(SPEC, ORIGIN, endpoints, np.array([True]))
    terminal = tuple(world_to_cell(SPEC, endpoints)[0])
    assert as_set(update.occupied_cells) == {(int(terminal[0]), int(terminal[1]))}
    assert update.free_cells.shape[0] == 6
    assert as_set(update.free_cells).isdisjoint(as_set(update.occupied_cells))
    assert update.placed_returns == 1


def test_a_maximum_range_beam_marks_free_space_and_nothing_occupied() -> None:
    """The classic bug. A beam at the range limit carries free space evidence only.

    The same beam is run twice, once flagged as a range return and once as a beam that
    reached the range limit. The traversal is identical; only the treatment of the last
    cell differs, and in the maximum range case that cell must be marked free like all
    the others.
    """
    endpoints = np.array([[6.2, 0.0]])
    with_return = scan_update(SPEC, ORIGIN, endpoints, np.array([True]))
    at_limit = scan_update(SPEC, ORIGIN, endpoints, np.array([False]))

    assert at_limit.occupied_cells.shape[0] == 0
    assert at_limit.placed_returns == 0
    assert at_limit.max_range_beams == 1
    assert as_set(at_limit.free_cells) == as_set(with_return.free_cells) | as_set(
        with_return.occupied_cells
    )


def test_a_full_sweep_of_maximum_range_beams_never_raises_a_cell_above_the_prior() -> None:
    """A sweep that returns nothing anywhere must leave a disc of free space, no more."""
    angles = np.linspace(0.0, 2.0 * math.pi, 721)[:-1]
    endpoints = np.stack((8.0 * np.cos(angles), 8.0 * np.sin(angles)), axis=1)
    grid = OccupancyGrid.from_prior(SPEC, MODEL)
    for _ in range(30):
        apply_scan(grid, MODEL, ORIGIN, endpoints, np.zeros(angles.size, dtype=np.bool_))
    assert np.all(grid.log_odds <= MODEL.l_prior)
    assert not np.any(grid.classify(MODEL) == int(CellState.OCCUPIED))
    assert int(np.count_nonzero(grid.classify(MODEL) == int(CellState.FREE))) > 150


def test_treating_the_range_limit_as_a_return_would_paint_a_phantom_arc() -> None:
    """The behaviour the correct handling avoids, demonstrated by asking for it.

    Passing ``is_hit=True`` for beams that actually reached the range limit is exactly
    the mistake. The result is a ring of occupied cells at the sensor horizon where the
    world is empty, and this test pins the difference between the two answers.
    """
    angles = np.linspace(0.0, 2.0 * math.pi, 721)[:-1]
    endpoints = np.stack((8.0 * np.cos(angles), 8.0 * np.sin(angles)), axis=1)
    correct = OccupancyGrid.from_prior(SPEC, MODEL)
    wrong = OccupancyGrid.from_prior(SPEC, MODEL)
    for _ in range(30):
        apply_scan(correct, MODEL, ORIGIN, endpoints, np.zeros(angles.size, dtype=np.bool_))
        apply_scan(wrong, MODEL, ORIGIN, endpoints, np.ones(angles.size, dtype=np.bool_))

    phantom = int(np.count_nonzero(wrong.classify(MODEL) == int(CellState.OCCUPIED)))
    assert phantom > 30
    assert int(np.count_nonzero(correct.classify(MODEL) == int(CellState.OCCUPIED))) == 0


def test_a_cell_receives_at_most_one_increment_per_scan() -> None:
    """Two beams crossing the same cell must not count as two observations of it."""
    endpoints = np.array([[6.0, 0.05], [6.0, -0.05], [6.0, 0.0]])
    update = scan_update(SPEC, ORIGIN, endpoints, np.array([True, True, True]))
    all_cells = list(as_set(update.free_cells)) + list(as_set(update.occupied_cells))
    assert len(all_cells) == len(set(all_cells))

    grid = OccupancyGrid.from_prior(SPEC, MODEL)
    apply_scan(grid, MODEL, ORIGIN, endpoints, np.array([True, True, True]))
    touched = grid.log_odds != MODEL.l_prior
    increments = grid.log_odds[touched] - MODEL.l_prior
    assert np.all(
        np.isclose(increments, MODEL.l_free) | np.isclose(increments, MODEL.l_occupied)
    )


def test_an_occupied_claim_beats_a_free_claim_within_one_scan() -> None:
    """When one beam crosses a cell that another beam terminates in, occupied wins."""
    endpoints = np.array([[3.4, 0.0], [8.0, 0.0]])
    update = scan_update(SPEC, ORIGIN, endpoints, np.array([True, False]))
    terminal = world_to_cell(SPEC, endpoints[:1])[0]
    key = (int(terminal[0]), int(terminal[1]))
    assert key in as_set(update.occupied_cells)
    assert key not in as_set(update.free_cells)


def test_an_unobserved_cell_keeps_exactly_the_prior() -> None:
    """Cells no beam reaches are untouched, bit for bit."""
    grid = OccupancyGrid.from_prior(SPEC, MODEL)
    endpoints = np.array([[3.2, 0.0]])
    update = apply_scan(grid, MODEL, ORIGIN, endpoints, np.array([True]))
    touched = as_set(update.free_cells) | as_set(update.occupied_cells)
    for row in range(SPEC.rows):
        for col in range(SPEC.cols):
            if (row, col) not in touched:
                assert grid.log_odds[row, col] == MODEL.l_prior


def test_repeated_scans_saturate_at_the_clamp_and_stop() -> None:
    """The same scan applied many times converges to the clamp and never passes it."""
    grid = OccupancyGrid.from_prior(SPEC, MODEL)
    endpoints = np.array([[6.2, 0.0], [0.0, 6.2]])
    is_hit = np.array([True, True])
    for _ in range(200):
        apply_scan(grid, MODEL, ORIGIN, endpoints, is_hit)
        assert np.all(grid.log_odds >= MODEL.l_min)
        assert np.all(grid.log_odds <= MODEL.l_max)
    update = scan_update(SPEC, ORIGIN, endpoints, is_hit)
    free = update.free_cells
    occupied = update.occupied_cells
    assert np.all(grid.log_odds[free[:, 0], free[:, 1]] == MODEL.l_min)
    assert np.all(grid.log_odds[occupied[:, 0], occupied[:, 1]] == MODEL.l_max)


def test_a_return_outside_the_grid_places_no_occupied_cell() -> None:
    """A beam that leaves the grid before its endpoint still marks the free part."""
    endpoints = np.array([[40.0, 0.0]])
    update = scan_update(SPEC, ORIGIN, endpoints, np.array([True]))
    assert update.occupied_cells.shape[0] == 0
    assert update.placed_returns == 0
    assert update.free_cells.shape[0] == 11


def test_scan_update_counts_match_the_traversal() -> None:
    rng = np.random.default_rng(41)
    angles = rng.uniform(0.0, 2.0 * math.pi, size=300)
    ranges = rng.uniform(1.0, 9.0, size=300)
    endpoints = np.stack((ranges * np.cos(angles), ranges * np.sin(angles)), axis=1)
    is_hit = rng.random(300) > 0.3
    update = scan_update(SPEC, ORIGIN, endpoints, is_hit)
    traversal = traverse_rays(SPEC, ORIGIN, endpoints)
    assert update.cell_visits == traversal.total_cells
    assert update.beams == 300
    assert update.hit_beams == int(np.count_nonzero(is_hit))
    assert update.max_range_beams == 300 - update.hit_beams
    assert update.touched_cells <= update.cell_visits


def test_shape_validation() -> None:
    with pytest.raises(ValueError, match="endpoints must have shape"):
        scan_update(SPEC, ORIGIN, np.zeros(3), np.array([True]))
    with pytest.raises(ValueError, match="is_hit must have shape"):
        scan_update(SPEC, ORIGIN, np.zeros((3, 2)), np.array([True]))
