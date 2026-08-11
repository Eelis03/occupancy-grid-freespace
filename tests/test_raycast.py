"""Tier one: the ray traversal visits exactly the cells the analytic segment crosses."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from freespace_grid.algorithm.raycast import traverse_rays
from freespace_grid.model.grid import GridSpec

UNIT = GridSpec(resolution=1.0, rows=10, cols=10)


def cells_of(traversal, ray: int = 0) -> list[tuple[int, int]]:
    """Return the cells visited by one ray, in traversal order."""
    start = int(traversal.offsets[ray])
    stop = int(traversal.offsets[ray + 1])
    return [(int(traversal.rows[i]), int(traversal.cols[i])) for i in range(start, stop)]


def test_hand_computed_shallow_ray() -> None:
    """A ray from (0.5, 0.5) to (3.5, 2.2) on a unit grid crosses six named cells.

    Worked by hand from the crossing parameters. Vertical grid lines at x = 1, 2, 3 are
    met at t = 1/6, 1/2, 5/6, horizontal lines at y = 1, 2 at t = 5/17 and t = 15/17.
    Sorting those six events gives the step order x, y, x, x, y.
    """
    traversal = traverse_rays(UNIT, np.array([0.5, 0.5]), np.array([[3.5, 2.2]]))
    assert cells_of(traversal) == [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 3)]
    assert bool(traversal.reached[0])
    assert list(traversal.counts) == [6]


def test_exact_diagonal_never_cuts_a_corner() -> None:
    """Through a grid corner the traversal steps x and then y, visiting both neighbours.

    Cutting the corner would let a beam pass diagonally between two occupied cells that
    share only a corner, so free space would leak through a wall one cell thick.
    """
    traversal = traverse_rays(UNIT, np.array([0.5, 0.5]), np.array([[3.5, 3.5]]))
    assert cells_of(traversal) == [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 3),
        (3, 3),
    ]


def test_endpoint_inside_the_origin_cell_gives_one_cell() -> None:
    traversal = traverse_rays(UNIT, np.array([2.2, 3.7]), np.array([[2.8, 3.1]]))
    assert cells_of(traversal) == [(3, 2)]
    assert bool(traversal.reached[0])


def test_degenerate_ray_of_zero_length() -> None:
    traversal = traverse_rays(UNIT, np.array([4.5, 6.5]), np.array([[4.5, 6.5]]))
    assert cells_of(traversal) == [(6, 4)]
    assert bool(traversal.reached[0])


def test_axis_aligned_rays_are_straight_lines_of_cells() -> None:
    east = traverse_rays(UNIT, np.array([0.5, 5.5]), np.array([[6.5, 5.5]]))
    assert cells_of(east) == [(5, c) for c in range(7)]
    north = traverse_rays(UNIT, np.array([5.5, 0.5]), np.array([[5.5, 6.5]]))
    assert cells_of(north) == [(r, 5) for r in range(7)]
    west = traverse_rays(UNIT, np.array([6.5, 5.5]), np.array([[0.5, 5.5]]))
    assert cells_of(west) == [(5, c) for c in range(6, -1, -1)]


def test_ray_leaving_the_grid_is_truncated_and_flagged() -> None:
    traversal = traverse_rays(UNIT, np.array([8.5, 5.5]), np.array([[20.0, 5.5]]))
    assert cells_of(traversal) == [(5, 8), (5, 9)]
    assert not bool(traversal.reached[0])


def test_ray_starting_outside_the_grid_contributes_nothing() -> None:
    traversal = traverse_rays(UNIT, np.array([-3.0, 5.5]), np.array([[5.0, 5.5]]))
    assert cells_of(traversal) == []
    assert not bool(traversal.reached[0])
    assert traversal.total_cells == 0


def test_empty_bundle_is_handled() -> None:
    traversal = traverse_rays(UNIT, np.zeros((0, 2)), np.zeros((0, 2)))
    assert traversal.total_cells == 0
    assert list(traversal.offsets) == [0]


def test_traversal_is_connected_and_ends_at_the_endpoint_cell() -> None:
    """Consecutive cells differ by one step on exactly one axis, and the last is the endpoint."""
    rng = np.random.default_rng(17)
    spec = GridSpec(resolution=0.3, rows=40, cols=55, origin_x=-4.0, origin_y=1.5)
    origins = rng.uniform([-3.5, 2.0], [11.5, 12.5], size=(200, 2))
    endpoints = origins + rng.uniform(-8.0, 8.0, size=(200, 2))
    traversal = traverse_rays(spec, origins, endpoints)
    for ray in range(origins.shape[0]):
        cells = cells_of(traversal, ray)
        if not cells:
            continue
        for before, after in pairwise(cells):
            assert abs(before[0] - after[0]) + abs(before[1] - after[1]) == 1
        if bool(traversal.reached[ray]):
            expected_row = int(np.floor((endpoints[ray, 1] - spec.origin_y) / spec.resolution))
            expected_col = int(np.floor((endpoints[ray, 0] - spec.origin_x) / spec.resolution))
            assert cells[-1] == (expected_row, expected_col)


def test_traversal_matches_a_dense_sampling_of_the_segment() -> None:
    """Every cell a densely sampled segment falls in is visited, and no others.

    Dense sampling is the analytic reference: it cannot invent a cell the segment does
    not enter, and at this sampling density it cannot miss one either.
    """
    rng = np.random.default_rng(29)
    spec = GridSpec(resolution=0.25, rows=48, cols=48, origin_x=-6.0, origin_y=-6.0)
    origins = rng.uniform(-5.0, 5.0, size=(120, 2))
    endpoints = origins + rng.uniform(-4.0, 4.0, size=(120, 2))
    traversal = traverse_rays(spec, origins, endpoints)
    samples = np.linspace(0.0, 1.0, 20001)[:, None]

    for ray in range(origins.shape[0]):
        visited = cells_of(traversal, ray)
        points = origins[ray] + samples * (endpoints[ray] - origins[ray])
        cols = np.floor((points[:, 0] - spec.origin_x) / spec.resolution).astype(np.int64)
        rows = np.floor((points[:, 1] - spec.origin_y) / spec.resolution).astype(np.int64)
        inside = (rows >= 0) & (rows < spec.rows) & (cols >= 0) & (cols < spec.cols)
        sampled = {(int(r), int(c)) for r, c in zip(rows[inside], cols[inside], strict=True)}
        if not bool(traversal.reached[ray]):
            # The ray left the grid, so the sampled set may include cells found after
            # re-entry that the traversal correctly stopped before.
            assert set(visited) <= sampled
            continue
        assert set(visited) == sampled


def test_a_bundle_gives_the_same_answer_as_the_rays_run_singly() -> None:
    """Vectorising over rays must not change any individual result."""
    rng = np.random.default_rng(5)
    spec = GridSpec(resolution=0.4, rows=30, cols=30)
    origins = rng.uniform(1.0, 11.0, size=(50, 2))
    endpoints = origins + rng.uniform(-6.0, 6.0, size=(50, 2))
    together = traverse_rays(spec, origins, endpoints)
    for ray in range(origins.shape[0]):
        alone = traverse_rays(spec, origins[ray], endpoints[ray : ray + 1])
        assert cells_of(together, ray) == cells_of(alone)
        assert bool(together.reached[ray]) == bool(alone.reached[0])


def test_terminal_mask_selects_the_last_cell_of_each_completed_ray() -> None:
    spec = GridSpec(resolution=1.0, rows=6, cols=6)
    origins = np.array([[0.5, 0.5], [0.5, 0.5]])
    endpoints = np.array([[4.5, 2.5], [30.0, 0.5]])
    traversal = traverse_rays(spec, origins, endpoints)
    mask = traversal.terminal_mask()
    assert int(mask.sum()) == 1
    row = int(traversal.rows[mask][0])
    col = int(traversal.cols[mask][0])
    assert (row, col) == (2, 4)


def test_shape_validation() -> None:
    with pytest.raises(ValueError, match="endpoints must have shape"):
        traverse_rays(UNIT, np.array([0.5, 0.5]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="origins must have shape"):
        traverse_rays(UNIT, np.zeros((3, 2)), np.zeros((2, 2)))
