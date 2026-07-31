"""Tier one: invariants of the grid coordinate system and the rigid transforms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from freespace_grid.model.grid import (
    GridSpec,
    cell_centers,
    cell_to_world,
    in_bounds,
    world_to_cell,
)
from freespace_grid.model.transform import Pose2D, compose, inverse, transform_points, wrap_angle

SPECS = (
    GridSpec(resolution=1.0, rows=8, cols=12),
    GridSpec(resolution=0.2, rows=200, cols=300),
    GridSpec(resolution=0.05, rows=64, cols=64, origin_x=-3.2, origin_y=7.35),
    GridSpec(resolution=0.37, rows=13, cols=29, origin_x=-100.5, origin_y=0.25),
)


@pytest.mark.parametrize("spec", SPECS)
def test_cell_round_trip_is_exact_at_cell_centres(spec: GridSpec) -> None:
    """Every cell index survives a trip through world coordinates and back."""
    rows, cols = np.meshgrid(
        np.arange(spec.rows, dtype=np.int64), np.arange(spec.cols, dtype=np.int64), indexing="ij"
    )
    cells = np.stack((rows.reshape(-1), cols.reshape(-1)), axis=1)
    recovered = world_to_cell(spec, cell_to_world(spec, cells))
    assert np.array_equal(recovered, cells)


@pytest.mark.parametrize("spec", SPECS)
def test_cell_centres_agree_with_cell_to_world(spec: GridSpec) -> None:
    """The bulk centre grid and the per-cell mapping give the same coordinates."""
    grid_x, grid_y = cell_centers(spec)
    rows, cols = np.meshgrid(
        np.arange(spec.rows, dtype=np.int64), np.arange(spec.cols, dtype=np.int64), indexing="ij"
    )
    cells = np.stack((rows.reshape(-1), cols.reshape(-1)), axis=1)
    points = cell_to_world(spec, cells)
    assert np.array_equal(points[:, 0], grid_x.reshape(-1))
    assert np.array_equal(points[:, 1], grid_y.reshape(-1))


def test_world_to_cell_uses_the_lower_left_convention() -> None:
    """A point exactly on a cell boundary belongs to the cell above and to the right."""
    spec = GridSpec(resolution=2.0, rows=4, cols=4, origin_x=-4.0, origin_y=-4.0)
    points = np.array([[-4.0, -4.0], [-3.9, -3.9], [-2.0, -2.0], [3.99, 3.99]])
    assert np.array_equal(
        world_to_cell(spec, points), np.array([[0, 0], [0, 0], [1, 1], [3, 3]], dtype=np.int64)
    )


def test_in_bounds_rejects_indices_outside_the_grid() -> None:
    spec = GridSpec(resolution=1.0, rows=3, cols=5)
    cells = np.array([[0, 0], [2, 4], [-1, 0], [0, 5], [3, 0]], dtype=np.int64)
    assert np.array_equal(in_bounds(spec, cells), np.array([True, True, False, False, False]))


def test_extent_and_recentering_are_consistent() -> None:
    spec = GridSpec(resolution=0.25, rows=40, cols=80, origin_x=1.0, origin_y=2.0)
    x_min, x_max, y_min, y_max = spec.extent
    assert (x_max - x_min, y_max - y_min) == (20.0, 10.0)
    moved = spec.recentered(0.0, 0.0)
    assert moved.extent == (-10.0, 10.0, -5.0, 5.0)
    assert moved.shape == spec.shape
    assert moved.resolution == spec.resolution


def test_grid_spec_rejects_degenerate_geometry() -> None:
    with pytest.raises(ValueError, match="resolution"):
        GridSpec(resolution=0.0, rows=2, cols=2)
    with pytest.raises(ValueError, match="rows and cols"):
        GridSpec(resolution=1.0, rows=0, cols=2)


def test_transform_then_inverse_recovers_the_points() -> None:
    """Composing a pose with its inverse is the identity, on poses and on points."""
    rng = np.random.default_rng(11)
    points = rng.normal(0.0, 5.0, size=(64, 2))
    for _ in range(20):
        pose = Pose2D(
            x=float(rng.normal(0.0, 10.0)),
            y=float(rng.normal(0.0, 10.0)),
            theta=float(rng.uniform(-math.pi, math.pi)),
        )
        recovered = transform_points(inverse(pose), transform_points(pose, points))
        assert np.allclose(recovered, points, atol=1e-12)
        identity = compose(pose, inverse(pose))
        assert abs(identity.x) < 1e-12
        assert abs(identity.y) < 1e-12
        assert abs(identity.theta) < 1e-12


def test_compose_matches_successive_application() -> None:
    rng = np.random.default_rng(3)
    points = rng.normal(0.0, 2.0, size=(32, 2))
    outer = Pose2D(1.5, -2.5, 0.7)
    inner = Pose2D(-0.25, 4.0, -1.9)
    stepwise = transform_points(outer, transform_points(inner, points))
    combined = transform_points(compose(outer, inner), points)
    assert np.allclose(stepwise, combined, atol=1e-12)


def test_rotation_matrix_is_orthonormal() -> None:
    pose = Pose2D(3.0, -1.0, 2.2)
    rotation = pose.rotation()
    assert np.allclose(rotation @ rotation.T, np.eye(2), atol=1e-14)
    assert abs(float(np.linalg.det(rotation)) - 1.0) < 1e-14


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0.0, 0.0), (math.pi, math.pi), (-math.pi, math.pi), (3.0 * math.pi, math.pi)],
)
def test_wrap_angle_maps_into_the_half_open_interval(angle: float, expected: float) -> None:
    assert wrap_angle(angle) == pytest.approx(expected, abs=1e-12)
