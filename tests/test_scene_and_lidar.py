"""Tier one: scene geometry and the lidar simulator, checked against closed forms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from freespace_grid.model.grid import GridSpec
from freespace_grid.model.transform import Pose2D
from freespace_grid.pipeline.lidar import LidarSpec, simulate_scan
from freespace_grid.pipeline.scene import (
    Circle,
    MovingCircle,
    Polygon,
    Scene,
    occupancy_truth,
    ray_ranges,
)
from freespace_grid.pipeline.trajectory import Trajectory, constant_twist, from_segments


def test_ray_circle_range_matches_the_closed_form() -> None:
    scene = Scene(name="one_disc", circles=(Circle(5.0, 0.0, 1.0),))
    directions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    ranges = ray_ranges(scene, np.array([0.0, 0.0]), directions)
    assert ranges[0] == pytest.approx(4.0)
    assert math.isinf(ranges[1])
    assert math.isinf(ranges[2])


def test_ray_circle_range_at_an_oblique_angle() -> None:
    """A ray offset from the centre meets the disc at the analytic chord distance."""
    scene = Scene(name="one_disc", circles=(Circle(0.0, 0.0, 2.0),))
    offset = 1.0
    origin = np.array([-10.0, offset])
    ranges = ray_ranges(scene, origin, np.array([[1.0, 0.0]]))
    assert ranges[0] == pytest.approx(10.0 - math.sqrt(4.0 - offset**2))


def test_ray_inside_a_circle_returns_the_far_intersection() -> None:
    scene = Scene(name="one_disc", circles=(Circle(0.0, 0.0, 3.0),))
    ranges = ray_ranges(scene, np.array([0.0, 0.0]), np.array([[1.0, 0.0]]))
    assert ranges[0] == pytest.approx(3.0)


def test_ray_polygon_range_matches_the_closed_form() -> None:
    scene = Scene(name="one_box", polygons=(Polygon.rectangle(2.0, -1.0, 4.0, 1.0),))
    directions = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    ranges = ray_ranges(scene, np.array([0.0, 0.0]), directions)
    assert ranges[0] == pytest.approx(2.0)
    assert math.isinf(ranges[1])
    assert math.isinf(ranges[2])


def test_nearest_of_several_obstacles_wins() -> None:
    scene = Scene(
        name="two",
        circles=(Circle(9.0, 0.0, 1.0),),
        polygons=(Polygon.rectangle(3.0, -1.0, 4.0, 1.0),),
    )
    ranges = ray_ranges(scene, np.array([0.0, 0.0]), np.array([[1.0, 0.0]]))
    assert ranges[0] == pytest.approx(3.0)


def test_occupancy_truth_matches_the_analytic_areas() -> None:
    """Counted cells recover the analytic area to within the discretisation error."""
    spec = GridSpec(resolution=0.05, rows=200, cols=200, origin_x=-5.0, origin_y=-5.0)
    scene = Scene(
        name="areas",
        circles=(Circle(-2.0, -2.0, 1.5),),
        polygons=(Polygon.rectangle(0.0, 0.0, 3.0, 2.0),),
    )
    truth = occupancy_truth(scene, spec)
    area = float(truth.sum()) * spec.cell_area
    expected = math.pi * 1.5**2 + 3.0 * 2.0
    assert area == pytest.approx(expected, rel=0.01)


def test_occupancy_truth_of_a_non_convex_polygon() -> None:
    """The crossing number test handles a concave outline, an L shape here."""
    spec = GridSpec(resolution=0.1, rows=60, cols=60, origin_x=-1.0, origin_y=-1.0)
    shape = Polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (1.0, 1.0), (1.0, 4.0), (0.0, 4.0)))
    truth = occupancy_truth(Scene(name="ell", polygons=(shape,)), spec)
    area = float(truth.sum()) * spec.cell_area
    assert area == pytest.approx(7.0, rel=0.02)


def test_polygon_needs_three_vertices() -> None:
    with pytest.raises(ValueError, match="three vertices"):
        Polygon(((0.0, 0.0), (1.0, 1.0)))


def test_circle_needs_a_positive_radius() -> None:
    with pytest.raises(ValueError, match="radius"):
        Circle(0.0, 0.0, 0.0)


def test_moving_circle_position_and_direction() -> None:
    mover = MovingCircle(1.0, 2.0, 3.0, -4.0, 0.5)
    assert mover.speed == pytest.approx(5.0)
    at_two = mover.at(2.0)
    assert (at_two.center_x, at_two.center_y) == pytest.approx((7.0, -6.0))
    assert mover.direction() == pytest.approx(np.array([0.6, -0.8]))


def test_lidar_beam_angles_span_the_field_of_view() -> None:
    lidar = LidarSpec(angular_resolution_deg=1.0, field_of_view_deg=90.0)
    angles = lidar.beam_angles()
    assert angles.size == 90
    assert angles[0] == pytest.approx(-math.radians(44.5))
    assert angles[-1] == pytest.approx(math.radians(44.5))


def test_lidar_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="max_range"):
        LidarSpec(max_range=0.1, min_range=0.2)
    with pytest.raises(ValueError, match="dropout_prob"):
        LidarSpec(dropout_prob=1.0)
    with pytest.raises(ValueError, match="field_of_view_deg"):
        LidarSpec(field_of_view_deg=400.0)


def test_noise_free_scan_reproduces_the_analytic_ranges() -> None:
    """With no noise and no dropout the simulator is the geometry, nothing else."""
    scene = Scene(name="box", polygons=(Polygon.rectangle(-6.0, -6.0, 6.0, 6.0),))
    lidar = LidarSpec(
        max_range=40.0,
        angular_resolution_deg=1.0,
        range_noise_std=0.0,
        dropout_prob=0.0,
    )
    pose = Pose2D(0.0, 0.0, 0.0)
    scan = simulate_scan(scene, pose, lidar, np.random.default_rng(0))
    assert scan.dropped == 0
    assert bool(np.all(scan.is_hit))
    world = scan.angles + pose.theta
    expected = np.minimum(
        6.0 / np.abs(np.cos(world)),
        6.0 / np.abs(np.sin(world)),
    )
    assert np.allclose(scan.ranges, expected, rtol=1e-9)


def test_maximum_range_beams_are_flagged_and_report_the_limit() -> None:
    scene = Scene(name="one_disc", circles=(Circle(4.0, 0.0, 0.5),))
    lidar = LidarSpec(
        max_range=10.0,
        angular_resolution_deg=2.0,
        range_noise_std=0.0,
        dropout_prob=0.0,
    )
    scan = simulate_scan(scene, Pose2D(0.0, 0.0, 0.0), lidar, np.random.default_rng(1))
    assert scan.max_range_count > 0
    assert np.all(scan.ranges[~scan.is_hit] == lidar.max_range)
    assert np.all(scan.ranges[scan.is_hit] < lidar.max_range)


def test_scan_endpoints_lie_at_the_reported_range_and_bearing() -> None:
    scene = Scene(name="one_disc", circles=(Circle(4.0, 3.0, 1.0),))
    lidar = LidarSpec(max_range=12.0, angular_resolution_deg=5.0, dropout_prob=0.0)
    pose = Pose2D(-1.0, 2.0, 0.6)
    scan = simulate_scan(scene, pose, lidar, np.random.default_rng(4))
    offsets = scan.endpoints() - scan.origin
    assert np.allclose(np.hypot(offsets[:, 0], offsets[:, 1]), scan.ranges, atol=1e-12)


def test_scan_is_reproducible_from_the_seed_and_dropout_is_geometry_independent() -> None:
    """The same seed gives the same scan, and the random stream does not depend on the scene.

    The second property matters because the trajectory runner draws from one generator
    across the whole run. If a scene change moved the stream, adding an obstacle would
    silently change the noise on every later sweep.
    """
    lidar = LidarSpec(angular_resolution_deg=2.0, dropout_prob=0.2)
    pose = Pose2D(0.0, 0.0, 0.0)
    sparse = Scene(name="sparse", circles=(Circle(6.0, 0.0, 1.0),))
    dense = Scene(
        name="dense",
        circles=(Circle(6.0, 0.0, 1.0), Circle(-8.0, 3.0, 2.0)),
        polygons=(Polygon.rectangle(2.0, 4.0, 5.0, 9.0),),
    )
    first = simulate_scan(sparse, pose, lidar, np.random.default_rng(9))
    again = simulate_scan(sparse, pose, lidar, np.random.default_rng(9))
    other = simulate_scan(dense, pose, lidar, np.random.default_rng(9))
    assert np.array_equal(first.ranges, again.ranges)
    assert np.array_equal(first.angles, again.angles)
    assert np.array_equal(first.angles, other.angles)
    assert first.dropped == other.dropped


def test_dropout_removes_about_the_configured_fraction_of_beams() -> None:
    lidar = LidarSpec(angular_resolution_deg=0.25, dropout_prob=0.1, range_noise_std=0.0)
    scene = Scene(name="empty")
    scan = simulate_scan(scene, Pose2D(), lidar, np.random.default_rng(6))
    assert scan.dropped + scan.angles.size == lidar.beam_count
    assert scan.dropped / lidar.beam_count == pytest.approx(0.1, abs=0.03)


def test_constant_twist_of_zero_yaw_rate_is_a_straight_line() -> None:
    trajectory = constant_twist(Pose2D(1.0, 2.0, 0.0), speed=2.0, yaw_rate=0.0, dt=0.5, steps=4)
    assert len(trajectory) == 5
    assert trajectory.poses[-1].x == pytest.approx(1.0 + 4.0)
    assert trajectory.poses[-1].y == pytest.approx(2.0)
    assert trajectory.duration == pytest.approx(2.0)


def test_constant_twist_follows_a_circle_of_the_published_radius() -> None:
    """A constant twist traces an arc of radius speed over yaw rate, exactly."""
    speed, yaw_rate = 3.0, 0.5
    radius = speed / yaw_rate
    trajectory = constant_twist(Pose2D(0.0, 0.0, 0.0), speed, yaw_rate, dt=0.1, steps=60)
    centre = np.array([0.0, radius])
    for pose in trajectory.poses:
        distance = math.hypot(pose.x - centre[0], pose.y - centre[1])
        assert distance == pytest.approx(radius, abs=1e-9)


def test_trajectory_subsampling_keeps_the_endpoints() -> None:
    trajectory = from_segments(Pose2D(), ((1.0, 0.0, 20), (1.0, 0.3, 10)), dt=0.25)
    reduced = trajectory.subsample(6)
    assert len(reduced) == 6
    assert reduced.poses[0] == trajectory.poses[0]
    assert reduced.poses[-1] == trajectory.poses[-1]
    assert reduced.times[-1] == trajectory.times[-1]
    assert trajectory.subsample(1000) is trajectory


def test_trajectory_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        Trajectory(poses=(Pose2D(),), times=(0.0, 1.0))
    with pytest.raises(ValueError, match="at least one pose"):
        Trajectory(poses=(), times=())
