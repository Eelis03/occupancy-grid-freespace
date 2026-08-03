"""Odometry drift, and the scan to map matching that corrects it.

This module is the evidence behind the claim that the pose limitation recorded in
``docs/design-notes.md`` was closed rather than described. It checks three separate
things: that the odometry model degrades the pose the way an odometry model should, that
the matcher recovers a pose it is given no other information about, and that switching
the whole apparatus off reproduces the exact pose path it replaced.

The last of those matters most. Every other result in this repository is measured with
exact poses, so a localiser that quietly perturbed that path would invalidate the rest
of the README.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from freespace_grid.algorithm.scan_match import (
    SearchWindow,
    likelihood_field,
    match_scan,
    scan_body_points,
)
from freespace_grid.analysis.metrics import score_grid
from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.model.occupancy import CellState, OccupancyGrid
from freespace_grid.model.transform import Pose2D, compose
from freespace_grid.pipeline.lidar import simulate_scan
from freespace_grid.pipeline.odometry import (
    OdometryNoise,
    dead_reckon,
    noisy_increments,
    pose_error,
    relative_motion,
)
from freespace_grid.pipeline.runner import MappingTrace, RunConfig, run_mapping
from freespace_grid.pipeline.scenarios import SCANNER_ODOMETRY, enclosed_room, urban_block

STEPS = 20


def _agreement(trace: MappingTrace) -> tuple[float, float, float]:
    """Decided fraction, free agreement and occupied agreement over the observed region."""
    model = urban_block().model
    result = score_grid(trace.grid, trace.truth, model, region=trace.observed_mask)
    return (result.decided_fraction, result.free_agreement, result.occupied_agreement)


# Relative motion and dead reckoning


def test_relative_motion_composes_back_to_the_pose_it_came_from() -> None:
    start = Pose2D(3.0, -2.0, 0.4)
    end = Pose2D(5.5, 1.25, -1.1)
    recovered = compose(start, relative_motion(start, end))
    assert recovered.x == pytest.approx(end.x, abs=1e-12)
    assert recovered.y == pytest.approx(end.y, abs=1e-12)
    assert recovered.theta == pytest.approx(end.theta, abs=1e-12)


def test_zero_noise_dead_reckoning_reproduces_the_trajectory() -> None:
    trajectory = urban_block().trajectory
    rng = np.random.default_rng(7)
    increments = noisy_increments(trajectory, OdometryNoise(), rng)
    estimates = dead_reckon(increments, trajectory.poses[0])
    assert len(estimates) == len(trajectory.poses)
    for estimate, truth in zip(estimates, trajectory.poses, strict=True):
        position, heading = pose_error(estimate, truth)
        assert position < 1e-9
        assert heading < 1e-12


def test_noise_coefficients_scale_the_same_random_walk() -> None:
    """Doubling every coefficient doubles the drift rather than redrawing it.

    Three variates are consumed per increment whatever the coefficients are, so two
    levels of noise differ only in amplitude. That is what makes the rows of the drift
    table comparable: they are the same accident of the seed, scaled.
    """
    trajectory = urban_block().trajectory
    errors = []
    for factor in (1.0, 2.0):
        rng = np.random.default_rng(11)
        increments = noisy_increments(trajectory, SCANNER_ODOMETRY.scaled(factor), rng)
        estimates = dead_reckon(increments, trajectory.poses[0])
        errors.append(pose_error(estimates[-1], trajectory.poses[-1])[0])
    assert errors[0] > 0.1
    assert errors[1] / errors[0] == pytest.approx(2.0, rel=0.05)


def test_odometry_noise_rejects_negative_coefficients() -> None:
    with pytest.raises(ValueError, match="translation_std_per_m"):
        OdometryNoise(translation_std_per_m=-0.1)
    with pytest.raises(ValueError, match="heading_std_per_rad"):
        OdometryNoise(heading_std_per_rad=-1e-9)
    with pytest.raises(ValueError, match="factor"):
        OdometryNoise().scaled(-1.0)


def test_is_exact_is_true_only_when_every_coefficient_is_zero() -> None:
    assert OdometryNoise().is_exact
    assert SCANNER_ODOMETRY.scaled(0.0).is_exact
    assert not SCANNER_ODOMETRY.is_exact


# The likelihood field


def test_likelihood_field_is_zero_where_the_map_holds_no_occupied_evidence() -> None:
    model = LogOddsModel()
    scenario = enclosed_room(steps=4)
    trace = run_mapping(scenario)
    field = likelihood_field(trace.grid, model, blur_cells=0.0)
    states = trace.grid.classify(model)
    assert np.all(field[states == int(CellState.FREE)] == 0.0)
    assert np.all(field[states == int(CellState.UNKNOWN)] == 0.0)
    assert np.all(field[states == int(CellState.OCCUPIED)] > 0.0)
    assert field.max() == pytest.approx(1.0, abs=1e-12)


def test_blurring_spreads_the_field_without_inventing_evidence() -> None:
    """The blur redistributes the field. It cannot add to it, and only the edge loses."""
    model = LogOddsModel()
    trace = run_mapping(enclosed_room(steps=4))
    sharp = likelihood_field(trace.grid, model, blur_cells=0.0)
    blurred = likelihood_field(trace.grid, model, blur_cells=1.0)
    assert np.count_nonzero(blurred) > np.count_nonzero(sharp)
    assert blurred.max() < sharp.max()
    assert blurred.min() >= 0.0
    assert blurred.sum() <= sharp.sum()
    assert blurred.sum() > 0.99 * sharp.sum()


def test_likelihood_field_rejects_a_negative_blur() -> None:
    with pytest.raises(ValueError, match="blur_cells"):
        likelihood_field(run_mapping(enclosed_room(steps=1)).grid, LogOddsModel(), blur_cells=-1.0)


# The matcher itself


def test_matcher_leaves_the_prediction_alone_on_an_empty_map() -> None:
    """With nothing to align against, every candidate ties and the prediction wins."""
    model = LogOddsModel()
    scenario = enclosed_room(steps=1)
    empty = OccupancyGrid.from_prior(scenario.grid, model)
    scan = simulate_scan(
        scenario.scene, scenario.trajectory.poses[0], scenario.lidar, np.random.default_rng(3)
    )
    points = scan_body_points(scan.angles, scan.ranges, scan.is_hit, stride=8)
    predicted = Pose2D(1.0, 2.0, 0.3)
    result = match_scan(empty, model, points, predicted)
    assert result.pose == predicted
    assert result.score == 0.0
    assert result.translation_correction == 0.0
    assert result.evaluations > 0


def test_matcher_recovers_a_pose_it_is_deliberately_given_wrong() -> None:
    """The core claim. A displaced prediction is pulled back to within half a cell."""
    model = LogOddsModel()
    scenario = enclosed_room(steps=8)
    trace = run_mapping(scenario)
    truth = scenario.trajectory.poses[0]
    scan = simulate_scan(scenario.scene, truth, scenario.lidar, np.random.default_rng(5))
    points = scan_body_points(scan.angles, scan.ranges, scan.is_hit, stride=4)

    resolution = scenario.grid.resolution
    window = SearchWindow(
        translation_radius=4.0 * resolution,
        heading_radius=math.radians(3.0),
        translation_step=resolution,
        heading_step=math.radians(0.5),
        refinements=2,
    )
    for offset in (
        Pose2D(3.0 * resolution, 0.0, 0.0),
        Pose2D(0.0, -3.0 * resolution, 0.0),
        Pose2D(2.0 * resolution, 2.0 * resolution, math.radians(1.5)),
    ):
        predicted = compose(truth, offset)
        before = pose_error(predicted, truth)[0]
        result = match_scan(trace.grid, model, points, predicted, window=window)
        after = pose_error(result.pose, truth)[0]
        assert after < 0.5 * resolution, (offset, before, after)
        assert result.score >= result.predicted_score
        assert result.points == points.shape[0]


def test_matcher_cannot_move_further_than_the_search_allows() -> None:
    """Three passes at halving radius bound the correction by twice the first radius."""
    model = LogOddsModel()
    scenario = enclosed_room(steps=8)
    trace = run_mapping(scenario)
    truth = scenario.trajectory.poses[0]
    scan = simulate_scan(scenario.scene, truth, scenario.lidar, np.random.default_rng(5))
    points = scan_body_points(scan.angles, scan.ranges, scan.is_hit, stride=8)

    window = SearchWindow(
        translation_radius=0.5,
        heading_radius=math.radians(1.0),
        translation_step=0.25,
        heading_step=math.radians(0.5),
        refinements=2,
    )
    predicted = compose(truth, Pose2D(4.0, 0.0, 0.0))
    result = match_scan(trace.grid, model, points, predicted, window=window)
    bound = 2.0 * window.translation_radius * math.sqrt(2.0)
    assert result.translation_correction <= bound + 1e-9
    assert abs(result.heading_correction) <= 2.0 * window.heading_radius + 1e-9


def test_search_window_rejects_impossible_settings() -> None:
    with pytest.raises(ValueError, match="steps must be positive"):
        SearchWindow(translation_step=0.0)
    with pytest.raises(ValueError, match="radii must not be negative"):
        SearchWindow(heading_radius=-0.1)
    with pytest.raises(ValueError, match="refinements"):
        SearchWindow(refinements=-1)


def test_scan_body_points_drops_range_limit_beams() -> None:
    angles = np.array([0.0, 0.5, 1.0, 1.5])
    ranges = np.array([2.0, 30.0, 3.0, 30.0])
    is_hit = np.array([True, False, True, False])
    points = scan_body_points(angles, ranges, is_hit)
    assert points.shape == (2, 2)
    assert points[0] == pytest.approx([2.0, 0.0], abs=1e-12)
    assert points[1] == pytest.approx([3.0 * math.cos(1.0), 3.0 * math.sin(1.0)], abs=1e-12)
    assert scan_body_points(angles, ranges, is_hit, stride=2).shape == (1, 2)
    with pytest.raises(ValueError, match="stride"):
        scan_body_points(angles, ranges, is_hit, stride=0)


# The runner, end to end


def test_exact_poses_stay_the_default_and_the_odometry_path_reproduces_them() -> None:
    """Switching the machinery on with zero noise must not move any published number."""
    scenario = urban_block()
    exact = run_mapping(scenario, RunConfig(frame="world", max_steps=STEPS))
    reckoned = run_mapping(
        scenario, RunConfig(frame="world", odometry=OdometryNoise(), max_steps=STEPS)
    )
    assert exact.final_position_error == 0.0
    assert reckoned.final_position_error < 1e-9
    assert reckoned.peak_position_error < 1e-9
    assert reckoned.pose_corrections == 0
    for produced, recorded in zip(_agreement(reckoned), _agreement(exact), strict=True):
        assert produced == pytest.approx(recorded, abs=1e-9)


def test_pose_correction_without_an_odometry_model_is_refused() -> None:
    with pytest.raises(ValueError, match="pose_correction requires an odometry model"):
        run_mapping(urban_block(), RunConfig(pose_correction=True, max_steps=4))


def test_drifting_poses_destroy_the_occupied_class_and_spare_the_free_one() -> None:
    """The shape of the damage, not only its size.

    A displaced sweep still reports the open ground as open, because free space is a
    thick region and moving it by a metre leaves most of it inside itself. A surface is
    one cell thick, so the same displacement puts it somewhere else entirely.
    """
    scenario = urban_block()
    exact = run_mapping(scenario, RunConfig(frame="world", max_steps=STEPS))
    drifted = run_mapping(
        scenario,
        RunConfig(frame="world", odometry=SCANNER_ODOMETRY.scaled(2.0), max_steps=STEPS),
    )
    assert drifted.final_position_error > 0.5

    _, exact_free, exact_occupied = _agreement(exact)
    _, drifted_free, drifted_occupied = _agreement(drifted)
    assert drifted_occupied < 0.6 * exact_occupied
    assert drifted_free > exact_free - 0.02


def test_scan_matching_recovers_most_of_what_the_drift_destroyed() -> None:
    scenario = urban_block()
    noise = SCANNER_ODOMETRY.scaled(4.0)
    drifted = run_mapping(scenario, RunConfig(frame="world", odometry=noise, max_steps=STEPS))
    matched = run_mapping(
        scenario,
        RunConfig(frame="world", odometry=noise, pose_correction=True, max_steps=STEPS),
    )

    assert matched.pose_corrections == STEPS - 1
    assert matched.final_position_error < 0.25 * drifted.final_position_error
    assert matched.final_heading_error < 0.25 * drifted.final_heading_error

    _, _, drifted_occupied = _agreement(drifted)
    _, _, matched_occupied = _agreement(matched)
    assert matched_occupied > drifted_occupied + 0.3
    assert sum(step.match_evaluations for step in matched.steps) > 0


def test_correcting_a_pose_that_needs_no_correction_is_not_free() -> None:
    """What the close cost, asserted rather than asserted about.

    The matcher aligns each sweep with the discretised map rather than with the truth,
    so it introduces a bias of its own. Run against exact odometry it therefore moves
    the pose off the truth and loses occupied agreement, which is why exact poses remain
    the default for every other result in the repository.
    """
    scenario = urban_block()
    exact = run_mapping(scenario, RunConfig(frame="world", max_steps=STEPS))
    corrected = run_mapping(
        scenario,
        RunConfig(
            frame="world",
            odometry=OdometryNoise(),
            pose_correction=True,
            max_steps=STEPS,
        ),
    )
    assert corrected.final_position_error > 0.0
    assert corrected.final_position_error < scenario.grid.resolution
    _, _, exact_occupied = _agreement(exact)
    _, _, corrected_occupied = _agreement(corrected)
    assert corrected_occupied < exact_occupied
