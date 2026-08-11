"""Tier one: convergence of the filter, and the properties of the scoring functions."""

from __future__ import annotations

import json
from itertools import pairwise

import numpy as np
import pytest

from freespace_grid.analysis.metrics import score_grid, threshold_sweep, tolerance_sweep
from freespace_grid.analysis.sharpness import boundary_band, measure_sharpness
from freespace_grid.analysis.smear import measure_smear, path_region
from freespace_grid.model.grid import GridSpec
from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.model.occupancy import CellState, OccupancyGrid
from freespace_grid.pipeline.runner import RunConfig, run_mapping
from freespace_grid.pipeline.scenarios import enclosed_room


def test_enclosed_room_converges_to_the_correct_occupancy() -> None:
    """A closed room seen from its centre converges: interior free, walls occupied.

    No beam reaches the range limit here, so every sweep terminates on a wall and the
    only thing that can stop convergence is a fault in the traversal or the update.
    """
    scenario = enclosed_room(steps=12)
    trace = run_mapping(scenario)
    states = trace.grid.classify(scenario.model)
    observed = trace.observed_mask

    agreement = score_grid(trace.grid, trace.truth, scenario.model, region=observed)
    assert agreement.decided_fraction == 1.0
    assert agreement.free_agreement == 1.0
    assert agreement.occupied_agreement == 1.0
    assert agreement.occupied_truth > 100

    interior = observed & ~trace.truth
    assert np.all(states[interior] == int(CellState.FREE))
    assert np.all(trace.grid.log_odds[interior] == scenario.model.l_min)

    wall_face = observed & trace.truth
    assert np.all(states[wall_face] == int(CellState.OCCUPIED))
    assert np.all(trace.grid.log_odds[wall_face] == scenario.model.l_max)
    assert np.all(trace.grid.log_odds <= scenario.model.l_max)
    assert np.all(trace.grid.log_odds >= scenario.model.l_min)


def test_enclosed_room_never_places_a_range_limit_return() -> None:
    scenario = enclosed_room(steps=4)
    trace = run_mapping(scenario)
    assert trace.totals()["max_range_beams"] == 0
    assert trace.totals()["hits"] == trace.totals()["beams"]


def test_more_sweeps_never_reduce_the_decided_fraction() -> None:
    """Evidence accumulates, so coverage is monotone in the number of sweeps."""
    scenario = enclosed_room(steps=12)
    previous = -1.0
    for count in (1, 2, 4, 8, 13):
        trace = run_mapping(scenario, RunConfig(max_steps=count))
        decided = score_grid(trace.grid, trace.truth, scenario.model).decided_fraction
        assert decided >= previous - 1e-12
        previous = decided


def test_the_run_trace_serialises_to_json_with_one_record_per_sweep() -> None:
    """The structured trace is the interface between the pipeline and everything else."""
    scenario = enclosed_room(steps=3)
    trace = run_mapping(scenario)
    document = json.loads(json.dumps(trace.to_dict()))

    assert document["scenario"] == "enclosed_room"
    assert document["totals"]["scans"] == len(trace.steps) == 4
    assert len(document["steps"]) == 4
    assert [step["index"] for step in document["steps"]] == [0, 1, 2, 3]
    assert document["grid_cells"] == trace.grid.spec.size
    assert document["observed_cells"] == int(np.count_nonzero(trace.observed))
    assert document["config"]["frame"] == "world"
    for step in document["steps"]:
        assert step["hits"] + step["max_range_beams"] == step["beams"]
        assert (
            step["free_cells"] + step["occupied_cells"] + step["unknown_cells"]
            == (document["grid_cells"])
        )
    assert trace.final is trace.steps[-1]


def test_a_copied_grid_shares_no_storage() -> None:
    model = LogOddsModel()
    spec = GridSpec(resolution=1.0, rows=4, cols=4)
    original = OccupancyGrid.from_prior(spec, model)
    duplicate = original.copy()
    duplicate.log_odds[0, 0] = model.l_max
    assert original.log_odds[0, 0] == model.l_prior
    assert duplicate.spec == original.spec


def test_scoring_a_grid_against_itself_is_perfect() -> None:
    """Ground truth taken from the map's own decision gives agreement one on both classes."""
    rng = np.random.default_rng(23)
    model = LogOddsModel()
    spec = GridSpec(resolution=0.4, rows=48, cols=61, origin_x=-3.0, origin_y=2.0)
    grid = OccupancyGrid(spec=spec, log_odds=rng.uniform(model.l_min, model.l_max, size=spec.shape))
    truth = np.asarray(grid.classify(model) == int(CellState.OCCUPIED), dtype=np.bool_)
    agreement = score_grid(grid, truth, model)
    assert agreement.free_agreement == 1.0
    assert agreement.occupied_agreement == 1.0
    assert agreement.balanced_agreement == 1.0
    assert agreement.free_called_occupied == 0
    assert agreement.occupied_called_free == 0
    assert agreement.decided_cells == agreement.free_truth + agreement.occupied_truth


def test_scoring_against_the_complement_is_the_worst_possible() -> None:
    rng = np.random.default_rng(24)
    model = LogOddsModel()
    spec = GridSpec(resolution=0.4, rows=30, cols=30)
    grid = OccupancyGrid(spec=spec, log_odds=rng.uniform(model.l_min, model.l_max, size=spec.shape))
    truth = np.asarray(grid.classify(model) == int(CellState.FREE), dtype=np.bool_)
    agreement = score_grid(grid, truth, model)
    assert agreement.free_agreement == 0.0
    assert agreement.occupied_agreement == 0.0


def test_raising_the_threshold_never_increases_the_decided_fraction() -> None:
    scenario = enclosed_room(steps=6)
    trace = run_mapping(scenario)
    sweep = threshold_sweep(
        trace.grid,
        trace.truth,
        scenario.model,
        (0.55, 0.62, 0.68, 0.74, 0.80, 0.86),
        region=trace.observed_mask,
    )
    fractions = [item.decided_fraction for item in sweep]
    assert all(later <= earlier + 1e-12 for earlier, later in pairwise(fractions))
    assert [item.decision_prob for item in sweep] == [0.55, 0.62, 0.68, 0.74, 0.80, 0.86]


def test_spatial_tolerance_never_reduces_occupied_agreement() -> None:
    scenario = enclosed_room(steps=6)
    trace = run_mapping(scenario)
    sweep = tolerance_sweep(
        trace.grid, trace.truth, scenario.model, (0, 1, 2, 3), region=trace.observed_mask
    )
    values = [item.occupied_agreement for item in sweep]
    assert all(later >= earlier - 1e-12 for earlier, later in pairwise(values))
    assert all(item.free_agreement == sweep[0].free_agreement for item in sweep)


def test_score_grid_validates_shapes_and_tolerance() -> None:
    model = LogOddsModel()
    spec = GridSpec(resolution=1.0, rows=4, cols=4)
    grid = OccupancyGrid.from_prior(spec, model)
    with pytest.raises(ValueError, match="truth shape"):
        score_grid(grid, np.zeros((3, 3), dtype=np.bool_), model)
    with pytest.raises(ValueError, match="region shape"):
        score_grid(
            grid,
            np.zeros((4, 4), dtype=np.bool_),
            model,
            region=np.zeros((2, 2), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="spatial_tolerance"):
        score_grid(grid, np.zeros((4, 4), dtype=np.bool_), model, spatial_tolerance=-1)


def test_path_region_is_a_capsule_around_the_segment() -> None:
    spec = GridSpec(resolution=0.1, rows=100, cols=200, origin_x=-2.0, origin_y=-2.0)
    region = path_region(spec, (0.0, 0.0), (4.0, 0.0), 1.0)
    area = float(region.sum()) * spec.cell_area
    assert area == pytest.approx(4.0 * 2.0 + np.pi * 1.0**2, rel=0.02)
    with pytest.raises(ValueError, match="radius"):
        path_region(spec, (0.0, 0.0), (1.0, 0.0), 0.0)


def test_measure_smear_recovers_a_known_rectangle() -> None:
    """Extents along and across a known direction match the shape that was planted."""
    model = LogOddsModel()
    spec = GridSpec(resolution=0.1, rows=120, cols=120, origin_x=-6.0, origin_y=-6.0)
    grid = OccupancyGrid.from_prior(spec, model)
    grid.log_odds[55:65, 40:80] = model.l_max
    truth = np.zeros(spec.shape, dtype=np.bool_)
    truth[55:65, 40:80] = True
    region = np.ones(spec.shape, dtype=np.bool_)
    metrics = measure_smear(grid, truth, model, region, np.array([1.0, 0.0]), label="planted")
    assert metrics.occupied_cells == 400
    assert metrics.extent_along == pytest.approx(4.0)
    assert metrics.extent_across == pytest.approx(1.0)
    assert metrics.stale_cells == 0
    assert metrics.false_free_cells == 0
    assert metrics.area_ratio == pytest.approx(1.0)


def test_measure_smear_rejects_a_zero_direction() -> None:
    model = LogOddsModel()
    spec = GridSpec(resolution=0.5, rows=8, cols=8)
    grid = OccupancyGrid.from_prior(spec, model)
    with pytest.raises(ValueError, match="non-zero"):
        measure_smear(
            grid,
            np.zeros(spec.shape, dtype=np.bool_),
            model,
            np.ones(spec.shape, dtype=np.bool_),
            np.zeros(2),
            label="bad",
        )


def test_boundary_band_surrounds_the_truth_edge() -> None:
    truth = np.zeros((21, 21), dtype=np.bool_)
    truth[8:13, 8:13] = True
    band = boundary_band(truth, width=1)
    assert bool(band[7, 10])
    assert bool(band[8, 8])
    assert not bool(band[10, 10])
    with pytest.raises(ValueError, match="width"):
        boundary_band(truth, width=0)


def test_sharpness_reports_full_saturation_for_a_clamped_map() -> None:
    model = LogOddsModel()
    spec = GridSpec(resolution=0.5, rows=20, cols=20)
    grid = OccupancyGrid(spec=spec, log_odds=np.full(spec.shape, model.l_min))
    truth = np.zeros(spec.shape, dtype=np.bool_)
    truth[8:12, 8:12] = True
    sharpness = measure_sharpness(grid, truth, model)
    assert sharpness.clamped_fraction == 1.0
    assert sharpness.edge_contrast == 0.0
    assert sharpness.mean_abs_evidence == pytest.approx(abs(model.l_min - model.l_prior))
