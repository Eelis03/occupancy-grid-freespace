"""Tier one: which free cells a planner can reach, and what is wrong inside them."""

from __future__ import annotations

import numpy as np
import pytest

from freespace_grid.analysis.reachability import measure_reachability, reachable_free
from freespace_grid.model.grid import GridSpec
from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.model.occupancy import OccupancyGrid
from freespace_grid.pipeline.runner import run_mapping
from freespace_grid.pipeline.scenarios import enclosed_room

MODEL = LogOddsModel()


def _unknown_grid(spec: GridSpec) -> OccupancyGrid:
    """A grid holding the prior everywhere, ready for free and occupied cells to be set."""
    return OccupancyGrid.from_prior(spec, MODEL)


def _divided_room() -> OccupancyGrid:
    """A nine by nine room of free cells split down the middle by a wall of occupied ones."""
    grid = _unknown_grid(GridSpec(resolution=1.0, rows=9, cols=9))
    grid.log_odds[:, :] = MODEL.l_min
    grid.log_odds[:, 4] = MODEL.l_max
    return grid


def test_a_wall_strands_the_free_space_on_its_far_side() -> None:
    """Free space behind a wall is free space the vehicle cannot use, and is counted apart."""
    grid = _divided_room()
    report = measure_reachability(
        grid, np.zeros(grid.spec.shape, dtype=np.bool_), MODEL, (1.5, 1.5)
    )
    assert report.free_cells == 72
    assert report.components == 2
    assert report.reachable_cells == 36
    assert report.stranded_cells == 36
    assert report.reachable_fraction == 0.5
    assert report.reachable_area == pytest.approx(36.0)


def test_free_space_does_not_leak_diagonally_between_two_occupied_cells() -> None:
    """Two obstacle cells meeting at a corner separate the map, as they do a vehicle.

    An eight connected fill walks straight through that corner and would report the
    isolated cell as reachable. The ray traversal refuses to cut the same corner, for
    the same reason.
    """
    grid = _unknown_grid(GridSpec(resolution=1.0, rows=5, cols=5))
    grid.log_odds[:, :] = MODEL.l_min
    grid.log_odds[0, 1] = MODEL.l_max
    grid.log_odds[1, 0] = MODEL.l_max
    mask = reachable_free(grid, MODEL, (1.5, 1.5))
    assert bool(mask[1, 1])
    assert not bool(mask[0, 0])
    assert int(np.count_nonzero(mask)) == 22


def test_the_frontier_is_the_reachable_free_edge_of_the_unknown_region() -> None:
    """A planner that treats unknown as impassable stops on exactly these cells."""
    grid = _unknown_grid(GridSpec(resolution=0.5, rows=6, cols=8))
    grid.log_odds[:, :4] = MODEL.l_min
    report = measure_reachability(
        grid, np.zeros(grid.spec.shape, dtype=np.bool_), MODEL, (0.25, 0.25)
    )
    assert report.free_cells == 24
    assert report.reachable_cells == 24
    assert report.frontier_cells == 6


def test_a_missed_obstacle_behind_a_wall_is_not_one_the_vehicle_can_hit() -> None:
    """Both cells are wrong in the same way and only one of them is on the vehicle's side."""
    grid = _divided_room()
    truth = np.zeros(grid.spec.shape, dtype=np.bool_)
    truth[2, 1] = True
    truth[2, 7] = True
    report = measure_reachability(grid, truth, MODEL, (1.5, 1.5))
    assert report.false_free_cells == 2
    assert report.reachable_false_free_cells == 1
    assert report.exposed_fraction == 0.5


def test_reachability_rejects_a_start_it_cannot_plan_from() -> None:
    grid = _divided_room()
    truth = np.zeros(grid.spec.shape, dtype=np.bool_)
    with pytest.raises(ValueError, match="outside the grid"):
        measure_reachability(grid, truth, MODEL, (-1.0, 1.5))
    with pytest.raises(ValueError, match="is not free"):
        measure_reachability(grid, truth, MODEL, (4.5, 1.5))
    with pytest.raises(ValueError, match="truth shape"):
        measure_reachability(grid, np.zeros((3, 3), dtype=np.bool_), MODEL, (1.5, 1.5))


def test_the_enclosed_room_interior_is_one_sealed_reachable_component() -> None:
    """Every beam terminates on a wall, so the free interior is closed by occupied cells.

    No free cell of that interior touches an unknown one, which is what makes the
    frontier count zero here and nonzero in any scene the sensor has not finished.
    """
    scenario = enclosed_room(steps=6)
    trace = run_mapping(scenario)
    pose = scenario.trajectory.poses[0]
    report = measure_reachability(trace.grid, trace.truth, scenario.model, (pose.x, pose.y))
    assert report.components == 1
    assert report.reachable_fraction == 1.0
    assert report.stranded_cells == 0
    assert report.frontier_cells == 0
    assert report.false_free_cells == 0
    assert report.exposed_fraction == 0.0
    assert report.reachable_area == pytest.approx(
        report.reachable_cells * trace.grid.spec.cell_area
    )
