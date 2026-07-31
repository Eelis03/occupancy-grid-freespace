"""Tier two: a recorded reference run, compared with numeric tolerance.

What is pinned, and why only that
---------------------------------

A regression baseline is only useful if it fails when the code changes and passes when
only the machine changes. Quantities are therefore pinned according to how they are
produced.

Pinned exactly.
    Sweep count, retained beam count, and dropped beam count. Dropout is decided by
    comparing draws from a seeded PCG64 stream against a constant, which is bit exact
    on every platform, so these are integers that cannot move.

Pinned to a count tolerance of two cells or one part in five hundred, whichever is
larger.
    Range return counts, cell visit counts, observed cell counts, classification
    counts, and stale cell counts. These come from geometry, and the library routines
    behind them, ``cos``, ``sin``, ``sqrt`` and division, are specified to within one
    unit in the last place rather than bit for bit. A range that differs in its last
    bit moves a beam endpoint by about ten to the minus fifteen metres, which changes
    the cell it lands in only if that endpoint sits within the same distance of a cell
    boundary. The tolerance covers that case without hiding a real change, which would
    move these counts by far more.

Pinned to an absolute tolerance of two parts in a thousand.
    Agreement fractions, decided fractions, and the sharpness figures. These are ratios
    of the counts above, so the same argument applies with the denominators dividing
    the tolerance down.

Pinned to half a cell.
    Extents and lengths in metres. Their quantum is one cell, 0.2 metres here, so half
    a cell separates "identical" from "changed".

Asserted qualitatively rather than pinned.
    The orderings between the four map maintenance policies, the monotonicity of the
    two sweeps, and the direction of each dynamic obstacle result. These are the
    properties the run exists to demonstrate, they follow from the mathematics rather
    than from any particular arithmetic, and they must hold on any machine.

Nothing here comes from an iterative solve. Every quantity is a count, a classification,
or a ratio of the two, reached in a fixed number of steps. Regenerate the baseline with
``tests/reference.py`` if the scenario definitions change on purpose.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from tests.reference import REFERENCE_THRESHOLDS, REFERENCE_TOLERANCES, build_reference

REFERENCE_PATH = Path(__file__).parent / "data" / "reference_run.json"
COUNT_KEYS = (
    "hits",
    "max_range_beams",
    "placed_returns",
    "cell_visits",
)
EXACT_KEYS = ("scans", "beams", "dropped")
FRACTION_TOLERANCE = 2e-3
LENGTH_TOLERANCE = 0.1
MAPPING_RUNS = (
    "urban_block_world",
    "urban_block_ego_snap",
    "urban_block_ego_bilinear",
    "urban_block_ego_nearest",
    "enclosed_room",
)


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fresh() -> dict[str, Any]:
    return build_reference()


def assert_count(actual: int, expected: int, what: str) -> None:
    """Compare a count within two units or one part in five hundred."""
    allowed = max(2.0, 0.002 * abs(expected))
    assert abs(actual - expected) <= allowed, f"{what}: {actual} against recorded {expected}"


@pytest.mark.parametrize("run", MAPPING_RUNS)
def test_beam_bookkeeping_is_reproduced(
    run: str, baseline: dict[str, Any], fresh: dict[str, Any]
) -> None:
    recorded = baseline[run]["totals"]
    produced = fresh[run]["totals"]
    for key in EXACT_KEYS:
        assert produced[key] == recorded[key], f"{run}.{key}"
    for key in COUNT_KEYS:
        assert_count(produced[key], recorded[key], f"{run}.{key}")
    assert produced["hits"] + produced["max_range_beams"] == produced["beams"]


@pytest.mark.parametrize("run", MAPPING_RUNS)
def test_map_contents_are_reproduced(
    run: str, baseline: dict[str, Any], fresh: dict[str, Any]
) -> None:
    recorded = baseline[run]
    produced = fresh[run]
    assert produced["grid_cells"] == recorded["grid_cells"]
    assert produced["resamples"] == recorded["resamples"]
    assert produced["lossless_resamples"] == recorded["lossless_resamples"]
    assert_count(produced["observed_cells"], recorded["observed_cells"], f"{run}.observed")
    for state in ("free", "occupied", "unknown"):
        assert_count(
            produced["states"][state], recorded["states"][state], f"{run}.states.{state}"
        )
    assert sum(produced["states"].values()) == produced["grid_cells"]


@pytest.mark.parametrize("run", MAPPING_RUNS)
def test_agreement_and_sharpness_are_reproduced(
    run: str, baseline: dict[str, Any], fresh: dict[str, Any]
) -> None:
    for key in ("decided_fraction", "free_agreement", "occupied_agreement"):
        assert fresh[run]["agreement"][key] == pytest.approx(
            baseline[run]["agreement"][key], abs=FRACTION_TOLERANCE
        ), f"{run}.{key}"
    assert fresh[run]["sharpness"]["clamped_fraction"] == pytest.approx(
        baseline[run]["sharpness"]["clamped_fraction"], abs=FRACTION_TOLERANCE
    )
    assert fresh[run]["sharpness"]["edge_contrast"] == pytest.approx(
        baseline[run]["sharpness"]["edge_contrast"], rel=5e-3
    )


def test_threshold_sweep_is_reproduced(baseline: dict[str, Any], fresh: dict[str, Any]) -> None:
    recorded = baseline["urban_block_threshold_sweep"]
    produced = fresh["urban_block_threshold_sweep"]
    assert [row["decision_prob"] for row in produced] == list(REFERENCE_THRESHOLDS)
    assert len(produced) == len(recorded)
    for before, after in zip(recorded, produced, strict=True):
        assert after["decision_prob"] == before["decision_prob"]
        for key in ("decided_fraction", "free_agreement", "occupied_agreement"):
            assert after[key] == pytest.approx(before[key], abs=FRACTION_TOLERANCE)


def test_tolerance_sweep_is_reproduced(baseline: dict[str, Any], fresh: dict[str, Any]) -> None:
    recorded = baseline["urban_block_tolerance_sweep"]
    produced = fresh["urban_block_tolerance_sweep"]
    assert [row["spatial_tolerance"] for row in produced] == list(REFERENCE_TOLERANCES)
    for before, after in zip(recorded, produced, strict=True):
        assert after["occupied_agreement"] == pytest.approx(
            before["occupied_agreement"], abs=FRACTION_TOLERANCE
        )


@pytest.mark.parametrize("direction", ["approaching", "receding", "crossing"])
def test_dynamic_obstacle_measurements_are_reproduced(
    direction: str, baseline: dict[str, Any], fresh: dict[str, Any]
) -> None:
    key = f"dynamic_{direction}"
    recorded = baseline[key]
    produced = fresh[key]
    assert produced["swept_distance"] == pytest.approx(recorded["swept_distance"], abs=1e-9)
    assert produced["footprint_cells"] == recorded["footprint_cells"]
    assert produced["region_cells"] == recorded["region_cells"]
    for name in ("parked_extent_along", "moving_extent_along", "smear_length"):
        assert produced[name] == pytest.approx(recorded[name], abs=LENGTH_TOLERANCE), name
    for name in (
        "moving_occupied_cells",
        "parked_occupied_cells",
        "moving_stale_cells",
        "moving_detected_cells",
        "moving_unknown_footprint_cells",
        "moving_peak_returns_per_cell",
    ):
        assert_count(produced[name], recorded[name], f"{key}.{name}")
    assert produced["missed_footprint_fraction"] == pytest.approx(
        recorded["missed_footprint_fraction"], abs=FRACTION_TOLERANCE
    )


def test_map_maintenance_policies_keep_their_published_ordering(fresh: dict[str, Any]) -> None:
    """Qualitative claims that hold on any machine, stated because they are the point.

    Snapping the window to whole cells never interpolates, so every one of its shifts
    is lossless while neither fractional policy manages any. Interpolating instead
    costs occupied structure: bilinear averages the surface with its free surroundings
    and nearest neighbour displaces it by up to half a cell every frame, and thin
    surfaces do not survive either treatment as well as they survive an exact copy.
    """
    snap = fresh["urban_block_ego_snap"]
    bilinear = fresh["urban_block_ego_bilinear"]
    nearest = fresh["urban_block_ego_nearest"]

    assert snap["lossless_resamples"] == snap["resamples"]
    assert bilinear["lossless_resamples"] == 0
    assert nearest["lossless_resamples"] == 0

    assert snap["agreement"]["occupied_agreement"] > bilinear["agreement"]["occupied_agreement"]
    assert bilinear["agreement"]["occupied_agreement"] > nearest["agreement"]["occupied_agreement"]
    assert snap["sharpness"]["edge_contrast"] > bilinear["sharpness"]["edge_contrast"]
    assert snap["sharpness"]["edge_contrast"] > nearest["sharpness"]["edge_contrast"]


def test_sweeps_are_monotone(fresh: dict[str, Any]) -> None:
    """Raising the threshold cannot decide more cells, and slack cannot lose agreement."""
    decided = [row["decided_fraction"] for row in fresh["urban_block_threshold_sweep"]]
    assert all(b <= a + 1e-12 for a, b in pairwise(decided))
    occupied = [row["occupied_agreement"] for row in fresh["urban_block_tolerance_sweep"]]
    assert all(b >= a - 1e-12 for a, b in pairwise(occupied))


def test_enclosed_room_stays_a_clean_convergence_case(fresh: dict[str, Any]) -> None:
    """Every beam terminates on a wall, and the map converges without error."""
    room = fresh["enclosed_room"]
    assert room["totals"]["max_range_beams"] == 0
    assert room["totals"]["hits"] == room["totals"]["beams"]
    assert room["agreement"]["decided_fraction"] == 1.0
    assert room["agreement"]["free_agreement"] == 1.0
    assert room["agreement"]["occupied_agreement"] == 1.0
    assert room["sharpness"]["clamped_fraction"] == 1.0


def test_dynamic_results_point_the_way_the_geometry_says_they_must(
    fresh: dict[str, Any],
) -> None:
    """The three motions differ in whether the trail can be re-observed.

    A receding obstacle leaves its trail between itself and the sensor, so later sweeps
    correct it and the map ends up reporting the obstacle where it is. An obstacle
    crossing the line of sight moves into cells whose free evidence has already reached
    the clamp, and under the default parameters the filter cannot overturn that within
    the time the obstacle spends there, so most of its footprint is still called free.
    """
    assert fresh["dynamic_receding"]["missed_footprint_fraction"] == 0.0
    assert fresh["dynamic_crossing"]["missed_footprint_fraction"] > 0.9
    assert (
        fresh["dynamic_approaching"]["missed_footprint_fraction"]
        > fresh["dynamic_receding"]["missed_footprint_fraction"]
    )
    for direction in ("approaching", "receding", "crossing"):
        record = fresh[f"dynamic_{direction}"]
        assert record["moving_stale_cells"] <= record["moving_occupied_cells"]
        assert (
            record["moving_detected_cells"]
            + record["moving_unknown_footprint_cells"]
            + round(record["missed_footprint_fraction"] * record["footprint_cells"])
            == record["footprint_cells"]
        )
    # Motion across the line of sight keeps a cell on the near surface of the obstacle
    # for longer than motion along it, so it deposits more evidence per cell.
    assert (
        fresh["dynamic_crossing"]["moving_peak_returns_per_cell"]
        > fresh["dynamic_approaching"]["moving_peak_returns_per_cell"]
    )
