"""Construction of the recorded reference run used by the tier two regression test.

Both the stored baseline and the freshly computed comparison come from this one
function, so the regression test cannot drift away from what was recorded.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from freespace_grid.analysis.metrics import score_grid, threshold_sweep, tolerance_sweep
from freespace_grid.analysis.sharpness import measure_sharpness
from freespace_grid.analysis.smear import run_smear_case
from freespace_grid.model.occupancy import CellState
from freespace_grid.pipeline.runner import MappingTrace, RunConfig, run_mapping
from freespace_grid.pipeline.scenarios import DYNAMIC_DIRECTIONS, enclosed_room, urban_block

__all__ = ["REFERENCE_THRESHOLDS", "REFERENCE_TOLERANCES", "build_reference"]

REFERENCE_THRESHOLDS: tuple[float, ...] = (0.55, 0.62, 0.65, 0.68, 0.72, 0.78, 0.82, 0.86)
REFERENCE_TOLERANCES: tuple[int, ...] = (0, 1, 2, 3)


def _state_counts(trace: MappingTrace, model: Any) -> dict[str, int]:
    states = trace.grid.classify(model)
    return {
        "free": int(np.count_nonzero(states == int(CellState.FREE))),
        "occupied": int(np.count_nonzero(states == int(CellState.OCCUPIED))),
        "unknown": int(np.count_nonzero(states == int(CellState.UNKNOWN))),
    }


def _mapping_record(trace: MappingTrace, model: Any) -> dict[str, Any]:
    observed = trace.observed_mask
    on_observed = score_grid(trace.grid, trace.truth, model, region=observed)
    sharpness = measure_sharpness(trace.grid, trace.truth, model, region=observed)
    return {
        "totals": trace.totals(),
        "observed_cells": int(np.count_nonzero(observed)),
        "grid_cells": int(trace.grid.spec.size),
        "resamples": trace.resamples,
        "lossless_resamples": trace.lossless_resamples,
        "states": _state_counts(trace, model),
        "agreement": {
            "decided_fraction": on_observed.decided_fraction,
            "free_agreement": on_observed.free_agreement,
            "occupied_agreement": on_observed.occupied_agreement,
            "free_called_occupied": on_observed.free_called_occupied,
            "occupied_called_free": on_observed.occupied_called_free,
        },
        "sharpness": {
            "clamped_fraction": sharpness.clamped_fraction,
            "edge_contrast": sharpness.edge_contrast,
            "band_cells": sharpness.band_cells,
        },
    }


def build_reference() -> dict[str, Any]:
    """Compute every quantity the regression test pins."""
    static = urban_block()
    world = run_mapping(static, RunConfig(frame="world"))
    observed = world.observed_mask

    record: dict[str, Any] = {
        "urban_block_world": _mapping_record(world, static.model),
        "urban_block_threshold_sweep": [
            {
                "decision_prob": item.decision_prob,
                "decided_fraction": item.decided_fraction,
                "free_agreement": item.free_agreement,
                "occupied_agreement": item.occupied_agreement,
            }
            for item in threshold_sweep(
                world.grid, world.truth, static.model, REFERENCE_THRESHOLDS, region=observed
            )
        ],
        "urban_block_tolerance_sweep": [
            {
                "spatial_tolerance": item.spatial_tolerance,
                "occupied_agreement": item.occupied_agreement,
            }
            for item in tolerance_sweep(
                world.grid, world.truth, static.model, REFERENCE_TOLERANCES, region=observed
            )
        ],
    }

    for policy in ("snap", "bilinear", "nearest"):
        trace = run_mapping(static, RunConfig(frame="ego", shift_policy=policy))
        record[f"urban_block_ego_{policy}"] = _mapping_record(trace, static.model)

    room = enclosed_room(steps=12)
    record["enclosed_room"] = _mapping_record(run_mapping(room), room.model)

    for direction in DYNAMIC_DIRECTIONS:
        case = run_smear_case(direction)
        record[f"dynamic_{direction}"] = {
            "swept_distance": case.report.swept_distance,
            "parked_extent_along": case.report.parked.extent_along,
            "moving_extent_along": case.report.moving.extent_along,
            "smear_length": case.report.smear_length,
            "moving_occupied_cells": case.report.moving.occupied_cells,
            "parked_occupied_cells": case.report.parked.occupied_cells,
            "moving_stale_cells": case.report.moving.stale_cells,
            "moving_detected_cells": case.report.moving.detected_cells,
            "moving_unknown_footprint_cells": case.report.moving.unknown_footprint_cells,
            "moving_peak_returns_per_cell": case.report.moving.peak_returns_per_cell,
            "footprint_cells": case.report.moving.truth_cells,
            "missed_footprint_fraction": case.report.missed_footprint_fraction,
            "region_cells": case.report.moving.roi_cells,
        }

    return record
