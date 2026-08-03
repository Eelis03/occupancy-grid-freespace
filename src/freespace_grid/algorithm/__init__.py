"""Mapping mathematics: ray traversal, inverse sensor model, accumulation. No plotting."""

from __future__ import annotations

from freespace_grid.algorithm.accumulation import (
    Accumulator,
    ShiftPolicy,
    is_whole_cell_shift,
    resample_grid,
)
from freespace_grid.algorithm.inverse_sensor import ScanUpdate, apply_scan, scan_update
from freespace_grid.algorithm.raycast import Traversal, traverse_rays
from freespace_grid.algorithm.scan_match import (
    MatchResult,
    SearchWindow,
    likelihood_field,
    match_scan,
    scan_body_points,
)

__all__ = [
    "Accumulator",
    "MatchResult",
    "ScanUpdate",
    "SearchWindow",
    "ShiftPolicy",
    "Traversal",
    "apply_scan",
    "is_whole_cell_shift",
    "likelihood_field",
    "match_scan",
    "resample_grid",
    "scan_body_points",
    "scan_update",
    "traverse_rays",
]
