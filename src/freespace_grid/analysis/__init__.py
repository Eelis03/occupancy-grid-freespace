"""Scoring, failure measurement, and figures."""

from __future__ import annotations

from freespace_grid.analysis.metrics import (
    Agreement,
    score_grid,
    threshold_sweep,
    tolerance_sweep,
)
from freespace_grid.analysis.reachability import (
    Reachability,
    measure_reachability,
    reachable_free,
)
from freespace_grid.analysis.report import agreement_table, render_table, smear_table
from freespace_grid.analysis.sharpness import Sharpness, boundary_band, measure_sharpness
from freespace_grid.analysis.smear import (
    SmearCase,
    SmearMetrics,
    SmearReport,
    compare_smear,
    measure_smear,
    path_region,
    run_smear_case,
)

__all__ = [
    "Agreement",
    "Reachability",
    "Sharpness",
    "SmearCase",
    "SmearMetrics",
    "SmearReport",
    "agreement_table",
    "boundary_band",
    "compare_smear",
    "measure_reachability",
    "measure_sharpness",
    "measure_smear",
    "path_region",
    "reachable_free",
    "render_table",
    "run_smear_case",
    "score_grid",
    "smear_table",
    "threshold_sweep",
    "tolerance_sweep",
]
