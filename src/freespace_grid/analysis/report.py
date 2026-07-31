"""Rendering measurement records as fixed width tables.

The example scripts print these tables and the README quotes them verbatim, so a
number in the documentation can be traced to the command that produced it.
"""

from __future__ import annotations

from collections.abc import Sequence

from freespace_grid.analysis.metrics import Agreement
from freespace_grid.analysis.smear import SmearReport

__all__ = ["agreement_table", "render_table", "smear_table"]


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a table with columns padded to their widest entry."""
    columns = list(headers)
    body = [list(row) for row in rows]
    for row in body:
        if len(row) != len(columns):
            raise ValueError(f"row has {len(row)} fields but there are {len(columns)} headers")
    widths = [
        max(len(columns[i]), *(len(row[i]) for row in body)) if body else len(columns[i])
        for i in range(len(columns))
    ]
    lines = [
        "  ".join(name.ljust(widths[i]) for i, name in enumerate(columns)),
        "  ".join("-" * widths[i] for i in range(len(columns))),
    ]
    lines.extend("  ".join(row[i].ljust(widths[i]) for i in range(len(columns))) for row in body)
    return "\n".join(lines)


def agreement_table(agreements: Sequence[Agreement], *, by: str = "decision_prob") -> str:
    """Render a sweep of :class:`Agreement` records.

    Args:
        agreements: The records to render, in sweep order.
        by: ``decision_prob`` labels rows by the decision threshold,
            ``spatial_tolerance`` labels them by the tolerance in cells.
    """
    if by == "decision_prob":
        label, values = "threshold", [f"{a.decision_prob:.2f}" for a in agreements]
    elif by == "spatial_tolerance":
        label, values = "tolerance", [f"{a.spatial_tolerance:d}" for a in agreements]
    else:
        raise ValueError(f"by must be decision_prob or spatial_tolerance, got {by!r}")

    headers = [label, "decided", "free agr", "occ agr", "balanced", "free as occ", "occ as free"]
    rows = [
        [
            values[index],
            f"{item.decided_fraction:.4f}",
            f"{item.free_agreement:.4f}",
            f"{item.occupied_agreement:.4f}",
            f"{item.balanced_agreement:.4f}",
            f"{item.free_called_occupied:d}",
            f"{item.occupied_called_free:d}",
        ]
        for index, item in enumerate(agreements)
    ]
    return render_table(headers, rows)


def smear_table(reports: Sequence[SmearReport], *, label_header: str = "case") -> str:
    """Render one row per moving obstacle case beside its parked control."""
    headers = [
        label_header,
        "swept (m)",
        "parked (m)",
        "moving (m)",
        "smear (m)",
        "smear/swept",
        "stale cells",
        "stale (m2)",
        "footprint",
        "found",
        "unknown",
        "called free",
        "peak returns",
    ]
    rows = [
        [
            report.label,
            f"{report.swept_distance:.2f}",
            f"{report.parked.extent_along:.2f}",
            f"{report.moving.extent_along:.2f}",
            f"{report.smear_length:.2f}",
            f"{report.smear_fraction_of_sweep:.3f}",
            f"{report.moving.stale_cells:d}",
            f"{report.moving.stale_area:.2f}",
            f"{report.moving.truth_cells:d}",
            f"{report.moving.detected_cells:d}",
            f"{report.moving.unknown_footprint_cells:d}",
            f"{report.moving.false_free_cells:d}",
            f"{report.moving.peak_returns_per_cell:d}",
        ]
        for report in reports
    ]
    return render_table(headers, rows)
