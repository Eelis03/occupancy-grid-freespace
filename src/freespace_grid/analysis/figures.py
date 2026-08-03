"""Figures. The only module in the package that imports matplotlib.

Two audiences are served here and they want different things. The example scripts want
a large figure that can be zoomed into, so they take the defaults. The three figures
tracked in ``docs/figures`` have to fit a byte budget and are read at the width of a
README column, so every entry point takes an explicit size, resolution and crop rather
than hard coding one. A three way decision map is a flat colour image and survives a
modest resolution intact, which is what makes the budget affordable without a
compression dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from freespace_grid.analysis.metrics import Agreement
from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.model.occupancy import CellState, OccupancyGrid
from freespace_grid.model.typing import BoolArray

if TYPE_CHECKING:  # pragma: no cover - import only for annotations
    from freespace_grid.pipeline.runner import MappingTrace

__all__ = [
    "Panel",
    "plot_decision_map",
    "plot_map",
    "plot_state_panels",
    "plot_threshold_sweep",
]

_STATE_COLORS = ("#9aa0a6", "#f5f5f5", "#1a1a1a")
_TRUTH_COLOR = "#d1495b"
_REGION_COLOR = "#2a9d8f"

Crop = tuple[float, float, float, float]
Panel = tuple[str, "MappingTrace", BoolArray | None]


def _state_image(grid: OccupancyGrid, model: LogOddsModel) -> np.ndarray:
    states = grid.classify(model)
    image = np.zeros((*grid.spec.shape, 3), dtype=np.float64)
    for code, color in zip(
        (CellState.UNKNOWN, CellState.FREE, CellState.OCCUPIED), _STATE_COLORS, strict=True
    ):
        rgb = matplotlib.colors.to_rgb(color)
        image[states == int(code)] = rgb
    return image


def plot_map(
    trace: MappingTrace,
    model: LogOddsModel,
    path: Path,
    *,
    title: str = "Occupancy map",
) -> Path:
    """Write a two panel figure: posterior probability, and the three way decision."""
    spec = trace.grid.spec
    extent = spec.extent
    width = max(6.0, 12.0 * spec.cols / max(spec.cols, spec.rows))
    height = max(3.0, 12.0 * spec.rows / max(spec.cols, spec.rows))
    fig, axes = plt.subplots(2, 1, figsize=(width, 2.0 * height + 1.2), constrained_layout=True)

    probability = trace.grid.probability()
    first = axes[0]
    image = first.imshow(
        probability,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    fig.colorbar(image, ax=first, label="posterior occupancy probability", shrink=0.85)
    first.set_title(f"{title}: posterior")

    second = axes[1]
    second.imshow(
        _state_image(trace.grid, model),
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    second.contour(
        np.asarray(trace.truth, dtype=np.float64),
        levels=[0.5],
        colors="#d1495b",
        linewidths=0.8,
        origin="lower",
        extent=extent,
    )
    second.set_title(
        f"{title}: decision at p >= {model.decision_prob:.2f}, "
        "ground truth outline in red"
    )

    xs = [step.x for step in trace.steps]
    ys = [step.y for step in trace.steps]
    for axis in axes:
        axis.plot(xs, ys, color="#2a9d8f", linewidth=1.4, label="vehicle path")
        axis.plot(xs[0], ys[0], marker="o", color="#2a9d8f", markersize=4)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.legend(loc="upper right", fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_threshold_sweep(
    agreements: tuple[Agreement, ...],
    path: Path,
    *,
    title: str = "Decision threshold sweep",
) -> Path:
    """Write the decided fraction and the two class agreements against the threshold."""
    thresholds = [a.decision_prob for a in agreements]
    fig, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    axis.plot(thresholds, [a.decided_fraction for a in agreements], marker="o", label="decided")
    axis.plot(
        thresholds, [a.free_agreement for a in agreements], marker="s", label="free agreement"
    )
    axis.plot(
        thresholds,
        [a.occupied_agreement for a in agreements],
        marker="^",
        label="occupied agreement",
    )
    axis.set_xlabel("decision threshold, probability")
    axis.set_ylabel("fraction")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.3)
    axis.legend()
    axis.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _draw_state_panel(
    axis: Axes,
    label: str,
    trace: MappingTrace,
    region: BoolArray | None,
    model: LogOddsModel,
    *,
    crop: Crop | None,
    fontsize: float,
) -> None:
    """Draw one three way decision map with the ground truth outlined on top of it."""
    extent = trace.grid.spec.extent
    axis.imshow(
        _state_image(trace.grid, model),
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    if region is not None:
        axis.contour(
            np.asarray(region, dtype=np.float64),
            levels=[0.5],
            colors=_REGION_COLOR,
            linewidths=0.8,
            origin="lower",
            extent=extent,
        )
    axis.contour(
        np.asarray(trace.truth, dtype=np.float64),
        levels=[0.5],
        colors=_TRUTH_COLOR,
        linewidths=0.8,
        origin="lower",
        extent=extent,
    )
    axis.plot(trace.steps[0].x, trace.steps[0].y, marker="o", color="#e9c46a", markersize=5)
    axis.set_title(label, fontsize=fontsize)
    axis.set_xlabel("x (m)", fontsize=fontsize)
    axis.set_ylabel("y (m)", fontsize=fontsize)
    axis.tick_params(labelsize=fontsize - 1.0)
    if crop is not None:
        axis.set_xlim(crop[0], crop[1])
        axis.set_ylim(crop[2], crop[3])


def plot_state_panels(
    panels: tuple[Panel, ...],
    model: LogOddsModel,
    path: Path,
    *,
    title: str = "Static world assumption under a moving obstacle",
    layout: Literal["rows", "columns"] = "rows",
    width: float = 11.0,
    panel_height: float = 2.6,
    dpi: int = 140,
    crop: Crop | None = None,
    fontsize: float = 10.0,
) -> Path:
    """Write one three way decision map per panel, ground truth outlined in red.

    Args:
        panels: ``(label, trace, region)`` triples. A region of ``None`` draws no
            region outline, which is what a panel that is about the whole map wants.
        model: Supplies the decision band.
        path: Destination file.
        title: Figure level title.
        layout: ``rows`` stacks the panels, ``columns`` places them side by side. Side
            by side is the right choice for a before and after pair, because the eye
            compares horizontally displaced images far more readily than stacked ones.
        width: Figure width in inches.
        panel_height: Height of one panel in inches.
        dpi: Output resolution.
        crop: Optional ``(x_min, x_max, y_min, y_max)`` window in metres. Applied to
            every panel, so the panels stay comparable.
        fontsize: Base font size for titles and axis labels.
    """
    count = len(panels)
    if count == 0:
        raise ValueError("at least one panel is required")
    rows, cols = (count, 1) if layout == "rows" else (1, count)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(width, panel_height * rows + 1.0),
        constrained_layout=True,
    )
    for axis, (label, trace, region) in zip(np.atleast_1d(axes), panels, strict=True):
        _draw_state_panel(
            axis, label, trace, region, model, crop=crop, fontsize=fontsize
        )
    fig.suptitle(title, fontsize=fontsize + 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_decision_map(
    trace: MappingTrace,
    model: LogOddsModel,
    path: Path,
    *,
    title: str = "Occupancy map",
    width: float = 8.0,
    dpi: int = 110,
    crop: Crop | None = None,
    fontsize: float = 9.0,
) -> Path:
    """Write a single panel decision map, sized from the aspect ratio of the grid."""
    spec = trace.grid.spec
    aspect = (spec.rows * spec.resolution) / (spec.cols * spec.resolution)
    height = max(2.0, width * aspect) + 0.9
    fig, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    _draw_state_panel(axis, title, trace, None, model, crop=crop, fontsize=fontsize)
    axis.plot(
        [step.x for step in trace.steps],
        [step.y for step in trace.steps],
        color=_REGION_COLOR,
        linewidth=1.2,
        label="vehicle path",
    )
    axis.legend(loc="upper right", fontsize=fontsize - 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
