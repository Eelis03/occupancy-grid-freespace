"""Figures. The only module in the package that imports matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from freespace_grid.analysis.metrics import Agreement
from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.model.occupancy import CellState, OccupancyGrid
from freespace_grid.model.typing import BoolArray

if TYPE_CHECKING:  # pragma: no cover - import only for annotations
    from freespace_grid.pipeline.runner import MappingTrace

__all__ = [
    "plot_map",
    "plot_smear_panels",
    "plot_threshold_sweep",
]

_STATE_COLORS = ("#9aa0a6", "#f5f5f5", "#1a1a1a")


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


def plot_smear_panels(
    panels: tuple[tuple[str, MappingTrace, BoolArray], ...],
    model: LogOddsModel,
    path: Path,
    *,
    title: str = "Static world assumption under a moving obstacle",
) -> Path:
    """Write one decision map per panel, with the region of interest outlined."""
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(11.0, 2.6 * len(panels) + 1.0), constrained_layout=True
    )
    axis_list = np.atleast_1d(axes)
    for axis, (label, trace, region) in zip(axis_list, panels, strict=True):
        extent = trace.grid.spec.extent
        axis.imshow(
            _state_image(trace.grid, model),
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        axis.contour(
            np.asarray(region, dtype=np.float64),
            levels=[0.5],
            colors="#2a9d8f",
            linewidths=0.8,
            origin="lower",
            extent=extent,
        )
        axis.contour(
            np.asarray(trace.truth, dtype=np.float64),
            levels=[0.5],
            colors="#d1495b",
            linewidths=0.8,
            origin="lower",
            extent=extent,
        )
        axis.plot(trace.steps[0].x, trace.steps[0].y, marker="o", color="#e9c46a", markersize=6)
        axis.set_title(label, fontsize=10)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
