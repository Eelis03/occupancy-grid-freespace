"""Correlative scan to map matching, the pose correction for a drifting odometry.

An occupancy map is only as good as the poses the sweeps are placed at. Given a map
built from earlier sweeps and an odometry prediction of where the vehicle now is, the
correction searched for here is the rigid transform that puts the current sweep's range
returns on top of the occupied structure the map already holds. This is the correlative
matcher of Olson, used against a map rather than against a previous scan, and it is the
front end of Hector SLAM in all but the gradient refinement.

Three choices make it work at this scale.

The objective is a likelihood field, not the log odds array. Scoring the raw log odds
would reward a candidate that pushes the sweep into unobserved territory, because an
unknown cell holds the prior and a correctly matched free cell holds a large negative
number, so the best score would be obtained by leaving the map. The field used instead
is zero wherever the map is free or unknown and positive only where it holds occupied
evidence, so a candidate is rewarded for agreement and never for absence.

The field is blurred by a Gaussian one cell wide. Without it the objective is a sum of
indicator functions and is flat almost everywhere: a candidate half a cell from the
optimum scores the same as one ten cells away whenever both miss every surface, and the
search has no gradient to follow. The blur is the standard remedy and it also absorbs
the one cell disagreement between where a range return is attributed and where the
surface actually lies.

The search is coarse to fine. A single pass fine enough to resolve a quarter of a cell
over the whole window would cost tens of thousands of candidate evaluations; three
passes, each halving the radius and the step around the previous winner, reach the same
resolution for a few hundred. The refinement can in principle settle into a local
optimum that the full search would have rejected, which is the price, and the blur is
what keeps the coarse pass wide enough for that to be rare.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.model.occupancy import OccupancyGrid
from freespace_grid.model.transform import Pose2D, wrap_angle
from freespace_grid.model.typing import BoolArray, FloatArray

__all__ = ["MatchResult", "SearchWindow", "likelihood_field", "match_scan", "scan_body_points"]


@dataclass(frozen=True, slots=True)
class SearchWindow:
    """Extent and resolution of the pose search.

    Args:
        translation_radius: Half width of the translation search, in metres, applied to
            both axes independently.
        heading_radius: Half width of the heading search, in radians.
        translation_step: Spacing of the translation candidates on the first pass, in
            metres. One cell is the natural choice.
        heading_step: Spacing of the heading candidates on the first pass, in radians.
        refinements: Number of extra passes. Each one halves the radius and the step
            around the winner of the pass before it, so ``refinements`` of two resolves
            a quarter of the initial step.
    """

    translation_radius: float = 0.8
    heading_radius: float = 0.035
    translation_step: float = 0.2
    heading_step: float = 0.00873
    refinements: int = 2

    def __post_init__(self) -> None:
        if self.translation_step <= 0.0 or self.heading_step <= 0.0:
            raise ValueError("search steps must be positive")
        if self.translation_radius < 0.0 or self.heading_radius < 0.0:
            raise ValueError("search radii must not be negative")
        if self.refinements < 0:
            raise ValueError(f"refinements must not be negative, got {self.refinements}")


@dataclass(frozen=True, slots=True)
class MatchResult:
    """The outcome of matching one sweep against the map.

    Attributes:
        pose: The corrected sensor pose.
        score: Objective value at ``pose``.
        predicted_score: Objective value at the pose the odometry predicted. The
            correction is worth having only when ``score`` exceeds it.
        translation_correction: Distance from the predicted pose to ``pose``, in metres.
        heading_correction: Heading change applied, in radians.
        evaluations: Candidate poses scored, summed over all passes.
        points: Range returns used, after subsampling.
    """

    pose: Pose2D
    score: float
    predicted_score: float
    translation_correction: float
    heading_correction: float
    evaluations: int
    points: int


def scan_body_points(
    angles: FloatArray, ranges: FloatArray, is_hit: BoolArray, *, stride: int = 1
) -> FloatArray:
    """Return the range returns of one sweep as ``(n, 2)`` points in the sensor frame.

    Beams that reached the range limit are dropped. They carry free space evidence, not
    a surface, and there is nothing in the map for them to be aligned against.

    Args:
        angles: Beam bearings in the sensor frame, radians.
        ranges: Reported ranges.
        is_hit: True where the beam carries a range return.
        stride: Keep every ``stride``-th return. Matching cost is linear in the number
            of points and a sweep of several hundred returns is far more than the three
            degrees of freedom need.
    """
    if stride < 1:
        raise ValueError(f"stride must be at least one, got {stride}")
    hit = np.asarray(is_hit, dtype=np.bool_)
    bearings = np.asarray(angles, dtype=np.float64)[hit][::stride]
    distances = np.asarray(ranges, dtype=np.float64)[hit][::stride]
    return np.stack((distances * np.cos(bearings), distances * np.sin(bearings)), axis=1)


def likelihood_field(
    grid: OccupancyGrid, model: LogOddsModel, *, blur_cells: float = 1.0
) -> FloatArray:
    """Return the field a candidate pose is scored against.

    The value is the occupancy evidence above the prior, rescaled so that a cell at the
    occupied clamp contributes one and a cell at or below the prior contributes nothing,
    then blurred by a Gaussian of ``blur_cells`` standard deviation.
    """
    if blur_cells < 0.0:
        raise ValueError(f"blur_cells must not be negative, got {blur_cells}")
    above = (grid.log_odds - model.l_prior) / (model.l_max - model.l_prior)
    evidence = np.clip(above, 0.0, 1.0)
    if blur_cells == 0.0:
        return np.ascontiguousarray(evidence, dtype=np.float64)
    blurred = ndimage.gaussian_filter(evidence, sigma=blur_cells, mode="constant", cval=0.0)
    return np.ascontiguousarray(blurred, dtype=np.float64)


def _score(
    field: FloatArray,
    grid: OccupancyGrid,
    points: FloatArray,
    candidates: FloatArray,
) -> FloatArray:
    """Score every candidate pose in ``candidates``, shape ``(c, 3)``, against ``field``."""
    spec = grid.spec
    cos = np.cos(candidates[:, 2])[:, None]
    sin = np.sin(candidates[:, 2])[:, None]
    body_x = points[None, :, 0]
    body_y = points[None, :, 1]
    world_x = cos * body_x - sin * body_y + candidates[:, 0][:, None]
    world_y = sin * body_x + cos * body_y + candidates[:, 1][:, None]

    col = (world_x - spec.origin_x) / spec.resolution - 0.5
    row = (world_y - spec.origin_y) / spec.resolution - 0.5
    sampled = ndimage.map_coordinates(
        field,
        np.stack((row.ravel(), col.ravel()), axis=0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return np.asarray(sampled.reshape(world_x.shape).sum(axis=1), dtype=np.float64)


def _candidates(
    center: Pose2D, translation_radius: float, heading_radius: float, steps: tuple[float, float]
) -> FloatArray:
    """Return the grid of candidate poses around ``center``, shape ``(c, 3)``."""
    translation_step, heading_step = steps
    span_xy = np.arange(
        -translation_radius, translation_radius + 0.5 * translation_step, translation_step
    )
    span_h = np.arange(-heading_radius, heading_radius + 0.5 * heading_step, heading_step)
    if span_xy.size == 0:
        span_xy = np.zeros(1, dtype=np.float64)
    if span_h.size == 0:
        span_h = np.zeros(1, dtype=np.float64)
    d_x, d_y, d_h = np.meshgrid(span_xy, span_xy, span_h, indexing="ij")
    return np.stack(
        (
            center.x + d_x.ravel(),
            center.y + d_y.ravel(),
            center.theta + d_h.ravel(),
        ),
        axis=1,
    )


def match_scan(
    grid: OccupancyGrid,
    model: LogOddsModel,
    points: FloatArray,
    predicted: Pose2D,
    *,
    window: SearchWindow | None = None,
    blur_cells: float = 1.0,
) -> MatchResult:
    """Return the pose that best aligns ``points`` with the occupied evidence in ``grid``.

    Args:
        grid: The map built from every earlier sweep.
        model: Supplies the prior and the occupied clamp used to normalise the field.
        points: Range returns in the sensor frame, shape ``(n, 2)``, as produced by
            :func:`scan_body_points`.
        predicted: The pose odometry believes the sensor is at.
        window: Search extent and resolution.
        blur_cells: Standard deviation of the Gaussian applied to the likelihood field,
            in cells.

    Returns:
        A :class:`MatchResult`. When the map holds no occupied evidence yet, every
        candidate scores zero and the predicted pose is returned unchanged, which is the
        correct behaviour for the first sweep of a run.
    """
    search = window if window is not None else SearchWindow()
    field = likelihood_field(grid, model, blur_cells=blur_cells)
    predicted_score = float(
        _score(field, grid, points, np.array([[predicted.x, predicted.y, predicted.theta]]))[0]
    )

    best = predicted
    best_score = predicted_score
    evaluations = 0
    translation_radius = search.translation_radius
    heading_radius = search.heading_radius
    translation_step = search.translation_step
    heading_step = search.heading_step

    for _ in range(search.refinements + 1):
        grid_of_poses = _candidates(
            best, translation_radius, heading_radius, (translation_step, heading_step)
        )
        scores = _score(field, grid, points, grid_of_poses)
        evaluations += int(grid_of_poses.shape[0])
        pick = _argbest(scores, grid_of_poses, best, translation_step, heading_step)
        if scores[pick] > best_score:
            best_score = float(scores[pick])
            best = Pose2D(
                x=float(grid_of_poses[pick, 0]),
                y=float(grid_of_poses[pick, 1]),
                theta=wrap_angle(float(grid_of_poses[pick, 2])),
            )
        translation_radius *= 0.5
        heading_radius *= 0.5
        translation_step *= 0.5
        heading_step *= 0.5

    return MatchResult(
        pose=best,
        score=best_score,
        predicted_score=predicted_score,
        translation_correction=math.hypot(best.x - predicted.x, best.y - predicted.y),
        heading_correction=wrap_angle(best.theta - predicted.theta),
        evaluations=evaluations,
        points=int(points.shape[0]),
    )


def _argbest(
    scores: FloatArray,
    candidates: FloatArray,
    center: Pose2D,
    translation_step: float,
    heading_step: float,
) -> int:
    """Index of the best candidate, ties broken towards the smallest correction.

    Ties are common: the field is zero over most of the map, so any two candidates that
    put every return on unobserved ground score identically. Breaking towards the centre
    keeps the matcher from inventing a correction it has no evidence for, and makes the
    result independent of the order the candidate grid happens to be built in.
    """
    best = float(np.max(scores))
    tied = np.flatnonzero(scores >= best - 1e-12)
    d_x = (candidates[tied, 0] - center.x) / translation_step
    d_y = (candidates[tied, 1] - center.y) / translation_step
    d_h = (candidates[tied, 2] - center.theta) / heading_step
    cost = d_x * d_x + d_y * d_y + d_h * d_h
    return int(tied[int(np.argmin(cost))])
