"""Dead reckoning with a drifting odometry model.

Every pose this simulator produces is exact, which is not a property any vehicle has.
Odometry is integrated from wheel or inertial measurements and each measurement carries
an error, so the estimate a mapper actually receives is the true motion plus a random
walk that never gets smaller on its own.

The corruption is applied to the *relative* motion between consecutive poses, in the
body frame, and not to the absolute pose. That is what makes the error accumulate:
adding independent noise to each absolute pose would produce a jittering estimate that
stays near the truth, while adding it to each increment produces a slow drift that a
mapper cannot distinguish from a change in the world. It is the second that damages a
map, so it is the second that is modelled. The three coefficients follow the odometry
motion model of Thrun, Burgard and Fox, chapter 5, reduced to the terms that matter at
this scale: translation error grows with distance travelled, and heading error grows
with both distance travelled and angle turned.

The first pose is never corrupted. The map frame is whatever the vehicle believes its
starting pose to be, so an error there is a change of coordinates rather than a
distortion, and every quantity in the results would inherit it for no useful reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from freespace_grid.model.transform import Pose2D, compose, inverse, wrap_angle
from freespace_grid.pipeline.trajectory import Trajectory

__all__ = [
    "OdometryNoise",
    "dead_reckon",
    "noisy_increments",
    "pose_error",
    "relative_motion",
]


def relative_motion(previous: Pose2D, current: Pose2D) -> Pose2D:
    """Return the motion from ``previous`` to ``current``, in the body frame of ``previous``."""
    return compose(inverse(previous), current)


def pose_error(estimate: Pose2D, truth: Pose2D) -> tuple[float, float]:
    """Return the ``(position, heading)`` error of ``estimate``, in metres and radians."""
    return (
        math.hypot(estimate.x - truth.x, estimate.y - truth.y),
        abs(wrap_angle(estimate.theta - truth.theta)),
    )


@dataclass(frozen=True, slots=True)
class OdometryNoise:
    """Gaussian corruption of one body frame motion increment.

    Args:
        translation_std_per_m: Standard deviation of the error added to each body frame
            translation component, per metre travelled in that increment.
        heading_std_per_m: Standard deviation of the heading error, per metre travelled.
            This is the term that dominates a long straight run: a heading error is
            carried by every later increment, so it displaces the estimate by an amount
            that grows with the distance still to be covered.
        heading_std_per_rad: Standard deviation of the heading error, per radian turned.

    All three default to zero, which makes :func:`dead_reckon` reproduce the trajectory
    exactly and lets the noiseless case share one code path with the noisy one rather
    than branching around it.
    """

    translation_std_per_m: float = 0.0
    heading_std_per_m: float = 0.0
    heading_std_per_rad: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("translation_std_per_m", self.translation_std_per_m),
            ("heading_std_per_m", self.heading_std_per_m),
            ("heading_std_per_rad", self.heading_std_per_rad),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must not be negative, got {value}")

    @property
    def is_exact(self) -> bool:
        """True when no coefficient is set, so dead reckoning reproduces the truth."""
        return (
            self.translation_std_per_m == 0.0
            and self.heading_std_per_m == 0.0
            and self.heading_std_per_rad == 0.0
        )

    def scaled(self, factor: float) -> OdometryNoise:
        """Return the same model with every coefficient multiplied by ``factor``."""
        if factor < 0.0:
            raise ValueError(f"factor must not be negative, got {factor}")
        return OdometryNoise(
            translation_std_per_m=self.translation_std_per_m * factor,
            heading_std_per_m=self.heading_std_per_m * factor,
            heading_std_per_rad=self.heading_std_per_rad * factor,
        )

    def corrupt(self, delta: Pose2D, rng: np.random.Generator) -> Pose2D:
        """Return ``delta`` with one draw of odometry error added.

        Three variates are drawn per call whatever the coefficients are, including when
        they are all zero, so the length of the random stream does not depend on the
        noise level and two runs at different levels stay comparable increment by
        increment.
        """
        distance = math.hypot(delta.x, delta.y)
        turn = abs(delta.theta)
        translation_std = self.translation_std_per_m * distance
        heading_std = self.heading_std_per_m * distance + self.heading_std_per_rad * turn
        draws = rng.normal(0.0, 1.0, size=3)
        return Pose2D(
            x=delta.x + float(draws[0]) * translation_std,
            y=delta.y + float(draws[1]) * translation_std,
            theta=wrap_angle(delta.theta + float(draws[2]) * heading_std),
        )


def noisy_increments(
    trajectory: Trajectory, noise: OdometryNoise, rng: np.random.Generator
) -> tuple[Pose2D, ...]:
    """Return one corrupted body frame increment per pose transition.

    The increments are produced ahead of the run rather than inside it so that a
    corrected run and an uncorrected one receive exactly the same odometry, which is
    what makes the comparison between them a comparison of the correction alone.
    """
    return tuple(
        noise.corrupt(relative_motion(previous, current), rng)
        for previous, current in pairwise(trajectory.poses)
    )


def dead_reckon(increments: tuple[Pose2D, ...], start: Pose2D) -> tuple[Pose2D, ...]:
    """Chain ``increments`` onto ``start`` to give the pose sequence the vehicle believes."""
    estimates = [start.normalized()]
    for increment in increments:
        estimates.append(compose(estimates[-1], increment).normalized())
    return tuple(estimates)
