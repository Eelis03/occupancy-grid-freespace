"""A planar lidar simulator.

The simulator produces the four things a mapping pipeline has to cope with: a range
return with noise, a beam that reaches its range limit without a return, a beam that
is dropped entirely, and a minimum range below which the sensor reports nothing
useful. The three failure modes are distinct and the mapper treats them differently,
so they are represented distinctly rather than folded into one sentinel value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from freespace_grid.model.transform import Pose2D
from freespace_grid.model.typing import BoolArray, FloatArray
from freespace_grid.pipeline.scene import Scene, ray_ranges

__all__ = ["LidarSpec", "Scan", "simulate_scan"]


@dataclass(frozen=True, slots=True)
class LidarSpec:
    """Configuration of the simulated sensor.

    Args:
        max_range: Range limit, in metres. A beam meeting no surface inside this
            distance reports the limit and is flagged as carrying no return.
        min_range: Distance below which a return is not reported. Returns closer than
            this are clamped to it.
        angular_resolution_deg: Angular spacing between beams, in degrees.
        field_of_view_deg: Total angular span, in degrees. 360 gives a full sweep.
        range_noise_std: Standard deviation of the additive Gaussian range error, in
            metres. Applied only to beams that carry a return.
        dropout_prob: Probability that a beam is lost entirely and reports nothing.
    """

    max_range: float = 30.0
    min_range: float = 0.25
    angular_resolution_deg: float = 0.5
    field_of_view_deg: float = 360.0
    range_noise_std: float = 0.03
    dropout_prob: float = 0.02

    def __post_init__(self) -> None:
        if self.max_range <= self.min_range:
            raise ValueError(f"max_range must exceed min_range, got {self.max_range}")
        if self.angular_resolution_deg <= 0.0:
            raise ValueError(
                f"angular_resolution_deg must be positive, got {self.angular_resolution_deg}"
            )
        if not 0.0 < self.field_of_view_deg <= 360.0:
            raise ValueError(
                f"field_of_view_deg must lie in (0, 360], got {self.field_of_view_deg}"
            )
        if self.range_noise_std < 0.0:
            raise ValueError(f"range_noise_std must not be negative, got {self.range_noise_std}")
        if not 0.0 <= self.dropout_prob < 1.0:
            raise ValueError(f"dropout_prob must lie in [0, 1), got {self.dropout_prob}")

    @property
    def beam_count(self) -> int:
        """Number of beams in one sweep."""
        return round(self.field_of_view_deg / self.angular_resolution_deg)

    def beam_angles(self) -> FloatArray:
        """Beam bearings in the sensor frame, in radians, evenly spaced from the centre."""
        count = self.beam_count
        step = math.radians(self.angular_resolution_deg)
        offsets = np.arange(count, dtype=np.float64) - 0.5 * (count - 1)
        return offsets * step


@dataclass(frozen=True, slots=True)
class Scan:
    """One sweep of the simulated sensor, with dropped beams already removed.

    Attributes:
        pose: Sensor pose in the world frame at the time of the sweep.
        angles: Beam bearings in the sensor frame, radians, shape ``(n,)``.
        ranges: Reported range of each beam, shape ``(n,)``. Beams without a return
            report the range limit.
        is_hit: True where the beam carries a range return.
        dropped: Number of beams lost to dropout and therefore absent from the arrays.
    """

    pose: Pose2D
    angles: FloatArray
    ranges: FloatArray
    is_hit: BoolArray
    dropped: int

    @property
    def origin(self) -> FloatArray:
        """Sensor position in the world frame, shape ``(2,)``."""
        return np.array([self.pose.x, self.pose.y], dtype=np.float64)

    @property
    def hit_count(self) -> int:
        """Number of beams carrying a range return."""
        return int(np.count_nonzero(self.is_hit))

    @property
    def max_range_count(self) -> int:
        """Number of beams that reached the range limit without a return."""
        return int(np.count_nonzero(~self.is_hit))

    def endpoints(self) -> FloatArray:
        """Beam endpoints in the world frame, shape ``(n, 2)``.

        For a beam with a return this is the measured surface point. For a beam at the
        range limit it is the point at the range limit, which bounds the free space
        the beam certifies but is not a surface.
        """
        world_angles = self.angles + self.pose.theta
        return np.stack(
            (
                self.pose.x + self.ranges * np.cos(world_angles),
                self.pose.y + self.ranges * np.sin(world_angles),
            ),
            axis=1,
        )


def simulate_scan(scene: Scene, pose: Pose2D, lidar: LidarSpec, rng: np.random.Generator) -> Scan:
    """Simulate one sweep against ``scene`` from ``pose``.

    The order of operations matters for reproducibility: dropout is drawn first for
    every beam of the sweep, then range noise is drawn for every beam of the sweep, so
    the number of variates consumed depends only on the beam count and not on the
    scene. A scene change therefore does not shift the random stream of later scans.

    Args:
        scene: Obstacles to measure.
        pose: Sensor pose in the world frame.
        lidar: Sensor configuration.
        rng: Source of the dropout and noise variates.

    Returns:
        A :class:`Scan` holding the surviving beams.
    """
    angles = lidar.beam_angles()
    world_angles = angles + pose.theta
    directions = np.stack((np.cos(world_angles), np.sin(world_angles)), axis=1)

    origin = np.array([pose.x, pose.y], dtype=np.float64)
    true_ranges = ray_ranges(scene, origin, directions)

    keep = rng.random(angles.size) >= lidar.dropout_prob
    noise = rng.normal(0.0, 1.0, size=angles.size) * lidar.range_noise_std

    is_hit = true_ranges < lidar.max_range
    measured = np.where(is_hit, true_ranges + noise, lidar.max_range)
    # A noisy return that lands beyond the range limit is reported as no return, which
    # is what the hardware would do.
    beyond = measured >= lidar.max_range
    is_hit = is_hit & ~beyond
    measured = np.where(is_hit, np.maximum(measured, lidar.min_range), lidar.max_range)

    return Scan(
        pose=pose,
        angles=angles[keep],
        ranges=measured[keep],
        is_hit=np.asarray(is_hit[keep], dtype=np.bool_),
        dropped=int(np.count_nonzero(~keep)),
    )
