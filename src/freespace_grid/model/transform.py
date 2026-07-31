"""Planar rigid transforms.

The vehicle observes in a body-fixed frame while the map is maintained in a world
frame, so every scan crosses one SE(2) transform. These are pure functions on plain
arrays; rotations are written out elementwise rather than as a matrix product so that
the result does not depend on which BLAS kernel happens to be linked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from freespace_grid.model.typing import FloatArray

__all__ = ["Pose2D", "compose", "inverse", "transform_points", "wrap_angle"]


def wrap_angle(angle: float) -> float:
    """Wrap ``angle`` into ``(-pi, pi]``."""
    wrapped = math.remainder(angle, 2.0 * math.pi)
    return math.pi if wrapped == -math.pi else wrapped


@dataclass(frozen=True, slots=True)
class Pose2D:
    """A rigid placement of one planar frame inside another.

    ``Pose2D(x, y, theta)`` maps a point expressed in the child frame to the parent
    frame by rotating it through ``theta`` and then translating by ``(x, y)``.
    """

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    @property
    def translation(self) -> FloatArray:
        """Translation part as a length-two array."""
        return np.array([self.x, self.y], dtype=np.float64)

    def rotation(self) -> FloatArray:
        """Rotation part as a two by two matrix."""
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)

    def normalized(self) -> Pose2D:
        """Return the same pose with its heading wrapped into ``(-pi, pi]``."""
        return Pose2D(self.x, self.y, wrap_angle(self.theta))


def transform_points(pose: Pose2D, points: FloatArray) -> FloatArray:
    """Map points from the child frame of ``pose`` to its parent frame.

    Args:
        pose: The placement of the child frame in the parent frame.
        points: Array of shape ``(n, 2)``.

    Returns:
        Array of shape ``(n, 2)`` in the parent frame.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (n, 2), got {pts.shape}")
    cos_t = math.cos(pose.theta)
    sin_t = math.sin(pose.theta)
    x = cos_t * pts[:, 0] - sin_t * pts[:, 1] + pose.x
    y = sin_t * pts[:, 0] + cos_t * pts[:, 1] + pose.y
    return np.stack((x, y), axis=1)


def compose(outer: Pose2D, inner: Pose2D) -> Pose2D:
    """Return the pose equivalent to applying ``inner`` and then ``outer``."""
    cos_t = math.cos(outer.theta)
    sin_t = math.sin(outer.theta)
    return Pose2D(
        x=outer.x + cos_t * inner.x - sin_t * inner.y,
        y=outer.y + sin_t * inner.x + cos_t * inner.y,
        theta=wrap_angle(outer.theta + inner.theta),
    )


def inverse(pose: Pose2D) -> Pose2D:
    """Return the pose that undoes ``pose``."""
    cos_t = math.cos(pose.theta)
    sin_t = math.sin(pose.theta)
    return Pose2D(
        x=-(cos_t * pose.x + sin_t * pose.y),
        y=-(-sin_t * pose.x + cos_t * pose.y),
        theta=wrap_angle(-pose.theta),
    )
