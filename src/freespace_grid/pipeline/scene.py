"""Scene geometry: the obstacles the simulated lidar measures and the ground truth.

Obstacles are circles and simple polygons. Both admit a closed form ray intersection
and a closed form point containment test, so the simulated range and the ground truth
occupancy come from the same geometry rather than from two approximations of it. A
scene that rasterised its obstacles first and then ray traced the raster would score
the mapper against its own discretisation and hide every error the discretisation
causes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from freespace_grid.model.grid import GridSpec, cell_centers
from freespace_grid.model.typing import BoolArray, FloatArray

__all__ = ["Circle", "MovingCircle", "Polygon", "Scene", "occupancy_truth", "ray_ranges"]


@dataclass(frozen=True, slots=True)
class Circle:
    """A filled disc obstacle."""

    center_x: float
    center_y: float
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"radius must be positive, got {self.radius}")


@dataclass(frozen=True, slots=True)
class Polygon:
    """A filled simple polygon, given by its vertices in order.

    The polygon is implicitly closed. Vertices may be listed clockwise or
    counter-clockwise; neither the intersection test nor the containment test depends
    on the winding.
    """

    vertices: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError(f"a polygon needs at least three vertices, got {len(self.vertices)}")

    def as_array(self) -> FloatArray:
        """Return the vertices as an ``(k, 2)`` array."""
        return np.asarray(self.vertices, dtype=np.float64)

    def edges(self) -> tuple[FloatArray, FloatArray]:
        """Return the edge start and end points as two ``(k, 2)`` arrays."""
        verts = self.as_array()
        return verts, np.roll(verts, -1, axis=0)

    @classmethod
    def rectangle(cls, x_min: float, y_min: float, x_max: float, y_max: float) -> Polygon:
        """Return an axis-aligned rectangle."""
        return cls(((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)))


@dataclass(frozen=True, slots=True)
class MovingCircle:
    """A disc translating at constant velocity, used for the dynamic obstacle case."""

    center_x: float
    center_y: float
    velocity_x: float
    velocity_y: float
    radius: float

    def at(self, time: float) -> Circle:
        """Return the disc as it stands at ``time`` seconds."""
        return Circle(
            center_x=self.center_x + self.velocity_x * time,
            center_y=self.center_y + self.velocity_y * time,
            radius=self.radius,
        )

    @property
    def speed(self) -> float:
        """Magnitude of the velocity, in metres per second."""
        return float(np.hypot(self.velocity_x, self.velocity_y))

    def direction(self) -> FloatArray:
        """Unit vector along the direction of travel."""
        speed = self.speed
        if speed == 0.0:
            return np.array([1.0, 0.0], dtype=np.float64)
        return np.array([self.velocity_x / speed, self.velocity_y / speed], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Scene:
    """A named collection of static obstacles."""

    name: str
    circles: tuple[Circle, ...] = ()
    polygons: tuple[Polygon, ...] = field(default_factory=tuple)


def ray_ranges(scene: Scene, origin: FloatArray, directions: FloatArray) -> FloatArray:
    """Return the distance to the first surface along each ray, or infinity.

    Args:
        scene: Obstacles to test against.
        origin: Ray origin in world coordinates, shape ``(2,)``.
        directions: Unit direction vectors, shape ``(n, 2)``.

    Returns:
        Array of shape ``(n,)``. Entry ``i`` is the smallest strictly positive
        distance along ray ``i`` at which a surface is met, or ``inf``.
    """
    start = np.asarray(origin, dtype=np.float64).reshape(2)
    dirs = np.asarray(directions, dtype=np.float64)
    if dirs.ndim != 2 or dirs.shape[1] != 2:
        raise ValueError(f"directions must have shape (n, 2), got {dirs.shape}")

    best = np.full(dirs.shape[0], np.inf, dtype=np.float64)
    for circle in scene.circles:
        np.minimum(best, _circle_ranges(circle, start, dirs), out=best)
    for polygon in scene.polygons:
        np.minimum(best, _polygon_ranges(polygon, start, dirs), out=best)
    return best


def _circle_ranges(circle: Circle, origin: FloatArray, dirs: FloatArray) -> FloatArray:
    """Smallest positive root of ``|origin + t d - c|^2 = r^2`` for each unit direction."""
    to_center = np.array([circle.center_x - origin[0], circle.center_y - origin[1]])
    projection = dirs[:, 0] * to_center[0] + dirs[:, 1] * to_center[1]
    gap = float(to_center[0] ** 2 + to_center[1] ** 2) - circle.radius**2
    discriminant = projection**2 - gap
    valid = discriminant >= 0.0
    root = np.sqrt(np.where(valid, discriminant, 0.0))
    near = projection - root
    far = projection + root
    distance = np.where(near > 0.0, near, far)
    return np.where(valid & (distance > 0.0), distance, np.inf)


def _polygon_ranges(polygon: Polygon, origin: FloatArray, dirs: FloatArray) -> FloatArray:
    """Smallest positive ray parameter meeting any edge of the polygon."""
    starts, ends = polygon.edges()
    edge = ends - starts
    # Solve origin + t d = a + s e for each ray and edge pair by Cramer's rule.
    denominator = np.outer(dirs[:, 0], edge[:, 1]) - np.outer(dirs[:, 1], edge[:, 0])
    offset = starts[None, :, :] - origin[None, None, :]
    t_num = offset[:, :, 0] * edge[None, :, 1] - offset[:, :, 1] * edge[None, :, 0]
    s_num = offset[:, :, 0] * dirs[:, None, 1] - offset[:, :, 1] * dirs[:, None, 0]
    parallel = denominator == 0.0
    safe = np.where(parallel, 1.0, denominator)
    t_value = t_num / safe
    s_value = s_num / safe
    hit = ~parallel & (t_value > 0.0) & (s_value >= 0.0) & (s_value <= 1.0)
    return np.min(np.where(hit, t_value, np.inf), axis=1)


def occupancy_truth(scene: Scene, spec: GridSpec) -> BoolArray:
    """Return the ground truth occupancy of every cell of ``spec``.

    A cell counts as occupied when its centre lies inside any obstacle. Sampling at
    the centre rather than integrating over the cell means a surface grazing a cell
    corner is not recorded, which matches what a mapper working at the same resolution
    can be expected to recover.
    """
    grid_x, grid_y = cell_centers(spec)
    xs = grid_x.reshape(-1)
    ys = grid_y.reshape(-1)
    inside = np.zeros(xs.shape, dtype=np.bool_)
    for circle in scene.circles:
        radial = (xs - circle.center_x) ** 2 + (ys - circle.center_y) ** 2
        inside |= radial <= circle.radius**2
    for polygon in scene.polygons:
        inside |= _points_in_polygon(polygon, xs, ys)
    return np.asarray(inside.reshape(spec.shape), dtype=np.bool_)


def _points_in_polygon(polygon: Polygon, xs: FloatArray, ys: FloatArray) -> BoolArray:
    """Crossing number test, vectorised over points and edges."""
    starts, ends = polygon.edges()
    crossings = np.zeros(xs.shape, dtype=np.int64)
    for (ax, ay), (bx, by) in zip(starts, ends, strict=True):
        if ay == by:
            continue
        straddles = (ay > ys) != (by > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at_y = (bx - ax) * (ys - ay) / (by - ay) + ax
        crossings += np.asarray(straddles & (xs < x_at_y), dtype=np.int64)
    return np.asarray(crossings % 2 == 1, dtype=np.bool_)
