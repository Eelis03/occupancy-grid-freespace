"""Named scenes, scenarios, and the grids and trajectories that go with them.

A :class:`Scenario` binds everything one run needs: the static geometry, any moving
obstacles, the grid the map is built on, the sensor, the vehicle trajectory, and the
seed. Keeping them together means an example script is a call and a print, and means
the regression test and the example describe the same run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from freespace_grid.model.grid import GridSpec
from freespace_grid.model.logodds import LogOddsModel
from freespace_grid.model.transform import Pose2D
from freespace_grid.pipeline.lidar import LidarSpec
from freespace_grid.pipeline.scene import Circle, MovingCircle, Polygon, Scene
from freespace_grid.pipeline.trajectory import Trajectory, constant_twist, from_segments

__all__ = [
    "DYNAMIC_DIRECTIONS",
    "SCENARIOS",
    "Scenario",
    "dynamic_corridor",
    "enclosed_room",
    "urban_block",
]


@dataclass(frozen=True, slots=True)
class Scenario:
    """A complete, reproducible mapping problem."""

    name: str
    scene: Scene
    grid: GridSpec
    lidar: LidarSpec
    trajectory: Trajectory
    model: LogOddsModel = field(default_factory=LogOddsModel)
    movers: tuple[MovingCircle, ...] = ()
    seed: int = 20260731

    def scene_at(self, time: float) -> Scene:
        """Return the scene as it stands at ``time`` seconds."""
        if not self.movers:
            return self.scene
        return Scene(
            name=self.scene.name,
            circles=(*self.scene.circles, *(mover.at(time) for mover in self.movers)),
            polygons=self.scene.polygons,
        )

    def frozen_at(self, time: float) -> Scenario:
        """Return a copy whose movers are replaced by static discs at their pose at ``time``.

        This is the control run for the dynamic obstacle study: identical geometry at
        the final instant, identical sensor, identical trajectory, no motion.
        """
        return Scenario(
            name=f"{self.name}_frozen",
            scene=Scene(
                name=self.scene.name,
                circles=(*self.scene.circles, *(mover.at(time) for mover in self.movers)),
                polygons=self.scene.polygons,
            ),
            grid=self.grid,
            lidar=self.lidar,
            trajectory=self.trajectory,
            model=self.model,
            movers=(),
            seed=self.seed,
        )


def urban_block() -> Scenario:
    """A 60 by 40 metre street block driven end to end by a vehicle at 5 metres per second.

    Five building footprints bound a twelve metre corridor. Two parked vehicles, four
    poles and a planter sit near the corridor edges. The corridor is open at both ends,
    so a useful fraction of every sweep reaches the range limit without a return, which
    is the case the inverse sensor model has to handle correctly.
    """
    scene = Scene(
        name="urban_block",
        polygons=(
            Polygon.rectangle(4.0, 26.0, 20.0, 38.0),
            Polygon.rectangle(26.0, 26.0, 42.0, 38.0),
            Polygon.rectangle(48.0, 26.0, 58.0, 38.0),
            Polygon.rectangle(6.0, 2.0, 22.0, 12.0),
            Polygon.rectangle(30.0, 2.0, 46.0, 12.0),
            Polygon.rectangle(10.0, 24.0, 14.5, 25.8),
            Polygon.rectangle(32.0, 24.0, 36.5, 25.8),
        ),
        circles=(
            Circle(24.0, 24.6, 0.3),
            Circle(46.0, 24.6, 0.3),
            Circle(28.0, 15.4, 0.6),
            Circle(52.0, 23.6, 1.2),
        ),
    )
    # 4.7 metres per second at a 0.2 second interval advances the vehicle 4.7 cells per
    # step, so no re-anchoring of an ego window ever lands on a whole cell offset. That
    # is deliberate: a speed that happened to be a whole number of cells per step would
    # make the interpolation comparison in examples/compare_grid_frames.py vacuous.
    trajectory = from_segments(
        start=Pose2D(4.0, 20.0, 0.0),
        segments=((4.7, 0.0, 18), (4.7, -0.12, 10), (4.7, 0.12, 10), (4.7, 0.0, 12)),
        dt=0.2,
    )
    return Scenario(
        name="urban_block",
        scene=scene,
        grid=GridSpec(resolution=0.2, rows=200, cols=300, origin_x=0.0, origin_y=0.0),
        lidar=LidarSpec(
            max_range=30.0,
            angular_resolution_deg=0.5,
            field_of_view_deg=360.0,
            range_noise_std=0.03,
            dropout_prob=0.02,
        ),
        trajectory=trajectory,
        model=LogOddsModel(),
    )


DYNAMIC_DIRECTIONS: tuple[str, ...] = ("approaching", "receding", "crossing")

# Start position and velocity of the disc, in metres and metres per second. Speeds are
# chosen so that a cell on the near surface of the disc is the terminal cell of a beam
# for two to five consecutive sweeps, which is the regime in which the static world
# assumption is under real strain: enough evidence to matter, not enough to converge.
_DYNAMIC_MOTION: dict[str, tuple[float, float, float, float]] = {
    "approaching": (26.0, 20.0, -1.2, 0.0),
    "receding": (16.4, 20.0, 1.2, 0.0),
    "crossing": (20.0, 13.6, 0.0, 1.6),
}


def dynamic_corridor(direction: str, model: LogOddsModel | None = None) -> Scenario:
    """A stationary observer watching one disc translate through a straight corridor.

    The vehicle is held still so that anything the map gets wrong is caused by the
    obstacle's motion alone and not by sensor motion. Three directions are offered,
    and they behave differently because the map can only be corrected where the sensor
    can still see:

    ``approaching``
        The disc moves towards the sensor. The cells it vacates fall into its own
        shadow, are never observed again, and keep whatever occupied evidence they were
        given. This is the worst case for stale evidence.
    ``receding``
        The disc moves away. The cells it vacates lie between the sensor and the disc,
        are observed free on every later sweep, and recover within a few frames.
    ``crossing``
        The disc moves across the line of sight. Part of the trail is re-observed and
        part is shadowed, so the residual falls between the other two.

    Args:
        direction: One of ``approaching``, ``receding``, ``crossing``.
        model: Optional replacement for the default log odds parameters, used by the
            clamp sweep in ``examples/dynamic_smear.py``.
    """
    if direction not in _DYNAMIC_MOTION:
        raise ValueError(
            f"direction must be one of {list(DYNAMIC_DIRECTIONS)}, got {direction!r}"
        )
    start_x, start_y, vel_x, vel_y = _DYNAMIC_MOTION[direction]

    scene = Scene(
        name="dynamic_corridor",
        polygons=(
            Polygon.rectangle(0.0, 9.0, 40.0, 10.0),
            Polygon.rectangle(0.0, 30.0, 40.0, 31.0),
        ),
        circles=(Circle(34.0, 14.0, 0.8),),
    )
    return Scenario(
        name=f"dynamic_corridor_{direction}",
        scene=scene,
        grid=GridSpec(resolution=0.2, rows=120, cols=200, origin_x=0.0, origin_y=8.0),
        lidar=LidarSpec(
            max_range=30.0,
            angular_resolution_deg=0.5,
            field_of_view_deg=360.0,
            range_noise_std=0.03,
            dropout_prob=0.02,
        ),
        trajectory=constant_twist(
            Pose2D(10.0, 20.0, 0.0), speed=0.0, yaw_rate=0.0, dt=0.2, steps=40
        ),
        model=model if model is not None else LogOddsModel(),
        movers=(MovingCircle(start_x, start_y, vel_x, vel_y, 1.0),),
    )


def enclosed_room(*, steps: int = 8) -> Scenario:
    """A closed square room observed from its centre, with no noise and no dropout.

    Every beam terminates on a wall, so no beam reaches the range limit, and the
    interior must converge to free and the visible wall face to occupied. This is the
    convergence fixture, and it is small enough to run in a fraction of a second.

    The inner faces of the walls are placed at 0.70 and 10.60 metres on a 0.25 metre
    grid, which is deliberate rather than arbitrary. A range return is attributed to
    the cell containing the measured point, while the ground truth labels a cell by
    whether its centre lies inside an obstacle. When a surface falls exactly on a cell
    boundary the two conventions disagree by one cell, and a convergence test built on
    such a room would be measuring that offset rather than convergence. Placing 0.70
    in the upper part of the cell that spans 0.50 to 0.75, and 10.60 in the lower part
    of the cell that spans 10.50 to 10.75, puts every terminal cell centre inside a
    wall on all four sides. The offset itself is not swept under the carpet: it is what
    the spatial tolerance sweep in ``examples/sweep_agreement.py`` measures.
    """
    inner_low, inner_high, outer = 0.70, 10.60, 11.30
    scene = Scene(
        name="enclosed_room",
        polygons=(
            Polygon.rectangle(0.0, 0.0, outer, inner_low),
            Polygon.rectangle(0.0, inner_high, outer, outer),
            Polygon.rectangle(0.0, 0.0, inner_low, outer),
            Polygon.rectangle(inner_high, 0.0, outer, outer),
        ),
    )
    center = 0.5 * (inner_low + inner_high)
    return Scenario(
        name="enclosed_room",
        scene=scene,
        grid=GridSpec(resolution=0.25, rows=46, cols=46),
        lidar=LidarSpec(
            max_range=40.0,
            angular_resolution_deg=0.25,
            field_of_view_deg=360.0,
            range_noise_std=0.0,
            dropout_prob=0.0,
        ),
        trajectory=constant_twist(
            Pose2D(center, center, 0.0), speed=0.0, yaw_rate=0.0, dt=0.2, steps=steps
        ),
        model=LogOddsModel(),
    )


SCENARIOS: dict[str, str] = {
    "urban_block": "Street block driven end to end, static obstacles.",
    "dynamic_corridor_approaching": "Stationary observer, one disc moving towards it.",
    "dynamic_corridor_receding": "Stationary observer, one disc moving away from it.",
    "dynamic_corridor_crossing": "Stationary observer, one disc crossing the line of sight.",
    "enclosed_room": "Closed square room observed from the centre.",
}
