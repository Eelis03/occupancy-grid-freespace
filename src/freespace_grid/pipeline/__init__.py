"""Simulation and orchestration: scenes, sensor, trajectories, and the run trace."""

from __future__ import annotations

from freespace_grid.pipeline.lidar import LidarSpec, Scan, simulate_scan
from freespace_grid.pipeline.runner import MappingTrace, RunConfig, StepRecord, run_mapping
from freespace_grid.pipeline.scenarios import (
    SCENARIOS,
    Scenario,
    dynamic_corridor,
    enclosed_room,
    urban_block,
)
from freespace_grid.pipeline.scene import (
    Circle,
    MovingCircle,
    Polygon,
    Scene,
    occupancy_truth,
    ray_ranges,
)
from freespace_grid.pipeline.trajectory import Trajectory, constant_twist, from_segments

__all__ = [
    "SCENARIOS",
    "Circle",
    "LidarSpec",
    "MappingTrace",
    "MovingCircle",
    "Polygon",
    "RunConfig",
    "Scan",
    "Scenario",
    "Scene",
    "StepRecord",
    "Trajectory",
    "constant_twist",
    "dynamic_corridor",
    "enclosed_room",
    "from_segments",
    "occupancy_truth",
    "ray_ranges",
    "run_mapping",
    "simulate_scan",
    "urban_block",
]
