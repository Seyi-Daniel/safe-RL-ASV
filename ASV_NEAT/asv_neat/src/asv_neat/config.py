"""Configuration dataclasses for the crossing scenario environment."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BoatParams:
    """Geometry and kinematic limits for a vessel."""

    length: float = 6.0
    width: float = 2.2

    max_speed: float = 7.0
    min_speed: float = 0.0
    accel_rate: float = 0.20
    decel_rate: float = 0.05


@dataclass
class RudderParams:
    """Mechanical limits for the continuous rudder controller."""

    max_rudder: float = math.radians(35.0)
    max_yaw_rate: float = 0.25
    max_rudder_rate: float = math.radians(40.0)


@dataclass
class EnvConfig:
    """Rendering and integration settings for the simplified environment."""

    world_w: float = 520.0
    world_h: float = 520.0
    dt: float = 0.05
    substeps: int = 1

    render: bool = False
    pixels_per_meter: float = 1.5
    show_grid: bool = True
    show_trails: bool = True
    show_hud: bool = True
