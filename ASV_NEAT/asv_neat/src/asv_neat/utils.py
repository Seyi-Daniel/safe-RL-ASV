"""Utility helpers for the simplified crossing scenario package."""
from __future__ import annotations

import math
from typing import Tuple


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to the inclusive range ``[lo, hi]``."""

    return max(lo, min(hi, value))


def wrap_pi(angle: float) -> float:
    """Wrap an angle to the ``[-π, π]`` interval."""

    return math.atan2(math.sin(angle), math.cos(angle))


def angle_deg(angle: float) -> float:
    """Convert radians to a canonical degree representation in ``[0, 360)``."""

    return math.degrees(angle) % 360.0


def euclidean_distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Return the planar Euclidean distance between two points."""

    return math.hypot(ax - bx, ay - by)


def goal_distance(state: dict) -> float:
    """Distance from the vessel to its goal, falling back to the current position."""

    gx_val = state.get("goal_x")
    gy_val = state.get("goal_y")
    gx = float(state["x"]) if gx_val is None else float(gx_val)
    gy = float(state["y"]) if gy_val is None else float(gy_val)
    return euclidean_distance(float(state["x"]), float(state["y"]), gx, gy)


def heading_error_deg(state: dict) -> float:
    """Absolute heading error between the vessel's course and the goal direction."""

    gx_val = state.get("goal_x")
    gy_val = state.get("goal_y")
    gx = float(state["x"]) if gx_val is None else float(gx_val)
    gy = float(state["y"]) if gy_val is None else float(gy_val)
    x = float(state["x"])
    y = float(state["y"])
    dx = gx - x
    dy = gy - y
    if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
        return 0.0
    desired_heading = math.atan2(dy, dx)
    current_heading = float(state["heading"])
    return abs(math.degrees(wrap_pi(desired_heading - current_heading)))


def tcpa_dcpa(agent: dict, stand_on: dict) -> Tuple[float, float]:
    """Compute time and distance to closest point of approach between two vessels."""

    ax = float(agent["x"])
    ay = float(agent["y"])
    ah = float(agent["heading"])
    au = float(agent.get("speed", 0.0))

    sx = float(stand_on["x"])
    sy = float(stand_on["y"])
    sh = float(stand_on["heading"])
    su = float(stand_on.get("speed", 0.0))

    avx = math.cos(ah) * au
    avy = math.sin(ah) * au
    svx = math.cos(sh) * su
    svy = math.sin(sh) * su

    rx = sx - ax
    ry = sy - ay
    rvx = svx - avx
    rvy = svy - avy

    rel_speed_sq = rvx * rvx + rvy * rvy
    if rel_speed_sq <= 1e-8:
        return float("inf"), euclidean_distance(ax, ay, sx, sy)

    tcpa = -((rx * rvx) + (ry * rvy)) / rel_speed_sq
    closest_x = sx + rvx * tcpa
    closest_y = sy + rvy * tcpa
    dcpa = euclidean_distance(ax, ay, closest_x, closest_y)
    return tcpa, dcpa


def relative_bearing_deg(observer: dict, target: dict) -> float:
    """Return the clockwise relative bearing from ``observer`` to ``target``."""

    dx = float(target["x"]) - float(observer["x"])
    dy = float(target["y"]) - float(observer["y"])
    heading = float(observer["heading"])
    ch = math.cos(heading)
    sh = math.sin(heading)
    x_rel = ch * dx + sh * dy
    y_rel = -sh * dx + ch * dy
    rel_port = math.degrees(math.atan2(y_rel, x_rel))
    rel_port = (rel_port + 360.0) % 360.0
    return (360.0 - rel_port) % 360.0


def helm_label_from_rudder_cmd(rudder_cmd: float, eps: float = 0.05) -> str:
    """
    Map a continuous rudder command in [-1, 1] to a categorical label:
    "turn_port", "turn_starboard", or "keep_straight".
    """

    if rudder_cmd > eps:
        return "turn_port"
    if rudder_cmd < -eps:
        return "turn_starboard"
    return "keep_straight"
