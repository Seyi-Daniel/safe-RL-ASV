"""Boat dynamics used by the crossing scenario environment."""
from __future__ import annotations

import math
from typing import Optional, Tuple

from .config import BoatParams, RudderParams
from .utils import clamp, wrap_pi


class Boat:
    """Vessel model with simple per-step helm inputs."""

    def __init__(
        self,
        boat_id: int,
        x: float,
        y: float,
        heading: float,
        speed: float,
        kin: BoatParams,
        rudder_cfg: RudderParams,
        goal: Optional[Tuple[float, float]] = None,
        name: Optional[str] = None,
    ) -> None:
        self.id = boat_id
        self.name = name
        self.x = float(x)
        self.y = float(y)
        self.h = float(heading)
        self.u = float(speed)
        self.kin = kin
        self.rudder_cfg = rudder_cfg
        if goal is not None:
            gx, gy = goal
            self.goal_x = float(gx)
            self.goal_y = float(gy)
        else:
            self.goal_x = None
            self.goal_y = None

        self.rudder = 0.0
        self.last_thr = 0
        self.prev_rudder_cmd = 0.0
        self.last_rudder_cmd = 0.0

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Return the observable state for this boat."""

        return {
            "id": self.id,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "heading": self.h,
            "speed": self.u,
            "rudder": self.rudder,
            "goal_x": self.goal_x,
            "goal_y": self.goal_y,
        }

    @staticmethod
    def decode_action(action) -> Tuple[float, int]:
        """Clamp and unpack ``(rudder_command, throttle)`` selections."""

        rudder_cmd, throttle = action
        rudder_cmd = clamp(float(rudder_cmd), -1.0, 1.0)
        throttle_i = clamp(int(round(throttle)), 0, 2)
        return rudder_cmd, throttle_i

    def apply_action(self, action) -> None:
        rudder_cmd, throttle = self.decode_action(action)
        self.prev_rudder_cmd = self.last_rudder_cmd
        self.last_rudder_cmd = rudder_cmd
        self.last_thr = throttle

    def begin_step(self) -> None:
        """Record the current command so deltas can be computed later."""

        self.prev_rudder_cmd = self.last_rudder_cmd

    def integrate(self, dt: float) -> None:
        max_step = self.rudder_cfg.max_rudder_rate * dt
        rudder_target = clamp(self.last_rudder_cmd, -1.0, 1.0) * self.rudder_cfg.max_rudder
        delta = rudder_target - self.rudder
        delta = clamp(delta, -max_step, max_step)
        self.rudder += delta

        yaw_rate = (self.rudder / self.rudder_cfg.max_rudder) * self.rudder_cfg.max_yaw_rate
        self.h = wrap_pi(self.h + yaw_rate * dt)

        if self.last_thr == 1:
            self.u += self.kin.accel_rate * dt
        elif self.last_thr == 2:
            self.u -= self.kin.decel_rate * dt
        self.u = clamp(self.u, self.kin.min_speed, self.kin.max_speed)

        self.x += math.cos(self.h) * self.u * dt
        self.y += math.sin(self.h) * self.u * dt
