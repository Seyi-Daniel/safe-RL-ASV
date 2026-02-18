#!/usr/bin/env python3
"""Feature-based two-vessel RL environment with COLREGs-shaped rewards.

This environment is intentionally standalone and combines ideas from:
- ASV_NEAT objective shaping (goal progress + COLREGs wrong-action penalties),
- feature-RL-ASV sector awareness + TCPA/DCPA risk features,
- RL_ASV discrete steer/throttle action coding.

Observation design (10 features):
1) ego_x_norm
2) ego_y_norm
3) ego_heading_norm
4) ego_speed_norm
5) ego_goal_x_norm
6) ego_goal_y_norm
7) target_heading_rel_norm
8) target_speed_norm
9) relative_bearing_norm
10) distance_norm

The target vessel has no goal and its absolute (x, y) is not exposed.
A 12-sector awareness model is used internally for COLREGs logic and reward shaping.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return hi if x > hi else lo if x < lo else x


def tcpa_dcpa(
    ax: float,
    ay: float,
    ah: float,
    aspd: float,
    bx: float,
    by: float,
    bh: float,
    bspd: float,
) -> Tuple[float, float]:
    """Closest point of approach for constant-velocity vessels."""
    rvx = math.cos(bh) * bspd - math.cos(ah) * aspd
    rvy = math.sin(bh) * bspd - math.sin(ah) * aspd
    rx, ry = (bx - ax), (by - ay)
    rv2 = rvx * rvx + rvy * rvy
    if rv2 < 1e-9:
        return 0.0, math.hypot(rx, ry)
    tcpa = -((rx * rvx + ry * rvy) / rv2)
    if tcpa < 0.0:
        tcpa = 0.0
    cx = rx + rvx * tcpa
    cy = ry + rvy * tcpa
    return tcpa, math.hypot(cx, cy)


@dataclass
class Vessel:
    x: float
    y: float
    h: float
    speed: float
    rudder: float = 0.0
    throttle: float = 0.0


@dataclass
class EnvConfig:
    world_size: float = 300.0
    dt: float = 0.2
    episode_seconds: float = 120.0

    # initialization
    spawn_margin: float = 20.0
    min_spawn_separation: float = 30.0
    target_min_speed: float = 0.5
    target_max_speed: float = 7.0
    ego_min_speed: float = 0.0
    ego_max_speed: float = 7.0

    # ASV_NEAT-style actions / dynamics
    rudder_max_angle: float = math.radians(35.0)      # ±35°
    rudder_max_yaw_rate: float = 0.25                 # rad/s
    rudder_slew_rate: float = math.radians(40.0)      # 40°/s
    accel_rate: float = 0.20                           # m/s^2 (full fwd)
    decel_rate: float = 0.05                           # m/s^2 (passive/coast)
    brake_rate: float = 0.20                           # m/s^2 (full reverse thrust)
    throttle_slew_rate: float = 0.4                    # 1/s (0->1 in 2.5 s)
    throttle_deadband: float = 0.02

    # goal
    goal_radius: float = 10.0
    goal_min_distance: float = 80.0

    # 12-sector awareness radius
    encounter_radius: float = 120.0

    # reward shaping
    living_penalty: float = -0.002
    progress_weight: float = 0.04
    goal_bonus: float = 8.0
    collision_penalty: float = -15.0
    out_of_bounds_penalty: float = -8.0

    cpa_horizon: float = 60.0
    risk_weight: float = 0.10
    dcpa_scale: float = 120.0
    tcpa_decay: float = 20.0

    colregs_correct_bonus: float = 0.08
    colregs_wrong_penalty: float = 0.08

    dcpa_improve_weight: float = 0.20  # "DCPA super action" reward
    dcpa_worse_weight: float = 0.10


class ColregsFeatureEnv:
    """Single-agent (ego) + random target vessel environment."""

    def __init__(self, cfg: EnvConfig = EnvConfig(), seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = random.Random(seed)

        self.ego: Optional[Vessel] = None
        self.target: Optional[Vessel] = None
        self.goal: Tuple[float, float] = (0.0, 0.0)

        self.max_steps = max(1, int(round(cfg.episode_seconds / cfg.dt)))
        self.step_count = 0

        self.prev_goal_distance = 0.0

    @staticmethod
    def decode_action(action: Union[int, np.ndarray, Tuple[float, float], list]) -> Tuple[float, float]:
        """Map action to continuous (rudder_cmd, throttle_cmd) in [-1, 1].

        Backward compatibility: integer actions [0..8] are still accepted and
        converted from the historical 3x3 discrete action table.
        """
        if isinstance(action, (int, np.integer)):
            steer = int(action) // 3
            throttle = int(action) % 3
            rudder_cmd = -1.0 if steer == 1 else (1.0 if steer == 2 else 0.0)
            throttle_cmd = -1.0 if throttle == 2 else (1.0 if throttle == 1 else 0.0)
            return rudder_cmd, throttle_cmd

        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            raise ValueError("Continuous action must provide at least 2 values: [rudder_cmd, throttle_cmd].")
        rudder_cmd = clamp(float(arr[0]), -1.0, 1.0)
        throttle_cmd = clamp(float(arr[1]), -1.0, 1.0)
        return rudder_cmd, throttle_cmd

    def _random_vessel(self, speed_lo: float, speed_hi: float) -> Vessel:
        m = self.cfg.spawn_margin
        return Vessel(
            x=self.rng.uniform(m, self.cfg.world_size - m),
            y=self.rng.uniform(m, self.cfg.world_size - m),
            h=self.rng.uniform(-math.pi, math.pi),
            speed=self.rng.uniform(speed_lo, speed_hi),
        )

    def _outside(self, v: Vessel) -> bool:
        return not (0.0 <= v.x <= self.cfg.world_size and 0.0 <= v.y <= self.cfg.world_size)

    def _distance(self, a: Vessel, b: Vessel) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _goal_distance(self) -> float:
        gx, gy = self.goal
        return math.hypot(gx - self.ego.x, gy - self.ego.y)

    def _apply_motion(self, v: Vessel, rudder_cmd: float, throttle_cmd: float, speed_constant: bool = False) -> None:
        # 1) Rudder dynamics (ASV_NEAT-style continuous rudder with slew limits)
        rudder_target = clamp(rudder_cmd, -1.0, 1.0) * self.cfg.rudder_max_angle
        rudder_step = self.cfg.rudder_slew_rate * self.cfg.dt
        rudder_delta = clamp(rudder_target - v.rudder, -rudder_step, rudder_step)
        v.rudder += rudder_delta

        # 2) Yaw rate cap from rudder authority
        yaw_rate = (v.rudder / max(1e-6, self.cfg.rudder_max_angle)) * self.cfg.rudder_max_yaw_rate
        v.h = wrap_pi(v.h + yaw_rate * self.cfg.dt)

        # 3) Continuous throttle with slew limits + forward/reverse/coast behavior
        thr_target = clamp(throttle_cmd, -1.0, 1.0)
        thr_step = self.cfg.throttle_slew_rate * self.cfg.dt
        v.throttle = clamp(v.throttle + clamp(thr_target - v.throttle, -thr_step, thr_step), -1.0, 1.0)

        if not speed_constant:
            if v.throttle > self.cfg.throttle_deadband:
                accel = v.throttle * self.cfg.accel_rate
            elif v.throttle < -self.cfg.throttle_deadband:
                accel = v.throttle * self.cfg.brake_rate
            else:
                accel = -self.cfg.decel_rate

            v.speed += accel * self.cfg.dt

            if v is self.ego:
                v.speed = clamp(v.speed, self.cfg.ego_min_speed, self.cfg.ego_max_speed)
            else:
                v.speed = clamp(v.speed, self.cfg.target_min_speed, self.cfg.target_max_speed)

        # kinematics
        v.x += v.speed * math.cos(v.h) * self.cfg.dt
        v.y += v.speed * math.sin(v.h) * self.cfg.dt

    def _relative_geometry(self) -> Tuple[float, float, int]:
        """Returns (relative_bearing_deg, distance, sector_12)."""
        dx = self.target.x - self.ego.x
        dy = self.target.y - self.ego.y
        dist = math.hypot(dx, dy)

        # body frame (x fwd, y port)
        ch = math.cos(self.ego.h)
        sh = math.sin(self.ego.h)
        x_rel = ch * dx + sh * dy
        y_rel = -sh * dx + ch * dy
        rel_brg = math.atan2(y_rel, x_rel)

        deg_360 = (math.degrees(rel_brg) + 360.0) % 360.0
        sector = int(deg_360 // 30.0)  # 12 sectors

        deg_signed = math.degrees(rel_brg)
        return deg_signed, dist, sector

    def _encounter_type(self, rel_bearing_deg: float) -> str:
        """Very lightweight COLREG encounter classification."""
        course_diff = abs(math.degrees(wrap_pi(self.target.h - self.ego.h)))
        abs_brg = abs(rel_bearing_deg)

        if abs_brg <= 15.0 and course_diff >= 150.0:
            return "head_on"
        if abs_brg >= 112.5 and self.ego.speed > self.target.speed:
            return "overtaking"
        return "crossing"

    def _colregs_reward(self, encounter: str, rel_bearing_deg: float, rudder_cmd: float) -> float:
        """Reward/penalty for action consistency with COLREG expectations."""
        # rudder_cmd<0 means starboard (right) turn, rudder_cmd>0 means port (left)
        if encounter == "head_on":
            return self.cfg.colregs_correct_bonus if rudder_cmd < 0 else -self.cfg.colregs_wrong_penalty

        if encounter == "crossing":
            # target on starboard side (give-way): prefer right turn
            if -112.5 <= rel_bearing_deg <= 0.0:
                return self.cfg.colregs_correct_bonus if rudder_cmd < 0 else -self.cfg.colregs_wrong_penalty
            # target on port side: neutral/slight preference to keep course
            if abs(rudder_cmd) <= 0.05:
                return 0.5 * self.cfg.colregs_correct_bonus
            return -0.25 * self.cfg.colregs_wrong_penalty

        # overtaking: reward starboard-side pass tendency
        return self.cfg.colregs_correct_bonus if rudder_cmd < 0 else -0.5 * self.cfg.colregs_wrong_penalty

    def _obs(self) -> np.ndarray:
        rel_bearing_deg, dist, _ = self._relative_geometry()

        return np.asarray(
            [
                self.ego.x / self.cfg.world_size,
                self.ego.y / self.cfg.world_size,
                self.ego.h / math.pi,
                self.ego.speed / self.cfg.ego_max_speed,
                self.goal[0] / self.cfg.world_size,
                self.goal[1] / self.cfg.world_size,
                wrap_pi(self.target.h - self.ego.h) / math.pi,
                self.target.speed / self.cfg.target_max_speed,
                rel_bearing_deg / 180.0,
                min(dist, self.cfg.encounter_radius) / self.cfg.encounter_radius,
            ],
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        # sample until separated
        while True:
            ego = self._random_vessel(self.cfg.ego_min_speed, self.cfg.ego_max_speed)
            target = self._random_vessel(self.cfg.target_min_speed, self.cfg.target_max_speed)
            if self._distance(ego, target) >= self.cfg.min_spawn_separation:
                self.ego, self.target = ego, target
                break

        # random goal for ego only (target has no goal)
        while True:
            gx = self.rng.uniform(self.cfg.spawn_margin, self.cfg.world_size - self.cfg.spawn_margin)
            gy = self.rng.uniform(self.cfg.spawn_margin, self.cfg.world_size - self.cfg.spawn_margin)
            if math.hypot(gx - self.ego.x, gy - self.ego.y) >= self.cfg.goal_min_distance:
                self.goal = (gx, gy)
                break

        self.step_count = 0
        self.prev_goal_distance = self._goal_distance()
        return self._obs()

    def step(self, action: Union[int, np.ndarray, Tuple[float, float], list]) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        rudder_cmd, throttle_cmd = self.decode_action(action)

        # pre-step CPA for DCPA-super-action shaping
        _, dcpa_before = tcpa_dcpa(
            self.ego.x,
            self.ego.y,
            self.ego.h,
            self.ego.speed,
            self.target.x,
            self.target.y,
            self.target.h,
            self.target.speed,
        )

        # ego action
        self._apply_motion(self.ego, rudder_cmd, throttle_cmd)

        # target: heading can wander, speed remains constant after reset
        t_rudder = self.rng.uniform(-1.0, 1.0) if self.rng.random() < 0.15 else 0.0
        self._apply_motion(self.target, t_rudder, 0.0, speed_constant=True)

        self.step_count += 1

        rel_bearing_deg, dist, sector = self._relative_geometry()
        encounter = self._encounter_type(rel_bearing_deg)

        tcpa, dcpa_after = tcpa_dcpa(
            self.ego.x,
            self.ego.y,
            self.ego.h,
            self.ego.speed,
            self.target.x,
            self.target.y,
            self.target.h,
            self.target.speed,
        )

        # terminals
        done = False
        reason = ""
        if dist <= 8.0:
            done, reason = True, "collision"
        elif self._outside(self.ego) or self._outside(self.target):
            done, reason = True, "out_of_bounds"
        elif self._goal_distance() <= self.cfg.goal_radius:
            done, reason = True, "goal"
        elif self.step_count >= self.max_steps:
            done, reason = True, "timeout"

        # reward
        reward = self.cfg.living_penalty

        # goal progress objective (from ASV-style objective shaping)
        d_now = self._goal_distance()
        reward += self.cfg.progress_weight * (self.prev_goal_distance - d_now)

        # risk shaping within horizon
        if 0.0 <= tcpa <= self.cfg.cpa_horizon:
            risk = math.exp(-tcpa / self.cfg.tcpa_decay) * math.exp(-dcpa_after / self.cfg.dcpa_scale)
            reward -= self.cfg.risk_weight * risk

        # apply COLREGs shaping only if target is in encounter radius
        if dist <= self.cfg.encounter_radius:
            reward += self._colregs_reward(encounter, rel_bearing_deg, rudder_cmd)

        # DCPA super-action shaping: increasing DCPA means safer
        dcpa_delta = dcpa_after - dcpa_before
        if dcpa_delta > 0.0:
            reward += self.cfg.dcpa_improve_weight * dcpa_delta
        else:
            reward += self.cfg.dcpa_worse_weight * dcpa_delta  # negative when worsening

        if reason == "goal":
            reward += self.cfg.goal_bonus
        elif reason == "collision":
            reward += self.cfg.collision_penalty
        elif reason == "out_of_bounds":
            reward += self.cfg.out_of_bounds_penalty

        self.prev_goal_distance = d_now

        info: Dict[str, object] = {
            "reason": reason,
            "encounter": encounter,
            "sector_12": sector,
            "relative_bearing_deg": rel_bearing_deg,
            "tcpa": tcpa,
            "dcpa_before": dcpa_before,
            "dcpa_after": dcpa_after,
            "dcpa_delta": dcpa_delta,
            "rudder_cmd": rudder_cmd,
            "throttle_cmd": throttle_cmd,
            "ego_rudder_angle_rad": self.ego.rudder,
            "ego_effective_throttle": self.ego.throttle,
        }
        return self._obs(), float(reward), done, info


if __name__ == "__main__":
    env = ColregsFeatureEnv(seed=1)
    obs = env.reset()
    ep_reward = 0.0
    while True:
        action = np.asarray([
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
        ], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        if done:
            print(f"Episode done | reason={info['reason']} | steps={env.step_count} | return={ep_reward:.3f}")
            obs = env.reset()
            ep_reward = 0.0
