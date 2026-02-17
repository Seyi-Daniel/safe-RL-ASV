from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from hyperparameters import EnvParams, RewardParams

try:
    import pygame

    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def mod2pi(a: float) -> float:
    return a % (2.0 * math.pi)


def clamp(x: float, lo: float, hi: float) -> float:
    return hi if x > hi else lo if x < lo else x


@dataclass
class Vessel:
    x: float
    y: float
    h: float
    speed: float
    goal_x: float
    goal_y: float
    rudder: float = 0.0
    throttle: float = 0.0


class SingleTargetFeatureEnv:
    """Two-vessel setup: vessel-1 straight center->circle goal, vessel-2 follows Dubins-style plan."""

    def __init__(
        self,
        env_params: EnvParams = EnvParams(),
        reward_params: RewardParams = RewardParams(),
        render: bool = False,
    ):
        self.envp = env_params
        self.rewp = reward_params
        self.rng = random.Random(self.envp.seed)

        self.agent: Optional[Vessel] = None
        self.target: Optional[Vessel] = None
        self.start_x = 0.5 * self.envp.world_w
        self.start_y = 0.5 * self.envp.world_h
        self.time = 0.0
        self.step_idx = 0
        self.max_steps = max(1, int(round(self.envp.episode_seconds / self.envp.dt)))
        self.prev_goal_d_agent = 0.0
        self.prev_goal_d_target = 0.0
        self.agent_reached = False
        self.target_reached = False

        # per-vessel telemetry
        self.agent_steps_taken = 0
        self.target_steps_taken = 0
        self.colregs_scenario = "safe"
        self.agent_role = "none"
        self.target_role = "none"
        self.risk_of_collision = False
        self.last_dcpa = float("inf")
        self.last_tcpa = float("inf")
        self.agent_rl_active = False
        self.target_rl_active = False
        self.agent_relative_bearing_deg = 0.0
        self.target_relative_bearing_deg = 0.0
        self.agent_start_speed = 0.0
        self.target_start_speed = 0.0
        self.agent_start_pos = (0.0, 0.0)
        self.target_start_pos = (0.0, 0.0)

        # vessel 2 planned path and tracking state
        self.target_plan: List[Tuple[str, float, float]] = []  # (segment_kind, duration_s, cmd)
        self.target_plan_idx = 0
        self.target_plan_elapsed = 0.0
        self.target_goal_heading = 0.0
        self.target_planner_mode = "shortest_path"
        self.target_path_word = ""
        self.target_plan_terminal_pos_err = 0.0
        self.target_plan_terminal_heading_err = 0.0

        # render-time planned path visualization
        self.show_planned_paths = True
        self.agent_planned_path: List[Tuple[float, float]] = []
        self.target_planned_path: List[Tuple[float, float]] = []

        self.render_enabled = render and HAS_PYGAME
        self._screen = None
        self._clock = None
        self._font = None
        if self.render_enabled:
            self._init_render()

    def _init_render(self) -> None:
        pygame.init()
        w = int(self.envp.world_w * self.envp.pixels_per_meter)
        h = int(self.envp.world_h * self.envp.pixels_per_meter)
        self._screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Unified Feature RL - ASV")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 18)

    def close(self) -> None:
        if self._screen is not None:
            pygame.quit()
            self._screen = None

    def sx(self, x: float) -> int:
        return int(round(x * self.envp.pixels_per_meter))

    def sy(self, y: float) -> int:
        return int(round(y * self.envp.pixels_per_meter))

    def _outside(self, v: Vessel) -> bool:
        return not (0.0 <= v.x <= self.envp.world_w and 0.0 <= v.y <= self.envp.world_h)

    def _goal_distance(self, v: Vessel) -> float:
        return math.hypot(v.goal_x - v.x, v.goal_y - v.y)

    def _distance_from_start(self, v: Vessel, start_xy: Tuple[float, float]) -> float:
        sx, sy = start_xy
        return math.hypot(v.x - sx, v.y - sy)

    def _relative_bearing_deg(self, observer: Vessel, target: Vessel) -> float:
        dx = target.x - observer.x
        dy = target.y - observer.y
        ch = math.cos(observer.h)
        sh = math.sin(observer.h)
        x_rel = ch * dx + sh * dy
        y_rel = -sh * dx + ch * dy
        rel_port = (math.degrees(math.atan2(y_rel, x_rel)) + 360.0) % 360.0
        return (360.0 - rel_port) % 360.0

    def _bearing_in_sector(self, bearing_deg: float, start_deg: float, end_deg: float) -> bool:
        b = bearing_deg % 360.0
        s = start_deg % 360.0
        e = end_deg % 360.0
        if s <= e:
            return s <= b <= e
        return b >= s or b <= e

    def _tcpa_dcpa(self, a: Vessel, b: Vessel) -> Tuple[float, float]:
        avx = math.cos(a.h) * a.speed
        avy = math.sin(a.h) * a.speed
        bvx = math.cos(b.h) * b.speed
        bvy = math.sin(b.h) * b.speed
        rx = b.x - a.x
        ry = b.y - a.y
        rvx = bvx - avx
        rvy = bvy - avy
        rv2 = rvx * rvx + rvy * rvy
        if rv2 <= 1e-8:
            return float("inf"), math.hypot(rx, ry)
        tcpa = -((rx * rvx) + (ry * rvy)) / rv2
        if tcpa < 0.0:
            tcpa = 0.0
        cx = rx + rvx * tcpa
        cy = ry + rvy * tcpa
        dcpa = math.hypot(cx, cy)
        return tcpa, dcpa

    def _classify_colregs(self) -> Dict[str, float | str]:
        own_bearing = self._relative_bearing_deg(self.agent, self.target)
        tgt_bearing = self._relative_bearing_deg(self.target, self.agent)

        head_on_half = self.envp.colregs_head_on_half_angle_deg
        head_on_min = (360.0 - head_on_half) % 360.0
        head_on_max = head_on_half
        crossing_max = self.envp.colregs_crossing_starboard_max_deg
        overtaking_max = self.envp.colregs_overtaking_aft_max_deg

        head_on = self._bearing_in_sector(own_bearing, head_on_min, head_on_max) and self._bearing_in_sector(
            tgt_bearing, head_on_min, head_on_max
        )
        if head_on:
            return {
                "scenario": "head_on",
                "agent_role": "give_way",
                "target_role": "give_way",
                "agent_bearing_deg": own_bearing,
                "target_bearing_deg": tgt_bearing,
            }

        speed_eps = self.envp.colregs_speed_eps
        agent_overtaking = self._bearing_in_sector(tgt_bearing, crossing_max, overtaking_max) and (
            self.agent.speed > self.target.speed + speed_eps
        )
        target_overtaking = self._bearing_in_sector(own_bearing, crossing_max, overtaking_max) and (
            self.target.speed > self.agent.speed + speed_eps
        )
        if agent_overtaking and not target_overtaking:
            return {
                "scenario": "overtaking",
                "agent_role": "give_way",
                "target_role": "stand_on",
                "agent_bearing_deg": own_bearing,
                "target_bearing_deg": tgt_bearing,
            }
        if target_overtaking and not agent_overtaking:
            return {
                "scenario": "overtaking",
                "agent_role": "stand_on",
                "target_role": "give_way",
                "agent_bearing_deg": own_bearing,
                "target_bearing_deg": tgt_bearing,
            }

        if self._bearing_in_sector(own_bearing, head_on_half, crossing_max):
            return {
                "scenario": "crossing",
                "agent_role": "give_way",
                "target_role": "stand_on",
                "agent_bearing_deg": own_bearing,
                "target_bearing_deg": tgt_bearing,
            }
        if self._bearing_in_sector(own_bearing, overtaking_max, head_on_min):
            return {
                "scenario": "crossing",
                "agent_role": "stand_on",
                "target_role": "give_way",
                "agent_bearing_deg": own_bearing,
                "target_bearing_deg": tgt_bearing,
            }
        return {
            "scenario": "safe",
            "agent_role": "none",
            "target_role": "none",
            "agent_bearing_deg": own_bearing,
            "target_bearing_deg": tgt_bearing,
        }

    def _point_on_big_circle(self, ang: float) -> Tuple[float, float]:
        r = self.envp.target_outer_radius
        return self.start_x + r * math.cos(ang), self.start_y + r * math.sin(ang)

    def _arc_gap(self, a0: float, a1: float) -> float:
        d = abs(wrap_pi(a1 - a0))
        return min(d, abs(2.0 * math.pi - d))

    def _sample_end_heading_candidates(self, sx: float, sy: float, gx: float, gy: float) -> List[float]:
        cx, cy = self.start_x, self.start_y
        radx = gx - cx
        rady = gy - cy
        tang_cw = math.atan2(rady, radx) - 0.5 * math.pi
        tang_ccw = math.atan2(rady, radx) + 0.5 * math.pi
        to_center = math.atan2(cy - gy, cx - gx)
        chord = math.atan2(gy - sy, gx - sx)

        base = [tang_cw, tang_ccw, to_center, chord]
        sweep = math.radians(self.envp.dubins_heading_sweep_deg)
        n = max(0, int(self.envp.dubins_heading_choices))

        out: List[float] = []
        for b in base:
            out.append(wrap_pi(b))
            for k in range(1, n + 1):
                out.append(wrap_pi(b + k * sweep))
                out.append(wrap_pi(b - k * sweep))
        return out

    # ----- exact Dubins words in normalized coordinates -----
    def _dubins_lsl(self, alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
        tmp = d + math.sin(alpha) - math.sin(beta)
        p2 = 2.0 + d * d - 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) - math.sin(beta))
        if p2 < 0.0:
            return None
        tmp2 = math.atan2(math.cos(beta) - math.cos(alpha), tmp)
        t = mod2pi(-alpha + tmp2)
        p = math.sqrt(p2)
        q = mod2pi(beta - tmp2)
        return t, p, q

    def _dubins_rsr(self, alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
        tmp = d - math.sin(alpha) + math.sin(beta)
        p2 = 2.0 + d * d - 2.0 * math.cos(alpha - beta) + 2.0 * d * (-math.sin(alpha) + math.sin(beta))
        if p2 < 0.0:
            return None
        tmp2 = math.atan2(math.cos(alpha) - math.cos(beta), tmp)
        t = mod2pi(alpha - tmp2)
        p = math.sqrt(p2)
        q = mod2pi(-beta + tmp2)
        return t, p, q

    def _dubins_lsr(self, alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
        p2 = -2.0 + d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) + math.sin(beta))
        if p2 < 0.0:
            return None
        p = math.sqrt(p2)
        tmp2 = math.atan2(-math.cos(alpha) - math.cos(beta), d + math.sin(alpha) + math.sin(beta)) - math.atan2(-2.0, p)
        t = mod2pi(-alpha + tmp2)
        q = mod2pi(-beta + tmp2)
        return t, p, q

    def _dubins_rsl(self, alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
        p2 = -2.0 + d * d + 2.0 * math.cos(alpha - beta) - 2.0 * d * (math.sin(alpha) + math.sin(beta))
        if p2 < 0.0:
            return None
        p = math.sqrt(p2)
        tmp2 = math.atan2(math.cos(alpha) + math.cos(beta), d - math.sin(alpha) - math.sin(beta)) - math.atan2(2.0, p)
        t = mod2pi(alpha - tmp2)
        q = mod2pi(beta - tmp2)
        return t, p, q

    def _dubins_rlr(self, alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
        tmp = (6.0 - d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) - math.sin(beta))) / 8.0
        if abs(tmp) > 1.0:
            return None
        p = mod2pi(2.0 * math.pi - math.acos(tmp))
        t = mod2pi(alpha - math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta)) + 0.5 * p)
        q = mod2pi(alpha - beta - t + p)
        return t, p, q

    def _dubins_lrl(self, alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
        tmp = (6.0 - d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (-math.sin(alpha) + math.sin(beta))) / 8.0
        if abs(tmp) > 1.0:
            return None
        p = mod2pi(2.0 * math.pi - math.acos(tmp))
        t = mod2pi(-alpha - math.atan2(math.cos(alpha) - math.cos(beta), d + math.sin(alpha) - math.sin(beta)) + 0.5 * p)
        q = mod2pi(beta - alpha - t + p)
        return t, p, q

    def _exact_dubins_candidates(self, sx: float, sy: float, sh: float, gx: float, gy: float, gh: float, radius: float) -> List[Tuple[str, Tuple[float, float, float], float]]:
        dx = gx - sx
        dy = gy - sy
        D = math.hypot(dx, dy)
        d = D / max(1e-9, radius)
        theta = math.atan2(dy, dx)
        alpha = mod2pi(sh - theta)
        beta = mod2pi(gh - theta)

        words = [
            ("LSL", self._dubins_lsl),
            ("RSR", self._dubins_rsr),
            ("LSR", self._dubins_lsr),
            ("RSL", self._dubins_rsl),
            ("RLR", self._dubins_rlr),
            ("LRL", self._dubins_lrl),
        ]
        out: List[Tuple[str, Tuple[float, float, float], float]] = []
        for word, fn in words:
            params = fn(alpha, beta, d)
            if params is None:
                continue
            lengths = (params[0] * radius, params[1] * radius, params[2] * radius)
            out.append((word, lengths, sum(lengths)))
        return out

    def _segment_cost(self, mode: str, lengths: Tuple[float, float, float], word: str, heading_err: float, pos_err: float, chord: float) -> float:
        total_len = sum(lengths)
        steering_len = lengths[0] + lengths[2] + (lengths[1] if word[1] in {"L", "R"} else 0.0)
        curvature_proxy = steering_len
        straight_dev = abs(total_len - chord)

        if mode == "minimum_steering_effort":
            return steering_len + 0.15 * total_len + 6.0 * pos_err + heading_err
        if mode == "minimum_curvature_change":
            return curvature_proxy + 0.2 * steering_len + 5.0 * pos_err + heading_err
        if mode == "closest_to_straight_line":
            return straight_dev + 0.2 * steering_len + 7.0 * pos_err + heading_err
        return total_len + 0.25 * steering_len + 5.0 * pos_err + heading_err

    def _simulate_plan_endpoint_ideal(self, sx: float, sy: float, sh: float, speed: float, plan: List[Tuple[str, float, float]]) -> Tuple[float, float, float]:
        x, y, h = sx, sy, sh
        yaw_rate = self.envp.rudder_max_yaw_rate_rad_s
        for _, duration, cmd in plan:
            t = 0.0
            while t < duration:
                step = min(0.05, duration - t)
                h = wrap_pi(h + cmd * yaw_rate * step)
                x += speed * math.cos(h) * step
                y += speed * math.sin(h) * step
                t += step
        return x, y, h

    def _sample_dubins_style_plan(self, sx: float, sy: float, sh: float, gx: float, gy: float, speed: float) -> Tuple[List[Tuple[str, float, float]], float, str, str, float, float]:
        yaw_rate = self.envp.rudder_max_yaw_rate_rad_s
        radius = max(1e-6, speed / max(1e-6, yaw_rate))
        chord = math.hypot(gx - sx, gy - sy)

        modes = [
            "shortest_path",
            "minimum_steering_effort",
            "minimum_curvature_change",
            "closest_to_straight_line",
        ]
        mode = self.rng.choice(modes)

        best_plan: List[Tuple[str, float, float]] = []
        best_goal_h = sh
        best_word = ""
        best_pos_err = float("inf")
        best_heading_err = float("inf")
        best_score = float("inf")

        for gh in self._sample_end_heading_candidates(sx, sy, gx, gy):
            for word, lengths, _ in self._exact_dubins_candidates(sx, sy, sh, gx, gy, gh, radius):
                # Convert geometric lengths to durations for constant speed execution.
                durations = tuple(l / max(1e-6, speed) for l in lengths)
                seg_cmd = {"L": 1.0, "R": -1.0, "S": 0.0}
                plan = [
                    (word[0], durations[0], seg_cmd[word[0]]),
                    (word[1], durations[1], seg_cmd[word[1]]),
                    (word[2], durations[2], seg_cmd[word[2]]),
                ]

                ex, ey, eh = self._simulate_plan_endpoint_ideal(sx, sy, sh, speed, plan)
                pos_err = math.hypot(gx - ex, gy - ey)
                heading_err = abs(wrap_pi(gh - eh))
                score = self._segment_cost(mode, lengths, word, heading_err, pos_err, chord) + 20.0 * pos_err
                if score < best_score:
                    best_score = score
                    best_plan = plan
                    best_goal_h = gh
                    best_word = word
                    best_pos_err = pos_err
                    best_heading_err = heading_err

        # Fallback (very rare): simple straight segment.
        if not best_plan:
            dur = chord / max(1e-6, speed)
            best_plan = [("S", dur, 0.0)]
            best_word = "S"
            best_goal_h = math.atan2(gy - sy, gx - sx)
            best_pos_err = 0.0
            best_heading_err = abs(wrap_pi(best_goal_h - sh))

        return best_plan, best_goal_h, mode, best_word, best_pos_err, best_heading_err

    def _integrate_rudder_heading(self, v: Vessel, rudder_cmd: float, dt: float) -> None:
        rudder_cmd = clamp(rudder_cmd, -1.0, 1.0)
        rudder_target = rudder_cmd * self.envp.rudder_max_angle_rad
        rudder_step = self.envp.rudder_max_rate_rad_s * dt
        v.rudder = clamp(
            v.rudder + clamp(rudder_target - v.rudder, -rudder_step, rudder_step),
            -self.envp.rudder_max_angle_rad,
            self.envp.rudder_max_angle_rad,
        )
        yaw_rate = (v.rudder / max(1e-6, self.envp.rudder_max_angle_rad)) * self.envp.rudder_max_yaw_rate_rad_s
        v.h = wrap_pi(v.h + yaw_rate * dt)

    def _advance_straight(self, v: Vessel, reached_attr: str, dt: float) -> None:
        if getattr(self, reached_attr):
            return

        d = self._goal_distance(v)
        if d <= self.envp.goal_radius:
            setattr(self, reached_attr, True)
            v.speed = 0.0
            return

        travel = min(v.speed * dt, d)
        v.x += math.cos(v.h) * travel
        v.y += math.sin(v.h) * travel

        if travel + 1e-9 >= d:
            setattr(self, reached_attr, True)
            v.speed = 0.0

    def _advance_controlled(self, v: Vessel, reached_attr: str, rudder_cmd: float, throttle_cmd: float, dt: float) -> None:
        if getattr(self, reached_attr):
            return

        d = self._goal_distance(v)
        if d <= self.envp.goal_radius:
            setattr(self, reached_attr, True)
            v.speed = 0.0
            return

        self._integrate_rudder_heading(v, rudder_cmd, dt)

        throttle_target = clamp(throttle_cmd, -1.0, 1.0)
        throttle_step = self.envp.throttle_slew_rate * dt
        v.throttle = clamp(v.throttle + clamp(throttle_target - v.throttle, -throttle_step, throttle_step), -1.0, 1.0)

        if abs(v.throttle) <= self.envp.throttle_deadband:
            accel = 0.0
        elif v.throttle > 0.0:
            accel = self.envp.accel_rate * v.throttle
        else:
            accel = self.envp.decel_rate * v.throttle

        v.speed = clamp(v.speed + accel * dt, self.envp.min_speed, self.envp.max_speed)

        travel = min(v.speed * dt, d)
        v.x += math.cos(v.h) * travel
        v.y += math.sin(v.h) * travel

        if travel + 1e-9 >= d:
            setattr(self, reached_attr, True)
            v.speed = 0.0

    def _advance_target_plan(self, dt: float) -> None:
        if self.target is None or self.target_reached:
            return

        remaining = dt
        while remaining > 1e-9 and not self.target_reached:
            d_goal = self._goal_distance(self.target)
            if d_goal <= self.envp.goal_radius:
                self.target_reached = True
                self.target.speed = 0.0
                break

            if self.target_plan_idx < len(self.target_plan):
                seg_kind, seg_dur, cmd = self.target_plan[self.target_plan_idx]
                seg_rem = max(0.0, seg_dur - self.target_plan_elapsed)
                if seg_rem <= 1e-9:
                    self.target_plan_idx += 1
                    self.target_plan_elapsed = 0.0
                    continue
                step = min(remaining, seg_rem)
                desired_cmd = cmd
            else:
                # Post-plan terminal guidance: smoothly steer toward goal heading.
                step = remaining
                goal_bearing = math.atan2(self.target.goal_y - self.target.y, self.target.goal_x - self.target.x)
                err = wrap_pi(goal_bearing - self.target.h)
                desired_cmd = clamp(err / math.radians(25.0), -1.0, 1.0)

            self._integrate_rudder_heading(self.target, desired_cmd, step)

            d_goal = self._goal_distance(self.target)
            travel = min(self.target.speed * step, d_goal)
            self.target.x += travel * math.cos(self.target.h)
            self.target.y += travel * math.sin(self.target.h)

            if travel + 1e-9 >= d_goal:
                self.target_reached = True
                self.target.speed = 0.0
                break

            remaining -= step
            if self.target_plan_idx < len(self.target_plan):
                self.target_plan_elapsed += step
                if self.target_plan_elapsed + 1e-9 >= self.target_plan[self.target_plan_idx][1]:
                    self.target_plan_idx += 1
                    self.target_plan_elapsed = 0.0


    def _build_agent_planned_path(self) -> None:
        if self.agent is None:
            self.agent_planned_path = []
            return
        self.agent_planned_path = [(self.start_x, self.start_y), (self.agent.goal_x, self.agent.goal_y)]

    def _build_target_planned_path(self, sx: float, sy: float, sh: float, speed: float, goal_x: float, goal_y: float) -> None:
        # Build a visualized path by replaying the same dynamics used during execution.
        sim = Vessel(sx, sy, sh, speed, goal_x, goal_y, rudder=0.0, throttle=0.0)
        pts: List[Tuple[float, float]] = [(sim.x, sim.y)]

        local_idx = 0
        local_elapsed = 0.0
        dt = self.envp.dt / max(1, self.envp.substeps)
        max_sim_steps = max(2000, int(2.0 * self.max_steps * max(1, self.envp.substeps)))

        for _ in range(max_sim_steps):
            d_goal = math.hypot(goal_x - sim.x, goal_y - sim.y)
            if d_goal <= self.envp.goal_radius:
                break

            if local_idx < len(self.target_plan):
                _, seg_dur, cmd = self.target_plan[local_idx]
                seg_rem = max(0.0, seg_dur - local_elapsed)
                if seg_rem <= 1e-9:
                    local_idx += 1
                    local_elapsed = 0.0
                    continue
                step = min(dt, seg_rem)
                desired_cmd = cmd
            else:
                step = dt
                goal_bearing = math.atan2(goal_y - sim.y, goal_x - sim.x)
                err = wrap_pi(goal_bearing - sim.h)
                desired_cmd = clamp(err / math.radians(25.0), -1.0, 1.0)

            self._integrate_rudder_heading(sim, desired_cmd, step)
            d_goal = math.hypot(goal_x - sim.x, goal_y - sim.y)
            travel = min(sim.speed * step, d_goal)
            sim.x += travel * math.cos(sim.h)
            sim.y += travel * math.sin(sim.h)
            pts.append((sim.x, sim.y))

            if travel + 1e-9 >= d_goal:
                break

            if local_idx < len(self.target_plan):
                local_elapsed += step
                if local_elapsed + 1e-9 >= self.target_plan[local_idx][1]:
                    local_idx += 1
                    local_elapsed = 0.0

        self.target_planned_path = pts

    def get_obs(self) -> np.ndarray:
        return np.asarray(
            [
                self.agent.x / self.envp.world_w,
                self.agent.y / self.envp.world_h,
                self.agent.h / math.pi,
                self.agent.speed / self.envp.max_speed,
                self.agent.goal_x / self.envp.world_w,
                self.agent.goal_y / self.envp.world_h,
                self.target.x / self.envp.world_w,
                self.target.y / self.envp.world_h,
                self.target.h / math.pi,
                self.target.speed / self.envp.target_max_speed,
                self.target.goal_x / self.envp.world_w,
                self.target.goal_y / self.envp.world_h,
            ],
            dtype=np.float32,
        )

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)

        # Vessel 1: center -> random point on big circle, straight-line heading.
        goal_ang_1 = self.rng.uniform(0.0, 2.0 * math.pi)
        agx, agy = self._point_on_big_circle(goal_ang_1)
        ah = math.atan2(agy - self.start_y, agx - self.start_x)
        aspeed = self.rng.uniform(self.envp.target_min_speed, self.envp.target_max_speed)
        self.agent = Vessel(self.start_x, self.start_y, ah, aspeed, agx, agy)
        self.agent_start_speed = aspeed
        self.agent_start_pos = (self.agent.x, self.agent.y)

        # Vessel 2: random start/goal on big circle + randomized initial heading.
        start_ang_2 = self.rng.uniform(0.0, 2.0 * math.pi)
        goal_ang_2 = self.rng.uniform(0.0, 2.0 * math.pi)
        tries = 0
        while self._arc_gap(start_ang_2, goal_ang_2) < math.radians(20.0) and tries < 40:
            goal_ang_2 = self.rng.uniform(0.0, 2.0 * math.pi)
            tries += 1

        sx2, sy2 = self._point_on_big_circle(start_ang_2)
        gx2, gy2 = self._point_on_big_circle(goal_ang_2)
        sh2 = self.rng.uniform(-math.pi, math.pi)
        sp2 = self.rng.uniform(self.envp.target_min_speed, self.envp.target_max_speed)
        self.target = Vessel(sx2, sy2, sh2, sp2, gx2, gy2)
        self.target_start_speed = sp2
        self.target_start_pos = (self.target.x, self.target.y)

        (
            self.target_plan,
            self.target_goal_heading,
            self.target_planner_mode,
            self.target_path_word,
            self.target_plan_terminal_pos_err,
            self.target_plan_terminal_heading_err,
        ) = self._sample_dubins_style_plan(sx2, sy2, sh2, gx2, gy2, sp2)

        self.target_plan_idx = 0
        self.target_plan_elapsed = 0.0

        self.time = 0.0
        self.step_idx = 0
        self.agent_reached = False
        self.target_reached = False

        self.prev_goal_d_agent = self._goal_distance(self.agent)
        self.prev_goal_d_target = self._goal_distance(self.target)
        self.agent_steps_taken = 0
        self.target_steps_taken = 0
        self.colregs_scenario = "safe"
        self.agent_role = "none"
        self.target_role = "none"
        self.risk_of_collision = False
        self.last_dcpa = float("inf")
        self.last_tcpa = float("inf")
        self.agent_rl_active = False
        self.target_rl_active = False
        self.agent_relative_bearing_deg = 0.0
        self.target_relative_bearing_deg = 0.0

        t1 = self.prev_goal_d_agent / max(1e-6, self.agent.speed)
        planned_t2 = sum(d for _, d, _ in self.target_plan)
        t2 = max(planned_t2, self.prev_goal_d_target / max(1e-6, self.target.speed))
        episode_time = max(self.envp.episode_seconds, 1.4 * max(t1, t2) + 20.0)
        self.max_steps = max(1, int(round(episode_time / self.envp.dt)))

        self._build_agent_planned_path()
        self._build_target_planned_path(sx2, sy2, sh2, sp2, gx2, gy2)

        return self.get_obs()

    def step(self, action: Union[np.ndarray, Tuple[float, float], list]) -> Tuple[np.ndarray, float, bool, Dict[str, float | str | int]]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size < 2:
            raise ValueError("Action must contain [rudder_cmd, throttle_cmd].")
        rudder_cmd = clamp(float(a[0]), -1.0, 1.0)
        throttle_cmd = clamp(float(a[1]), -1.0, 1.0)

        encounter = self._classify_colregs()
        self.colregs_scenario = str(encounter["scenario"])
        self.agent_role = str(encounter["agent_role"])
        self.target_role = str(encounter["target_role"])
        self.agent_relative_bearing_deg = float(encounter["agent_bearing_deg"])
        self.target_relative_bearing_deg = float(encounter["target_bearing_deg"])

        tcpa, dcpa = self._tcpa_dcpa(self.agent, self.target)
        self.last_tcpa = tcpa
        self.last_dcpa = dcpa
        self.risk_of_collision = (0.0 <= tcpa <= self.envp.tcpa_risk_threshold) and (dcpa <= self.envp.dcpa_risk_threshold)

        agent_dist = self._distance_from_start(self.agent, self.agent_start_pos)
        target_dist = self._distance_from_start(self.target, self.target_start_pos)

        self.agent_rl_active = self.risk_of_collision and self.agent_role == "give_way" and agent_dist >= self.envp.rl_takeover_distance
        self.target_rl_active = self.risk_of_collision and self.target_role == "give_way" and target_dist >= self.envp.rl_takeover_distance

        h = self.envp.dt / max(1, self.envp.substeps)
        was_agent_active = not self.agent_reached
        was_target_active = not self.target_reached
        for _ in range(max(1, self.envp.substeps)):
            if self.agent_role == "give_way":
                if self.agent_rl_active:
                    self._advance_controlled(self.agent, "agent_reached", rudder_cmd, throttle_cmd, h)
                else:
                    self._advance_straight(self.agent, "agent_reached", h)
            else:
                self._advance_straight(self.agent, "agent_reached", h)

            if self.target_role == "give_way":
                if self.target_rl_active:
                    self._advance_controlled(self.target, "target_reached", rudder_cmd, throttle_cmd, h)
                else:
                    self._advance_straight(self.target, "target_reached", h)
            else:
                self._advance_target_plan(h)

        if was_agent_active:
            self.agent_steps_taken += 1
        if was_target_active:
            self.target_steps_taken += 1

        self.time += self.envp.dt
        self.step_idx += 1

        done = False
        reason = ""
        if self._outside(self.agent) or self._outside(self.target):
            done, reason = True, "out_of_bounds"
        elif self.step_idx >= self.max_steps:
            done, reason = True, "timeout"
        elif self.agent_reached and self.target_reached:
            done, reason = True, "both_reached"

        reward = self.rewp.living_penalty
        d_agent = self._goal_distance(self.agent)
        d_target = self._goal_distance(self.target)
        reward += self.rewp.progress_weight * (self.prev_goal_d_agent - d_agent)
        reward += self.rewp.progress_weight * (self.prev_goal_d_target - d_target)

        if self.agent_reached and self.target_reached and reason == "both_reached":
            reward += self.rewp.goal_bonus

        if reason == "out_of_bounds":
            reward += self.rewp.out_of_bounds_penalty

        self.prev_goal_d_agent = d_agent
        self.prev_goal_d_target = d_target

        durations = ",".join(f"{d:.2f}" for _, d, _ in self.target_plan)
        info: Dict[str, float | str | int] = {
            "reason": reason,
            "agent_goal_distance": d_agent,
            "target_goal_distance": d_target,
            "agent_reached": int(self.agent_reached),
            "target_reached": int(self.target_reached),
            "rudder_cmd": rudder_cmd,
            "throttle_cmd": throttle_cmd,
            "target_planner_mode": self.target_planner_mode,
            "target_path_word": self.target_path_word,
            "target_plan_durations": durations,
            "target_plan_terminal_pos_err": float(self.target_plan_terminal_pos_err),
            "target_plan_terminal_heading_err": float(self.target_plan_terminal_heading_err),
            "agent_steps_taken": int(self.agent_steps_taken),
            "target_steps_taken": int(self.target_steps_taken),
            "agent_start_speed": float(self.agent_start_speed),
            "target_start_speed": float(self.target_start_speed),
            "agent_heading_deg": float(math.degrees(self.agent.h)),
            "target_heading_deg": float(math.degrees(self.target.h)),
            "agent_rudder_deg": float(math.degrees(self.agent.rudder)),
            "target_rudder_deg": float(math.degrees(self.target.rudder)),
            "dcpa": float(dcpa),
            "tcpa": float(tcpa),
            "risk_of_collision": int(self.risk_of_collision),
            "colregs_scenario": self.colregs_scenario,
            "agent_role": self.agent_role,
            "target_role": self.target_role,
            "agent_rl_active": int(self.agent_rl_active),
            "target_rl_active": int(self.target_rl_active),
            "agent_distance_from_start": float(agent_dist),
            "target_distance_from_start": float(target_dist),
            "agent_relative_bearing_deg": float(self.agent_relative_bearing_deg),
            "target_relative_bearing_deg": float(self.target_relative_bearing_deg),
        }
        return self.get_obs(), float(reward), done, info
    def render(self) -> None:
        if not self.render_enabled or self._screen is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.show_planned_paths = not self.show_planned_paths

        surf = self._screen
        surf.fill((17, 58, 92))

        pygame.draw.rect(
            surf,
            (170, 170, 170),
            (0, 0, self.sx(self.envp.world_w), self.sy(self.envp.world_h)),
            2,
        )

        if self.envp.show_grid:
            step = 50
            for x in range(0, int(self.envp.world_w) + 1, step):
                pygame.draw.line(surf, (40, 80, 110), (self.sx(x), 0), (self.sx(x), self.sy(self.envp.world_h)))
            for y in range(0, int(self.envp.world_h) + 1, step):
                pygame.draw.line(surf, (40, 80, 110), (0, self.sy(y)), (self.sx(self.envp.world_w), self.sy(y)))

        self._draw_goal(self.agent.goal_x, self.agent.goal_y, (250, 215, 60))
        self._draw_goal(self.target.goal_x, self.target.goal_y, (255, 140, 90))

        if self.envp.show_spawn_rings:
            pygame.draw.circle(
                self._screen,
                (255, 225, 120),
                (self.sx(self.start_x), self.sy(self.start_y)),
                int(round(self.envp.target_outer_radius * self.envp.pixels_per_meter)),
                1,
            )

        if self.show_planned_paths:
            self._draw_planned_path(self.agent_planned_path, (150, 210, 255))
            self._draw_planned_path(self.target_planned_path, (255, 170, 170))

        self._draw_vessel(self.agent, (95, 170, 255), "V1")
        self._draw_vessel(self.target, (255, 120, 120), "V2")

        tcpa_txt = "inf" if math.isinf(self.last_tcpa) else f"{self.last_tcpa:.1f}s"
        hud0 = self._font.render(
            f"sim_step={self.step_idx}  t={self.time:.1f}s  paths[P]={int(self.show_planned_paths)}  mode={self.target_planner_mode[:4].upper()}  word={self.target_path_word}",
            True,
            (255, 255, 255),
        )
        hud1 = self._font.render(
            (
                f"COLREGS={self.colregs_scenario}  risk={'YES' if self.risk_of_collision else 'NO'}  "
                f"DCPA={self.last_dcpa:.2f}m  TCPA={tcpa_txt}  BRG={self.agent_relative_bearing_deg:.1f}deg"
            ),
            True,
            (255, 240, 170),
        )
        hud2 = self._font.render(
            (
                f"V1 role={self.agent_role} rl={int(self.agent_rl_active)} dist={self._distance_from_start(self.agent, self.agent_start_pos):.1f}/{self.envp.rl_takeover_distance:.1f} "
                f"spd={self.agent.speed:.2f} hdg={math.degrees(self.agent.h):.1f} rud={math.degrees(self.agent.rudder):.1f}"
            ),
            True,
            (170, 220, 255),
        )
        hud3 = self._font.render(
            (
                f"V2 role={self.target_role} rl={int(self.target_rl_active)} dist={self._distance_from_start(self.target, self.target_start_pos):.1f}/{self.envp.rl_takeover_distance:.1f} "
                f"spd={self.target.speed:.2f} hdg={math.degrees(self.target.h):.1f} rud={math.degrees(self.target.rudder):.1f}"
            ),
            True,
            (255, 190, 190),
        )
        surf.blit(hud0, (10, 8))
        surf.blit(hud1, (10, 26))
        surf.blit(hud2, (10, 44))
        surf.blit(hud3, (10, 62))

        pygame.display.flip()
        self._clock.tick(self.envp.render_fps)

    def _draw_planned_path(self, pts: List[Tuple[float, float]], color: Tuple[int, int, int]) -> None:
        if len(pts) < 2:
            return
        pix = [(self.sx(x), self.sy(y)) for x, y in pts]
        pygame.draw.lines(self._screen, color, False, pix, 2)

    def _draw_goal(self, gx: float, gy: float, color: Tuple[int, int, int]) -> None:
        pygame.draw.circle(self._screen, color, (self.sx(gx), self.sy(gy)), 6)

    def _draw_vessel(self, v: Vessel, color: Tuple[int, int, int], label: str) -> None:
        L, W = 6.0, 2.2
        verts = [(0.5 * L, 0.0), (-0.5 * L, -0.5 * W), (-0.5 * L, 0.5 * W)]
        ch, sh = math.cos(v.h), math.sin(v.h)
        pts = []
        for vx, vy in verts:
            wx = v.x + vx * ch - vy * sh
            wy = v.y + vx * sh + vy * ch
            pts.append((self.sx(wx), self.sy(wy)))
        pygame.draw.polygon(self._screen, color, pts)
        pygame.draw.circle(
            self._screen,
            (255, 255, 255),
            (self.sx(v.x), self.sy(v.y)),
            int(max(2, self.envp.vessel_outline_radius * self.envp.pixels_per_meter)),
            1,
        )
        txt = self._font.render(label, True, (255, 255, 255))
        self._screen.blit(txt, (self.sx(v.x) + 6, self.sy(v.y) - 8))
