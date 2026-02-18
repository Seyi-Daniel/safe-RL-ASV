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
    """Two-vessel setup: vessel-1 straight center->circle goal, vessel-2 follows Bézier path."""

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

        # Bézier path state
        self.target_bezier_waypoints: List[Tuple[float, float, float]] = []  # (x, y, heading_rad)
        self.target_bezier_wp_idx: int = 0
        self.target_bezier_tangent_scale: float = 1.0
        self.target_end_heading: float = 0.0

        # render-time planned path visualization
        self.show_planned_paths = True
        self.agent_planned_path: List[Tuple[float, float]] = []
        self.target_planned_path: List[Tuple[float, float]] = []

        self.render_enabled = render and HAS_PYGAME
        self.paused = False
        self.risk_overlay_active = False
        self.risk_overlay_payload: Dict[str, float | str | int] = {}
        self.rl_ever_triggered: bool = False  # latches True when RL first activates, never resets within episode
        self.rl_overlay_shown: bool = False  # True after overlay has been shown once this episode
        self.prev_agent_rl_active = False
        self.prev_target_rl_active = False
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
        # Do not classify if either vessel has already reached its goal.
        if self.agent_reached or self.target_reached:
            return {
                "scenario": "safe",
                "agent_role": "none",
                "target_role": "none",
                "agent_bearing_deg": 0.0,
                "target_bearing_deg": 0.0,
            }

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

        # Crossing: target on agent's starboard bow (agent gives way per Rule 15).
        if self._bearing_in_sector(own_bearing, head_on_half, crossing_max):
            return {
                "scenario": "crossing",
                "agent_role": "give_way",
                "target_role": "stand_on",
                "agent_bearing_deg": own_bearing,
                "target_bearing_deg": tgt_bearing,
            }

        # Crossing: target on agent's port bow (agent stands on, target gives way per Rule 15).
        if self._bearing_in_sector(own_bearing, 360.0 - crossing_max, head_on_min):
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

    def _inward_facing_heading(self, pos_x: float, pos_y: float) -> float:
        """Sample a heading that points into the circle interior."""
        cx, cy = self.start_x, self.start_y
        to_center_angle = math.atan2(cy - pos_y, cx - pos_x)
        offset = self.rng.uniform(-0.5 * math.pi, 0.5 * math.pi)
        return wrap_pi(to_center_angle + offset)

    def _outward_facing_heading(self, pos_x: float, pos_y: float) -> float:
        """Sample a heading that approaches the boundary point from inside the circle."""
        cx, cy = self.start_x, self.start_y
        from_center_angle = math.atan2(pos_y - cy, pos_x - cx)
        offset = self.rng.uniform(-0.5 * math.pi, 0.5 * math.pi)
        return wrap_pi(from_center_angle + offset)


    def _cubic_bezier_point(self, t: float, p0, p1, p2, p3) -> Tuple[float, float]:
        u = 1.0 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        return x, y

    def _cubic_bezier_derivative(self, t: float, p0, p1, p2, p3) -> Tuple[float, float]:
        u = 1.0 - t
        dx = 3 * u**2 * (p1[0] - p0[0]) + 6 * u * t * (p2[0] - p1[0]) + 3 * t**2 * (p3[0] - p2[0])
        dy = 3 * u**2 * (p1[1] - p0[1]) + 6 * u * t * (p2[1] - p1[1]) + 3 * t**2 * (p3[1] - p2[1])
        return dx, dy

    def _cubic_bezier_second_derivative(self, t: float, p0, p1, p2, p3) -> Tuple[float, float]:
        u = 1.0 - t
        ddx = 6 * u * (p2[0] - 2 * p1[0] + p0[0]) + 6 * t * (p3[0] - 2 * p2[0] + p1[0])
        ddy = 6 * u * (p2[1] - 2 * p1[1] + p0[1]) + 6 * t * (p3[1] - 2 * p2[1] + p1[1])
        return ddx, ddy

    def _bezier_curvature(self, t: float, p0, p1, p2, p3) -> float:
        dx, dy = self._cubic_bezier_derivative(t, p0, p1, p2, p3)
        ddx, ddy = self._cubic_bezier_second_derivative(t, p0, p1, p2, p3)
        denom = (dx * dx + dy * dy) ** 1.5
        if denom < 1e-12:
            return 0.0
        return abs(dx * ddy - dy * ddx) / denom

    def _build_bezier_waypoints(
        self,
        sx: float,
        sy: float,
        sh: float,
        gx: float,
        gy: float,
        gh: float,
        speed: float,
    ) -> List[Tuple[float, float, float]]:
        chord = math.hypot(gx - sx, gy - sy)
        if chord < 1e-6:
            return [(gx, gy, gh)]

        max_kappa = self.envp.rudder_max_yaw_rate_rad_s / max(speed, 1e-6)
        max_dkappa_ds = self.envp.rudder_max_rate_rad_s / max(speed, 1e-6)

        path_style = self.rng.random()
        single_turn_threshold = self.envp.bezier_style_straight_prob + self.envp.bezier_style_single_turn_prob
        if path_style < self.envp.bezier_style_straight_prob:
            f1 = self.rng.uniform(0.6, 0.9)
            f2 = self.rng.uniform(0.6, 0.9)
        elif path_style < single_turn_threshold:
            if self.rng.random() < 0.5:
                f1 = self.rng.uniform(self.envp.bezier_tangent_max_fraction, 0.85)
                f2 = self.rng.uniform(self.envp.bezier_tangent_min_fraction, 0.35)
            else:
                f1 = self.rng.uniform(self.envp.bezier_tangent_min_fraction, 0.35)
                f2 = self.rng.uniform(self.envp.bezier_tangent_max_fraction, 0.85)
        else:
            f1 = self.rng.uniform(self.envp.bezier_tangent_min_fraction, self.envp.bezier_tangent_max_fraction)
            f2 = self.rng.uniform(self.envp.bezier_tangent_min_fraction, self.envp.bezier_tangent_max_fraction)

        p0 = (sx, sy)
        p3 = (gx, gy)
        n_check = 200
        cx, cy = self.start_x, self.start_y
        r = self.envp.target_outer_radius

        def _check_scale(scale: float) -> Tuple[bool, Tuple[float, float], Tuple[float, float]]:
            t1 = f1 * chord * scale
            t2 = f2 * chord * scale
            p1 = (sx + t1 * math.cos(sh), sy + t1 * math.sin(sh))
            p2 = (gx - t2 * math.cos(gh), gy - t2 * math.sin(gh))

            kappas: List[float] = []
            for i in range(n_check + 1):
                t = i / n_check
                k = self._bezier_curvature(t, p0, p1, p2, p3)
                kappas.append(k)
                if k > max_kappa:
                    return False, p1, p2

            for i in range(1, len(kappas)):
                t_mid = (i - 0.5) / n_check
                dx, dy = self._cubic_bezier_derivative(t_mid, p0, p1, p2, p3)
                ds_dt = math.hypot(dx, dy)
                if ds_dt < 1e-9:
                    continue
                dkappa_dt = abs(kappas[i] - kappas[i - 1])
                dkappa_ds = dkappa_dt / ds_dt * n_check
                if dkappa_ds > max_dkappa_ds:
                    return False, p1, p2

            for i in range(n_check + 1):
                t = i / n_check
                px, py = self._cubic_bezier_point(t, p0, p1, p2, p3)
                if math.hypot(px - cx, py - cy) > r + 1e-6:
                    return False, p1, p2

            return True, p1, p2

        step = self.envp.bezier_curvature_scale_step
        scales: List[float] = [1.0]
        for k in range(1, self.envp.bezier_max_scale_iterations + 1):
            scales.append(step**k)
            scales.append(step**(-k))

        selected_scale = None
        selected_p1 = p0
        selected_p2 = p3
        for scale in scales[: self.envp.bezier_max_scale_iterations]:
            valid, p1, p2 = _check_scale(scale)
            if valid:
                selected_scale = scale
                selected_p1 = p1
                selected_p2 = p2
                break

        spacing = self.envp.bezier_waypoint_spacing_m
        if selected_scale is None:
            self.target_bezier_tangent_scale = 1.0
            n_straight = max(10, int(chord / max(1e-6, spacing)))
            straight_heading = math.atan2(gy - sy, gx - sx)
            pts = [
                (
                    sx + (gx - sx) * i / n_straight,
                    sy + (gy - sy) * i / n_straight,
                    straight_heading,
                )
                for i in range(n_straight + 1)
            ]
        else:
            self.target_bezier_tangent_scale = selected_scale
            n_sample = max(500, int(chord * 10))
            pts = []
            for i in range(n_sample + 1):
                t = i / n_sample
                x, y = self._cubic_bezier_point(t, p0, selected_p1, selected_p2, p3)
                dx, dy = self._cubic_bezier_derivative(t, p0, selected_p1, selected_p2, p3)
                heading = math.atan2(dy, dx) if (abs(dx) > 1e-9 or abs(dy) > 1e-9) else sh
                pts.append((x, y, heading))

        waypoints = [pts[0]]
        accumulated = 0.0
        for i in range(1, len(pts)):
            seg = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            accumulated += seg
            if accumulated >= spacing:
                waypoints.append(pts[i])
                accumulated = 0.0
        if waypoints[-1] != pts[-1]:
            waypoints.append(pts[-1])

        return waypoints

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

    def _advance_bezier_path(self, dt: float) -> None:
        if self.target is None or self.target_reached:
            return

        d_goal = self._goal_distance(self.target)
        if d_goal <= self.envp.goal_radius:
            self.target_reached = True
            self.target.speed = 0.0
            return

        if not self.target_bezier_waypoints:
            self.target_reached = True
            return

        distance_this_step = self.target.speed * dt
        remaining = distance_this_step

        while remaining > 1e-9 and self.target_bezier_wp_idx < len(self.target_bezier_waypoints):
            wp = self.target_bezier_waypoints[self.target_bezier_wp_idx]
            dist_to_wp = math.hypot(wp[0] - self.target.x, wp[1] - self.target.y)

            if dist_to_wp <= remaining:
                self.target.x = wp[0]
                self.target.y = wp[1]
                self.target.h = wp[2]
                remaining -= dist_to_wp
                self.target_bezier_wp_idx = min(self.target_bezier_wp_idx + 1, len(self.target_bezier_waypoints) - 1)
            else:
                self.target.h = wp[2]
                self.target.x += remaining * math.cos(self.target.h)
                self.target.y += remaining * math.sin(self.target.h)
                remaining = 0.0

        d_goal = self._goal_distance(self.target)
        if d_goal <= self.envp.goal_radius:
            self.target_reached = True
            self.target.speed = 0.0

    def _build_agent_planned_path(self) -> None:
        if self.agent is None:
            self.agent_planned_path = []
            return
        self.agent_planned_path = [(self.start_x, self.start_y), (self.agent.goal_x, self.agent.goal_y)]

    def _build_target_planned_path(self, *args, **kwargs) -> None:
        self.target_planned_path = [(wp[0], wp[1]) for wp in self.target_bezier_waypoints]

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
        sh2 = self._inward_facing_heading(sx2, sy2)
        gh2 = self._outward_facing_heading(gx2, gy2)
        self.target_end_heading = gh2
        sp2 = self.rng.uniform(self.envp.target_min_speed, self.envp.target_max_speed)
        self.target = Vessel(sx2, sy2, sh2, sp2, gx2, gy2)
        self.target_start_speed = sp2
        self.target_start_pos = (self.target.x, self.target.y)

        self.target_bezier_waypoints = self._build_bezier_waypoints(sx2, sy2, sh2, gx2, gy2, gh2, sp2)
        self.target_bezier_wp_idx = 0

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
        self.paused = False
        self.risk_overlay_active = False
        self.risk_overlay_payload = {}
        self.rl_ever_triggered = False
        self.rl_overlay_shown = False
        self.prev_agent_rl_active = False
        self.prev_target_rl_active = False

        bezier_path_length = sum(
            math.hypot(
                self.target_bezier_waypoints[i + 1][0] - self.target_bezier_waypoints[i][0],
                self.target_bezier_waypoints[i + 1][1] - self.target_bezier_waypoints[i][1],
            )
            for i in range(len(self.target_bezier_waypoints) - 1)
        ) if len(self.target_bezier_waypoints) > 1 else self.prev_goal_d_target

        t1 = self.prev_goal_d_agent / max(1e-6, self.agent.speed)
        t2 = bezier_path_length / max(1e-6, sp2)
        episode_time = max(self.envp.episode_seconds, 1.4 * max(t1, t2) + 20.0)
        self.max_steps = max(1, int(round(episode_time / self.envp.dt)))

        self._build_agent_planned_path()
        self._build_target_planned_path()

        return self.get_obs()

    def step(self, action: Union[np.ndarray, Tuple[float, float], list]) -> Tuple[np.ndarray, float, bool, Dict[str, float | str | int]]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size < 2:
            raise ValueError("Action must contain [rudder_cmd, throttle_cmd].")
        rudder_cmd = clamp(float(a[0]), -1.0, 1.0)
        throttle_cmd = clamp(float(a[1]), -1.0, 1.0)


        if self.paused:
            info: Dict[str, float | str | int] = {
                "reason": "paused",
                "agent_goal_distance": self._goal_distance(self.agent),
                "target_goal_distance": self._goal_distance(self.target),
                "agent_reached": int(self.agent_reached),
                "target_reached": int(self.target_reached),
                "rudder_cmd": rudder_cmd,
                "throttle_cmd": throttle_cmd,
                "dcpa": float(self.last_dcpa),
                "tcpa": float(self.last_tcpa),
                "risk_of_collision": int(self.risk_of_collision),
                "colregs_scenario": self.colregs_scenario,
                "agent_role": self.agent_role,
                "target_role": self.target_role,
                "agent_rl_active": int(self.agent_rl_active),
                "target_rl_active": int(self.target_rl_active),
                "agent_distance_from_start": float(self._distance_from_start(self.agent, self.agent_start_pos)),
                "target_distance_from_start": float(self._distance_from_start(self.target, self.target_start_pos)),
                "agent_relative_bearing_deg": float(self.agent_relative_bearing_deg),
                "target_relative_bearing_deg": float(self.target_relative_bearing_deg),
            }
            return self.get_obs(), 0.0, False, info

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

        # Compute whether RL should trigger this step (un-latched).
        agent_rl_trigger = (
            self.risk_of_collision
            and self.agent_role == "give_way"
            and agent_dist >= self.envp.rl_takeover_distance
            and not self.agent_reached
        )
        target_rl_trigger = (
            self.risk_of_collision
            and self.target_role == "give_way"
            and target_dist >= self.envp.rl_takeover_distance
            and not self.target_reached
        )

        # Latch: once RL activates for the first time, it stays active until episode end.
        if agent_rl_trigger or target_rl_trigger:
            self.rl_ever_triggered = True

        self.agent_rl_active = self.rl_ever_triggered and not self.agent_reached
        self.target_rl_active = self.rl_ever_triggered and not self.target_reached

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

            if self.target_role == "give_way" and self.target_rl_active:
                self._advance_controlled(self.target, "target_reached", rudder_cmd, throttle_cmd, h)
            else:
                # Keep vessel-2 on its nominal Bézier path unless RL takeover is active.
                self._advance_bezier_path(h)

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

        info: Dict[str, float | str | int] = {
            "reason": reason,
            "agent_goal_distance": d_agent,
            "target_goal_distance": d_target,
            "agent_reached": int(self.agent_reached),
            "target_reached": int(self.target_reached),
            "rudder_cmd": rudder_cmd,
            "throttle_cmd": throttle_cmd,
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

        # Show the RL takeover overlay exactly once per episode, on the first step RL activates.
        if (
            self.render_enabled
            and not self.rl_overlay_shown
            and self.rl_ever_triggered
            and (self.agent_rl_active or self.target_rl_active)
        ):
            self.rl_overlay_shown = True
            self.paused = True
            self.risk_overlay_active = True
            self.risk_overlay_payload = {
                "step": int(self.step_idx),
                "time": float(self.time),
                "scenario": self.colregs_scenario,
                "agent_role": self.agent_role,
                "target_role": self.target_role,
                "dcpa": float(self.last_dcpa),
                "tcpa": float(self.last_tcpa),
                "agent_bearing": float(self.agent_relative_bearing_deg),
                "target_bearing": float(self.target_relative_bearing_deg),
                "agent_rl_active": int(self.agent_rl_active),
                "target_rl_active": int(self.target_rl_active),
                "agent_distance": float(agent_dist),
                "target_distance": float(target_dist),
                "takeover_distance": float(self.envp.rl_takeover_distance),
            }

        self.prev_agent_rl_active = self.agent_rl_active
        self.prev_target_rl_active = self.target_rl_active
        return self.get_obs(), float(reward), done, info

    def _draw_risk_overlay(self, surf) -> None:
        if not self.risk_overlay_active or not self._font:
            return

        w = self.sx(self.envp.world_w)
        h = self.sy(self.envp.world_h)
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 155))

        p = self.risk_overlay_payload
        tcpa = p.get("tcpa", float("inf"))
        tcpa_txt = "inf" if (isinstance(tcpa, float) and math.isinf(tcpa)) else f"{float(tcpa):.1f}s"
        give_way_vessel = "V1 (agent)" if p.get("agent_rl_active", 0) else "V2 (target)" if p.get("target_rl_active", 0) else "unknown"
        lines = [
            "⚠  RL TAKEOVER — COLLISION AVOIDANCE ACTIVE",
            f"Step {int(p.get('step', self.step_idx))}   Sim time {float(p.get('time', self.time)):.1f}s",
            f"COLREGS scenario: {p.get('scenario', self.colregs_scenario).upper()}",
            f"Give-way vessel: {give_way_vessel}   Stand-on vessel: {'V2 (target)' if p.get('agent_rl_active', 0) else 'V1 (agent)'}",
            f"DCPA = {float(p.get('dcpa', self.last_dcpa)):.1f}m   TCPA = {tcpa_txt}",
            f"V1→V2 bearing = {float(p.get('agent_bearing', self.agent_relative_bearing_deg)):.1f}°   V2→V1 bearing = {float(p.get('target_bearing', self.target_relative_bearing_deg)):.1f}°",
            "RL model now controls give-way vessel for remainder of episode.",
            "Press SPACE or ENTER to dismiss and continue.",
        ]

        box_w = int(0.88 * w)
        box_h = 28 + 24 * len(lines)
        box_x = (w - box_w) // 2
        box_y = (h - box_h) // 2
        pygame.draw.rect(panel, (20, 20, 28, 235), (box_x, box_y, box_w, box_h), border_radius=10)
        pygame.draw.rect(panel, (255, 210, 90, 255), (box_x, box_y, box_w, box_h), width=2, border_radius=10)

        y = box_y + 14
        for idx, line in enumerate(lines):
            color = (255, 230, 130) if idx == 0 else (240, 240, 240)
            txt = self._font.render(line, True, color)
            panel.blit(txt, (box_x + 14, y))
            y += 24

        surf.blit(panel, (0, 0))

    def render(self) -> None:
        if not self.render_enabled or self._screen is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.show_planned_paths = not self.show_planned_paths
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_RETURN):
                if self.risk_overlay_active:
                    self.risk_overlay_active = False
                    self.risk_overlay_payload = {}
                    self.paused = False
                else:
                    self.paused = not self.paused

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
            f"step={self.step_idx}  t={self.time:.1f}s  paused={int(self.paused)}[SPACE]  paths[P]={int(self.show_planned_paths)}  rl_latched={int(self.rl_ever_triggered)}",
            True, (255, 255, 255),
        )
        hud1 = self._font.render(
            f"COLREGS={self.colregs_scenario}  risk={'YES' if self.risk_of_collision else 'NO'}  DCPA={self.last_dcpa:.1f}m  TCPA={tcpa_txt}  V1→V2_BRG={self.agent_relative_bearing_deg:.1f}°  V2→V1_BRG={self.target_relative_bearing_deg:.1f}°",
            True, (255, 240, 170),
        )
        hud2 = self._font.render(
            f"V1(agent) role={self.agent_role}  rl_active={int(self.agent_rl_active)}  reached={int(self.agent_reached)}  spd={self.agent.speed:.2f}m/s  hdg={math.degrees(self.agent.h):.1f}°",
            True, (170, 220, 255),
        )
        hud3 = self._font.render(
            f"V2(target) role={self.target_role}  rl_active={int(self.target_rl_active)}  reached={int(self.target_reached)}  spd={self.target.speed:.2f}m/s  hdg={math.degrees(self.target.h):.1f}°",
            True, (255, 190, 190),
        )
        surf.blit(hud0, (10, 8))
        surf.blit(hud1, (10, 26))
        surf.blit(hud2, (10, 44))
        surf.blit(hud3, (10, 62))

        if self.risk_overlay_active:
            self._draw_risk_overlay(surf)

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
