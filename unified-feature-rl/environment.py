from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

from hyperparameters import EnvParams, RewardParams

try:
    import pygame

    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


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
    """Single learning ASV with a moving target vessel that traverses an inner arc."""

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
        self.prev_goal_d = 0.0

        # target arc trajectory state (quadratic bezier)
        self.target_ctrl_x = 0.0
        self.target_ctrl_y = 0.0
        self.target_end_heading = 0.0
        self.target_prog = 0.0
        self.target_prog_rate = 0.0
        self.target_start_x = 0.0
        self.target_start_y = 0.0
        self.target_cruise_speed = 0.0

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

    def _sample_goal_on_ring(self, cx: float, cy: float, radius: float) -> Tuple[float, float]:
        ang = self.rng.uniform(0.0, 2.0 * math.pi)
        gx = cx + radius * math.cos(ang)
        gy = cy + radius * math.sin(ang)
        m = self.envp.spawn_margin
        gx = clamp(gx, m, self.envp.world_w - m)
        gy = clamp(gy, m, self.envp.world_h - m)
        return gx, gy

    def _outside(self, v: Vessel) -> bool:
        return not (0.0 <= v.x <= self.envp.world_w and 0.0 <= v.y <= self.envp.world_h)

    def _goal_distance(self, v: Vessel) -> float:
        return math.hypot(v.goal_x - v.x, v.goal_y - v.y)

    def _apply_control(self, v: Vessel, rudder_cmd: float, throttle_cmd: float, dt: float) -> None:
        rudder_cmd = clamp(rudder_cmd, -1.0, 1.0)
        throttle_cmd = clamp(throttle_cmd, -1.0, 1.0)

        # Rudder dynamics: command -> target angle with slew limit.
        rudder_target = rudder_cmd * self.envp.rudder_max_angle_rad
        rudder_step = self.envp.rudder_max_rate_rad_s * dt
        v.rudder = clamp(v.rudder + clamp(rudder_target - v.rudder, -rudder_step, rudder_step), -self.envp.rudder_max_angle_rad, self.envp.rudder_max_angle_rad)

        # Rudder-limited yaw: yaw rate saturates at max when rudder is at max angle.
        yaw_rate = (v.rudder / max(1e-6, self.envp.rudder_max_angle_rad)) * self.envp.rudder_max_yaw_rate_rad_s
        v.h = wrap_pi(v.h + yaw_rate * dt)

        # Throttle stick dynamics: command -> effective throttle with slew limit.
        thr_step = self.envp.throttle_slew_rate * dt
        v.throttle = clamp(v.throttle + clamp(throttle_cmd - v.throttle, -thr_step, thr_step), -1.0, 1.0)

        if v.throttle > self.envp.throttle_deadband:
            accel = v.throttle * self.envp.accel_rate
        elif v.throttle < -self.envp.throttle_deadband:
            accel = v.throttle * self.envp.brake_rate
        else:
            accel = -self.envp.decel_rate

        v.speed = clamp(v.speed + accel * dt, self.envp.min_speed, self.envp.max_speed)

        v.x += v.speed * math.cos(v.h) * dt
        v.y += v.speed * math.sin(v.h) * dt

    def _bezier_point(self, t: float, sx: float, sy: float, cx: float, cy: float, ex: float, ey: float) -> Tuple[float, float]:
        u = 1.0 - t
        x = u * u * sx + 2.0 * u * t * cx + t * t * ex
        y = u * u * sy + 2.0 * u * t * cy + t * t * ey
        return x, y

    def _bezier_tangent(self, t: float, sx: float, sy: float, cx: float, cy: float, ex: float, ey: float) -> Tuple[float, float]:
        tx = 2.0 * (1.0 - t) * (cx - sx) + 2.0 * t * (ex - cx)
        ty = 2.0 * (1.0 - t) * (cy - sy) + 2.0 * t * (ey - cy)
        return tx, ty

    def _sample_target_arc(self) -> Vessel:
        cxw = 0.5 * self.envp.world_w
        cyw = 0.5 * self.envp.world_h
        radius = self.envp.target_outer_radius

        start_ang = self.rng.uniform(0.0, 2.0 * math.pi)
        turn_sign = self.rng.choice([-1.0, 1.0])
        arc_deg = self.rng.uniform(self.envp.target_arc_min_deg, self.envp.target_arc_max_deg)
        end_ang = start_ang + turn_sign * math.radians(arc_deg)

        sx = cxw + radius * math.cos(start_ang)
        sy = cyw + radius * math.sin(start_ang)
        ex = cxw + radius * math.cos(end_ang)
        ey = cyw + radius * math.sin(end_ang)

        mid_ang = 0.5 * (start_ang + end_ang)
        inner_radius = self.rng.uniform(0.2 * radius, 0.8 * radius)
        self.target_ctrl_x = cxw + inner_radius * math.cos(mid_ang)
        self.target_ctrl_y = cyw + inner_radius * math.sin(mid_ang)

        self.target_prog = 0.0
        speed = self.rng.uniform(self.envp.target_min_speed, self.envp.target_max_speed)
        self.target_cruise_speed = speed
        # Coarse conversion to normalized bezier progress / second
        chord = math.hypot(ex - sx, ey - sy)
        travel_dist = max(0.35 * radius, chord)
        self.target_prog_rate = clamp(speed / max(1e-6, travel_dist), 0.05, 0.5)

        tanx0, tany0 = self._bezier_tangent(0.0, sx, sy, self.target_ctrl_x, self.target_ctrl_y, ex, ey)
        heading = math.atan2(tany0, tanx0)
        tanx1, tany1 = self._bezier_tangent(1.0, sx, sy, self.target_ctrl_x, self.target_ctrl_y, ex, ey)
        self.target_end_heading = math.atan2(tany1, tanx1)

        self.target_start_x = sx
        self.target_start_y = sy
        return Vessel(sx, sy, wrap_pi(heading), speed, ex, ey)

    def _advance_target(self, dt: float) -> None:
        if self.target is None:
            return
        if self.target_prog >= 1.0:
            self.target.h = wrap_pi(self.target_end_heading)
            self.target.speed = 0.0
            return

        ex, ey = self.target.goal_x, self.target.goal_y
        next_prog = clamp(self.target_prog + self.target_prog_rate * dt, 0.0, 1.0)
        px, py = self._bezier_point(
            next_prog,
            self.target_start_x,
            self.target_start_y,
            self.target_ctrl_x,
            self.target_ctrl_y,
            ex,
            ey,
        )
        tx, ty = self._bezier_tangent(
            next_prog,
            self.target_start_x,
            self.target_start_y,
            self.target_ctrl_x,
            self.target_ctrl_y,
            ex,
            ey,
        )

        self.target.x = px
        self.target.y = py
        self.target.h = wrap_pi(math.atan2(ty, tx))
        self.target.speed = self.target_cruise_speed if next_prog < 1.0 else 0.0
        self.target_prog = next_prog

        if self.target_prog >= 1.0:
            self.target.x = ex
            self.target.y = ey
            self.target.h = wrap_pi(self.target_end_heading)
            self.target.speed = 0.0

    def get_obs(self) -> np.ndarray:
        # 10 features: ego + goal + target absolute state
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
            ],
            dtype=np.float32,
        )

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)

        ax = self.start_x
        ay = self.start_y
        ah = self.rng.uniform(-math.pi, math.pi)
        agx, agy = self._sample_goal_on_ring(ax, ay, self.envp.goal_ring_radius)

        self.agent = Vessel(ax, ay, ah, self.rng.uniform(0.0, 0.5 * self.envp.max_speed), agx, agy)
        self.target = self._sample_target_arc()
        self.time = 0.0
        self.step_idx = 0
        self.prev_goal_d = self._goal_distance(self.agent)
        return self.get_obs()

    def step(self, action: Union[np.ndarray, Tuple[float, float], list]) -> Tuple[np.ndarray, float, bool, Dict[str, float | str | int]]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size < 2:
            raise ValueError("Action must contain [rudder_cmd, throttle_cmd].")
        rudder_cmd = float(a[0])
        throttle_cmd = float(a[1])

        h = self.envp.dt / max(1, self.envp.substeps)
        for _ in range(max(1, self.envp.substeps)):
            self._apply_control(self.agent, rudder_cmd, throttle_cmd, h)
            self._advance_target(h)

        self.time += self.envp.dt
        self.step_idx += 1

        done = False
        reason = ""
        agent_reached = self._goal_distance(self.agent) <= self.envp.goal_radius
        if self._outside(self.agent):
            done, reason = True, "out_of_bounds"
        elif self.step_idx >= self.max_steps:
            done, reason = True, "timeout"
        elif self._outside(self.target):
            done, reason = True, "target_out_of_bounds"
        elif agent_reached:
            done, reason = True, "goal"

        reward = self.rewp.living_penalty
        d_now = self._goal_distance(self.agent)
        reward += self.rewp.progress_weight * (self.prev_goal_d - d_now)

        if agent_reached and self.prev_goal_d > self.envp.goal_radius:
            reward += self.rewp.goal_bonus

        if reason in {"out_of_bounds", "target_out_of_bounds"}:
            reward += self.rewp.out_of_bounds_penalty

        self.prev_goal_d = d_now

        info: Dict[str, float | str | int] = {
            "reason": reason,
            "agent_goal_distance": d_now,
            "rudder_cmd": rudder_cmd,
            "throttle_cmd": throttle_cmd,
            "agent_rudder_angle_rad": self.agent.rudder,
            "agent_effective_throttle": self.agent.throttle,
            "target_speed_constant": self.target.speed,
        }
        return self.get_obs(), float(reward), done, info

    def render(self) -> None:
        if not self.render_enabled or self._screen is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

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
            self._draw_dotted_circle(self.start_x, self.start_y, self.envp.goal_ring_radius, (250, 215, 60))
            self._draw_dotted_circle(self.start_x, self.start_y, self.envp.target_outer_radius, (255, 140, 90))

        self._draw_vessel(self.agent, (95, 170, 255), "A")
        self._draw_vessel(self.target, (255, 120, 120), "T")

        hud = self._font.render(f"step={self.step_idx} t={self.time:.1f}s", True, (255, 255, 255))
        surf.blit(hud, (10, 10))

        pygame.display.flip()
        self._clock.tick(self.envp.render_fps)

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

    def _draw_dotted_circle(self, cx: float, cy: float, radius: float, color: Tuple[int, int, int]) -> None:
        points = 60
        dash = 2
        gap = 2
        for i in range(points):
            if i % (dash + gap) >= dash:
                continue
            ang = 2.0 * math.pi * i / points
            px = cx + radius * math.cos(ang)
            py = cy + radius * math.sin(ang)
            pygame.draw.circle(self._screen, color, (self.sx(px), self.sy(py)), 2)
