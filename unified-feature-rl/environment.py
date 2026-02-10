from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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
    goal_x: float
    goal_y: float


class SingleTargetFeatureEnv:
    """One learning ASV (agent) and one target vessel with its own goal."""

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
        self.time = 0.0
        self.step_idx = 0
        self.max_steps = max(1, int(round(self.envp.episode_seconds / self.envp.dt)))

        self.prev_goal_d = 0.0

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
        pygame.display.set_caption("Unified Feature RL - ASV/Target")
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

    def _random_pose(self) -> Tuple[float, float, float]:
        m = self.envp.spawn_margin
        x = self.rng.uniform(m, self.envp.world_w - m)
        y = self.rng.uniform(m, self.envp.world_h - m)
        h = self.rng.uniform(-math.pi, math.pi)
        return x, y, h

    def _random_goal_for(self, x: float, y: float) -> Tuple[float, float]:
        m = self.envp.spawn_margin
        while True:
            gx = self.rng.uniform(m, self.envp.world_w - m)
            gy = self.rng.uniform(m, self.envp.world_h - m)
            if math.hypot(gx - x, gy - y) >= self.envp.goal_min_distance:
                return gx, gy

    def _distance(self, a: Vessel, b: Vessel) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _outside(self, v: Vessel) -> bool:
        return not (0.0 <= v.x <= self.envp.world_w and 0.0 <= v.y <= self.envp.world_h)

    def _goal_distance(self, v: Vessel) -> float:
        return math.hypot(v.goal_x - v.x, v.goal_y - v.y)

    def _relative_geometry(self) -> Tuple[float, float, int]:
        dx, dy = self.target.x - self.agent.x, self.target.y - self.agent.y
        dist = math.hypot(dx, dy)
        ch, sh = math.cos(self.agent.h), math.sin(self.agent.h)
        x_rel = ch * dx + sh * dy
        y_rel = -sh * dx + ch * dy
        rel_brg = math.atan2(y_rel, x_rel)
        rel_deg = math.degrees(rel_brg)
        rel_deg_360 = (rel_deg + 360.0) % 360.0
        sector = int(rel_deg_360 // (360.0 / self.envp.sectors))
        return rel_deg, dist, sector

    def _encounter_type(self, rel_bearing_deg: float) -> str:
        diff = abs(math.degrees(wrap_pi(self.target.h - self.agent.h)))
        if abs(rel_bearing_deg) <= 15.0 and diff >= 150.0:
            return "head_on"
        if abs(rel_bearing_deg) >= 112.5 and self.agent.speed > self.target.speed:
            return "overtaking"
        return "crossing"

    def _colregs_reward(self, encounter: str, rel_bearing_deg: float, rudder_cmd: float) -> float:
        turning_right = rudder_cmd < 0.0
        if encounter == "head_on":
            return self.rewp.colregs_correct_bonus if turning_right else -self.rewp.colregs_wrong_penalty

        if encounter == "crossing":
            # target on starboard side => agent give-way with starboard turn
            if -self.rewp.colregs_window_deg <= rel_bearing_deg <= 0.0:
                return self.rewp.colregs_correct_bonus if turning_right else -self.rewp.colregs_wrong_penalty
            return 0.0

        # overtaking: keep a starboard bias as simple shaping
        return self.rewp.colregs_correct_bonus if turning_right else -0.5 * self.rewp.colregs_wrong_penalty

    def _apply_control(self, v: Vessel, rudder_cmd: float, throttle_cmd: float, dt: float) -> None:
        rudder_cmd = clamp(rudder_cmd, -1.0, 1.0)
        throttle_cmd = clamp(throttle_cmd, -1.0, 1.0)

        if v.speed > 1e-4:
            v.h = wrap_pi(v.h + rudder_cmd * self.envp.turn_rate_rad_s * dt)

        accel = self.envp.accel_rate * max(0.0, throttle_cmd)
        decel = self.envp.decel_rate * max(0.0, -throttle_cmd)
        v.speed = clamp(v.speed + (accel - decel) * dt, self.envp.min_speed, self.envp.max_speed)

        v.x += v.speed * math.cos(v.h) * dt
        v.y += v.speed * math.sin(v.h) * dt

    def _target_autopilot(self) -> Tuple[float, float]:
        # proportional heading control toward target goal + cruise throttle
        dx, dy = self.target.goal_x - self.target.x, self.target.goal_y - self.target.y
        goal_h = math.atan2(dy, dx)
        err = wrap_pi(goal_h - self.target.h)
        rud = clamp(err / math.radians(60.0), -1.0, 1.0)

        desired_speed = 0.6 * self.envp.max_speed
        throttle = clamp((desired_speed - self.target.speed) / max(1e-6, self.envp.max_speed), -1.0, 1.0)
        return rud, throttle

    def get_obs(self) -> np.ndarray:
        # 10 features: 6 for agent + 4 for target (goal omitted for target)
        rel_brg_deg, dist, _ = self._relative_geometry()
        rng = max(1e-6, self.envp.sensor_range)
        return np.asarray(
            [
                self.agent.x / self.envp.world_w,
                self.agent.y / self.envp.world_h,
                self.agent.h / math.pi,
                self.agent.speed / self.envp.max_speed,
                self.agent.goal_x / self.envp.world_w,
                self.agent.goal_y / self.envp.world_h,
                wrap_pi(self.target.h - self.agent.h) / math.pi,
                self.target.speed / self.envp.max_speed,
                rel_brg_deg / 180.0,
                min(dist, rng) / rng,
            ],
            dtype=np.float32,
        )

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)

        while True:
            ax, ay, ah = self._random_pose()
            tx, ty, th = self._random_pose()
            if math.hypot(ax - tx, ay - ty) >= self.envp.min_start_separation:
                break

        agx, agy = self._random_goal_for(ax, ay)
        tgx, tgy = self._random_goal_for(tx, ty)

        self.agent = Vessel(ax, ay, ah, self.rng.uniform(0.0, 0.5 * self.envp.max_speed), agx, agy)
        self.target = Vessel(tx, ty, th, self.rng.uniform(0.0, 0.5 * self.envp.max_speed), tgx, tgy)

        self.time = 0.0
        self.step_idx = 0
        self.prev_goal_d = self._goal_distance(self.agent)
        return self.get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, float | str | int]]:
        rudder_cmd = float(action[0])
        throttle_cmd = float(action[1])

        _, dcpa_before = tcpa_dcpa(
            self.agent.x,
            self.agent.y,
            self.agent.h,
            self.agent.speed,
            self.target.x,
            self.target.y,
            self.target.h,
            self.target.speed,
        )

        h = self.envp.dt / max(1, self.envp.substeps)
        for _ in range(max(1, self.envp.substeps)):
            self._apply_control(self.agent, rudder_cmd, throttle_cmd, h)
            tr, tt = self._target_autopilot()
            self._apply_control(self.target, tr, tt, h)

        self.time += self.envp.dt
        self.step_idx += 1

        rel_brg_deg, dist, sector = self._relative_geometry()
        encounter = self._encounter_type(rel_brg_deg)

        tcpa, dcpa_after = tcpa_dcpa(
            self.agent.x,
            self.agent.y,
            self.agent.h,
            self.agent.speed,
            self.target.x,
            self.target.y,
            self.target.h,
            self.target.speed,
        )

        done = False
        reason = ""
        agent_reached = self._goal_distance(self.agent) <= self.envp.goal_radius
        target_reached = self._goal_distance(self.target) <= self.envp.goal_radius

        if dist <= self.rewp.collision_radius:
            done, reason = True, "collision"
        elif self._outside(self.agent) or self._outside(self.target):
            done, reason = True, "out_of_bounds"
        elif self.step_idx >= self.max_steps:
            done, reason = True, "timeout"
        elif agent_reached and target_reached:
            done, reason = True, "both_goals"

        reward = self.rewp.living_penalty

        d_now = self._goal_distance(self.agent)
        reward += self.rewp.progress_weight * (self.prev_goal_d - d_now)

        if 0.0 <= tcpa <= self.rewp.cpa_horizon:
            risk = math.exp(-tcpa / self.rewp.tcpa_decay) * math.exp(-dcpa_after / self.rewp.dcpa_scale)
            reward -= self.rewp.risk_weight * risk

        if dist <= self.envp.sensor_range:
            reward += self._colregs_reward(encounter, rel_brg_deg, rudder_cmd)

        dcpa_delta = dcpa_after - dcpa_before
        if dcpa_delta >= 0.0:
            reward += self.rewp.dcpa_improve_weight * dcpa_delta
        else:
            reward += self.rewp.dcpa_worse_weight * dcpa_delta

        if agent_reached and self.prev_goal_d > self.envp.goal_radius:
            reward += self.rewp.goal_bonus

        if reason == "collision":
            reward += self.rewp.collision_penalty
        elif reason == "out_of_bounds":
            reward += self.rewp.out_of_bounds_penalty

        self.prev_goal_d = d_now

        info: Dict[str, float | str | int] = {
            "reason": reason,
            "encounter": encounter,
            "sector_12": sector,
            "rel_bearing_deg": rel_brg_deg,
            "tcpa": tcpa,
            "dcpa_before": dcpa_before,
            "dcpa_after": dcpa_after,
            "dcpa_delta": dcpa_delta,
            "agent_goal_distance": d_now,
            "target_goal_distance": self._goal_distance(self.target),
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

        # border
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
        self._draw_goal(self.target.goal_x, self.target.goal_y, (255, 150, 60))

        if self.envp.show_sectors:
            self._draw_sectors(self.agent)

        self._draw_vessel(self.agent, (95, 170, 255), "A")
        self._draw_vessel(self.target, (70, 210, 120), "T")

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
            int(max(2, self.rewp.collision_radius * self.envp.pixels_per_meter * 0.5)),
            1,
        )
        txt = self._font.render(label, True, (255, 255, 255))
        self._screen.blit(txt, (self.sx(v.x) + 6, self.sy(v.y) - 8))

    def _draw_sectors(self, v: Vessel) -> None:
        L = self.envp.sensor_range
        for k in range(self.envp.sectors):
            ang = v.h + 2.0 * math.pi * k / self.envp.sectors
            x2 = v.x + L * math.cos(ang)
            y2 = v.y + L * math.sin(ang)
            pygame.draw.line(self._screen, (215, 215, 215), (self.sx(v.x), self.sy(v.y)), (self.sx(x2), self.sy(y2)), 1)
