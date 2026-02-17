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

        self.agent: Optional[Vessel] = None   # vessel 1 (center -> circumference straight)
        self.target: Optional[Vessel] = None  # vessel 2 (circumference -> circumference dubins-style)
        self.start_x = 0.5 * self.envp.world_w
        self.start_y = 0.5 * self.envp.world_h
        self.time = 0.0
        self.step_idx = 0
        self.max_steps = max(1, int(round(self.envp.episode_seconds / self.envp.dt)))
        self.prev_goal_d_agent = 0.0
        self.prev_goal_d_target = 0.0
        self.agent_reached = False
        self.target_reached = False

        # vessel 2 Dubins-style open-loop sequence (turn/straight/turn)
        self.target_plan: List[Tuple[float, float]] = []  # list of (turn_cmd in {-1,0,1}, duration_s)
        self.target_plan_idx = 0
        self.target_plan_elapsed = 0.0
        self.target_goal_heading = 0.0
        self.target_planner_mode = "shortest_path"

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

    def _segment_cost(self, mode: str, plan: List[Tuple[float, float]], speed: float, yaw_rate: float, heading_err: float, pos_err: float, chord: float) -> float:
        total_time = sum(d for _, d in plan)
        total_len = total_time * speed
        steering_effort = sum(abs(cmd) * d for cmd, d in plan)
        curvature_proxy = steering_effort * yaw_rate / max(1e-6, speed)
        straight_dev = abs(total_len - chord)

        if mode == "minimum_steering_effort":
            return steering_effort + 0.15 * total_len + 6.0 * pos_err + heading_err
        if mode == "minimum_curvature_change":
            return curvature_proxy + 0.2 * steering_effort + 5.0 * pos_err + heading_err
        if mode == "closest_to_straight_line":
            return straight_dev + 0.2 * steering_effort + 7.0 * pos_err + heading_err
        return total_len + 0.25 * steering_effort + 5.0 * pos_err + heading_err

    def _simulate_plan_endpoint(self, sx: float, sy: float, sh: float, speed: float, yaw_rate: float, plan: List[Tuple[float, float]]) -> Tuple[float, float, float]:
        x, y, h = sx, sy, sh
        dt = 0.1
        for cmd, duration in plan:
            t = 0.0
            while t < duration:
                step = min(dt, duration - t)
                h = wrap_pi(h + cmd * yaw_rate * step)
                x += speed * math.cos(h) * step
                y += speed * math.sin(h) * step
                t += step
        return x, y, h

    def _sample_dubins_style_plan(self, sx: float, sy: float, sh: float, gx: float, gy: float, speed: float) -> Tuple[List[Tuple[float, float]], float, str]:
        yaw_rate = self.envp.rudder_max_yaw_rate_rad_s
        min_turn = 1.0 / max(1e-6, yaw_rate)
        chord = math.hypot(gx - sx, gy - sy)

        modes = [
            "shortest_path",
            "minimum_steering_effort",
            "minimum_curvature_change",
            "closest_to_straight_line",
        ]
        mode = self.rng.choice(modes)

        headings = self._sample_end_heading_candidates(sx, sy, gx, gy)
        path_types = [(-1.0, 0.0, -1.0), (1.0, 0.0, 1.0), (-1.0, 0.0, 1.0), (1.0, 0.0, -1.0)]

        best_plan: List[Tuple[float, float]] = []
        best_goal_h = headings[0]
        best_score = float("inf")

        for gh in headings:
            for t0, t1, t2 in path_types:
                for _ in range(30):
                    turn1 = self.rng.uniform(0.0, 1.4 * math.pi) * min_turn
                    turn2 = self.rng.uniform(0.0, 1.4 * math.pi) * min_turn
                    straight = self.rng.uniform(0.0, max(chord * 1.3 / max(1e-6, speed), 1.5))
                    plan = [(t0, turn1), (t1, straight), (t2, turn2)]
                    ex, ey, eh = self._simulate_plan_endpoint(sx, sy, sh, speed, yaw_rate, plan)
                    pos_err = math.hypot(gx - ex, gy - ey)
                    heading_err = abs(wrap_pi(gh - eh))
                    score = self._segment_cost(mode, plan, speed, yaw_rate, heading_err, pos_err, chord) + 12.0 * pos_err
                    if score < best_score:
                        best_score = score
                        best_plan = plan
                        best_goal_h = gh

        return best_plan, best_goal_h, mode

    def _advance_agent_straight(self, dt: float) -> None:
        if self.agent is None or self.agent_reached:
            return
        d = self._goal_distance(self.agent)
        if d <= self.envp.goal_radius:
            self.agent_reached = True
            self.agent.x = self.agent.goal_x
            self.agent.y = self.agent.goal_y
            self.agent.speed = 0.0
            return

        step = self.agent.speed * dt
        if step >= d:
            self.agent.x = self.agent.goal_x
            self.agent.y = self.agent.goal_y
            self.agent.speed = 0.0
            self.agent_reached = True
            return

        self.agent.x += math.cos(self.agent.h) * step
        self.agent.y += math.sin(self.agent.h) * step

    def _advance_target_plan(self, dt: float) -> None:
        if self.target is None or self.target_reached:
            return

        if self._goal_distance(self.target) <= self.envp.goal_radius:
            self.target.x = self.target.goal_x
            self.target.y = self.target.goal_y
            self.target.h = wrap_pi(self.target_goal_heading)
            self.target.speed = 0.0
            self.target_reached = True
            return

        remaining = dt
        yaw_rate = self.envp.rudder_max_yaw_rate_rad_s
        while remaining > 1e-9 and not self.target_reached:
            if self.target_plan_idx >= len(self.target_plan):
                # fallback after planned segments: yaw toward remaining goal bearing.
                goal_bearing = math.atan2(self.target.goal_y - self.target.y, self.target.goal_x - self.target.x)
                err = wrap_pi(goal_bearing - self.target.h)
                cmd = 0.0 if abs(err) < math.radians(1.0) else (1.0 if err > 0.0 else -1.0)
                seg_rem = remaining
            else:
                cmd, seg_dur = self.target_plan[self.target_plan_idx]
                seg_rem = max(0.0, seg_dur - self.target_plan_elapsed)
                if seg_rem <= 1e-9:
                    self.target_plan_idx += 1
                    self.target_plan_elapsed = 0.0
                    continue

            step = min(remaining, seg_rem)
            self.target.h = wrap_pi(self.target.h + cmd * yaw_rate * step)
            self.target.x += self.target.speed * math.cos(self.target.h) * step
            self.target.y += self.target.speed * math.sin(self.target.h) * step

            remaining -= step
            if self.target_plan_idx < len(self.target_plan):
                self.target_plan_elapsed += step
                if self.target_plan_elapsed + 1e-9 >= self.target_plan[self.target_plan_idx][1]:
                    self.target_plan_idx += 1
                    self.target_plan_elapsed = 0.0

            if self._goal_distance(self.target) <= self.envp.goal_radius:
                self.target.x = self.target.goal_x
                self.target.y = self.target.goal_y
                self.target.h = wrap_pi(self.target_goal_heading)
                self.target.speed = 0.0
                self.target_reached = True

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

        self.target_plan, self.target_goal_heading, self.target_planner_mode = self._sample_dubins_style_plan(
            sx2, sy2, sh2, gx2, gy2, sp2
        )
        self.target_plan_idx = 0
        self.target_plan_elapsed = 0.0

        self.time = 0.0
        self.step_idx = 0
        self.agent_reached = False
        self.target_reached = False
        self.prev_goal_d_agent = self._goal_distance(self.agent)
        self.prev_goal_d_target = self._goal_distance(self.target)

        t1 = self.prev_goal_d_agent / max(1e-6, self.agent.speed)
        planned_t2 = sum(d for _, d in self.target_plan)
        t2 = max(planned_t2, self.prev_goal_d_target / max(1e-6, self.target.speed))
        episode_time = max(self.envp.episode_seconds, 1.4 * max(t1, t2) + 20.0)
        self.max_steps = max(1, int(round(episode_time / self.envp.dt)))
        return self.get_obs()

    def step(self, action: Union[np.ndarray, Tuple[float, float], list]) -> Tuple[np.ndarray, float, bool, Dict[str, float | str | int]]:
        # Inputs are still consumed for compatibility, but not used for control yet.
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size < 2:
            raise ValueError("Action must contain [rudder_cmd, throttle_cmd].")
        rudder_cmd = float(a[0])
        throttle_cmd = float(a[1])

        h = self.envp.dt / max(1, self.envp.substeps)
        for _ in range(max(1, self.envp.substeps)):
            self._advance_agent_straight(h)
            self._advance_target_plan(h)

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
            "target_planner_mode": self.target_planner_mode,
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
            pygame.draw.circle(
                self._screen,
                (255, 225, 120),
                (self.sx(self.start_x), self.sy(self.start_y)),
                int(round(self.envp.target_outer_radius * self.envp.pixels_per_meter)),
                1,
            )

        self._draw_vessel(self.agent, (95, 170, 255), "V1")
        self._draw_vessel(self.target, (255, 120, 120), "V2")

        hud = self._font.render(
            f"step={self.step_idx} t={self.time:.1f}s mode={self.target_planner_mode} reached=({int(self.agent_reached)},{int(self.target_reached)})",
            True,
            (255, 255, 255),
        )
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
