#!/usr/bin/env python3
"""
Multi-Boat Sectors Environment (Feature-based, Turn-Session Control)
===================================================================

• Units: meters, seconds. Simple *kinematics* (no hydrodynamic yaw model).
• Observation (per boat): 12 sectors × 8 features + 4 ego = 100 floats.
  Sector features: [x_rel, y_rel, rel_brg, dist, tcpa, dcpa, tgt_speed, tgt_rel_h] (all normalized).
  Ego features: [x/world_w, y/world_h, speed/max_speed, heading/pi].
• Actions (Discrete 9): steer×throttle → steer∈{0 none,1 right,2 left}, throttle∈{0 coast,1 accel,2 decel}.
• Turn-Session controller (heading-chunk):
  - When a steer≠0 arrives *and no session active*, we latch a session to rotate by `turn_deg` at `yaw_rate_degps`.
  - While active, steer commands are ignored (optionally allow cancel via steer=0).
  - Throttle always passes through each step.
  - Hysteresis kills chatter near target.
• Goals: For each boat, goal = fixed distance straight ahead of its start heading (clamped in bounds).
• Rewards (per boat):
  - +progress_weight * (previous_distance − current_distance)
  - +goal_bonus once on arrival (<= goal_radius)
  - +living_penalty each step
  - −collision_penalty on collision (episode ends)
  - −oob_penalty if any boat exits bounds (episode ends)
  - Optional CPA risk shaping (small penalty under near-term, close-approach).
• Termination: collision, out-of-bounds, all reached, or max_steps.

Rendering:
  Visuals use pixels_per_meter for scale; physics stays in meters.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

# ---------------- Utilities ----------------
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def angle_deg(a: float) -> float:
    return a * 180.0 / math.pi

def clamp(x: float, lo: float, hi: float) -> float:
    return hi if x > hi else lo if x < lo else x

def tcpa_dcpa(ax, ay, ah, aspd, bx, by, bh, bspd) -> Tuple[float, float]:
    """Constant-velocity CPA metrics used for shaping/features (seconds, meters)."""
    rvx = math.cos(bh) * bspd - math.cos(ah) * aspd
    rvy = math.sin(bh) * bspd - math.sin(ah) * aspd
    rx, ry = (bx - ax), (by - ay)
    rv2 = rvx*rvx + rvy*rvy
    if rv2 < 1e-9:
        return 0.0, math.hypot(rx, ry)
    tcpa = - (rx*rvx + ry*rvy) / rv2
    if tcpa < 0.0:
        tcpa = 0.0
    cx = rx + rvx * tcpa
    cy = ry + rvy * tcpa
    return tcpa, math.hypot(cx, cy)

# ---------------- Turn-Session Controller ----------------
class HeadingSession:
    """Latch a discrete turn by angle at a fixed yaw rate; throttle is unaffected."""
    def __init__(self,
                 turn_deg: float = 15.0,
                 yaw_rate_degps: float = 45.0,
                 hysteresis_deg: float = 2.0,
                 allow_cancel: bool = False):
        self.turn_rad = math.radians(turn_deg)
        self.yaw_rate = math.radians(yaw_rate_degps)
        self.hys = math.radians(hysteresis_deg)
        self.allow_cancel = allow_cancel

        self.active = False
        self.dir = 0       # -1 right, +1 left
        self.target = 0.0  # rad

    def start_if_idle(self, h: float, steer: int):
        """steer: 0 none, 1 right, 2 left"""
        if self.active:
            return
        if steer == 1:
            self.dir = -1
        elif steer == 2:
            self.dir = +1
        else:
            return
        self.target = wrap_pi(h + self.dir * self.turn_rad)
        self.active = True

    def step(self, h: float, steer: int, dt: float) -> float:
        """Advance heading during session; optionally cancel if steer==0 and allowed."""
        if not self.active:
            # Optionally latch a new session immediately when not active.
            self.start_if_idle(h, steer)
            return h

        if self.allow_cancel and steer == 0:
            self.active = False
            return h

        err = wrap_pi(self.target - h)
        if abs(err) <= self.hys:
            self.active = False
            return wrap_pi(self.target)

        sgn = +1.0 if err > 0 else -1.0
        dh = sgn * self.yaw_rate * dt
        # clamp to not overshoot beyond target ±hys
        if abs(dh) >= abs(err):
            self.active = False
            return wrap_pi(self.target)
        return wrap_pi(h + dh)

# ---------------- Boat (kinematics only) ----------------
@dataclass
class BoatParams:
    length_m: float = 6.0
    width_m:  float = 2.2
    max_speed: float = 18.0           # m/s
    accel_rate: float = 1.6           # m/s^2 (forward)
    decel_rate: float = 1.0           # m/s^2 (braking)

@dataclass
class TurnSessionConfig:
    turn_deg: float = 15.0
    yaw_rate_degps: float = 45.0
    hysteresis_deg: float = 2.0
    allow_cancel: bool = False

class Boat:
    def __init__(self, boat_id: int, x: float, y: float, heading: float, speed: float,
                 kin: BoatParams, tcfg: TurnSessionConfig):
        self.id = boat_id
        self.x = float(x)
        self.y = float(y)
        self.h = float(heading)
        self.u = float(speed)
        self.kin = kin
        self.session = HeadingSession(
            tcfg.turn_deg, tcfg.yaw_rate_degps, tcfg.hysteresis_deg, tcfg.allow_cancel
        )
        self.last_steer = 0      # for logging
        self.last_throttle = 0   # for logging

    @staticmethod
    def decode_action(a: int) -> Tuple[int, int]:
        steer = a // 3
        throttle = a % 3
        return steer, throttle  # steer:0/1/2, throttle:0/1/2

    def apply_action(self, a: int):
        steer, throttle = self.decode_action(a)
        self.last_steer, self.last_throttle = steer, throttle
        # Try to start a turn session if idle; otherwise we ignore steer until session completes.
        self.session.start_if_idle(self.h, steer)

    def integrate(self, dt: float):
        # 1) Heading from session
        self.h = self.session.step(self.h, self.last_steer, dt)

        # 2) Speed (passthrough throttle)
        if self.last_throttle == 1:
            self.u = min(self.u + self.kin.accel_rate * dt, self.kin.max_speed)
        elif self.last_throttle == 2:
            self.u = max(self.u - self.kin.decel_rate * dt, 0.0)

        # 3) Position (pure kinematics)
        self.x += math.cos(self.h) * self.u * dt
        self.y += math.sin(self.h) * self.u * dt

# ---------------- Environment ----------------
@dataclass
class EnvConfig:
    # world (meters) & time
    world_w: float = 300.0
    world_h: float = 300.0
    dt: float = 0.05
    physics_substeps: int = 1
    max_steps: int = 3000

    # rendering
    render: bool = False
    pixels_per_meter: float = 3.0
    show_grid: bool = True

    # sensors
    sensor_range: Optional[float] = None  # default = diagonal
    seed: Optional[int] = None

    # goals
    goal_ahead_distance: float = 450.0
    goal_radius: float = 10.0

    # rewards
    progress_weight: float = 0.01
    living_penalty: float = -0.001
    goal_bonus: float = 6.0
    collision_penalty: float = -12.0
    oob_penalty: float = -3.0

    # CPA shaping
    cpa_horizon: float = 60.0
    dcpa_scale: float = 120.0
    tcpa_decay: float = 20.0
    risk_weight: float = 0.10

@dataclass
class SpawnConfig:
    n_boats: int = 2
    start_speed: float = 0.0
    min_sep_factor: float = 4.0  # × collision_radius

class MultiBoatSectorsEnv:
    """
    One shared policy controls all boats:
      - At each env step, call agent for each boat i with its 100-d observation.
      - Apply all actions (steer gets latched if idle; throttle passes through).
      - Integrate all boats together; compute per-boat rewards & terminals.
    """
    def __init__(self,
                 cfg: EnvConfig = EnvConfig(),
                 kin: BoatParams = BoatParams(),
                 tcfg: TurnSessionConfig = TurnSessionConfig(),
                 spawn: SpawnConfig = SpawnConfig()):
        self.cfg = cfg
        self.kin = kin
        self.tcfg = tcfg
        self.spawn = spawn
        self.rng = random.Random(cfg.seed)

        self.ships: List[Boat] = []
        self.goals: List[Tuple[float, float]] = []
        self.prev_goal_d: List[float] = []
        self.reached: List[bool] = []
        self.time = 0.0
        self.step_idx = 0
        self.steps = 0

        # rendering
        self._screen = None
        self._font = None
        self._clock = None
        self.ppm = float(self.cfg.pixels_per_meter)
        if self.cfg.render and HAS_PYGAME:
            self._setup_render()

    # ---------- Rendering ----------
    def _setup_render(self):
        pygame.init()
        width_px = max(200, int(round(self.cfg.world_w * self.ppm)))
        height_px = max(200, int(round(self.cfg.world_h * self.ppm)))
        self._screen = pygame.display.set_mode((width_px, height_px))
        pygame.display.set_caption("Multi-Boat (Sectors) — Turn-Session Control")
        self._font = pygame.font.Font(None, 18)
        self._clock = pygame.time.Clock()

    def enable_render(self):
        if not self._screen and HAS_PYGAME:
            self.cfg.render = True
            self._setup_render()

    def sx(self, xm: float) -> int: return int(round(xm * self.ppm))
    def sy(self, ym: float) -> int: return int(round(ym * self.ppm))

    def _draw_boat(self, surf, b: Boat):
        L, W = self.kin.length_m, self.kin.width_m
        verts = [(+0.5*L, 0.0), (-0.5*L, -0.5*W), (-0.5*L, +0.5*W)]
        ch, sh = math.cos(b.h), math.sin(b.h)
        pts=[]
        for vx, vy in verts:
            wx = b.x + vx*ch - vy*sh
            wy = b.y + vx*sh + vy*ch
            pts.append((self.sx(wx), self.sy(wy)))
        color = (90, 160, 255) if b.id % 2 == 0 else (70, 200, 120)
        pygame.draw.polygon(surf, color, pts)
        # collision radius (visual)
        rad_px = max(1, int(round(self.collision_radius() * self.ppm)))
        pygame.draw.circle(surf, (255,255,255), (self.sx(b.x), self.sy(b.y)), rad_px, 1)
        label = self._font.render(f"{b.id}", True, (255,255,255))
        surf.blit(label, (self.sx(b.x)+8, self.sy(b.y)-8))

    def render(self):
        if not self._screen:
            return
        surf = self._screen
        surf.fill((18,52,86))
        # border
        pygame.draw.rect(surf, (180,180,180),
                         (0,0,self.sx(self.cfg.world_w), self.sy(self.cfg.world_h)), 2)
        # grid
        if self.cfg.show_grid:
            step_m = 100
            for xm in range(0, int(self.cfg.world_w)+1, step_m):
                x = self.sx(xm)
                pygame.draw.line(surf, (40,70,100), (x,0), (x, self.sy(self.cfg.world_h)))
            for ym in range(0, int(self.cfg.world_h)+1, step_m):
                y = self.sy(ym)
                pygame.draw.line(surf, (40,70,100), (0,y), (self.sx(self.cfg.world_w), y))
        # goals
        goal_r_px = max(4, int(4 * self.ppm * 0.75))
        for i, (gx, gy) in enumerate(self.goals):
            pygame.draw.circle(surf, (250,200,50), (self.sx(gx), self.sy(gy)), goal_r_px)
            sh = self.ships[i]
            pygame.draw.line(surf, (200,200,80), (self.sx(sh.x), self.sy(sh.y)), (self.sx(gx), self.sy(gy)), 1)
        # boats
        for b in self.ships:
            self._draw_boat(surf, b)

        pygame.display.flip()
        self._clock.tick(60)

    # ---------- Setup / Reset ----------
    def collision_radius(self) -> float:
        return self.kin.length_m * 0.8

    def _outside(self, b: Boat) -> bool:
        return not (0 <= b.x <= self.cfg.world_w and 0 <= b.y <= self.cfg.world_h)

    def reset(self, seed: Optional[int]=None):
        if seed is not None:
            self.rng.seed(seed)
        self.ships=[]
        self.goals=[]
        self.prev_goal_d=[]
        self.reached=[False]*self.spawn.n_boats
        self.time=0.0
        self.step_idx=0
        self.steps=0

        margin = 80.0
        for i in range(self.spawn.n_boats):
            while True:
                x = self.rng.uniform(margin, self.cfg.world_w - margin)
                y = self.rng.uniform(margin, self.cfg.world_h - margin)
                h = self.rng.uniform(-math.pi, math.pi)
                u = float(self.spawn.start_speed)

                ok=True
                for sh in self.ships:
                    if (sh.x - x)**2 + (sh.y - y)**2 < (self.spawn.min_sep_factor * self.collision_radius())**2:
                        ok=False; break
                if ok:
                    boat = Boat(i, x, y, h, u, self.kin, self.tcfg)
                    self.ships.append(boat)
                    # goal straight ahead
                    gx = x + self.cfg.goal_ahead_distance * math.cos(h)
                    gy = y + self.cfg.goal_ahead_distance * math.sin(h)
                    gx = clamp(gx, 20, self.cfg.world_w - 20)
                    gy = clamp(gy, 20, self.cfg.world_h - 20)
                    self.goals.append((gx, gy))
                    self.prev_goal_d.append(math.hypot(gx - x, gy - y))
                    break

        return self.get_obs_all()

    # ---------- Observations ----------
    def _one_obs(self, idx: int) -> np.ndarray:
        ego = self.ships[idx]
        others = [s for s in self.ships if s is not ego]

        N_SECT = 12
        buckets: Dict[int, List[Tuple[float, dict]]] = {k: [] for k in range(N_SECT)}
        diag = math.hypot(self.cfg.world_w, self.cfg.world_h)
        rng = self.cfg.sensor_range or diag

        for tgt in others:
            dx, dy = tgt.x - ego.x, tgt.y - ego.y
            dist = math.hypot(dx, dy)
            if self.cfg.sensor_range and dist > self.cfg.sensor_range:
                continue
            ch, sh = math.cos(ego.h), math.sin(ego.h)
            x_rel =  ch * dx + sh * dy
            y_rel = -sh * dx + ch * dy
            rel_brg = math.atan2(y_rel, x_rel)

            rel_deg = (angle_deg(rel_brg) + 360.0) % 360.0
            sector = int(rel_deg // (360.0 / N_SECT))

            tcpa, dcpa = tcpa_dcpa(ego.x, ego.y, ego.h, ego.u, tgt.x, tgt.y, tgt.h, tgt.u)
            w_tcpa = math.exp(-tcpa / max(1e-6, self.cfg.tcpa_decay))
            w_dcpa = math.exp(-dcpa / max(1e-6, self.cfg.dcpa_scale))
            score = -(w_tcpa * w_dcpa)

            buckets[sector].append((score, dict(
                x_rel=x_rel, y_rel=y_rel,
                rel_brg=rel_brg, dist=dist,
                tcpa=tcpa, dcpa=dcpa,
                tgt_speed=tgt.u,
                tgt_rel_h=wrap_pi(tgt.h - ego.h),
            )))

        feats = []
        for k in range(N_SECT):
            if buckets[k]:
                buckets[k].sort(key=lambda t: t[0])
                d = buckets[k][0][1]
                feats.extend([
                    d["x_rel"]/rng, d["y_rel"]/rng,
                    d["rel_brg"]/math.pi,
                    d["dist"]/rng,
                    clamp(d["tcpa"], 0.0, self.cfg.cpa_horizon)/self.cfg.cpa_horizon,
                    clamp(d["dcpa"], 0.0, rng)/rng,
                    d["tgt_speed"]/self.kin.max_speed,
                    d["tgt_rel_h"]/math.pi
                ])
            else:
                feats.extend([0.0]*8)

        feats.extend([
            ego.x / self.cfg.world_w,
            ego.y / self.cfg.world_h,
            ego.u / self.kin.max_speed,
            ego.h / math.pi,
        ])
        assert len(feats) == 100
        return np.asarray(feats, dtype=np.float32)

    def get_obs_all(self) -> List[np.ndarray]:
        return [self._one_obs(i) for i in range(len(self.ships))]

    # ---------- Step ----------
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        """
        actions: list of length n_boats, one Discrete(9) action per boat.
        Returns: next_obs_list, reward_list, done, info
        """
        assert len(actions) == len(self.ships)
        self.steps += 1

        # remember previous goal distances
        d_prev = []
        for i in range(len(self.ships)):
            gx, gy = self.goals[i]
            sh = self.ships[i]
            d_prev.append(math.hypot(gx - sh.x, gy - sh.y))

        # apply all actions
        for i, a in enumerate(actions):
            self.ships[i].apply_action(a)

        # integrate all together
        dt = self.cfg.dt / max(1, self.cfg.physics_substeps)
        for _ in range(self.cfg.physics_substeps):
            for s in self.ships:
                s.integrate(dt)
        self.time += self.cfg.dt
        self.step_idx += 1

        # terminals
        done = False
        reason = ""
        # collision (simple circle)
        cr = self.collision_radius()
        for i in range(len(self.ships)):
            for j in range(i+1, len(self.ships)):
                dx = self.ships[i].x - self.ships[j].x
                dy = self.ships[i].y - self.ships[j].y
                if (dx*dx + dy*dy) <= (2 * cr)**2:
                    done = True; reason = "collision"; break
            if done: break
        # out of bounds
        if not done:
            for s in self.ships:
                if self._outside(s):
                    done = True; reason = "out_of_bounds"; break

        # goals reached
        reached_now = []
        for i in range(len(self.ships)):
            gx, gy = self.goals[i]
            sh = self.ships[i]
            d = math.hypot(gx - sh.x, gy - sh.y)
            if (not self.reached[i]) and d <= self.cfg.goal_radius:
                self.reached[i] = True
            reached_now.append(self.reached[i])
        if not done and all(self.reached):
            done = True; reason = "goals_reached"
        if not done and self.steps >= self.cfg.max_steps:
            done = True; reason = "max_steps"

        # rewards (per boat)
        rewards: List[float] = []
        for i in range(len(self.ships)):
            ego = self.ships[i]
            # progress
            gx, gy = self.goals[i]
            d_now = math.hypot(gx - ego.x, gy - ego.y)
            r = 0.0
            r += self.cfg.progress_weight * (d_prev[i] - d_now)
            r += self.cfg.living_penalty
            # arrival bonus (once)
            if self.reached[i] and d_prev[i] > self.cfg.goal_radius and d_now <= self.cfg.goal_radius:
                r += self.cfg.goal_bonus
            # CPA risk shaping (single most risky other)
            worst = 0.0
            for j, tgt in enumerate(self.ships):
                if j == i: continue
                tcpa, dcpa = tcpa_dcpa(ego.x, ego.y, ego.h, ego.u, tgt.x, tgt.y, tgt.h, tgt.u)
                if 0.0 <= tcpa <= self.cfg.cpa_horizon:
                    w = math.exp(-tcpa / self.cfg.tcpa_decay) * math.exp(-dcpa / self.cfg.dcpa_scale)
                    worst = max(worst, w)
            r -= self.cfg.risk_weight * worst
            # terminals
            if reason == "collision": r += self.cfg.collision_penalty
            elif reason == "out_of_bounds": r += self.cfg.oob_penalty

            rewards.append(float(r))

        info = dict(reason=reason, reached=tuple(self.reached))
        return self.get_obs_all(), rewards, done, info
