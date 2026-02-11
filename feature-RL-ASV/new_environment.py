#!/usr/bin/env python3
"""
Two-Boat Sectors Environment with Goals + Smooth Realistic Dynamics
(1 unit = 1 meter) + Render Scaling via pixels_per_meter
-------------------------------------------------------------------

Set EnvConfig.pixels_per_meter to enlarge visuals without touching physics.
Example: pixels_per_meter=2.0 makes a 6 m boat draw as ~12 screen pixels long.

Everything physical (positions, speeds, collisions) stays in METERS.
Only the RENDERING uses a world->screen transform with ppm.

Changes in this version (more visible turning):
- nomoto_K0 -> 0.12   (sets full-rudder turning radius ~20 m for δ_max=25°)
- nomoto_T_min -> 0.6 (snappier yaw build-up at speed)
- r_max -> 45°/s      (allows tighter arcs before clamping)
- prop_wash_gain -> 2.5 (rudder has some authority while accelerating from rest)
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


# =============================
# Utils
# =============================
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def angle_deg(a: float) -> float:
    return a * 180.0 / math.pi

def clamp(x: float, lo: float, hi: float) -> float:
    return hi if x > hi else lo if x < lo else x

def tcpa_dcpa(ax, ay, ah, aspd, bx, by, bh, bspd) -> Tuple[float, float]:
    """CPA for constant-velocity points (used for shaping/features)."""
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


# =============================
# Ship & Physics  (meters)
# =============================
@dataclass
class ShipParams:
    # ~20 ft runabout geometry (realistic)
    length: float = 6.0     # meters
    width:  float = 2.2     # meters

    # Surge (m/s, m/s^2)
    max_speed:  float = 18.0   # ≈ 35 kn top
    min_speed:  float = 0.0
    accel_rate: float = 1.6    # ~0–30 mph in ~8–9 s
    decel_rate: float = 1.0    # reverse thrust; weaker than forward

    # Yaw/turning (radians & seconds)
    rudder_max:   float = math.radians(25.0)   # ±25°
    rudder_rate:  float = math.radians(40.0)   # 40°/s slew
    nomoto_T0:    float = 4.0                  # baseline time constant
    nomoto_T_min: float = 0.6                  # faster low-speed response floor
    nomoto_K0:    float = 0.12                 # << key: sets full-rudder radius ~1/(K0*δ_max) ≈ 19–20 m
    r_max:        float = math.radians(45.0)   # higher yaw cap -> tighter turns before clamp

    # Mild prop wash for tiny yaw at near-0 speed (only when accelerating)
    prop_wash_gain: float = 2.5

    # Hydrodynamics
    quad_drag_k:     float = 0.0040   # a ≈ -k*u*|u| → ~0.4 m/s² @ 10 m/s
    leeway_k:        float = 0.25     # slip angle ≈ k * yaw_rate
    turn_drag_coeff: float = 0.05     # extra drag vs rudder deflection

class Ship:
    def __init__(self, ship_id: int, x: float, y: float, heading: float, speed: float, params: ShipParams):
        self.id = ship_id
        self.x = float(x)   # meters
        self.y = float(y)   # meters
        self.h = float(heading)  # rad
        self.u = float(speed)    # m/s

        self.r = 0.0             # yaw rate (rad/s)
        self.delta = 0.0         # rudder angle (rad)
        self.params = params

        # discrete commands (held between substeps)
        self.last_helm = 0  # -1 right, 0 midships, +1 left
        self.last_thr  = 0  # -1 decel, 0 coast, +1 accel

    @staticmethod
    def decode_action(a: int) -> Tuple[int, int]:
        steer = a // 3
        throttle = a % 3
        helm = (-1 if steer == 1 else +1 if steer == 2 else 0)
        thr  = (-1 if throttle == 2 else +1 if throttle == 1 else 0)
        return helm, thr

    def apply_action(self, a: int):
        self.last_helm, self.last_thr = self.decode_action(a)

    def integrate(self, dt: float):
        """Advance ship dynamics by dt using current commands (meters, seconds)."""
        p = self.params
        helm, thr = self.last_helm, self.last_thr

        # 1) Rudder slewing to target
        target_delta = helm * p.rudder_max
        dmax = p.rudder_rate * dt
        if self.delta < target_delta:
            self.delta = min(self.delta + dmax, target_delta)
        elif self.delta > target_delta:
            self.delta = max(self.delta - dmax, target_delta)

        # 2) Effective flow speed for yaw dynamics
        U_eff = max(self.u, p.prop_wash_gain * max(thr, 0))

        # 3) Nomoto-like yaw
        T = max(p.nomoto_T0 / max(U_eff, 1e-3), p.nomoto_T_min)
        K = p.nomoto_K0 * U_eff
        r_dot = (K * self.delta - self.r) / T
        self.r += r_dot * dt
        self.r = clamp(self.r, -p.r_max, p.r_max)

        # 4) Surge with quadratic drag and turn-induced extra drag
        a_cmd = (p.accel_rate if thr > 0 else -p.decel_rate if thr < 0 else 0.0)
        turn_drag = p.turn_drag_coeff * (abs(self.delta) / max(1e-6, p.rudder_max)) * self.u
        self.u += (a_cmd - p.quad_drag_k * self.u * abs(self.u) - turn_drag) * dt
        self.u = clamp(self.u, p.min_speed, p.max_speed)

        # 5) Position update (include small leeway slip proportional to yaw rate)
        beta = p.leeway_k * self.r
        self.h = wrap_pi(self.h + self.r * dt)
        self.x += math.cos(self.h + beta) * self.u * dt
        self.y += math.sin(self.h + beta) * self.u * dt


# =============================
# Environment
# =============================
@dataclass
class EnvConfig:
    # Physical world size (meters) — keep your edits
    world_w: int = 300
    world_h: int = 300

    dt: float = 0.05                # main step (seconds)
    physics_substeps: int = 4       # substeps per env step
    sensor_range: Optional[float] = None
    seed: Optional[int] = None

    # Rendering — keep your edits
    render: bool = False
    show_sectors: bool = False
    show_grid: bool = True
    pixels_per_meter: float = 3.0   # visual scale only

    # Goals (meters)
    goal_ahead_distance: float = 450.0  # will clamp inside world
    goal_radius: float = 10.0

    # Rewards (dimensionless)
    progress_weight: float = 0.01
    living_penalty: float = -0.001
    goal_bonus: float = 6.0
    collision_penalty: float = -12.0
    oob_penalty: float = -3.0

    # CPA risk shaping
    cpa_horizon: float = 60.0
    dcpa_scale: float = 120.0
    tcpa_decay: float = 20.0
    risk_weight: float = 0.10

    # Simple COLREGs-inspired shaping
    colregs_penalty: float = 0.05

class TwoBoatSectorsEnv:
    """
    One policy acts twice per env step (boat 0, then boat 1). We apply both,
    integrate together with small substeps, then compute rewards and terminals.

    Physics in meters. Rendering uses pixels_per_meter (ppm) for world->screen.
    """
    def __init__(self, cfg: EnvConfig = EnvConfig(), ship_params: ShipParams = ShipParams()):
        self.cfg = cfg
        self.ship_params = ship_params
        self.rng = random.Random(cfg.seed)

        self.world_w = cfg.world_w
        self.world_h = cfg.world_h

        self.ships: List[Ship] = []
        self.goals: List[Tuple[float, float]] = []       # (gx, gy) per ship
        self.prev_goal_d: List[float] = []
        self.reached: List[bool] = []
        self.time = 0.0
        self.step_index = 0

        # rendering
        self._screen = None
        self._font = None
        self._clock = None
        self.ppm = float(cfg.pixels_per_meter)
        if self.cfg.render and HAS_PYGAME:
            self._setup_render()

    # ---------- Rendering ----------
    def _setup_render(self):
        pygame.init()
        width_px  = max(200, int(round(self.world_w * self.ppm)))
        height_px = max(200, int(round(self.world_h * self.ppm)))
        self._screen = pygame.display.set_mode((width_px, height_px))
        pygame.display.set_caption("Two-Boat (Goals) — Realistic Dynamics (meters)")
        self._font = pygame.font.Font(None, 18)
        self._clock = pygame.time.Clock()

    def enable_render(self):
        if not self._screen and HAS_PYGAME:
            self.cfg.render = True
            self._setup_render()

    # world->screen transforms (meters -> pixels)
    def sx(self, x_m: float) -> int:
        return int(round(x_m * self.ppm))
    def sy(self, y_m: float) -> int:
        return int(round(y_m * self.ppm))

    def render(self):
        if not self._screen:
            return
        surf = self._screen
        surf.fill((18, 52, 86))

        # border
        pygame.draw.rect(surf, (180, 180, 180),
                         (0, 0, self.sx(self.world_w), self.sy(self.world_h)), 2)

        # grid (every 100 m)
        if self.cfg.show_grid:
            step_m = 100
            for xm in range(0, self.world_w + 1, step_m):
                x = self.sx(xm)
                pygame.draw.line(surf, (40, 70, 100), (x, 0), (x, self.sy(self.world_h)))
            for ym in range(0, self.world_h + 1, step_m):
                y = self.sy(ym)
                pygame.draw.line(surf, (40, 70, 100), (0, y), (self.sx(self.world_w), y))

        # goals
        goal_r_px = max(4, int(4 * self.ppm * 0.75))  # visible but modest
        for i, (gx, gy) in enumerate(self.goals):
            pygame.draw.circle(surf, (250, 200, 50), (self.sx(gx), self.sy(gy)), goal_r_px)
            sh = self.ships[i]
            pygame.draw.line(surf, (200, 200, 80), (self.sx(sh.x), self.sy(sh.y)), (self.sx(gx), self.sy(gy)), 1)

        # ships
        for sh in self.ships:
            self._draw_ship(surf, sh)
            if self.cfg.show_sectors:
                self._draw_sector_rays(surf, sh)

        pygame.display.flip()
        self._clock.tick(60)

    def _draw_ship(self, surf, sh: Ship):
        Lm, Wm = self.ship_params.length, self.ship_params.width  # meters
        # local vertices in METERS
        verts_local_m = [(+0.5*Lm, 0.0), (-0.5*Lm, -0.5*Wm), (-0.5*Lm, +0.5*Wm)]
        ch, shn = math.cos(sh.h), math.sin(sh.h)

        pts = []
        for vx_m, vy_m in verts_local_m:
            wx_m = sh.x + vx_m*ch - vy_m*shn
            wy_m = sh.y + vx_m*shn + vy_m*ch
            pts.append((self.sx(wx_m), self.sy(wy_m)))

        color = (90, 160, 255) if sh.id == 0 else (70, 200, 120)
        pygame.draw.polygon(surf, color, pts)

        # collision circle (visual) — scale radius in meters by ppm
        rad_px = max(1, int(round(self.collision_radius() * self.ppm)))
        pygame.draw.circle(surf, (255, 255, 255), (self.sx(sh.x), self.sy(sh.y)), rad_px, 1)

        label = self._font.render(f"{sh.id}", True, (255, 255, 255))
        surf.blit(label, (self.sx(sh.x) + 8, self.sy(sh.y) - 8))

    def _draw_sector_rays(self, surf, sh: Ship, n=12, ray_len_m: Optional[float] = None):
        Lm = ray_len_m or 320.0  # meters
        for k in range(n):
            ang = sh.h + 2.0 * math.pi * k / n
            x2 = self.sx(sh.x + Lm * math.cos(ang))
            y2 = self.sy(sh.y + Lm * math.sin(ang))
            pygame.draw.line(surf, (220, 220, 220), (self.sx(sh.x), self.sy(sh.y)), (x2, y2), 1)

    # ---------- Core Env ----------
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.ships = []
        self.goals = []
        self.prev_goal_d = []
        self.reached = [False, False]
        self.time = 0.0
        self.step_index = 0

        margin = 80  # meters
        for i in range(2):
            while True:
                x = self.rng.uniform(margin, self.world_w - margin)
                y = self.rng.uniform(margin, self.world_h - margin)
                h = self.rng.uniform(-math.pi, math.pi)
                u = 0.0  # start from rest

                # keep starts sufficiently apart
                ok = True
                for sh in self.ships:
                    if (sh.x - x)**2 + (sh.y - y)**2 < (4 * self.collision_radius())**2:
                        ok = False; break
                if ok:
                    self.ships.append(Ship(i, x, y, h, u, self.ship_params))
                    # goal straight ahead from start heading
                    gx = x + self.cfg.goal_ahead_distance * math.cos(h)
                    gy = y + self.cfg.goal_ahead_distance * math.sin(h)
                    gx = clamp(gx, 20, self.world_w - 20)
                    gy = clamp(gy, 20, self.world_h - 20)
                    self.goals.append((gx, gy))
                    self.prev_goal_d.append(math.hypot(gx - x, gy - y))
                    break

        return self.get_obs(0), self.get_obs(1)

    def collision_radius(self) -> float:
        # generous 80% of length (visual & simple) — in METERS
        return self.ship_params.length * 0.8

    def _outside(self, sh: Ship) -> bool:
        return not (0 <= sh.x <= self.world_w and 0 <= sh.y <= self.world_h)

    # ---- Observations (100-d) ----
    def get_obs(self, i: int) -> np.ndarray:
        ego = self.ships[i]
        others = [s for s in self.ships if s is not ego]

        N_SECT = 12
        buckets: Dict[int, List[Tuple[float, dict]]] = {k: [] for k in range(N_SECT)}

        diag = math.hypot(self.world_w, self.world_h)  # meters
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
                    d["tgt_speed"]/self.ship_params.max_speed,
                    d["tgt_rel_h"]/math.pi
                ])
            else:
                feats.extend([0.0]*8)

        feats.extend([
            ego.x / self.world_w,
            ego.y / self.world_h,
            ego.u / self.ship_params.max_speed,
            ego.h / math.pi,
        ])
        assert len(feats) == 100
        return np.asarray(feats, dtype=np.float32)

    # ---- Step ----
    def step(self, actions: Tuple[int, int]) -> Tuple[bool, dict]:
        a0, a1 = actions
        sh0, sh1 = self.ships[0], self.ships[1]

        # prior distances for progress reward
        d0_prev = self.prev_goal_d[0]
        d1_prev = self.prev_goal_d[1]

        # apply commands
        sh0.apply_action(a0)
        sh1.apply_action(a1)

        # integrate together in substeps
        dt = self.cfg.dt / max(1, self.cfg.physics_substeps)
        for _ in range(self.cfg.physics_substeps):
            sh0.integrate(dt)
            sh1.integrate(dt)

        self.time += self.cfg.dt
        self.step_index += 1

        # terminals
        done = False
        reason = ""
        # collision (circle, meters)
        dx, dy = (sh0.x - sh1.x), (sh0.y - sh1.y)
        if dx*dx + dy*dy <= (2 * self.collision_radius())**2:
            done, reason = True, "collision"
        # out-of-bounds
        if not done and (self._outside(sh0) or self._outside(sh1)):
            done, reason = True, "out_of_bounds"

        # goal distances (meters)
        def goal_dist(i: int) -> float:
            gx, gy = self.goals[i]; sh = self.ships[i]
            return math.hypot(gx - sh.x, gy - sh.y)

        d0 = goal_dist(0)
        d1 = goal_dist(1)

        just0 = (d0 <= self.cfg.goal_radius) and not hasattr(sh0, "_arrived")
        just1 = (d1 <= self.cfg.goal_radius) and not hasattr(sh1, "_arrived")
        if just0: setattr(sh0, "_arrived", True)
        if just1: setattr(sh1, "_arrived", True)

        if not done and hasattr(sh0, "_arrived") and hasattr(sh1, "_arrived"):
            done, reason = True, "goals_reached"

        # rewards
        def per_boat_reward(ego: Ship, tgt: Ship, d_prev: float, d_now: float, just_arrived: bool) -> float:
            r = 0.0
            # progress (meters → reward)
            r += self.cfg.progress_weight * (d_prev - d_now)
            # living
            r += self.cfg.living_penalty
            # arrival bonus (once)
            if just_arrived and (d_prev > self.cfg.goal_radius):
                r += self.cfg.goal_bonus
            # CPA risk shaping + simple COLREGs nudge
            tcpa, dcpa = tcpa_dcpa(ego.x, ego.y, ego.h, ego.u, tgt.x, tgt.y, tgt.h, tgt.u)
            if 0.0 <= tcpa <= self.cfg.cpa_horizon:
                w = math.exp(-tcpa / self.cfg.tcpa_decay) * math.exp(-dcpa / self.cfg.dcpa_scale)
                r -= self.cfg.risk_weight * w
                # starboard-forward quarter + turning to PORT → small penalty
                rel_brg = math.atan2(
                    -(math.sin(ego.h) * (tgt.x - ego.x) - math.cos(ego.h) * (tgt.y - ego.y)),
                    (math.cos(ego.h) * (tgt.x - ego.x) + math.sin(ego.h) * (tgt.y - ego.y))
                )
                rel_deg = (angle_deg(rel_brg) + 360.0) % 360.0
                if (0.0 <= rel_deg <= 112.5) and (ego.last_helm > 0):  # helm>0 → porting
                    r -= self.cfg.colregs_penalty * w
            # terminals
            if reason == "collision":
                r += self.cfg.collision_penalty
            elif reason == "out_of_bounds":
                r += self.cfg.oob_penalty
            return float(r)

        r0 = per_boat_reward(sh0, sh1, d0_prev, d0, just0)
        r1 = per_boat_reward(sh1, sh0, d1_prev, d1, just1)

        # update stored prev distances
        self.prev_goal_d[0] = d0
        self.prev_goal_d[1] = d1

        info = dict(rewards=(r0, r1), reason=reason,
                    reached=(hasattr(sh0, "_arrived"), hasattr(sh1, "_arrived")))
        return done, info


# --------- Quick manual demo ----------
if __name__ == "__main__":
    # Keep your render scale / world size edits
    cfg = EnvConfig(render=True, pixels_per_meter=3.0, show_sectors=False, dt=0.05, physics_substeps=4, seed=0)
    env = TwoBoatSectorsEnv(cfg)
    _ = env.reset()
    import time
    while True:
        # action index: steer*3 + throttle
        # 0 = (steer=0, throttle=0) straight + coast
        # 7 = (steer=2, throttle=1) turn left + accelerate  ← try to SEE curvature
        done, info = env.step((7, 7))
        env.render()
        if done:
            print("Ended:", info)
            time.sleep(1.0)
            _ = env.reset()
