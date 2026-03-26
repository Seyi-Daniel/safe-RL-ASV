from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from trainings.dcpa_sampled.hyperparameters import EnvParams, RewardParams

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


class SingleVessel2FeatureEnv:
    """Decentralized multi-vessel environment with per-vessel 96-dim radar observations."""

    def __init__(
        self,
        env_params: EnvParams = EnvParams(),
        reward_params: RewardParams = RewardParams(),
        render: bool = False,
    ):
        self.envp = env_params
        self.rewp = reward_params
        self.rng = random.Random(self.envp.seed)

        self.num_vessels = max(2, int(getattr(self.envp, "num_vessels", 2)))
        self.vessel_ids: List[str] = [f"vessel{i + 1}" for i in range(self.num_vessels)]
        self.vessels: Dict[str, Vessel] = {}

        self.start_x = 0.5 * self.envp.world_w
        self.start_y = 0.5 * self.envp.world_h

        self.time = 0.0
        self.step_idx = 0
        self.max_steps = max(1, int(round(self.envp.episode_seconds / self.envp.dt)))

        self.reached_by_vessel: Dict[str, bool] = {vid: False for vid in self.vessel_ids}
        self.model_control_latched: Dict[str, bool] = {vid: False for vid in self.vessel_ids}
        self.rl_active_by_vessel: Dict[str, bool] = {vid: False for vid in self.vessel_ids}
        self.role_by_vessel: Dict[str, str] = {vid: "none" for vid in self.vessel_ids}
        self.scenario_by_vessel: Dict[str, str] = {vid: "safe" for vid in self.vessel_ids}
        self.start_pos_by_vessel: Dict[str, Tuple[float, float]] = {vid: (0.0, 0.0) for vid in self.vessel_ids}
        self.start_speed_by_vessel: Dict[str, float] = {vid: 0.0 for vid in self.vessel_ids}
        self.prev_goal_dist_by_vessel: Dict[str, float] = {vid: 0.0 for vid in self.vessel_ids}
        self.prev_heading_err_by_vessel: Dict[str, float] = {vid: 0.0 for vid in self.vessel_ids}
        self.steps_taken_by_vessel: Dict[str, int] = {vid: 0 for vid in self.vessel_ids}
        self.control_source_by_vessel: Dict[str, str] = {vid: "scripted" for vid in self.vessel_ids}
        self.prev_rudder_sign_by_vessel: Dict[str, int] = {vid: 0 for vid in self.vessel_ids}

        self.risk_of_collision = False
        self.last_dcpa = float("inf")
        self.last_tcpa = float("inf")
        self.colregs_scenario = "safe"
        self.any_rl_ever_triggered = False
        self.encounter_was_risky = False
        self.safe_pass_awarded = False

        self.render_enabled = render and HAS_PYGAME
        self._screen = None
        self._clock = None
        self._font = None
        self.paused = False
        if self.render_enabled:
            self._init_render()

        self._sync_compat_attrs()

    def _sync_compat_attrs(self) -> None:
        self.vessel1 = self.vessels.get("vessel1")
        self.vessel2 = self.vessels.get("vessel2")
        self.vessel1_reached = self.reached_by_vessel.get("vessel1", True)
        self.vessel2_reached = self.reached_by_vessel.get("vessel2", True)
        self.vessel1_rl_active = self.rl_active_by_vessel.get("vessel1", False)
        self.vessel2_rl_active = self.rl_active_by_vessel.get("vessel2", False)
        self.vessel1_model_control_latched = self.model_control_latched.get("vessel1", False)
        self.vessel2_model_control_latched = self.model_control_latched.get("vessel2", False)
        self.vessel1_role = self.role_by_vessel.get("vessel1", "none")
        self.vessel2_role = self.role_by_vessel.get("vessel2", "none")
        self.vessel1_start_pos = self.start_pos_by_vessel.get("vessel1", (0.0, 0.0))
        self.vessel2_start_pos = self.start_pos_by_vessel.get("vessel2", (0.0, 0.0))
        self.vessel1_start_speed = self.start_speed_by_vessel.get("vessel1", 0.0)
        self.vessel2_start_speed = self.start_speed_by_vessel.get("vessel2", 0.0)

    def _init_render(self) -> None:
        pygame.init()
        w = int(self.envp.world_w * self.envp.pixels_per_meter)
        h = int(self.envp.world_h * self.envp.pixels_per_meter)
        self._screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("DCPA Sampled (N-vessel)")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 18)

    def close(self) -> None:
        if self._screen is not None:
            pygame.quit()
            self._screen = None

    def set_secondary_policy(self, fn) -> None:
        _ = fn

    def sx(self, x: float) -> int:
        return int(round(x * self.envp.pixels_per_meter))

    def sy(self, y: float) -> int:
        return int(round(y * self.envp.pixels_per_meter))

    def _outside(self, v: Vessel) -> bool:
        return not (0.0 <= v.x <= self.envp.world_w and 0.0 <= v.y <= self.envp.world_h)

    def _goal_distance(self, v: Vessel) -> float:
        return math.hypot(v.goal_x - v.x, v.goal_y - v.y)

    def _goal_heading_error(self, v: Vessel) -> float:
        goal_bearing = math.atan2(v.goal_y - v.y, v.goal_x - v.x)
        return abs(wrap_pi(goal_bearing - v.h))

    def _point_on_big_circle(self, ang: float) -> Tuple[float, float]:
        r = self.envp.vessel2_outer_radius
        return self.start_x + r * math.cos(ang), self.start_y + r * math.sin(ang)

    def _inward_facing_heading(self, pos_x: float, pos_y: float) -> float:
        to_center_angle = math.atan2(self.start_y - pos_y, self.start_x - pos_x)
        offset = self.rng.uniform(-0.5 * math.pi, 0.5 * math.pi)
        return wrap_pi(to_center_angle + offset)

    def _sample_ring_vessel(self) -> Vessel:
        start_ang = self.rng.uniform(0.0, 2.0 * math.pi)
        goal_ang = self.rng.uniform(0.0, 2.0 * math.pi)
        sx, sy = self._point_on_big_circle(start_ang)
        gx, gy = self._point_on_big_circle(goal_ang)
        heading = self._inward_facing_heading(sx, sy)
        speed = self.rng.uniform(self.envp.vessel2_min_speed, self.envp.vessel2_max_speed)
        return Vessel(sx, sy, heading, speed, gx, gy)

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

    def _pure_pursuit_rudder_cmd(self, v: Vessel) -> float:
        bearing_to_goal = math.atan2(v.goal_y - v.y, v.goal_x - v.x)
        heading_error = wrap_pi(bearing_to_goal - v.h)
        return clamp(heading_error / math.radians(self.envp.pp_heading_gain_deg), -1.0, 1.0)

    def _advance_scripted(self, vessel_id: str, dt: float) -> None:
        if self.reached_by_vessel[vessel_id]:
            return
        v = self.vessels[vessel_id]
        d = self._goal_distance(v)
        if d <= self.envp.goal_radius:
            self.reached_by_vessel[vessel_id] = True
            v.speed = 0.0
            return

        if vessel_id == "vessel1":
            rudder_cmd = 0.0
            v.speed = clamp(v.speed, self.envp.min_speed, self.envp.max_speed)
            source = "straight"
        else:
            rudder_cmd = self._pure_pursuit_rudder_cmd(v)
            v.speed = clamp(v.speed, self.envp.vessel2_min_speed, self.envp.vessel2_max_speed)
            source = "pure_pursuit"

        self._integrate_rudder_heading(v, rudder_cmd, dt)
        travel = min(v.speed * dt, d)
        v.x += travel * math.cos(v.h)
        v.y += travel * math.sin(v.h)
        self.control_source_by_vessel[vessel_id] = source
        if self._goal_distance(v) <= self.envp.goal_radius:
            self.reached_by_vessel[vessel_id] = True
            v.speed = 0.0

    def _advance_controlled(self, vessel_id: str, rudder_cmd: float, throttle_cmd: float, dt: float) -> None:
        if self.reached_by_vessel[vessel_id]:
            return
        v = self.vessels[vessel_id]
        d = self._goal_distance(v)
        if d <= self.envp.goal_radius:
            self.reached_by_vessel[vessel_id] = True
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
        v.x += travel * math.cos(v.h)
        v.y += travel * math.sin(v.h)
        self.control_source_by_vessel[vessel_id] = "rl_external"

        if self._goal_distance(v) <= self.envp.goal_radius:
            self.reached_by_vessel[vessel_id] = True
            v.speed = 0.0

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
        tcpa = max(tcpa, 0.0)
        cx = rx + rvx * tcpa
        cy = ry + rvy * tcpa
        return tcpa, math.hypot(cx, cy)

    def _relative_bearing_deg(self, observer: Vessel, target: Vessel) -> float:
        dx = target.x - observer.x
        dy = target.y - observer.y
        global_bearing = math.atan2(dy, dx)
        rel_bearing_rad = wrap_pi(global_bearing - observer.h)
        return (math.degrees(rel_bearing_rad) + 360.0) % 360.0

    @staticmethod
    def _bearing_to_signed_deg(bearing_360: float) -> float:
        return ((bearing_360 + 180.0) % 360.0) - 180.0

    def _bearing_in_sector(self, bearing_deg: float, start_deg: float, end_deg: float, inclusive: bool = True) -> bool:
        b = bearing_deg % 360.0
        s = start_deg % 360.0
        e = end_deg % 360.0
        if s <= e:
            return s <= b <= e if inclusive else s < b < e
        return (b >= s or b <= e) if inclusive else (b > s or b < e)

    def classify_geometry(self, own: Vessel, other: Vessel) -> Tuple[str, float, float]:
        rb_own = self._relative_bearing_deg(own, other)
        rb_other = self._relative_bearing_deg(other, own)

        head_on_half = self.envp.colregs_head_on_half_angle_deg
        head_on_min = (360.0 - head_on_half) % 360.0
        crossing_max = self.envp.colregs_crossing_starboard_max_deg
        overtaking_max = self.envp.colregs_overtaking_aft_max_deg

        head_on = self._bearing_in_sector(rb_own, head_on_min, head_on_half) and self._bearing_in_sector(
            rb_other, head_on_min, head_on_half
        )
        if head_on:
            return "head_on", rb_own, rb_other

        own_aft = self._bearing_in_sector(rb_own, crossing_max, overtaking_max)
        other_aft = self._bearing_in_sector(rb_other, crossing_max, overtaking_max)
        if own_aft or other_aft:
            return "overtaking", rb_own, rb_other

        return "crossing", rb_own, rb_other

    def assign_roles(self, scenario: str, rb_own: float, rb_other: float) -> Tuple[str, str]:
        crossing_max = self.envp.colregs_crossing_starboard_max_deg
        overtaking_max = self.envp.colregs_overtaking_aft_max_deg

        if scenario == "head_on":
            return "give_way", "give_way"
        if scenario == "overtaking":
            own_aft = self._bearing_in_sector(rb_own, crossing_max, overtaking_max)
            other_aft = self._bearing_in_sector(rb_other, crossing_max, overtaking_max)
            if own_aft and not other_aft:
                return "stand_on", "give_way"
            if other_aft and not own_aft:
                return "give_way", "stand_on"
            return "give_way", "give_way"

        rb_own_signed = self._bearing_to_signed_deg(rb_own)
        rb_other_signed = self._bearing_to_signed_deg(rb_other)
        own_starboard = (-crossing_max < rb_own_signed) and (rb_own_signed < 0.0)
        other_starboard = (-crossing_max < rb_other_signed) and (rb_other_signed < 0.0)
        if own_starboard and not other_starboard:
            return "give_way", "stand_on"
        if other_starboard and not own_starboard:
            return "stand_on", "give_way"
        return "give_way", "stand_on"

    def _pairwise_assessment(self) -> Dict[str, object]:
        roles = {vid: "none" for vid in self.vessel_ids}
        scenarios = {vid: "safe" for vid in self.vessel_ids}
        in_risk_pairs = []
        min_dcpa = float("inf")
        min_tcpa_pos = float("inf")

        for i, a_id in enumerate(self.vessel_ids):
            for b_id in self.vessel_ids[i + 1 :]:
                a = self.vessels[a_id]
                b = self.vessels[b_id]
                tcpa, dcpa = self._tcpa_dcpa(a, b)
                risk = (0.0 <= tcpa <= self.envp.tcpa_risk_threshold) and (dcpa <= self.envp.dcpa_risk_threshold)
                min_dcpa = min(min_dcpa, dcpa)
                if tcpa > 0.0:
                    min_tcpa_pos = min(min_tcpa_pos, tcpa)
                if not risk:
                    continue

                scenario, rb_a, rb_b = self.classify_geometry(a, b)
                role_a, role_b = self.assign_roles(scenario, rb_a, rb_b)
                in_risk_pairs.append((a_id, b_id, scenario, role_a, role_b, tcpa, dcpa))

                if role_a == "give_way":
                    roles[a_id] = "give_way"
                    scenarios[a_id] = scenario
                elif roles[a_id] == "none":
                    roles[a_id] = "stand_on"
                    scenarios[a_id] = scenario

                if role_b == "give_way":
                    roles[b_id] = "give_way"
                    scenarios[b_id] = scenario
                elif roles[b_id] == "none":
                    roles[b_id] = "stand_on"
                    scenarios[b_id] = scenario

        return {
            "roles": roles,
            "scenarios": scenarios,
            "risk": len(in_risk_pairs) > 0,
            "pairs": in_risk_pairs,
            "dcpa": min_dcpa,
            "tcpa": min_tcpa_pos,
        }

    def _compute_progress_reward_for_vessel(
        self,
        prev_dist: float,
        curr_dist: float,
        prev_heading_err: float,
        curr_heading_err: float,
    ) -> float:
        dist_eps = 1e-4
        heading_eps = 1e-4
        heading_shaping = 0.20 * self.rewp.progress_weight
        dist_delta = prev_dist - curr_dist
        heading_delta = prev_heading_err - curr_heading_err
        base_term = self.rewp.progress_weight * dist_delta

        heading_term = heading_shaping if heading_delta >= -heading_eps else -heading_shaping
        if dist_delta > dist_eps or dist_delta < -dist_eps:
            return base_term + heading_term
        if heading_delta > heading_eps or heading_delta < -heading_eps:
            return heading_term
        return 0.0

    def _apply_head_on_shaping(self, rudder: float, tcpa: float) -> float:
        reward = 0.0
        if rudder > self.rewp.starboard_min_rudder:
            reward += 0.3
            if tcpa > self.rewp.early_action_tcpa_threshold:
                reward += 0.2
            elif tcpa < self.rewp.late_action_tcpa_threshold:
                reward -= 0.2
        elif rudder < self.rewp.port_max_rudder:
            reward -= 0.5
        else:
            if tcpa < self.rewp.late_action_tcpa_threshold:
                reward -= 0.3
        return reward

    def _apply_crossing_shaping(self, rudder: float, tcpa: float, dcpa: float) -> float:
        reward = 0.0
        if rudder > self.rewp.starboard_min_rudder:
            reward += 0.25
            if tcpa > self.rewp.early_action_tcpa_threshold:
                reward += 0.15
            elif tcpa < self.rewp.late_action_tcpa_threshold:
                reward -= 0.15
        elif rudder < self.rewp.port_max_rudder:
            reward -= 0.4
        if dcpa < self.rewp.danger_dcpa_threshold and tcpa > 0.0:
            reward -= 0.4
        return reward

    def _apply_overtaking_shaping(self, tcpa: float, dcpa: float) -> float:
        reward = 0.0
        if dcpa > self.rewp.safe_dcpa_threshold:
            reward += 0.2
        if dcpa < self.rewp.danger_dcpa_threshold:
            reward -= 0.3
        if dcpa < self.rewp.safe_dcpa_threshold and tcpa < self.rewp.late_action_tcpa_threshold:
            reward -= 0.2
        return reward

    def _scenario_local_shaping(self, scenario: str, rl_active: bool, rudder: float, tcpa: float, dcpa: float) -> float:
        if not rl_active:
            return 0.0
        if scenario == "head_on":
            return self._apply_head_on_shaping(rudder, tcpa)
        if scenario == "crossing":
            return self._apply_crossing_shaping(rudder, tcpa, dcpa)
        if scenario == "overtaking":
            return self._apply_overtaking_shaping(tcpa, dcpa)
        return 0.0

    def _normalize_action_vector(self, action: Union[np.ndarray, Tuple[float, float], list, None]) -> Optional[np.ndarray]:
        if action is None:
            return None
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size < 2:
            raise ValueError("Action must contain [rudder_cmd, throttle_cmd].")
        return np.asarray([clamp(float(a[0]), -1.0, 1.0), clamp(float(a[1]), -1.0, 1.0)], dtype=np.float32)

    def _resolve_step_actions(
        self,
        action: Union[np.ndarray, Tuple[float, float], list, Dict[str, Union[np.ndarray, Tuple[float, float], list]]],
    ) -> Dict[str, Optional[np.ndarray]]:
        per_vessel = {vid: None for vid in self.vessel_ids}
        if isinstance(action, dict):
            for vid in self.vessel_ids:
                per_vessel[vid] = self._normalize_action_vector(action.get(vid))
            return per_vessel

        shared = self._normalize_action_vector(action)
        for vid in self.vessel_ids:
            per_vessel[vid] = shared
        return per_vessel

    def _get_sector_index(self, relative_bearing_deg: float) -> int:
        b = relative_bearing_deg % 360.0
        if b >= 350.0 or b < 10.0:
            return 0
        if b < 40.0:
            return 1
        if b < 75.0:
            return 2
        if b < 112.5:
            return 3
        if b < 180.0:
            return 4
        if b < 247.5:
            return 5
        if b < 285.0:
            return 6
        if b < 320.0:
            return 7
        return 8

    def _build_sector_features(self, own_vessel: Vessel, target_vessel: Vessel, distance: float, relative_bearing_deg: float) -> List[float]:
        distance_norm = clamp(distance / max(1e-6, self.envp.sensor_range), 0.0, 1.0)
        bearing_rad = math.radians(relative_bearing_deg)
        bearing_sin = math.sin(bearing_rad)
        bearing_cos = math.cos(bearing_rad)
        relative_heading = wrap_pi(target_vessel.h - own_vessel.h)
        relative_heading_sin = math.sin(relative_heading)
        relative_heading_cos = math.cos(relative_heading)
        target_speed_norm = target_vessel.speed / max(1e-6, self.envp.max_speed)

        los_x = (target_vessel.x - own_vessel.x) / max(1e-6, distance)
        los_y = (target_vessel.y - own_vessel.y) / max(1e-6, distance)
        rvx = math.cos(target_vessel.h) * target_vessel.speed - math.cos(own_vessel.h) * own_vessel.speed
        rvy = math.sin(target_vessel.h) * target_vessel.speed - math.sin(own_vessel.h) * own_vessel.speed
        closing_speed = -(rvx * los_x + rvy * los_y)
        closing_speed_norm = math.tanh(closing_speed / max(1e-6, self.envp.max_speed))

        tcpa, dcpa = self._tcpa_dcpa(own_vessel, target_vessel)
        tcpa_norm = 1.0 if not math.isfinite(tcpa) else clamp(tcpa / max(1e-6, self.envp.tcpa_risk_threshold), 0.0, 1.0)
        dcpa_norm = 1.0 if not math.isfinite(dcpa) else clamp(dcpa / max(1e-6, self.envp.dcpa_risk_threshold), 0.0, 1.0)

        return [1.0, distance_norm, bearing_sin, bearing_cos, relative_heading_sin, relative_heading_cos, target_speed_norm, closing_speed_norm, tcpa_norm, dcpa_norm]

    def _build_radar_observation(self, own_vessel: Vessel) -> List[float]:
        nearest_by_sector: Dict[int, Tuple[float, Vessel, float]] = {}
        for candidate in self.vessels.values():
            if candidate is own_vessel:
                continue
            distance = math.hypot(candidate.x - own_vessel.x, candidate.y - own_vessel.y)
            if distance > self.envp.sensor_range:
                continue
            bearing = self._relative_bearing_deg(own_vessel, candidate)
            sector_idx = self._get_sector_index(bearing)
            prev = nearest_by_sector.get(sector_idx)
            if prev is None or distance < prev[0]:
                nearest_by_sector[sector_idx] = (distance, candidate, bearing)

        features: List[float] = []
        for sector_idx in range(9):
            if sector_idx not in nearest_by_sector:
                features.extend([0.0] * 10)
            else:
                distance, target_vessel, bearing = nearest_by_sector[sector_idx]
                features.extend(self._build_sector_features(own_vessel, target_vessel, distance, bearing))
        return features

    def _build_obs(self, own_vessel: Vessel) -> np.ndarray:
        sector_features = self._build_radar_observation(own_vessel)
        own_speed_norm = own_vessel.speed / max(1e-6, self.envp.max_speed)
        goal_dx = own_vessel.goal_x - own_vessel.x
        goal_dy = own_vessel.goal_y - own_vessel.y
        goal_distance_norm = clamp(math.hypot(goal_dx, goal_dy) / max(1e-6, self.envp.sensor_range), 0.0, 1.0)
        goal_bearing = wrap_pi(math.atan2(goal_dy, goal_dx) - own_vessel.h)

        own_features = [
            own_speed_norm,
            goal_distance_norm,
            math.sin(goal_bearing),
            math.cos(goal_bearing),
            clamp(own_vessel.rudder, -1.0, 1.0),
            clamp(own_vessel.throttle, -1.0, 1.0),
        ]
        return np.asarray(sector_features + own_features, dtype=np.float32)

    def get_obs_for_vessel(self, vessel_id: str) -> np.ndarray:
        if vessel_id not in self.vessels:
            raise ValueError(f"Unknown vessel_id: {vessel_id!r}")
        return self._build_obs(self.vessels[vessel_id])

    def get_obs(self) -> np.ndarray:
        return self.get_obs_for_vessel("vessel1")

    def get_rl_controlled_vessel_ids(self) -> List[str]:
        return [vid for vid in self.vessel_ids if self.rl_active_by_vessel[vid]]

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)

        self.vessels = {}

        goal_ang_1 = self.rng.uniform(0.0, 2.0 * math.pi)
        gx1, gy1 = self._point_on_big_circle(goal_ang_1)
        h1 = math.atan2(gy1 - self.start_y, gx1 - self.start_x)
        sp1 = self.rng.uniform(self.envp.min_speed, self.envp.max_speed)
        self.vessels["vessel1"] = Vessel(self.start_x, self.start_y, h1, sp1, gx1, gy1)

        for vid in self.vessel_ids[1:]:
            self.vessels[vid] = self._sample_ring_vessel()

        self.time = 0.0
        self.step_idx = 0
        self.max_steps = max(1, int(round(self.envp.episode_seconds / self.envp.dt)))

        self.reached_by_vessel = {vid: False for vid in self.vessel_ids}
        self.model_control_latched = {vid: False for vid in self.vessel_ids}
        self.rl_active_by_vessel = {vid: False for vid in self.vessel_ids}
        self.role_by_vessel = {vid: "none" for vid in self.vessel_ids}
        self.scenario_by_vessel = {vid: "safe" for vid in self.vessel_ids}
        self.control_source_by_vessel = {"vessel1": "straight", **{vid: "pure_pursuit" for vid in self.vessel_ids[1:]}}
        self.steps_taken_by_vessel = {vid: 0 for vid in self.vessel_ids}
        self.prev_rudder_sign_by_vessel = {vid: 0 for vid in self.vessel_ids}

        self.start_pos_by_vessel = {vid: (self.vessels[vid].x, self.vessels[vid].y) for vid in self.vessel_ids}
        self.start_speed_by_vessel = {vid: self.vessels[vid].speed for vid in self.vessel_ids}
        self.prev_goal_dist_by_vessel = {vid: self._goal_distance(self.vessels[vid]) for vid in self.vessel_ids}
        self.prev_heading_err_by_vessel = {vid: self._goal_heading_error(self.vessels[vid]) for vid in self.vessel_ids}

        self.risk_of_collision = False
        self.last_dcpa = float("inf")
        self.last_tcpa = float("inf")
        self.colregs_scenario = "safe"
        self.any_rl_ever_triggered = False
        self.encounter_was_risky = False
        self.safe_pass_awarded = False
        self.reset_has_takeover_path = True

        self._sync_compat_attrs()
        return self.get_obs()

    def step(
        self,
        action: Union[np.ndarray, Tuple[float, float], list, Dict[str, Union[np.ndarray, Tuple[float, float], list]]],
    ) -> Tuple[np.ndarray, float, bool, Dict[str, float | str | int | dict]]:
        if self.paused:
            return self.get_obs(), 0.0, False, {"reason": "paused", "dcpa": self.last_dcpa, "tcpa": self.last_tcpa}

        actions_by_vessel = self._resolve_step_actions(action)
        pairwise = self._pairwise_assessment()

        self.role_by_vessel = dict(pairwise["roles"])
        self.scenario_by_vessel = dict(pairwise["scenarios"])
        self.risk_of_collision = bool(pairwise["risk"])
        self.last_dcpa = float(pairwise["dcpa"])
        self.last_tcpa = float(pairwise["tcpa"])
        self.colregs_scenario = "risk" if self.risk_of_collision else "safe"

        give_way_in_risk = {
            vid
            for vid in self.vessel_ids
            if self.role_by_vessel[vid] == "give_way" and (not self.reached_by_vessel[vid]) and self.risk_of_collision
        }

        for vid in give_way_in_risk:
            self.model_control_latched[vid] = True
        for vid in self.vessel_ids:
            if self.reached_by_vessel[vid]:
                self.model_control_latched[vid] = False
            self.rl_active_by_vessel[vid] = self.model_control_latched[vid] and (not self.reached_by_vessel[vid])

        self.any_rl_ever_triggered = self.any_rl_ever_triggered or any(self.rl_active_by_vessel.values())

        h = self.envp.dt / max(1, self.envp.substeps)
        for _ in range(max(1, self.envp.substeps)):
            frozen_rl_set = {vid for vid in self.vessel_ids if self.rl_active_by_vessel[vid]}
            frozen_actions = {vid: actions_by_vessel.get(vid) for vid in frozen_rl_set}
            for vid in self.vessel_ids:
                if vid in frozen_rl_set and frozen_actions[vid] is not None:
                    self._advance_controlled(vid, float(frozen_actions[vid][0]), float(frozen_actions[vid][1]), h)
                else:
                    self._advance_scripted(vid, h)

        for vid in self.vessel_ids:
            if not self.reached_by_vessel[vid]:
                self.steps_taken_by_vessel[vid] += 1

        min_pair_distance = float("inf")
        collision = False
        near_miss = False
        for i, a_id in enumerate(self.vessel_ids):
            for b_id in self.vessel_ids[i + 1 :]:
                a, b = self.vessels[a_id], self.vessels[b_id]
                d = math.hypot(b.x - a.x, b.y - a.y)
                min_pair_distance = min(min_pair_distance, d)
                collision = collision or (d <= self.envp.collision_distance)
                near_miss = near_miss or (d <= self.envp.near_miss_distance)

        self.time += self.envp.dt
        self.step_idx += 1

        done = False
        reason = ""
        if collision:
            done, reason = True, "collision"
        elif self.step_idx >= self.max_steps:
            done, reason = True, "timeout"
        elif all(self.reached_by_vessel.values()):
            done, reason = True, "all_reached"

        shared_reward = self.rewp.living_penalty
        if collision:
            shared_reward += self.rewp.collision_penalty
        if near_miss and not collision:
            shared_reward += self.rewp.near_miss_penalty
        if min_pair_distance < self.envp.safe_pass_distance:
            shared_reward -= self.rewp.unsafe_proximity_penalty_weight * (self.envp.safe_pass_distance - min_pair_distance)

        if self.risk_of_collision:
            self.encounter_was_risky = True
        if self.encounter_was_risky and (not self.risk_of_collision) and (min_pair_distance > self.envp.safe_pass_distance) and (not self.safe_pass_awarded):
            shared_reward += self.rewp.safe_pass_bonus
            self.safe_pass_awarded = True

        reward_by_vessel: Dict[str, float] = {}
        for vid in self.vessel_ids:
            v = self.vessels[vid]
            d_now = self._goal_distance(v)
            h_now = self._goal_heading_error(v)
            local = self._compute_progress_reward_for_vessel(
                self.prev_goal_dist_by_vessel[vid],
                d_now,
                self.prev_heading_err_by_vessel[vid],
                h_now,
            )
            if self.reached_by_vessel[vid]:
                local += self.rewp.goal_bonus

            local += self._scenario_local_shaping(
                self.scenario_by_vessel[vid],
                self.rl_active_by_vessel[vid],
                v.rudder,
                self.last_tcpa,
                self.last_dcpa,
            )

            rudder_sign = 1 if v.rudder > 1e-3 else -1 if v.rudder < -1e-3 else 0
            prev_sign = self.prev_rudder_sign_by_vessel[vid]
            if prev_sign != 0 and rudder_sign != 0 and rudder_sign != prev_sign:
                local -= self.rewp.oscillation_penalty_weight
            self.prev_rudder_sign_by_vessel[vid] = rudder_sign

            reward_by_vessel[vid] = local + shared_reward
            self.prev_goal_dist_by_vessel[vid] = d_now
            self.prev_heading_err_by_vessel[vid] = h_now

        total_reward = sum(reward_by_vessel.values()) - (len(self.vessel_ids) - 1) * shared_reward

        info: Dict[str, float | str | int | dict] = {
            "reason": reason,
            "dcpa": self.last_dcpa,
            "tcpa": self.last_tcpa,
            "risk_of_collision": int(self.risk_of_collision),
            "colregs_scenario": self.colregs_scenario,
            "reward_by_vessel": dict(reward_by_vessel),
            "rl_active_by_vessel": {vid: int(self.rl_active_by_vessel[vid]) for vid in self.vessel_ids},
            "role_by_vessel": dict(self.role_by_vessel),
            "controlled_vessel_ids": self.get_rl_controlled_vessel_ids(),
            "min_pair_distance": min_pair_distance,
        }

        for vid in self.vessel_ids:
            info[f"{vid}_reached"] = int(self.reached_by_vessel[vid])
            info[f"{vid}_rl_active"] = int(self.rl_active_by_vessel[vid])
            info[f"{vid}_model_control_latched"] = int(self.model_control_latched[vid])
            info[f"{vid}_role"] = self.role_by_vessel[vid]
            info[f"{vid}_distance_from_start"] = float(
                math.hypot(
                    self.vessels[vid].x - self.start_pos_by_vessel[vid][0],
                    self.vessels[vid].y - self.start_pos_by_vessel[vid][1],
                )
            )
            info[f"reward_{vid}"] = float(reward_by_vessel[vid])

        info["reward_v1"] = float(reward_by_vessel.get("vessel1", 0.0))
        info["reward_v2"] = float(reward_by_vessel.get("vessel2", 0.0))

        self._sync_compat_attrs()
        return self.get_obs(), float(total_reward), done, info

    def render(self) -> None:
        if not self.render_enabled or self._screen is None:
            return
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.close()
                return
        self._screen.fill((15, 20, 35))

        for idx, vid in enumerate(self.vessel_ids):
            v = self.vessels[vid]
            color = (100 + (idx * 40) % 155, 120, 220 - (idx * 30) % 120)
            pygame.draw.circle(self._screen, color, (self.sx(v.x), self.sy(v.y)), 5)
            pygame.draw.circle(self._screen, (240, 240, 120), (self.sx(v.goal_x), self.sy(v.goal_y)), 3, 1)
            label = self._font.render(f"{vid} {'RL' if self.rl_active_by_vessel[vid] else 'S'}", True, (255, 255, 255))
            self._screen.blit(label, (self.sx(v.x) + 6, self.sy(v.y) + 4))

        hud = self._font.render(
            f"step={self.step_idx} risk={self.risk_of_collision} dcpa={self.last_dcpa:.1f} tcpa={self.last_tcpa:.1f}",
            True,
            (220, 220, 220),
        )
        self._screen.blit(hud, (8, 8))
        pygame.display.flip()
        self._clock.tick(self.envp.render_fps)
