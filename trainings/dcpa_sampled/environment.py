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


class SingleVessel2FeatureEnv:
    """Two-vessel environment with vessel-centric radar observations (96-dim per vessel)."""

    def __init__(
        self,
        env_params: EnvParams = EnvParams(),
        reward_params: RewardParams = RewardParams(),
        render: bool = False,
    ):
        self.envp = env_params
        self.rewp = reward_params
        self.rng = random.Random(self.envp.seed)

        self.vessel1: Optional[Vessel] = None
        self.vessel2: Optional[Vessel] = None
        self.extra_vessels: Dict[str, Vessel] = {}
        self.extra_vessel_reached: Dict[str, bool] = {}
        self.extra_prev_goal_d: Dict[str, float] = {}
        self.extra_prev_goal_heading_err: Dict[str, float] = {}
        self.extra_steps_taken: Dict[str, int] = {}
        self.extra_vessel_start_pos: Dict[str, Tuple[float, float]] = {}
        self.extra_vessel_start_heading: Dict[str, float] = {}
        self.extra_vessel_start_speed: Dict[str, float] = {}
        self.extra_vessel_rl_active: Dict[str, bool] = {}
        self.extra_model_control_latched: Dict[str, bool] = {}
        self.extra_control_source: Dict[str, str] = {}
        self.start_x = 0.5 * self.envp.world_w
        self.start_y = 0.5 * self.envp.world_h
        self.time = 0.0
        self.step_idx = 0
        self.max_steps = max(1, int(round(self.envp.episode_seconds / self.envp.dt)))
        self.prev_goal_d_vessel1 = 0.0
        self.prev_goal_d_vessel2 = 0.0
        self.prev_goal_heading_err_vessel1 = 0.0
        self.prev_goal_heading_err_vessel2 = 0.0
        self.vessel1_reached = False
        self.vessel2_reached = False

        # per-vessel telemetry
        self.vessel1_steps_taken = 0
        self.vessel2_steps_taken = 0
        self.colregs_scenario = "safe"
        self.vessel1_role = "none"
        self.vessel2_role = "none"
        self.risk_of_collision = False
        self.last_dcpa = float("inf")
        self.last_tcpa = float("inf")
        self.vessel1_rl_active = False
        self.vessel2_rl_active = False
        self.vessel1_relative_bearing_deg = 0.0
        self.vessel2_relative_bearing_deg = 0.0
        self.vessel1_start_speed = 0.0
        self.vessel2_start_speed = 0.0
        self.vessel1_start_pos = (0.0, 0.0)
        self.vessel2_start_pos = (0.0, 0.0)
        self.vessel1_start_heading = 0.0
        self.vessel2_start_heading = 0.0

        # Vessel-2 scripted path state

        # render-time planned path visualization
        self.show_planned_paths = True
        self.vessel1_planned_path: List[Tuple[float, float]] = []
        self.vessel2_planned_path: List[Tuple[float, float]] = []
        self.extra_vessel_planned_paths: Dict[str, List[Tuple[float, float]]] = {}

        self.render_enabled = render and HAS_PYGAME
        self.paused = False
        self.risk_overlay_active = False
        self.risk_overlay_payload: Dict[str, float | str | int] = {}
        self.manual_sector_overlay_enabled = False
        self.risk_sector_overlay_active = False
        self.rl_ever_triggered: bool = False  # latches True when RL first activates, never resets within episode
        self.rl_overlay_shown: bool = False  # True after overlay has been shown once this episode
        self.prev_vessel1_rl_active = False
        self.prev_vessel2_rl_active = False
        self.overtaking_latched = False
        self.latched_scenario = "safe"
        self.latched_vessel1_role = "none"
        self.latched_vessel2_role = "none"
        self.overtaking_clear_steps = 0
        self.encounter_latched = False
        self.latched_encounter_active = False
        self.latched_geometry = "none"
        self.encounter_clear_steps = 0
        self.designated_vessel1_role = "none"
        self.designated_vessel2_role = "none"
        self.rl_controlled_vessel = "none"
        self.candidate_scenario = "safe"
        self.candidate_vessel1_role = "none"
        self.candidate_vessel2_role = "none"
        self.candidate_steps = 0
        self.active_non_overtaking_scenario = "safe"
        self.active_non_overtaking_vessel1_role = "none"
        self.active_non_overtaking_vessel2_role = "none"
        self.active_non_overtaking_exit_steps = 0
        self.geometry_scenario = "none"
        self.hud_scenario = "none"
        # Deprecated: stand-on escalation-to-control has been removed.
        self.vessel1_control_source = "straight"
        self.vessel2_control_source = "pure_pursuit"
        # Persistent per-vessel model-control latches.
        # A latch may be set only when the vessel is currently designated give-way.
        # Once set, control remains with the model until that vessel reaches its goal
        # (or the episode ends and reset() clears state).
        self.vessel1_model_control_latched = False
        self.vessel2_model_control_latched = False
        self.any_rl_ever_triggered = False
        self.locked = False
        self.locked_scenario = "safe"
        self.locked_role_v1 = "none"
        self.locked_role_v2 = "none"
        self.lock_candidate_steps = 0
        self.latched_encounter_active = False
        self.latched_geometry = "none"
        self.encounter_clear_steps = 0
        self.designated_vessel1_role = "none"
        self.designated_vessel2_role = "none"
        self.rl_controlled_vessel = "none"
        self.secondary_policy_fn = None
        self.last_inter_vessel_distance = float("inf")
        self.encounter_was_risky = False
        self.safe_pass_awarded = False
        self.vessel1_giveway_action_awarded = False
        self.vessel2_giveway_action_awarded = False
        self.prev_vessel1_rudder_sign = 0
        self.prev_vessel2_rudder_sign = 0
        self.candidate_scenario = "safe"
        self.candidate_vessel1_role = "none"
        self.candidate_vessel2_role = "none"
        self.candidate_steps = 0
        self.active_non_overtaking_scenario = "safe"
        self.active_non_overtaking_vessel1_role = "none"
        self.active_non_overtaking_vessel2_role = "none"
        self.active_non_overtaking_exit_steps = 0
        self._screen = None
        self._clock = None
        self._font = None
        self._validate_radar_bearing_convention()
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

    def set_secondary_policy(self, fn) -> None:
        # Compatibility no-op: only externally supplied per-vessel actions are used.
        self.secondary_policy_fn = None

    def sx(self, x: float) -> int:
        return int(round(x * self.envp.pixels_per_meter))

    def sy(self, y: float) -> int:
        return int(round(y * self.envp.pixels_per_meter))

    def _outside(self, v: Vessel) -> bool:
        return not (0.0 <= v.x <= self.envp.world_w and 0.0 <= v.y <= self.envp.world_h)

    def _goal_distance(self, v: Vessel) -> float:
        return math.hypot(v.goal_x - v.x, v.goal_y - v.y)

    def _goal_heading_error(self, v: Vessel) -> float:
        """Absolute wrapped heading error to own goal in [0, pi]."""
        goal_bearing = math.atan2(v.goal_y - v.y, v.goal_x - v.x)
        return abs(wrap_pi(goal_bearing - v.h))

    def _compute_progress_reward_for_vessel(
        self,
        prev_dist: float,
        curr_dist: float,
        prev_heading_err: float,
        curr_heading_err: float,
    ) -> float:
        # Tolerances to avoid floating-point jitter around neutral transitions.
        dist_eps = 1e-4
        heading_eps = 1e-4
        # Keep heading shaping smaller than distance-base term.
        heading_shaping = 0.20 * self.rewp.progress_weight

        dist_delta = prev_dist - curr_dist
        heading_delta = prev_heading_err - curr_heading_err

        base_term = self.rewp.progress_weight * dist_delta

        if heading_delta >= -heading_eps:
            heading_term = heading_shaping
        else:
            heading_term = -heading_shaping

        if dist_delta > dist_eps:
            return base_term + heading_term
        if dist_delta < -dist_eps:
            return base_term + heading_term

        if heading_delta > heading_eps:
            return heading_term
        if heading_delta < -heading_eps:
            return heading_term
        return 0.0

    def _distance_from_start(self, v: Vessel, start_xy: Tuple[float, float]) -> float:
        sx, sy = start_xy
        return math.hypot(v.x - sx, v.y - sy)

    def _inter_vessel_distance(self) -> float:
        return math.hypot(self.vessel2.x - self.vessel1.x, self.vessel2.y - self.vessel1.y)

    def _min_pair_distance(self) -> float:
        vessels = [(vessel_id, vessel) for vessel_id, vessel in self.get_vessel_map().items() if vessel is not None]
        if len(vessels) < 2:
            return float("inf")
        min_dist = float("inf")
        for i in range(len(vessels)):
            for j in range(i + 1, len(vessels)):
                _, va = vessels[i]
                _, vb = vessels[j]
                min_dist = min(min_dist, math.hypot(vb.x - va.x, vb.y - va.y))
        return min_dist

    def _update_extra_vessel_control_latches(self) -> None:
        vessel_map = {k: v for k, v in self.get_vessel_map().items() if v is not None}
        give_way_in_risk: Dict[str, bool] = {vessel_id: False for vessel_id in vessel_map}
        any_risk = False
        for i, (id_a, va) in enumerate(vessel_map.items()):
            for id_b, vb in list(vessel_map.items())[i + 1:]:
                risk, _, _ = self.assess_risk(va, vb)
                if not risk:
                    continue
                any_risk = True
                scenario, rb_a, rb_b = self.classify_geometry(va, vb)
                role_a, role_b = self.assign_roles(scenario, rb_a, rb_b)
                if role_a == "give_way":
                    give_way_in_risk[id_a] = True
                if role_b == "give_way":
                    give_way_in_risk[id_b] = True

        for vessel_id in self.extra_vessels:
            if give_way_in_risk.get(vessel_id, False) and (not self.extra_vessel_reached[vessel_id]):
                self.extra_model_control_latched[vessel_id] = True
            if self.extra_vessel_reached[vessel_id]:
                self.extra_model_control_latched[vessel_id] = False

        self.risk_of_collision = bool(self.risk_of_collision or any_risk)

    def get_vessel_ids(self) -> List[str]:
        """Return stable IDs for the current world."""
        ids = ["vessel1", "vessel2"]
        ids.extend(sorted(self.extra_vessels.keys(), key=lambda x: int(x.replace("vessel", ""))))
        return ids

    def get_vessel_map(self) -> Dict[str, Optional[Vessel]]:
        """Return the current vessel objects keyed by stable vessel ID."""
        vessel_map: Dict[str, Optional[Vessel]] = {
            "vessel1": self.vessel1,
            "vessel2": self.vessel2,
        }
        vessel_map.update(self.extra_vessels)
        return vessel_map

    def _get_vessel_map(self) -> Dict[str, Optional[Vessel]]:
        # Backward-compatible internal alias; not a new source of truth.
        return self.get_vessel_map()

    def get_vessel_by_id(self, vessel_id: str) -> Optional[Vessel]:
        vessel_map = self.get_vessel_map()
        if vessel_id not in vessel_map:
            raise KeyError(f"Unknown vessel_id: {vessel_id}")
        return vessel_map[vessel_id]

    def get_other_vessel_ids(self, vessel_id: str) -> List[str]:
        if vessel_id not in self.get_vessel_ids():
            raise KeyError(f"Unknown vessel_id: {vessel_id}")
        return [other_id for other_id in self.get_vessel_ids() if other_id != vessel_id]

    def get_other_vessels(self, vessel_id: str) -> List[Optional[Vessel]]:
        return [self.get_vessel_by_id(other_id) for other_id in self.get_other_vessel_ids(vessel_id)]

    def is_vessel_rl_active(self, vessel_id: str) -> bool:
        if vessel_id == "vessel1":
            return self.vessel1_rl_active
        if vessel_id == "vessel2":
            return self.vessel2_rl_active
        if vessel_id in self.extra_vessel_rl_active:
            return self.extra_vessel_rl_active[vessel_id]
        raise KeyError(f"Unknown vessel_id: {vessel_id}")

    def get_model_control_latched(self, vessel_id: str) -> bool:
        """Return current model-control latch state for the given vessel ID."""
        if vessel_id == "vessel1":
            return self.vessel1_model_control_latched
        if vessel_id == "vessel2":
            return self.vessel2_model_control_latched
        if vessel_id in self.extra_model_control_latched:
            return self.extra_model_control_latched[vessel_id]
        raise KeyError(f"Unknown vessel_id: {vessel_id}")

    def get_vessel_role(self, vessel_id: str) -> str:
        """Return current COLREGS role for the given vessel ID."""
        if vessel_id == "vessel1":
            return self.vessel1_role
        if vessel_id == "vessel2":
            return self.vessel2_role
        # Extra-vessel roles are computed pairwise in N-vessel mode and only used for control gating.
        return "none"

    def is_vessel_reached(self, vessel_id: str) -> bool:
        """Return whether the given vessel has reached its goal."""
        if vessel_id == "vessel1":
            return self.vessel1_reached
        if vessel_id == "vessel2":
            return self.vessel2_reached
        if vessel_id in self.extra_vessel_reached:
            return self.extra_vessel_reached[vessel_id]
        raise KeyError(f"Unknown vessel_id: {vessel_id}")

    def get_rl_controlled_vessel_ids(self) -> List[str]:
        """Return currently RL-controlled vessel IDs in deterministic order."""
        return [vessel_id for vessel_id in self.get_vessel_ids() if self.is_vessel_rl_active(vessel_id)]

    def _build_pairwise_interactions(self) -> List[Dict[str, float | str | int]]:
        vessel_map = {vessel_id: vessel for vessel_id, vessel in self.get_vessel_map().items() if vessel is not None}
        vessel_ids = list(vessel_map.keys())
        interactions: List[Dict[str, float | str | int]] = []
        for i, id_a in enumerate(vessel_ids):
            va = vessel_map[id_a]
            for id_b in vessel_ids[i + 1:]:
                vb = vessel_map[id_b]
                risk, tcpa, dcpa = self.assess_risk(va, vb)
                scenario, rb_a, rb_b = self.classify_geometry(va, vb)
                role_a, role_b = self.assign_roles(scenario, rb_a, rb_b)
                interactions.append(
                    {
                        "a": id_a,
                        "b": id_b,
                        "risk": int(risk),
                        "scenario": scenario,
                        "role_a": role_a,
                        "role_b": role_b,
                        "tcpa": float(tcpa),
                        "dcpa": float(dcpa),
                    }
                )
        return interactions

    def _build_vessel_status(self, pairwise_interactions: List[Dict[str, float | str | int]]) -> Dict[str, Dict[str, float | str | int]]:
        role_by_vessel: Dict[str, str] = {vessel_id: "none" for vessel_id in self.get_vessel_ids()}
        role_candidates: Dict[str, set[str]] = {vessel_id: set() for vessel_id in self.get_vessel_ids()}

        # Keep vessel1/vessel2 role reporting aligned with existing environment role fields.
        role_by_vessel["vessel1"] = self.vessel1_role
        role_by_vessel["vessel2"] = self.vessel2_role

        for entry in pairwise_interactions:
            if int(entry["risk"]) != 1:
                continue
            a = str(entry["a"])
            b = str(entry["b"])
            role_candidates[a].add(str(entry["role_a"]))
            role_candidates[b].add(str(entry["role_b"]))

        for vessel_id in self.extra_vessels:
            roles = role_candidates[vessel_id]
            if "give_way" in roles:
                role_by_vessel[vessel_id] = "give_way"
            elif "stand_on" in roles:
                role_by_vessel[vessel_id] = "stand_on"
            else:
                role_by_vessel[vessel_id] = "none"

        status: Dict[str, Dict[str, float | str | int]] = {}
        for vessel_id in self.get_vessel_ids():
            status[vessel_id] = {
                "role": role_by_vessel[vessel_id],
                "rl_active": int(self.is_vessel_rl_active(vessel_id)),
                "model_control_latched": int(self.get_model_control_latched(vessel_id)),
                "reached": int(self.is_vessel_reached(vessel_id)),
            }
        return status

    def _append_multi_vessel_debug_info(self, info: Dict[str, object]) -> None:
        pairwise_interactions = self._build_pairwise_interactions()
        vessel_status = self._build_vessel_status(pairwise_interactions)
        info["pairwise_interactions"] = pairwise_interactions
        info["vessel_status"] = vessel_status

    def _maybe_print_multi_vessel_debug(self, info: Dict[str, object]) -> None:
        if not self.envp.debug_multi_vessel_status:
            return
        pairwise_interactions = info.get("pairwise_interactions", [])
        vessel_status = info.get("vessel_status", {})
        risky_pairs = sum(1 for row in pairwise_interactions if int(row.get("risk", 0)) == 1)
        controlled = [vessel_id for vessel_id, row in vessel_status.items() if int(row.get("rl_active", 0)) == 1]
        role_latch = ", ".join(
            f"{vessel_id}:{row.get('role', 'none')}/L{int(row.get('model_control_latched', 0))}"
            for vessel_id, row in vessel_status.items()
        )
        print(
            "[MULTI_VESSEL_DEBUG] "
            f"step={self.step_idx} controlled={controlled} risky_pairs={risky_pairs} "
            f"roles_latches=[{role_latch}]"
        )

    def get_reward_by_vessel(self, reward_v1: float, reward_v2: float, extras: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Package current rewards into a stable vessel-id mapping."""
        rewards = {
            "vessel1": float(reward_v1),
            "vessel2": float(reward_v2),
        }
        if extras:
            for vessel_id, reward in extras.items():
                rewards[vessel_id] = float(reward)
        return rewards

    def _get_relative_bearing(self, observer: Vessel, target: Vessel) -> float:
        """Relative bearing in degrees [0, 360): 0=head-ahead, +CCW(port), starboard near 360."""
        dx = target.x - observer.x
        dy = target.y - observer.y
        global_bearing = math.atan2(dy, dx)
        rel_bearing_rad = wrap_pi(global_bearing - observer.h)
        return (math.degrees(rel_bearing_rad) + 360.0) % 360.0

    def _get_sector_index(self, relative_bearing_deg: float) -> int:
        """Map relative bearing to one of 9 radar sectors.

        Sector boundaries (deg): [350,10), [10,40), [40,75), [75,112.5), [112.5,180),
        [180,247.5), [247.5,285), [285,320), [320,350).
        """
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

    def _validate_radar_bearing_convention(self) -> None:
        """Lightweight self-check of radar bearing convention and sector wrap behavior."""
        eps = 1e-6
        observer = Vessel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ahead = Vessel(10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        port = Vessel(0.0, 10.0, 0.0, 0.0, 0.0, 0.0)
        starboard = Vessel(0.0, -10.0, 0.0, 0.0, 0.0, 0.0)
        assert abs(self._get_relative_bearing(observer, ahead) - 0.0) < eps
        assert abs(self._get_relative_bearing(observer, port) - 90.0) < eps
        assert abs(self._get_relative_bearing(observer, starboard) - 270.0) < eps
        assert self._get_sector_index(359.0) == 0
        assert self._get_sector_index(0.0) == 0
        assert self._get_sector_index(1.0) == 0

    def _build_sector_features(self, own_vessel: Vessel, target_vessel: Vessel, distance: float, relative_bearing_deg: float) -> List[float]:
        # distance_norm = clip(distance / sensor_range, 0, 1)
        distance_norm = clamp(distance / max(1e-6, self.envp.sensor_range), 0.0, 1.0)

        # bearing_rad = radians(relative_bearing_deg), then sin/cos encoding
        bearing_rad = math.radians(relative_bearing_deg)
        bearing_sin = math.sin(bearing_rad)
        bearing_cos = math.cos(bearing_rad)

        # relative_heading = wrap_angle(target_heading - own_heading), then sin/cos encoding
        relative_heading = wrap_pi(target_vessel.h - own_vessel.h)
        relative_heading_sin = math.sin(relative_heading)
        relative_heading_cos = math.cos(relative_heading)

        # target_speed_norm = clip(target_speed / max_speed, 0, 1)
        target_speed_norm = clamp(target_vessel.speed / max(1e-6, self.envp.max_speed), 0.0, 1.0)

        # closing_speed = relative_velocity_along_line_of_sight (positive means closing here)
        los_x = (target_vessel.x - own_vessel.x) / max(1e-6, distance)
        los_y = (target_vessel.y - own_vessel.y) / max(1e-6, distance)
        rvx = math.cos(target_vessel.h) * target_vessel.speed - math.cos(own_vessel.h) * own_vessel.speed
        rvy = math.sin(target_vessel.h) * target_vessel.speed - math.sin(own_vessel.h) * own_vessel.speed
        range_rate = rvx * los_x + rvy * los_y
        closing_speed = -range_rate
        # closing_speed_norm = tanh(closing_speed / max_speed), signed and bounded in [-1, 1]
        closing_speed_norm = math.tanh(closing_speed / max(1e-6, self.envp.max_speed))

        # Reuse existing TCPA/DCPA computation for pair.
        tcpa, dcpa = self._tcpa_dcpa(own_vessel, target_vessel)
        # tcpa_norm = 1 for non-future encounters (tcpa<=0) or non-finite values,
        # otherwise clip(tcpa / tcpa_horizon, 0, 1) with tcpa_horizon = tcpa_risk_threshold.
        tcpa_norm = 1.0
        if math.isfinite(tcpa) and tcpa > 0.0:
            tcpa_norm = clamp(tcpa / max(1e-6, self.envp.tcpa_risk_threshold), 0.0, 1.0)

        # dcpa_norm = clip(dcpa / dcpa_scale, 0, 1) where dcpa_scale = max(dcpa_risk_threshold, collision_distance).
        dcpa_scale = max(self.envp.dcpa_risk_threshold, self.envp.collision_distance)
        dcpa_norm = 1.0 if not math.isfinite(dcpa) else clamp(dcpa / max(1e-6, dcpa_scale), 0.0, 1.0)

        return [
            1.0,  # occupied_flag
            distance_norm,
            bearing_sin,
            bearing_cos,
            relative_heading_sin,
            relative_heading_cos,
            target_speed_norm,
            closing_speed_norm,
            tcpa_norm,
            dcpa_norm,
        ]

    def _build_radar_observation(self, own_vessel_id: str) -> List[float]:
        # Vessel-centric radar: 9 sectors, 10 features each.
        # Keep only the closest in-range contact per sector; empty sectors are zero-filled.
        if own_vessel_id not in self.get_vessel_ids():
            raise ValueError(f"Unknown vessel_id: {own_vessel_id!r}")
        own_vessel = self.get_vessel_by_id(own_vessel_id)
        if own_vessel is None:
            raise ValueError(f"Unknown vessel_id: {own_vessel_id!r}")

        nearest_by_sector: Dict[int, Tuple[float, Vessel, float]] = {}
        for candidate in self.get_other_vessels(own_vessel_id):
            if candidate is None:
                continue
            distance = math.hypot(candidate.x - own_vessel.x, candidate.y - own_vessel.y)
            if distance > self.envp.sensor_range:
                continue
            bearing = self._get_relative_bearing(own_vessel, candidate)
            sector_idx = self._get_sector_index(bearing)
            prev = nearest_by_sector.get(sector_idx)
            if prev is None or distance < prev[0]:
                nearest_by_sector[sector_idx] = (distance, candidate, bearing)

        features: List[float] = []
        for sector_idx in range(9):
            if sector_idx not in nearest_by_sector:
                # Empty sector rule: occupied_flag=0, all other entries=0.
                features.extend([0.0] * 10)
                continue
            distance, target_vessel, bearing = nearest_by_sector[sector_idx]
            features.extend(self._build_sector_features(own_vessel, target_vessel, distance, bearing))
        return features

    def _relative_bearing_deg(self, observer: Vessel, vessel2: Vessel) -> float:
        dx = vessel2.x - observer.x
        dy = vessel2.y - observer.y
        ch = math.cos(observer.h)
        sh = math.sin(observer.h)
        x_rel = ch * dx + sh * dy
        y_rel = -sh * dx + ch * dy
        rel_port = (math.degrees(math.atan2(y_rel, x_rel)) + 360.0) % 360.0
        return (360.0 - rel_port) % 360.0

    @staticmethod
    def _bearing_to_signed_deg(bearing_360: float) -> float:
        """Convert [0,360) relative bearing to signed bearing in [-180,180)."""
        return ((bearing_360 + 180.0) % 360.0) - 180.0

    def _bearing_in_sector(self, bearing_deg: float, start_deg: float, end_deg: float, inclusive: bool = True) -> bool:
        b = bearing_deg % 360.0
        s = start_deg % 360.0
        e = end_deg % 360.0
        if s <= e:
            if inclusive:
                return s <= b <= e
            return s < b < e
        if inclusive:
            return b >= s or b <= e
        return b > s or b < e

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

    def assess_risk(self, vessel1: Vessel, vessel2: Vessel) -> Tuple[bool, float, float]:
        """Pure risk gate from TCPA/DCPA thresholds.

        Returns:
            (risk_of_collision, tcpa, dcpa)
        """
        tcpa, dcpa = self._tcpa_dcpa(vessel1, vessel2)
        risk_of_collision = (0.0 <= tcpa <= self.envp.tcpa_risk_threshold) and (dcpa <= self.envp.dcpa_risk_threshold)
        return risk_of_collision, tcpa, dcpa

    def _apply_head_on_shaping(self, rudder: float, tcpa: float) -> float:
        # Current shaping is rudder-direction based.
        # Legacy timing-based shaping (e.g., early/late action) is deprecated/inactive.
        reward = 0.0
        if rudder > self.rewp.starboard_min_rudder:
            reward += 0.3
        elif rudder < self.rewp.port_max_rudder:
            reward -= 0.5
        return reward

    def _apply_crossing_shaping(self, rudder: float, tcpa: float, dcpa: float) -> float:
        # Current shaping is rudder-direction based.
        # Legacy crossing-ahead/timing shaping is deprecated/inactive.
        reward = 0.0
        if rudder > self.rewp.starboard_min_rudder:
            reward += 0.25
        elif rudder < self.rewp.port_max_rudder:
            reward -= 0.4

        return reward

    def _apply_overtaking_shaping(self, tcpa: float, dcpa: float) -> float:
        reward = 0.0
        if dcpa > self.rewp.safe_dcpa_threshold:
            reward += 0.2
        if dcpa < self.rewp.danger_dcpa_threshold:
            reward -= 0.3
        return reward

    def _scenario_local_shaping(
        self,
        *,
        scenario: str,
        rl_active: bool,
        rudder: float,
        tcpa: float,
        dcpa: float,
    ) -> float:
        if not rl_active:
            return 0.0
        if scenario == "head_on":
            return self._apply_head_on_shaping(rudder, tcpa)
        if scenario == "crossing":
            return self._apply_crossing_shaping(rudder, tcpa, dcpa)
        if scenario == "overtaking":
            return self._apply_overtaking_shaping(tcpa, dcpa)
        return 0.0

    def classify_geometry(self, vessel1: Vessel, vessel2: Vessel) -> Tuple[str, float, float]:
        """Classify encounter scenario from pure geometry.

        Returns:
            (scenario, rb_1, rb_2)
            - scenario in {"head_on", "overtaking", "crossing"}
            - rb_1: relative bearing of vessel2 seen from vessel1, [0, 360)
            - rb_2: relative bearing of vessel1 seen from vessel2, [0, 360)
        """
        rb_1 = self._relative_bearing_deg(vessel1, vessel2)
        rb_2 = self._relative_bearing_deg(vessel2, vessel1)

        head_on_half = self.envp.colregs_head_on_half_angle_deg
        head_on_min = (360.0 - head_on_half) % 360.0
        crossing_max = self.envp.colregs_crossing_starboard_max_deg
        overtaking_max = self.envp.colregs_overtaking_aft_max_deg

        head_on = self._bearing_in_sector(rb_1, head_on_min, head_on_half) and self._bearing_in_sector(
            rb_2, head_on_min, head_on_half
        )
        if head_on:
            return "head_on", rb_1, rb_2

        vessel1_sees_vessel2_in_aft = self._bearing_in_sector(rb_1, crossing_max, overtaking_max)
        vessel2_sees_vessel1_in_aft = self._bearing_in_sector(rb_2, crossing_max, overtaking_max)
        if vessel1_sees_vessel2_in_aft or vessel2_sees_vessel1_in_aft:
            return "overtaking", rb_1, rb_2

        return "crossing", rb_1, rb_2

    def _classify_pair_geometry(self, vessel1: Vessel, vessel2: Vessel) -> Dict[str, float | str]:
        scenario, own_bearing, tgt_bearing = self.classify_geometry(vessel1, vessel2)

        head_on_half = self.envp.colregs_head_on_half_angle_deg
        head_on_min = (360.0 - head_on_half) % 360.0
        crossing_max = self.envp.colregs_crossing_starboard_max_deg
        overtaking_max = self.envp.colregs_overtaking_aft_max_deg

        if scenario == "head_on":
            return {
                "geometry": "head_on_geom",
                "vessel1_bearing_deg": own_bearing,
                "vessel2_bearing_deg": tgt_bearing,
            }

        if scenario == "overtaking":
            if self._bearing_in_sector(own_bearing, crossing_max, overtaking_max):
                geometry = "overtaking_vessel2_geom"
            elif self._bearing_in_sector(tgt_bearing, crossing_max, overtaking_max):
                geometry = "overtaking_vessel1_geom"
            else:
                geometry = "overtaking_vessel1_geom"
            return {
                "geometry": geometry,
                "vessel1_bearing_deg": own_bearing,
                "vessel2_bearing_deg": tgt_bearing,
            }

        if self._bearing_in_sector(own_bearing, head_on_half, crossing_max):
            return {
                "geometry": "crossing_vessel1_stand_on_geom",
                "vessel1_bearing_deg": own_bearing,
                "vessel2_bearing_deg": tgt_bearing,
            }

        if self._bearing_in_sector(own_bearing, 360.0 - crossing_max, head_on_min):
            return {
                "geometry": "crossing_vessel1_give_way_geom",
                "vessel1_bearing_deg": own_bearing,
                "vessel2_bearing_deg": tgt_bearing,
            }

        return {
            "geometry": "crossing_vessel1_stand_on_geom",
            "vessel1_bearing_deg": own_bearing,
            "vessel2_bearing_deg": tgt_bearing,
        }

    def _assess_pair_risk(self, vessel1: Vessel, vessel2: Vessel) -> Dict[str, float | bool]:
        risk_of_collision, tcpa, dcpa = self.assess_risk(vessel1, vessel2)
        return {
            "tcpa": tcpa,
            "dcpa": dcpa,
            "risk_of_collision": risk_of_collision,
        }

    def assign_roles(self, scenario: str, rb_1: float, rb_2: float) -> Tuple[str, str]:
        """Pure COLREGS role assignment from scenario + relative bearings."""
        crossing_max = self.envp.colregs_crossing_starboard_max_deg
        overtaking_max = self.envp.colregs_overtaking_aft_max_deg

        if scenario == "head_on":
            return "give_way", "give_way"

        if scenario == "overtaking":
            vessel1_sees_vessel2_in_aft = self._bearing_in_sector(rb_1, crossing_max, overtaking_max)
            vessel2_sees_vessel1_in_aft = self._bearing_in_sector(rb_2, crossing_max, overtaking_max)
            if vessel1_sees_vessel2_in_aft and not vessel2_sees_vessel1_in_aft:
                return "stand_on", "give_way"
            if vessel2_sees_vessel1_in_aft and not vessel1_sees_vessel2_in_aft:
                return "give_way", "stand_on"
            # Degenerate symmetric aft-sector case: both maneuver conservatively.
            return "give_way", "give_way"

        # crossing: vessel that sees the other on its starboard side is give-way.
        # Bearing convention here maps signed negatives to starboard/right and positives to port/left.
        rb1_signed = self._bearing_to_signed_deg(rb_1)
        rb2_signed = self._bearing_to_signed_deg(rb_2)
        vessel1_starboard = (-crossing_max < rb1_signed) and (rb1_signed < 0.0)
        vessel2_starboard = (-crossing_max < rb2_signed) and (rb2_signed < 0.0)
        if vessel1_starboard and not vessel2_starboard:
            return "give_way", "stand_on"
        if vessel2_starboard and not vessel1_starboard:
            return "stand_on", "give_way"
        # Boundary/degenerate cases: default to vessel1 give-way for determinism.
        return "give_way", "stand_on"

    def _resolve_colregs_pair(self, vessel1: Vessel, vessel2: Vessel) -> Dict[str, float | str | bool]:
        geom = self._classify_pair_geometry(vessel1, vessel2)
        risk = self._assess_pair_risk(vessel1, vessel2)
        scenario_now, rb_1, rb_2 = self.classify_geometry(vessel1, vessel2)
        role1_now, role2_now = self.assign_roles(scenario_now, rb_1, rb_2)
        geometry = str(geom["geometry"])
        risk_of_collision = bool(risk["risk_of_collision"])

        if not self.locked:
            if risk_of_collision:
                self.lock_candidate_steps += 1
            else:
                self.lock_candidate_steps = 0

            if self.lock_candidate_steps >= max(1, int(self.envp.lock_enter_persistence_steps)):
                self.locked = True
                self.locked_scenario = scenario_now
                self.locked_role_v1 = role1_now
                self.locked_role_v2 = role2_now
                self.encounter_was_risky = True
                self.safe_pass_awarded = False
                self.latched_encounter_active = True
                self.encounter_latched = True
                self.latched_scenario = self.locked_scenario
                self.latched_vessel1_role = self.locked_role_v1
                self.latched_vessel2_role = self.locked_role_v2
                self.designated_vessel1_role = self.locked_role_v1
                self.designated_vessel2_role = self.locked_role_v2
                self.overtaking_latched = self.locked_scenario == "overtaking"

        if self.locked:
            scenario = self.locked_scenario
            vessel1_role = self.locked_role_v1
            vessel2_role = self.locked_role_v2
            encounter_latched = True
        else:
            scenario = scenario_now if risk_of_collision else "no_risk"
            vessel1_role = role1_now if risk_of_collision else "none"
            vessel2_role = role2_now if risk_of_collision else "none"
            encounter_latched = False

        return {
            "geometry": geometry,
            "scenario_now": scenario_now,
            "scenario": scenario,
            "vessel1_role": vessel1_role,
            "vessel2_role": vessel2_role,
            "vessel1_bearing_deg": float(geom["vessel1_bearing_deg"]),
            "vessel2_bearing_deg": float(geom["vessel2_bearing_deg"]),
            "tcpa": float(risk["tcpa"]),
            "dcpa": float(risk["dcpa"]),
            "risk_of_collision": risk_of_collision,
            "encounter_latched": encounter_latched,
            "overtaking_latched": int(self.locked and self.locked_scenario == "overtaking"),
        }

    def _classify_colregs(self) -> Dict[str, float | str | bool]:
        if self.vessel1_reached or self.vessel2_reached:
            fallback_vessel1_role = self.designated_vessel1_role if self.encounter_was_risky else "none"
            fallback_vessel2_role = self.designated_vessel2_role if self.encounter_was_risky else "none"
            if self.rl_controlled_vessel == "vessel1":
                fallback_vessel1_role = "give_way"
                fallback_vessel2_role = "stand_on" if fallback_vessel2_role == "none" else fallback_vessel2_role
            elif self.rl_controlled_vessel == "vessel2":
                fallback_vessel2_role = "give_way"
                fallback_vessel1_role = "stand_on" if fallback_vessel1_role == "none" else fallback_vessel1_role
            elif self.rl_controlled_vessel == "both":
                fallback_vessel1_role = "give_way"
                fallback_vessel2_role = "give_way"
            return {
                "geometry": "none",
                "scenario": "safe",
                "vessel1_role": self.locked_role_v1 if self.locked else fallback_vessel1_role,
                "vessel2_role": self.locked_role_v2 if self.locked else fallback_vessel2_role,
                "vessel1_bearing_deg": 0.0,
                "vessel2_bearing_deg": 0.0,
                "tcpa": float("inf"),
                "dcpa": float("inf"),
                "risk_of_collision": False,
                "encounter_latched": self.encounter_latched,
                "overtaking_latched": self.overtaking_latched,
            }
        return self._resolve_colregs_pair(self.vessel1, self.vessel2)

    def _point_on_big_circle(self, ang: float) -> Tuple[float, float]:
        r = self.envp.vessel2_outer_radius
        return self.start_x + r * math.cos(ang), self.start_y + r * math.sin(ang)

    def _inward_facing_heading(self, pos_x: float, pos_y: float) -> float:
        """Sample a heading that points into the circle interior."""
        cx, cy = self.start_x, self.start_y
        to_center_angle = math.atan2(cy - pos_y, cx - pos_x)
        offset = self.rng.uniform(-0.5 * math.pi, 0.5 * math.pi)
        return wrap_pi(to_center_angle + offset)

    def _pure_pursuit_rudder_cmd(self, v: Vessel, goal_x: float, goal_y: float) -> float:
        """Pure-pursuit rudder command that steers vessel directly toward goal position."""
        turning_radius = v.speed / max(self.envp.rudder_max_yaw_rate_rad_s, 1e-6)
        lookahead_dist = max(1e-6, self.envp.pp_lookahead_factor * turning_radius)

        bearing_to_goal = math.atan2(goal_y - v.y, goal_x - v.x)
        vessel2_x = v.x + lookahead_dist * math.cos(bearing_to_goal)
        vessel2_y = v.y + lookahead_dist * math.sin(bearing_to_goal)

        bearing = math.atan2(vessel2_y - v.y, vessel2_x - v.x)
        heading_error = wrap_pi(bearing - v.h)
        return clamp(
            heading_error / math.radians(self.envp.pp_heading_gain_deg),
            -1.0,
            1.0,
        )

    def _sample_vessel2_path(self) -> Vessel:
        # Vessel 2: random start/goal on big circle with inward-facing start heading.
        # Goal is sampled to remain beyond a minimum start->goal distance for local pure-pursuit behavior checks.
        start_ang_2 = self.rng.uniform(0.0, 2.0 * math.pi)
        goal_ang_2 = self.rng.uniform(0.0, 2.0 * math.pi)

        sx2, sy2 = self._point_on_big_circle(start_ang_2)
        gx2, gy2 = self._point_on_big_circle(goal_ang_2)

        sh2 = self._inward_facing_heading(sx2, sy2)
        sp2 = self.rng.uniform(self.envp.vessel2_min_speed, self.envp.vessel2_max_speed)

        tries = 0
        min_goal_chord_dist = max(0.0, float(self.envp.vessel2_min_goal_arc_distance_from_start))
        # Backward compatibility: allow legacy parameter names to override when explicitly provided.
        if self.envp.vessel2_max_goal_arc_distance_from_start is not None:
            min_goal_chord_dist = max(0.0, float(self.envp.vessel2_max_goal_arc_distance_from_start))
        if self.envp.vessel2_max_goal_distance_from_start is not None:
            min_goal_chord_dist = max(0.0, float(self.envp.vessel2_max_goal_distance_from_start))

        if bool(self.envp.adaptive_vessel2_min_goal_arc_from_speed):
            omega_max = max(1e-9, float(self.envp.rudder_max_yaw_rate_rad_s))
            dcrit_chord = max(0.0, float(self.envp.vessel2_min_goal_dcrit_factor)) * (2.0 * sp2 / omega_max)
            min_goal_chord_dist = max(min_goal_chord_dist, dcrit_chord)

        while math.hypot(gx2 - sx2, gy2 - sy2) < min_goal_chord_dist and tries < 40:
            goal_ang_2 = self.rng.uniform(0.0, 2.0 * math.pi)
            gx2, gy2 = self._point_on_big_circle(goal_ang_2)
            tries += 1

        return Vessel(sx2, sy2, sh2, sp2, gx2, gy2)

    def _sample_extra_vessel_path(self) -> Vessel:
        """
        Additive extra-vessel traffic sampling path.
        Defaults intentionally mirror vessel2's perimeter/path-following style.
        """
        spawn_mode = str(self.envp.extra_vessel_spawn_mode).lower()
        if spawn_mode != "perimeter":
            raise ValueError(
                f"Unsupported extra_vessel_spawn_mode={self.envp.extra_vessel_spawn_mode!r}; "
                "supported modes: 'perimeter'"
            )

        # Keep default behavior vessel2-like; only speed envelope is independently configurable.
        vessel = self._sample_vessel2_path()
        speed_lo = float(self.envp.extra_vessel_min_speed)
        speed_hi = float(self.envp.extra_vessel_max_speed)
        if speed_lo > speed_hi:
            speed_lo, speed_hi = speed_hi, speed_lo

        if (speed_lo != float(self.envp.vessel2_min_speed)) or (speed_hi != float(self.envp.vessel2_max_speed)):
            vessel.speed = self.rng.uniform(speed_lo, speed_hi)
        return vessel

    def _spawn_extra_vessels(self) -> Dict[str, Vessel]:
        """Spawn vessels 3..N via the additive extra-vessel extension path."""
        extra_vessels: Dict[str, Vessel] = {}
        if int(self.envp.num_vessels) <= 2:
            return extra_vessels
        for idx in range(3, int(self.envp.num_vessels) + 1):
            extra_vessels[f"vessel{idx}"] = self._sample_extra_vessel_path()
        return extra_vessels

    def _advance_target(self, dt: float) -> None:
        if self.vessel2 is None or self.vessel2_reached:
            return

        d_goal = self._goal_distance(self.vessel2)
        if d_goal <= self.envp.goal_radius:
            self.vessel2_reached = True
            self.vessel2.speed = 0.0
            return

        rudder_cmd = self._pure_pursuit_rudder_cmd(self.vessel2, self.vessel2.goal_x, self.vessel2.goal_y)

        self._integrate_rudder_heading(self.vessel2, rudder_cmd, dt)

        # hold constant scripted speed in nominal mode
        self.vessel2.speed = clamp(self.vessel2.speed, self.envp.vessel2_min_speed, self.envp.vessel2_max_speed)

        d_goal = self._goal_distance(self.vessel2)
        travel = min(self.vessel2.speed * dt, d_goal)
        self.vessel2.x += travel * math.cos(self.vessel2.h)
        self.vessel2.y += travel * math.sin(self.vessel2.h)

        if self._goal_distance(self.vessel2) <= self.envp.goal_radius:
            self.vessel2_reached = True
            self.vessel2.speed = 0.0

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

    def _get_reached_flag(self, reached_attr: str) -> bool:
        if reached_attr.startswith("extra_vessel_reached:"):
            vessel_id = reached_attr.split(":", 1)[1]
            return bool(self.extra_vessel_reached.get(vessel_id, False))
        return bool(getattr(self, reached_attr))

    def _set_reached_flag(self, reached_attr: str, value: bool) -> None:
        if reached_attr.startswith("extra_vessel_reached:"):
            vessel_id = reached_attr.split(":", 1)[1]
            self.extra_vessel_reached[vessel_id] = bool(value)
            return
        setattr(self, reached_attr, value)

    def _advance_straight(self, v: Vessel, reached_attr: str, dt: float) -> None:
        if self._get_reached_flag(reached_attr):
            return

        d = self._goal_distance(v)
        if d <= self.envp.goal_radius:
            self._set_reached_flag(reached_attr, True)
            v.speed = 0.0
            return

        travel = min(v.speed * dt, d)
        v.x += math.cos(v.h) * travel
        v.y += math.sin(v.h) * travel

        if self._goal_distance(v) <= self.envp.goal_radius:
            self._set_reached_flag(reached_attr, True)
            v.speed = 0.0

    def _advance_controlled(self, v: Vessel, reached_attr: str, rudder_cmd: float, throttle_cmd: float, dt: float) -> None:
        if self._get_reached_flag(reached_attr):
            return

        d = self._goal_distance(v)
        if d <= self.envp.goal_radius:
            self._set_reached_flag(reached_attr, True)
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

        if self._goal_distance(v) <= self.envp.goal_radius:
            self._set_reached_flag(reached_attr, True)
            v.speed = 0.0

    def _advance_extra_target(self, vessel_id: str, dt: float) -> None:
        vessel = self.extra_vessels[vessel_id]
        reached_attr = f"extra_vessel_reached:{vessel_id}"
        if self._get_reached_flag(reached_attr):
            return
        d_goal = self._goal_distance(vessel)
        if d_goal <= self.envp.goal_radius:
            self._set_reached_flag(reached_attr, True)
            vessel.speed = 0.0
            return

        rudder_cmd = self._pure_pursuit_rudder_cmd(vessel, vessel.goal_x, vessel.goal_y)
        self._integrate_rudder_heading(vessel, rudder_cmd, dt)
        vessel.speed = clamp(vessel.speed, self.envp.vessel2_min_speed, self.envp.vessel2_max_speed)

        d_goal = self._goal_distance(vessel)
        travel = min(vessel.speed * dt, d_goal)
        vessel.x += travel * math.cos(vessel.h)
        vessel.y += travel * math.sin(vessel.h)

        if self._goal_distance(vessel) <= self.envp.goal_radius:
            self._set_reached_flag(reached_attr, True)
            vessel.speed = 0.0

    def _build_vessel1_planned_path(self) -> None:
        if self.vessel1 is None:
            self.vessel1_planned_path = []
            return
        self.vessel1_planned_path = [(self.start_x, self.start_y), (self.vessel1.goal_x, self.vessel1.goal_y)]

    def _generate_scripted_vessel_planned_path(
        self, sx: float, sy: float, sh: float, speed: float, goal_x: float, goal_y: float
    ) -> List[Tuple[float, float]]:
        sim = Vessel(sx, sy, sh, speed, goal_x, goal_y, rudder=0.0, throttle=0.0)
        pts: List[Tuple[float, float]] = [(sim.x, sim.y)]
        dt = self.envp.dt / max(1, self.envp.substeps)
        max_sim_steps = max(2000, int(2.0 * self.max_steps * max(1, self.envp.substeps)))

        for _ in range(max_sim_steps):
            d_goal = math.hypot(goal_x - sim.x, goal_y - sim.y)
            if d_goal <= self.envp.goal_radius:
                break

            rudder_cmd = self._pure_pursuit_rudder_cmd(sim, goal_x, goal_y)
            self._integrate_rudder_heading(sim, rudder_cmd, dt)

            travel = min(sim.speed * dt, d_goal)
            sim.x += travel * math.cos(sim.h)
            sim.y += travel * math.sin(sim.h)
            pts.append((sim.x, sim.y))

            if travel + 1e-9 >= d_goal:
                break

        return pts

    def _build_vessel2_planned_path(self, sx: float, sy: float, sh: float, speed: float, goal_x: float, goal_y: float) -> None:
        self.vessel2_planned_path = self._generate_scripted_vessel_planned_path(sx, sy, sh, speed, goal_x, goal_y)

    def _build_extra_vessel_planned_paths(self) -> None:
        self.extra_vessel_planned_paths = {}
        for vessel_id, vessel in self.extra_vessels.items():
            self.extra_vessel_planned_paths[vessel_id] = self._generate_scripted_vessel_planned_path(
                vessel.x, vessel.y, vessel.h, vessel.speed, vessel.goal_x, vessel.goal_y
            )

    def _build_obs(self, own_vessel: Vessel, own_vessel_id: str) -> np.ndarray:
        # Observation layout: radar(9 sectors × 10 features = 90) + own-vessel features(6) = 96.
        sector_features = self._build_radar_observation(own_vessel_id)

        # own_speed_norm = own_speed / max_speed
        own_speed_norm = own_vessel.speed / max(1e-6, self.envp.max_speed)

        goal_dx = own_vessel.goal_x - own_vessel.x
        goal_dy = own_vessel.goal_y - own_vessel.y
        goal_distance = math.hypot(goal_dx, goal_dy)
        # goal_distance_norm = clip(goal_distance / sensor_range, 0, 1)
        goal_distance_norm = clamp(goal_distance / max(1e-6, self.envp.sensor_range), 0.0, 1.0)

        goal_bearing = wrap_pi(math.atan2(goal_dy, goal_dx) - own_vessel.h)
        goal_bearing_sin = math.sin(goal_bearing)
        goal_bearing_cos = math.cos(goal_bearing)

        own_features = [
            own_speed_norm,
            goal_distance_norm,
            goal_bearing_sin,
            goal_bearing_cos,
            clamp(own_vessel.rudder, -1.0, 1.0),   # own_rudder_norm
            clamp(own_vessel.throttle, -1.0, 1.0),  # own_throttle_norm
        ]
        obs = np.asarray(sector_features + own_features, dtype=np.float32)
        assert obs.shape[0] == 96
        return obs

    def get_obs_for_vessel(self, vessel_id: str) -> np.ndarray:
        if vessel_id not in self.get_vessel_ids():
            raise ValueError(f"Unknown vessel_id: {vessel_id!r}")
        own_vessel = self.get_vessel_map().get(vessel_id)
        if own_vessel is not None:
            return self._build_obs(own_vessel, vessel_id)
        raise ValueError(f"Unknown vessel_id: {vessel_id!r}")

    def get_obs(self) -> np.ndarray:
        # Compatibility path for callers that still expect a single observation tensor.
        return self.get_obs_for_vessel("vessel1")

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)

        goal_ang_1 = self.rng.uniform(0.0, 2.0 * math.pi)
        agx, agy = self._point_on_big_circle(goal_ang_1)
        ah = math.atan2(agy - self.start_y, agx - self.start_x)
        aspeed = self.rng.uniform(self.envp.min_speed, self.envp.max_speed)
        sampled_vessel1 = Vessel(self.start_x, self.start_y, ah, aspeed, agx, agy)
        sampled_vessel2 = self._sample_vessel2_path()

        # Episode acceptance is handled by the training-side scripted screening
        # pipeline in train.py; reset() intentionally avoids hidden viability
        # filtering.

        self.vessel1 = sampled_vessel1
        self.vessel2 = sampled_vessel2
        # Foundational pair (vessel1/vessel2) stays on dedicated setup above.
        # Extra traffic (vessel3+) is an explicit additive extension path.
        self.extra_vessels = self._spawn_extra_vessels()

        sx2, sy2 = self.vessel2.x, self.vessel2.y
        sh2 = self.vessel2.h
        sp2 = self.vessel2.speed
        gx2, gy2 = self.vessel2.goal_x, self.vessel2.goal_y

        self.vessel1_start_speed = self.vessel1.speed
        self.vessel1_start_pos = (self.vessel1.x, self.vessel1.y)
        self.vessel1_start_heading = self.vessel1.h
        self.vessel2_start_speed = sp2
        self.vessel2_start_pos = (self.vessel2.x, self.vessel2.y)
        self.vessel2_start_heading = self.vessel2.h

        self.time = 0.0
        self.step_idx = 0
        self.vessel1_reached = False
        self.vessel2_reached = False

        self.prev_goal_d_vessel1 = self._goal_distance(self.vessel1)
        self.prev_goal_d_vessel2 = self._goal_distance(self.vessel2)
        self.prev_goal_heading_err_vessel1 = self._goal_heading_error(self.vessel1)
        self.prev_goal_heading_err_vessel2 = self._goal_heading_error(self.vessel2)
        self.vessel1_steps_taken = 0
        self.vessel2_steps_taken = 0
        self.colregs_scenario = "safe"
        self.vessel1_role = "none"
        self.vessel2_role = "none"
        self.risk_of_collision = False
        self.last_dcpa = float("inf")
        self.last_tcpa = float("inf")
        self.vessel1_rl_active = False
        self.vessel2_rl_active = False
        self.vessel1_relative_bearing_deg = 0.0
        self.vessel2_relative_bearing_deg = 0.0
        self.paused = False
        self.risk_overlay_active = False
        self.risk_overlay_payload = {}
        self.manual_sector_overlay_enabled = False
        self.risk_sector_overlay_active = False
        self.rl_ever_triggered = False
        self.rl_overlay_shown = False
        self.prev_vessel1_rl_active = False
        self.prev_vessel2_rl_active = False
        self.overtaking_latched = False
        self.latched_scenario = "safe"
        self.latched_vessel1_role = "none"
        self.latched_vessel2_role = "none"
        self.overtaking_clear_steps = 0
        self.encounter_latched = False
        self.geometry_scenario = "none"
        self.hud_scenario = "none"
        # Deprecated: stand-on escalation-to-control has been removed.
        self.vessel1_control_source = "straight"
        self.vessel2_control_source = "pure_pursuit"
        self.vessel1_model_control_latched = False
        self.vessel2_model_control_latched = False
        self.extra_vessel_reached = {vessel_id: False for vessel_id in self.extra_vessels}
        self.extra_prev_goal_d = {vessel_id: self._goal_distance(v) for vessel_id, v in self.extra_vessels.items()}
        self.extra_prev_goal_heading_err = {vessel_id: self._goal_heading_error(v) for vessel_id, v in self.extra_vessels.items()}
        self.extra_steps_taken = {vessel_id: 0 for vessel_id in self.extra_vessels}
        self.extra_vessel_start_pos = {vessel_id: (v.x, v.y) for vessel_id, v in self.extra_vessels.items()}
        self.extra_vessel_start_heading = {vessel_id: v.h for vessel_id, v in self.extra_vessels.items()}
        self.extra_vessel_start_speed = {vessel_id: v.speed for vessel_id, v in self.extra_vessels.items()}
        self.extra_vessel_rl_active = {vessel_id: False for vessel_id in self.extra_vessels}
        self.extra_model_control_latched = {vessel_id: False for vessel_id in self.extra_vessels}
        self.extra_control_source = {vessel_id: "pure_pursuit" for vessel_id in self.extra_vessels}
        self.any_rl_ever_triggered = False
        self.locked = False
        self.locked_scenario = "safe"
        self.locked_role_v1 = "none"
        self.locked_role_v2 = "none"
        self.lock_candidate_steps = 0
        self.latched_encounter_active = False
        self.latched_geometry = "none"
        self.encounter_clear_steps = 0
        self.designated_vessel1_role = "none"
        self.designated_vessel2_role = "none"
        self.rl_controlled_vessel = "none"
        self.last_inter_vessel_distance = float("inf")
        self.encounter_was_risky = False
        self.safe_pass_awarded = False
        self.vessel1_giveway_action_awarded = False
        self.vessel2_giveway_action_awarded = False
        self.prev_vessel1_rudder_sign = 0
        self.prev_vessel2_rudder_sign = 0
        self.candidate_scenario = "safe"
        self.candidate_vessel1_role = "none"
        self.candidate_vessel2_role = "none"
        self.candidate_steps = 0
        self.active_non_overtaking_scenario = "safe"
        self.active_non_overtaking_vessel1_role = "none"
        self.active_non_overtaking_vessel2_role = "none"
        self.active_non_overtaking_exit_steps = 0

        # Episode termination is collision, fixed-time timeout, or both vessels reaching goals.
        self.max_steps = max(1, int(round(self.envp.episode_seconds / self.envp.dt)))

        self._build_vessel1_planned_path()
        self._build_vessel2_planned_path(sx2, sy2, sh2, sp2, gx2, gy2)
        self._build_extra_vessel_planned_paths()
        self.last_inter_vessel_distance = self._inter_vessel_distance()

        return self.get_obs()

    def _select_rl_action_for_vessel(
        self, vessel_name: str, external_action: Optional[np.ndarray]
    ) -> Tuple[Optional[Tuple[float, float]], str]:
        if not self.is_vessel_rl_active(vessel_name):
            return None, ""
        if external_action is None:
            return None, ""
        a = np.asarray(external_action, dtype=np.float32).reshape(-1)
        if a.size < 2:
            return None, ""
        return (clamp(float(a[0]), -1.0, 1.0), clamp(float(a[1]), -1.0, 1.0)), "rl_external"

    @staticmethod
    def _normalize_action_vector(action: Union[np.ndarray, Tuple[float, float], list]) -> Optional[np.ndarray]:
        if action is None:
            return None
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size < 2:
            raise ValueError("Action must contain [rudder_cmd, throttle_cmd].")
        return np.asarray([clamp(float(a[0]), -1.0, 1.0), clamp(float(a[1]), -1.0, 1.0)], dtype=np.float32)

    def _resolve_step_actions(
        self,
        action: Union[np.ndarray, Tuple[float, float], list, Dict[str, Union[np.ndarray, Tuple[float, float], list]]],
    ) -> Tuple[Dict[str, Optional[np.ndarray]], float, float]:
        vessel_actions: Dict[str, Optional[np.ndarray]] = {vessel_id: None for vessel_id in self.get_vessel_ids()}
        if isinstance(action, dict):
            for vessel_id in vessel_actions:
                vessel_actions[vessel_id] = self._normalize_action_vector(action.get(vessel_id))
            # Backward-compatible scalar fields for info payloads.
            info_action = None
            for vessel_id in self.get_vessel_ids():
                if vessel_actions[vessel_id] is not None:
                    info_action = vessel_actions[vessel_id]
                    break
        else:
            shared_action = self._normalize_action_vector(action)
            # Backward-compatible path: a single action vector is applied to both vessels.
            # Current training passes a dict and controls only RL-active give-way vessels.
            for vessel_id in vessel_actions:
                vessel_actions[vessel_id] = shared_action
            info_action = shared_action

        if info_action is None:
            rudder_cmd, throttle_cmd = 0.0, 0.0
        else:
            rudder_cmd = float(info_action[0])
            throttle_cmd = float(info_action[1])
        return vessel_actions, rudder_cmd, throttle_cmd

    def step(
        self,
        action: Union[np.ndarray, Tuple[float, float], list, Dict[str, Union[np.ndarray, Tuple[float, float], list]]],
    ) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        action_by_vessel, rudder_cmd, throttle_cmd = self._resolve_step_actions(action)
        give_way_vessel = "vessel1" if self.vessel1_role == "give_way" else "vessel2" if self.vessel2_role == "give_way" else "none"
        stand_on_vessel = "vessel1" if self.vessel1_role == "stand_on" else "vessel2" if self.vessel2_role == "stand_on" else "none"
        stand_on_nominal_mode = "pure_pursuit" if stand_on_vessel == "vessel2" else "straight" if stand_on_vessel == "vessel1" else "none"

        if self.paused:
            reward_by_vessel = self.get_reward_by_vessel(0.0, 0.0, extras={vessel_id: 0.0 for vessel_id in self.extra_vessels})
            info: Dict[str, object] = {
                "reason": "paused",
                "vessel1_goal_distance": self._goal_distance(self.vessel1),
                "vessel2_goal_distance": self._goal_distance(self.vessel2),
                "vessel1_reached": int(self.vessel1_reached),
                "vessel2_reached": int(self.vessel2_reached),
                "rudder_cmd": rudder_cmd,
                "throttle_cmd": throttle_cmd,
                "dcpa": float(self.last_dcpa),
                "tcpa": float(self.last_tcpa),
                "risk_of_collision": int(self.risk_of_collision),
                "colregs_scenario": self.colregs_scenario,
                "geometry_scenario": self.geometry_scenario,
                "encounter_latched": int(self.encounter_latched),
                "overtaking_latched": int(self.overtaking_latched),
                "latched_scenario": self.latched_scenario,
                "vessel1_role": self.vessel1_role,
                "vessel2_role": self.vessel2_role,
                "designated_give_way_vessel": give_way_vessel,
                "designated_stand_on_vessel": stand_on_vessel,
                "stand_on_nominal_mode": stand_on_nominal_mode,
                "vessel1_rl_active": int(self.vessel1_rl_active),
                "vessel2_rl_active": int(self.vessel2_rl_active),
                "vessel1_model_control_latched": int(self.vessel1_model_control_latched),
                "vessel2_model_control_latched": int(self.vessel2_model_control_latched),
                # Backward-compatible aliases.
                "vessel1_rl_latched": int(self.vessel1_model_control_latched),
                "vessel2_rl_latched": int(self.vessel2_model_control_latched),
                "vessel1_distance_from_start": float(self._distance_from_start(self.vessel1, self.vessel1_start_pos)),
                "vessel2_distance_from_start": float(self._distance_from_start(self.vessel2, self.vessel2_start_pos)),
                "vessel1_relative_bearing_deg": float(self.vessel1_relative_bearing_deg),
                "vessel2_relative_bearing_deg": float(self.vessel2_relative_bearing_deg),
                "vessel1_control_source": self.vessel1_control_source,
                "vessel2_control_source": self.vessel2_control_source,
                "inter_vessel_distance": float(self._inter_vessel_distance()),
                "collision": 0,
                "near_miss": int(self._inter_vessel_distance() <= self.envp.near_miss_distance),
                "safe_pass_awarded": int(self.safe_pass_awarded),
                "reward_v1": reward_by_vessel["vessel1"],
                "reward_v2": reward_by_vessel["vessel2"],
                "reward_by_vessel": reward_by_vessel,
            }
            self._append_multi_vessel_debug_info(info)
            self._maybe_print_multi_vessel_debug(info)
            return self.get_obs(), 0.0, False, info

        encounter = self._classify_colregs()
        self.colregs_scenario = str(encounter["scenario"])
        self.hud_scenario = str(encounter.get("scenario_now", "none"))
        self.geometry_scenario = str(encounter["geometry"])
        self.vessel1_role = str(encounter["vessel1_role"])
        self.vessel2_role = str(encounter["vessel2_role"])
        self.overtaking_latched = bool(encounter["overtaking_latched"])
        self.encounter_latched = bool(encounter["encounter_latched"])
        self.vessel1_relative_bearing_deg = float(encounter["vessel1_bearing_deg"])
        self.vessel2_relative_bearing_deg = float(encounter["vessel2_bearing_deg"])
        self.last_tcpa = float(encounter["tcpa"])
        self.last_dcpa = float(encounter["dcpa"])
        self.risk_of_collision = bool(encounter["risk_of_collision"])
        tcpa = self.last_tcpa
        dcpa = self.last_dcpa
        tcpa_ok = 0.0 <= tcpa <= self.envp.tcpa_risk_threshold
        dcpa_ok = dcpa <= self.envp.dcpa_risk_threshold
        if self.envp.enable_step_risk_logging:
            print(
                "[RISK TRACE] "
                f"step={self.step_idx} t={self.time:.1f}s "
                f"tcpa={tcpa:.2f}s (<= {self.envp.tcpa_risk_threshold:.1f}? {tcpa_ok}) "
                f"dcpa={dcpa:.2f}m (<= {self.envp.dcpa_risk_threshold:.1f}? {dcpa_ok}) "
                f"risk={self.risk_of_collision} "
                f"scenario={self.colregs_scenario} geometry={self.geometry_scenario}"
            )
        give_way_vessel = "vessel1" if self.vessel1_role == "give_way" else "vessel2" if self.vessel2_role == "give_way" else "none"
        stand_on_vessel = "vessel1" if self.vessel1_role == "stand_on" else "vessel2" if self.vessel2_role == "stand_on" else "none"
        stand_on_nominal_mode = "pure_pursuit" if stand_on_vessel == "vessel2" else "straight" if stand_on_vessel == "vessel1" else "none"

        vessel1_dist = self._distance_from_start(self.vessel1, self.vessel1_start_pos)
        vessel2_dist = self._distance_from_start(self.vessel2, self.vessel2_start_pos)

        encounter_active = bool(self.locked)

        # Model-control takeover can start only on currently give-way vessels.
        vessel1_takeover_trigger = (
            encounter_active and self.vessel1_role == "give_way" and (not self.vessel1_reached)
        )
        vessel2_takeover_trigger = (
            encounter_active and self.vessel2_role == "give_way" and (not self.vessel2_reached)
        )
        if vessel1_takeover_trigger:
            self.vessel1_model_control_latched = True
        if vessel2_takeover_trigger:
            self.vessel2_model_control_latched = True
        if int(self.envp.num_vessels) > 2:
            self._update_extra_vessel_control_latches()

        # Release only when the vessel reaches goal; do not release when current risk disappears.
        if self.vessel1_reached:
            self.vessel1_model_control_latched = False
        if self.vessel2_reached:
            self.vessel2_model_control_latched = False
        for vessel_id in self.extra_vessels:
            if self.extra_vessel_reached[vessel_id]:
                self.extra_model_control_latched[vessel_id] = False

        # Active control is derived from persistent per-vessel latches.
        # Roles remain pure COLREGS outputs from geometry/risk classification.
        self.vessel1_rl_active = self.vessel1_model_control_latched and (not self.vessel1_reached)
        self.vessel2_rl_active = self.vessel2_model_control_latched and (not self.vessel2_reached)
        for vessel_id in self.extra_vessels:
            self.extra_vessel_rl_active[vessel_id] = self.extra_model_control_latched[vessel_id] and (not self.extra_vessel_reached[vessel_id])
        if self.vessel1_rl_active and self.vessel2_rl_active:
            self.rl_controlled_vessel = "both"
        elif self.vessel1_rl_active:
            self.rl_controlled_vessel = "vessel1"
        elif self.vessel2_rl_active:
            self.rl_controlled_vessel = "vessel2"
        else:
            self.rl_controlled_vessel = "none"
        self.any_rl_ever_triggered = self.any_rl_ever_triggered or any(self.is_vessel_rl_active(vessel_id) for vessel_id in self.get_vessel_ids())
        self.rl_ever_triggered = self.any_rl_ever_triggered

        give_way_vessel = "vessel1" if self.vessel1_role == "give_way" else "vessel2" if self.vessel2_role == "give_way" else "none"
        stand_on_vessel = "vessel1" if self.vessel1_role == "stand_on" else "vessel2" if self.vessel2_role == "stand_on" else "none"
        stand_on_nominal_mode = "pure_pursuit" if stand_on_vessel == "vessel2" else "straight" if stand_on_vessel == "vessel1" else "none"

        h = self.envp.dt / max(1, self.envp.substeps)
        was_vessel1_active = not self.vessel1_reached
        was_vessel2_active = not self.vessel2_reached
        for _ in range(max(1, self.envp.substeps)):
            if self.vessel1_rl_active:
                vessel1_rl_cmd, vessel1_rl_src = self._select_rl_action_for_vessel("vessel1", action_by_vessel.get("vessel1"))
                if vessel1_rl_cmd is not None:
                    self.vessel1_control_source = vessel1_rl_src
                    self._advance_controlled(self.vessel1, "vessel1_reached", vessel1_rl_cmd[0], vessel1_rl_cmd[1], h)
                else:
                    self.vessel1_control_source = "straight"
                    self._advance_straight(self.vessel1, "vessel1_reached", h)
            else:
                self.vessel1_control_source = "straight"
                self._advance_straight(self.vessel1, "vessel1_reached", h)

            if self.vessel2_rl_active:
                vessel2_rl_cmd, vessel2_rl_src = self._select_rl_action_for_vessel("vessel2", action_by_vessel.get("vessel2"))
                if vessel2_rl_cmd is not None:
                    self.vessel2_control_source = vessel2_rl_src
                    self._advance_controlled(self.vessel2, "vessel2_reached", vessel2_rl_cmd[0], vessel2_rl_cmd[1], h)
                else:
                    self.vessel2_control_source = "pure_pursuit"
                    self._advance_target(h)
            else:
                self.vessel2_control_source = "pure_pursuit"
                self._advance_target(h)

            for vessel_id, vessel in self.extra_vessels.items():
                reached_attr = f"extra_vessel_reached:{vessel_id}"
                if self.extra_vessel_rl_active[vessel_id]:
                    extra_rl_cmd, extra_rl_src = self._select_rl_action_for_vessel(vessel_id, action_by_vessel.get(vessel_id))
                    if extra_rl_cmd is not None:
                        self.extra_control_source[vessel_id] = extra_rl_src
                        self._advance_controlled(vessel, reached_attr, extra_rl_cmd[0], extra_rl_cmd[1], h)
                    else:
                        self.extra_control_source[vessel_id] = "pure_pursuit"
                        self._advance_extra_target(vessel_id, h)
                else:
                    self.extra_control_source[vessel_id] = "pure_pursuit"
                    self._advance_extra_target(vessel_id, h)

        if was_vessel1_active:
            self.vessel1_steps_taken += 1
        if was_vessel2_active:
            self.vessel2_steps_taken += 1
        for vessel_id in self.extra_vessels:
            if not self.extra_vessel_reached[vessel_id]:
                self.extra_steps_taken[vessel_id] += 1

        inter_vessel_distance = self._inter_vessel_distance()
        min_pair_distance = self._min_pair_distance()
        collision = min_pair_distance <= self.envp.collision_distance
        near_miss = (not collision) and (min_pair_distance <= self.envp.near_miss_distance)

        self.time += self.envp.dt
        self.step_idx += 1

        done = False
        reason = ""
        if collision:
            done, reason = True, "collision"
        elif self.step_idx >= self.max_steps:
            done, reason = True, "timeout"
        elif all(self.is_vessel_reached(vessel_id) for vessel_id in self.get_vessel_ids()):
            done, reason = True, "both_reached"

        vessel1_local_reward = 0.0
        vessel2_local_reward = 0.0
        shared_reward = self.rewp.living_penalty
        d_vessel1 = self._goal_distance(self.vessel1)
        d_vessel2 = self._goal_distance(self.vessel2)
        h_err_vessel1 = self._goal_heading_error(self.vessel1)
        h_err_vessel2 = self._goal_heading_error(self.vessel2)
        vessel1_local_reward += self._compute_progress_reward_for_vessel(
            self.prev_goal_d_vessel1,
            d_vessel1,
            self.prev_goal_heading_err_vessel1,
            h_err_vessel1,
        )
        vessel2_local_reward += self._compute_progress_reward_for_vessel(
            self.prev_goal_d_vessel2,
            d_vessel2,
            self.prev_goal_heading_err_vessel2,
            h_err_vessel2,
        )

        if self.vessel1_reached:
            vessel1_local_reward += self.rewp.goal_bonus
        if self.vessel2_reached:
            vessel2_local_reward += self.rewp.goal_bonus

        if reason == "collision":
            shared_reward += self.rewp.collision_penalty

        if near_miss:
            shared_reward += self.rewp.near_miss_penalty

        if min_pair_distance < self.envp.safe_pass_distance:
            shared_reward -= self.rewp.unsafe_proximity_penalty_weight * (self.envp.safe_pass_distance - min_pair_distance)

        if self.risk_of_collision:
            self.encounter_was_risky = True

        if self.encounter_was_risky and (not self.risk_of_collision) and (min_pair_distance > self.envp.safe_pass_distance) and (not self.safe_pass_awarded):
            shared_reward += self.rewp.safe_pass_bonus
            self.safe_pass_awarded = True

        vessel1_local_reward += self._scenario_local_shaping(
            scenario=self.colregs_scenario,
            rl_active=self.vessel1_rl_active,
            rudder=self.vessel1.rudder,
            tcpa=tcpa,
            dcpa=dcpa,
        )
        vessel2_local_reward += self._scenario_local_shaping(
            scenario=self.colregs_scenario,
            rl_active=self.vessel2_rl_active,
            rudder=self.vessel2.rudder,
            tcpa=tcpa,
            dcpa=dcpa,
        )

        vessel1_rudder_sign = 1 if self.vessel1.rudder > 1e-3 else -1 if self.vessel1.rudder < -1e-3 else 0
        vessel2_rudder_sign = 1 if self.vessel2.rudder > 1e-3 else -1 if self.vessel2.rudder < -1e-3 else 0
        if self.prev_vessel1_rudder_sign != 0 and vessel1_rudder_sign != 0 and vessel1_rudder_sign != self.prev_vessel1_rudder_sign:
            vessel1_local_reward -= self.rewp.oscillation_penalty_weight
        if self.prev_vessel2_rudder_sign != 0 and vessel2_rudder_sign != 0 and vessel2_rudder_sign != self.prev_vessel2_rudder_sign:
            vessel2_local_reward -= self.rewp.oscillation_penalty_weight
        self.prev_vessel1_rudder_sign = vessel1_rudder_sign
        self.prev_vessel2_rudder_sign = vessel2_rudder_sign
        self.last_inter_vessel_distance = min_pair_distance
        reward_v1 = vessel1_local_reward + shared_reward
        reward_v2 = vessel2_local_reward + shared_reward
        extra_rewards: Dict[str, float] = {}
        for vessel_id, vessel in self.extra_vessels.items():
            vessel_reward = self._compute_progress_reward_for_vessel(
                self.extra_prev_goal_d[vessel_id],
                self._goal_distance(vessel),
                self.extra_prev_goal_heading_err[vessel_id],
                self._goal_heading_error(vessel),
            )
            if self.extra_vessel_reached[vessel_id]:
                vessel_reward += self.rewp.goal_bonus
            extra_rewards[vessel_id] = vessel_reward + shared_reward
            self.extra_prev_goal_d[vessel_id] = self._goal_distance(vessel)
            self.extra_prev_goal_heading_err[vessel_id] = self._goal_heading_error(vessel)
        reward_by_vessel = self.get_reward_by_vessel(reward_v1, reward_v2, extras=extra_rewards)
        # Backward-compatible scalar return for legacy callers/logging.
        # Training updates consume per-vessel rewards from info["reward_v1"/"reward_v2"].
        reward = reward_v1 + reward_v2 - shared_reward

        self.prev_goal_d_vessel1 = d_vessel1
        self.prev_goal_d_vessel2 = d_vessel2
        self.prev_goal_heading_err_vessel1 = h_err_vessel1
        self.prev_goal_heading_err_vessel2 = h_err_vessel2

        info: Dict[str, object] = {
            "reason": reason,
            "vessel1_goal_distance": d_vessel1,
            "vessel2_goal_distance": d_vessel2,
            "vessel1_reached": int(self.vessel1_reached),
            "vessel2_reached": int(self.vessel2_reached),
            "rudder_cmd": rudder_cmd,
            "throttle_cmd": throttle_cmd,
            "vessel1_steps_taken": int(self.vessel1_steps_taken),
            "vessel2_steps_taken": int(self.vessel2_steps_taken),
            "vessel1_start_speed": float(self.vessel1_start_speed),
            "vessel2_start_speed": float(self.vessel2_start_speed),
            "vessel1_heading_deg": float(math.degrees(self.vessel1.h)),
            "vessel2_heading_deg": float(math.degrees(self.vessel2.h)),
            "vessel1_rudder_deg": float(math.degrees(self.vessel1.rudder)),
            "vessel2_rudder_deg": float(math.degrees(self.vessel2.rudder)),
            "dcpa": float(dcpa),
            "tcpa": float(tcpa),
            "risk_of_collision": int(self.risk_of_collision),
            "colregs_scenario": self.colregs_scenario,
            "geometry_scenario": self.geometry_scenario,
            "encounter_latched": int(self.encounter_latched),
            "overtaking_latched": int(self.overtaking_latched),
            "vessel1_role": self.vessel1_role,
            "vessel2_role": self.vessel2_role,
            "latched_scenario": self.latched_scenario,
            "designated_give_way_vessel": give_way_vessel,
            "designated_stand_on_vessel": stand_on_vessel,
            "stand_on_nominal_mode": stand_on_nominal_mode,
            "vessel1_rl_active": int(self.vessel1_rl_active),
            "vessel2_rl_active": int(self.vessel2_rl_active),
            "vessel1_model_control_latched": int(self.vessel1_model_control_latched),
            "vessel2_model_control_latched": int(self.vessel2_model_control_latched),
            # Backward-compatible aliases.
            "vessel1_rl_latched": int(self.vessel1_model_control_latched),
            "vessel2_rl_latched": int(self.vessel2_model_control_latched),
            "vessel1_distance_from_start": float(vessel1_dist),
            "vessel2_distance_from_start": float(vessel2_dist),
            "vessel1_relative_bearing_deg": float(self.vessel1_relative_bearing_deg),
            "vessel2_relative_bearing_deg": float(self.vessel2_relative_bearing_deg),
            "inter_vessel_distance": float(min_pair_distance),
            "collision": int(collision),
            "near_miss": int(near_miss),
            "safe_pass_awarded": int(self.safe_pass_awarded),
            "vessel1_control_source": self.vessel1_control_source,
            "vessel2_control_source": self.vessel2_control_source,
            "reward_v1": reward_by_vessel["vessel1"],
            "reward_v2": reward_by_vessel["vessel2"],
            "reward_by_vessel": reward_by_vessel,
        }
        self._append_multi_vessel_debug_info(info)
        self._maybe_print_multi_vessel_debug(info)

        # Trigger the takeover pause event exactly once per episode, on the first step RL activates.
        if (
            self.render_enabled
            and not self.rl_overlay_shown
            and self.rl_ever_triggered
            and (self.vessel1_rl_active or self.vessel2_rl_active)
        ):
            self.rl_overlay_shown = True
            self.paused = True
            if self.envp.show_risk_overlay:
                self.risk_overlay_active = True
                self.risk_overlay_payload = {
                    "step": int(self.step_idx),
                    "time": float(self.time),
                    "geometry": self.geometry_scenario,
                    "scenario": self.colregs_scenario,
                    "risk_of_collision": int(self.risk_of_collision),
                    "encounter_latched": int(self.encounter_latched),
                    "overtaking_latched": int(self.overtaking_latched),
                    "vessel1_role": self.vessel1_role,
                    "vessel2_role": self.vessel2_role,
                    "dcpa": float(self.last_dcpa),
                    "tcpa": float(self.last_tcpa),
                    "vessel1_bearing": float(self.vessel1_relative_bearing_deg),
                    "vessel2_bearing": float(self.vessel2_relative_bearing_deg),
                    "vessel1_rl_active": int(self.vessel1_rl_active),
                    "vessel2_rl_active": int(self.vessel2_rl_active),
                    "vessel1_model_control_latched": int(self.vessel1_model_control_latched),
                    "vessel2_model_control_latched": int(self.vessel2_model_control_latched),
                    # Backward-compatible aliases.
                    "vessel1_rl_latched": int(self.vessel1_model_control_latched),
                    "vessel2_rl_latched": int(self.vessel2_model_control_latched),
                    "vessel1_control_source": self.vessel1_control_source,
                    "vessel2_control_source": self.vessel2_control_source,
                    "vessel1_distance": float(vessel1_dist),
                    "vessel2_distance": float(vessel2_dist),
                }
            if self.envp.auto_show_risk_sector_overlay:
                self.risk_sector_overlay_active = True
            else:
                self.risk_sector_overlay_active = False

        self.prev_vessel1_rl_active = self.vessel1_rl_active
        self.prev_vessel2_rl_active = self.vessel2_rl_active
        return self.get_obs(), float(reward), done, info

    def _draw_risk_overlay(self, surf) -> None:
        if not self.risk_overlay_active or not self._font:
            return

        w = self.sx(self.envp.world_w)
        h = self.sy(self.envp.world_h)
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 155))

        p = self.risk_overlay_payload
        tcpa = float(p.get("tcpa", float("inf")))
        tcpa_txt = "inf" if math.isinf(tcpa) else f"{tcpa:.1f}s"
        vessel1_active = int(p.get("vessel1_rl_active", 0))
        vessel2_active = int(p.get("vessel2_rl_active", 0))
        if vessel1_active and vessel2_active:
            rl_summary = "RL control active on BOTH vessels"
        elif vessel1_active:
            rl_summary = "RL control active on V1; V2 uses fallback"
        elif vessel2_active:
            rl_summary = "RL control active on V2; V1 uses fallback"
        else:
            rl_summary = "No vessel currently under RL control"

        lines = [
            "⚠  RL TAKEOVER / ENCOUNTER STATUS",
            f"Step {int(p.get('step', self.step_idx))}   Sim time {float(p.get('time', self.time)):.1f}s",
            f"Scenario={str(p.get('scenario', self.colregs_scenario)).upper()}",
            f"V1 role={p.get('vessel1_role', self.vessel1_role)}",
            f"V2 role={p.get('vessel2_role', self.vessel2_role)}",
            f"DCPA={float(p.get('dcpa', self.last_dcpa)):.1f}m  TCPA={tcpa_txt}  V1→V2={float(p.get('vessel1_bearing', self.vessel1_relative_bearing_deg)):.1f}°/{self._bearing_to_signed_deg(float(p.get('vessel1_bearing', self.vessel1_relative_bearing_deg))):+.1f}°  V2→V1={float(p.get('vessel2_bearing', self.vessel2_relative_bearing_deg)):.1f}°/{self._bearing_to_signed_deg(float(p.get('vessel2_bearing', self.vessel2_relative_bearing_deg))):+.1f}°",
            rl_summary,
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

    def _sector_boundaries_deg(self) -> List[float]:
        """True radar-sector boundaries in relative-bearing coordinates."""
        return [350.0, 10.0, 40.0, 75.0, 112.5, 180.0, 247.5, 285.0, 320.0, 350.0]

    def _sector_spans_with_category(self) -> List[Tuple[float, float, str]]:
        # Category mapping:
        # - head_on: 350→10
        # - crossing: 10→40, 40→75, 75→112.5, 247.5→285, 285→320, 320→350
        # - overtaking: 112.5→180, 180→247.5
        return [
            (350.0, 10.0, "head_on"),
            (10.0, 40.0, "crossing"),
            (40.0, 75.0, "crossing"),
            (75.0, 112.5, "crossing"),
            (112.5, 180.0, "overtaking"),
            (180.0, 247.5, "overtaking"),
            (247.5, 285.0, "crossing"),
            (285.0, 320.0, "crossing"),
            (320.0, 350.0, "crossing"),
        ]

    def _sector_span_for_index(self, sector_idx: int) -> Tuple[float, float, str]:
        spans = self._sector_spans_with_category()
        clamped_idx = max(0, min(8, int(sector_idx)))
        return spans[clamped_idx]

    def _risk_occupied_spans_for_vessel(self, vessel_id: str) -> List[Tuple[float, float, str]]:
        if vessel_id == "vessel1":
            observer = self.vessel1
            target = self.vessel2
        elif vessel_id == "vessel2":
            observer = self.vessel2
            target = self.vessel1
        else:
            return []
        if observer is None or target is None:
            return []
        rel_bearing = self._get_relative_bearing(observer, target)
        sector_idx = self._get_sector_index(rel_bearing)
        return [self._sector_span_for_index(sector_idx)]

    def _sector_spans_for_vessel_overlay(self, vessel_id: str) -> List[Tuple[float, float, str]]:
        if self.manual_sector_overlay_enabled:
            return self._sector_spans_with_category()
        if self.risk_sector_overlay_active:
            return self._risk_occupied_spans_for_vessel(vessel_id)
        return []

    def _sector_overlay_is_active(self) -> bool:
        return bool(self.manual_sector_overlay_enabled or self.risk_sector_overlay_active)

    def _sector_style_for_category(self, category: str) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int]]:
        if category == "head_on":
            return (255, 220, 120, 36), (255, 225, 150)
        if category == "overtaking":
            return (255, 160, 140, 30), (255, 185, 165)
        # crossing sectors (default)
        return (140, 205, 255, 30), (170, 220, 255)

    def _bearing_arc_points(self, vessel: Vessel, start_deg: float, end_deg: float, radius: float, steps: int) -> List[Tuple[int, int]]:
        start = start_deg
        end = end_deg
        if end < start:
            end += 360.0
        pts: List[Tuple[int, int]] = []
        for i in range(steps + 1):
            bdeg = start + (end - start) * (i / max(1, steps))
            world_ang = vessel.h + math.radians(bdeg % 360.0)
            wx = vessel.x + radius * math.cos(world_ang)
            wy = vessel.y + radius * math.sin(world_ang)
            pts.append((self.sx(wx), self.sy(wy)))
        return pts

    def _draw_sector_overlay(self, surf) -> None:
        if not self._sector_overlay_is_active():
            return

        w = self.sx(self.envp.world_w)
        h = self.sy(self.envp.world_h)
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        ray_len = float(self.envp.sensor_range)
        for vessel_id in ("vessel1", "vessel2"):
            vessel = self.get_vessel_by_id(vessel_id)
            if vessel is None:
                continue
            spans_to_draw = self._sector_spans_for_vessel_overlay(vessel_id)
            if not spans_to_draw:
                continue
            x0 = self.sx(vessel.x)
            y0 = self.sy(vessel.y)
            for start_deg, end_deg, category in spans_to_draw:
                fill_rgba, _ = self._sector_style_for_category(category)
                span = (end_deg - start_deg) % 360.0
                steps = max(4, int(round(span / 6.0)))
                arc_pts = self._bearing_arc_points(vessel, start_deg, end_deg, ray_len, steps)
                poly = [(x0, y0)] + arc_pts
                pygame.draw.polygon(overlay, fill_rgba, poly)
                line_bearings = [start_deg, end_deg]
                for bdeg in line_bearings:
                    world_ang = vessel.h + math.radians(bdeg % 360.0)
                    x1 = self.sx(vessel.x + ray_len * math.cos(world_ang))
                    y1 = self.sy(vessel.y + ray_len * math.sin(world_ang))
                    category = "head_on" if category == "head_on" else ("overtaking" if category == "overtaking" else "crossing")
                    _, line_rgb = self._sector_style_for_category(category)
                    pygame.draw.line(surf, line_rgb, (x0, y0), (x1, y1), 1)
            if self.manual_sector_overlay_enabled:
                for bdeg in self._sector_boundaries_deg():
                    world_ang = vessel.h + math.radians(bdeg % 360.0)
                    x1 = self.sx(vessel.x + ray_len * math.cos(world_ang))
                    y1 = self.sy(vessel.y + ray_len * math.sin(world_ang))
                    _, line_rgb = self._sector_style_for_category("crossing")
                    pygame.draw.line(surf, line_rgb, (x0, y0), (x1, y1), 1)
        surf.blit(overlay, (0, 0))

    def render(self) -> None:
        if not self.render_enabled or self._screen is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.show_planned_paths = not self.show_planned_paths
            if event.type == pygame.KEYDOWN and event.key == pygame.K_o:
                self.manual_sector_overlay_enabled = not self.manual_sector_overlay_enabled
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_RETURN):
                if self.risk_overlay_active:
                    self.risk_overlay_active = False
                    self.risk_overlay_payload = {}
                    self.risk_sector_overlay_active = False
                    self.paused = False
                elif self.paused and self.risk_sector_overlay_active:
                    self.risk_sector_overlay_active = False
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

        self._draw_goal(self.vessel1.goal_x, self.vessel1.goal_y, (250, 215, 60))
        self._draw_goal(self.vessel2.goal_x, self.vessel2.goal_y, (255, 140, 90))
        for vessel_id, vessel in self.extra_vessels.items():
            self._draw_goal(vessel.goal_x, vessel.goal_y, (255, 185, 120))

        if self.envp.show_spawn_rings:
            pygame.draw.circle(
                self._screen,
                (255, 225, 120),
                (self.sx(self.start_x), self.sy(self.start_y)),
                int(round(self.envp.vessel2_outer_radius * self.envp.pixels_per_meter)),
                1,
            )

        if self.show_planned_paths:
            self._draw_planned_path(self.vessel1_planned_path, (150, 210, 255))
            self._draw_planned_path(self.vessel2_planned_path, (255, 170, 170))
            for vessel_id in sorted(self.extra_vessel_planned_paths, key=lambda x: int(x.replace("vessel", ""))):
                self._draw_planned_path(self.extra_vessel_planned_paths[vessel_id], (255, 200, 150))

        self._draw_vessel(self.vessel1, (95, 170, 255), "V1")
        self._draw_vessel(self.vessel2, (255, 120, 120), "V2")
        for vessel_id, vessel in self.extra_vessels.items():
            label = f"V{vessel_id.replace('vessel', '')}"
            self._draw_vessel(vessel, (255, 150, 120), label)
        self._draw_sector_overlay(surf)

        tcpa_txt = "inf" if math.isinf(self.last_tcpa) else f"{self.last_tcpa:.1f}s"

        hud_scenario = "heading" if self.hud_scenario == "head_on" else self.hud_scenario
        hud0 = self._font.render(
            f"step={self.step_idx} t={self.time:.1f}s scenario={hud_scenario} risk={self.risk_of_collision}",
            True, (255, 255, 255),
        )
        hud1 = self._font.render(
            f"DCPA={self.last_dcpa:.1f}m TCPA={tcpa_txt} BRG V1→V2={self.vessel1_relative_bearing_deg:.1f}°/{self._bearing_to_signed_deg(self.vessel1_relative_bearing_deg):+.1f}° V2→V1={self.vessel2_relative_bearing_deg:.1f}°/{self._bearing_to_signed_deg(self.vessel2_relative_bearing_deg):+.1f}°",
            True, (255, 240, 170),
        )
        hud2 = self._font.render(
            f"V1 spd={self.vessel1.speed:.2f}",
            True, (170, 220, 255),
        )
        hud3 = self._font.render(
            f"V2 spd={self.vessel2.speed:.2f}",
            True, (255, 190, 190),
        )
        hud4 = self._font.render(
            f"V1 start=({self.vessel1_start_pos[0]:.1f},{self.vessel1_start_pos[1]:.1f}) goal=({self.vessel1.goal_x:.1f},{self.vessel1.goal_y:.1f}) h0={math.degrees(self.vessel1_start_heading):.1f}° v0={self.vessel1_start_speed:.2f}",
            True, (170, 220, 255),
        )
        hud5 = self._font.render(
            f"V2 start=({self.vessel2_start_pos[0]:.1f},{self.vessel2_start_pos[1]:.1f}) goal=({self.vessel2.goal_x:.1f},{self.vessel2.goal_y:.1f}) h0={math.degrees(self.vessel2_start_heading):.1f}° v0={self.vessel2_start_speed:.2f}",
            True, (255, 190, 190),
        )

        surf.blit(hud0, (10, 8))
        surf.blit(hud1, (10, 26))
        surf.blit(hud2, (10, 44))
        surf.blit(hud3, (10, 62))
        surf.blit(hud4, (10, 80))
        surf.blit(hud5, (10, 98))

        if self.envp.show_risk_overlay and self.risk_overlay_active:
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
