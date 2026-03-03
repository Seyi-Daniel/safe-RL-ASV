from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from simulations.hyperparameters_perimeter_start import EnvParams, RewardParams

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
    """Two-vessel setup: both vessels spawn on the big-circle circumference; vessel-2 follows pure pursuit path."""

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
        self.start_x = 0.5 * self.envp.world_w
        self.start_y = 0.5 * self.envp.world_h
        self.time = 0.0
        self.step_idx = 0
        self.max_steps = max(1, int(round(self.envp.episode_seconds / self.envp.dt)))
        self.prev_goal_d_vessel1 = 0.0
        self.prev_goal_d_vessel2 = 0.0
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

        self.render_enabled = render and HAS_PYGAME
        self.paused = False
        self.risk_overlay_active = False
        self.risk_overlay_payload: Dict[str, float | str | int] = {}
        self.rl_ever_triggered: bool = False  # latches True when RL first activates, never resets within episode
        self.rl_overlay_shown: bool = False  # True after overlay has been shown once this episode
        self.prev_vessel1_rl_active = False
        self.prev_vessel2_rl_active = False
        self.overtaking_latched = False
        self.locked = False
        self.locked_scenario = "safe"
        self.locked_role_v1 = "none"
        self.locked_role_v2 = "none"
        self.lock_candidate_steps = 0
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
        self.locked = False
        self.locked_scenario = "safe"
        self.locked_role_v1 = "none"
        self.locked_role_v2 = "none"
        self.lock_candidate_steps = 0
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
        self.vessel1_standon_escalated = False
        self.vessel2_standon_escalated = False
        self.vessel1_standon_risk_steps = 0
        self.vessel2_standon_risk_steps = 0
        self.vessel1_control_source = "straight"
        self.vessel2_control_source = "pure_pursuit"
        self.vessel1_rl_latched = False
        self.vessel2_rl_latched = False
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
        self.vessel1_standon_hold_awarded = False
        self.vessel2_standon_hold_awarded = False
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
        # Dual-control is disabled: policy controls only one give-way vessel via external action.
        self.secondary_policy_fn = None

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

    def _inter_vessel_distance(self) -> float:
        return math.hypot(self.vessel2.x - self.vessel1.x, self.vessel2.y - self.vessel1.y)

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

    def _is_closing(self, from_v: Vessel, to_v: Vessel, eps: float = 1e-6) -> bool:
        rx = to_v.x - from_v.x
        ry = to_v.y - from_v.y
        r = math.hypot(rx, ry)
        if r <= eps:
            return False
        rvx = math.cos(to_v.h) * to_v.speed - math.cos(from_v.h) * from_v.speed
        rvy = math.sin(to_v.h) * to_v.speed - math.sin(from_v.h) * from_v.speed
        range_rate = (rx * rvx + ry * rvy) / r
        return range_rate < -eps

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

    def _reset_sample_triggers_takeover(self, vessel1: Vessel, vessel2: Vessel) -> bool:
        saved_latch_state = (
            self.overtaking_latched,
            self.latched_scenario,
            self.latched_vessel1_role,
            self.latched_vessel2_role,
            self.overtaking_clear_steps,
            self.encounter_latched,
            self.latched_encounter_active,
            self.latched_geometry,
            self.encounter_clear_steps,
            self.designated_vessel1_role,
            self.designated_vessel2_role,
            self.rl_controlled_vessel,
            self.candidate_scenario,
            self.candidate_vessel1_role,
            self.candidate_vessel2_role,
            self.candidate_steps,
            self.active_non_overtaking_scenario,
            self.active_non_overtaking_vessel1_role,
            self.active_non_overtaking_vessel2_role,
            self.active_non_overtaking_exit_steps,
            self.locked,
            self.locked_scenario,
            self.locked_role_v1,
            self.locked_role_v2,
            self.lock_candidate_steps,
        )

        def _restore_latch_state() -> None:
            (
                self.overtaking_latched,
                self.latched_scenario,
                self.latched_vessel1_role,
                self.latched_vessel2_role,
                self.overtaking_clear_steps,
                self.encounter_latched,
                self.latched_encounter_active,
                self.latched_geometry,
                self.encounter_clear_steps,
                self.designated_vessel1_role,
                self.designated_vessel2_role,
                self.rl_controlled_vessel,
                self.candidate_scenario,
                self.candidate_vessel1_role,
                self.candidate_vessel2_role,
                self.candidate_steps,
                self.active_non_overtaking_scenario,
                self.active_non_overtaking_vessel1_role,
                self.active_non_overtaking_vessel2_role,
                self.active_non_overtaking_exit_steps,
                self.locked,
                self.locked_scenario,
                self.locked_role_v1,
                self.locked_role_v2,
                self.lock_candidate_steps,
            ) = saved_latch_state

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

        vessel1_sim = Vessel(vessel1.x, vessel1.y, vessel1.h, vessel1.speed, vessel1.goal_x, vessel1.goal_y, vessel1.rudder, vessel1.throttle)
        vessel2_sim = Vessel(vessel2.x, vessel2.y, vessel2.h, vessel2.speed, vessel2.goal_x, vessel2.goal_y, vessel2.rudder, vessel2.throttle)

        vessel1_reached = False
        vessel2_reached = False
        vessel1_start = (vessel1_sim.x, vessel1_sim.y)
        vessel2_start = (vessel2_sim.x, vessel2_sim.y)
        h = self.envp.dt / max(1, self.envp.substeps)

        takeover_viable = False
        min_separation = float("inf")

        for _ in range(self.max_steps):
            if vessel1_reached or vessel2_reached:
                encounter = {
                    "vessel1_role": "none",
                    "vessel2_role": "none",
                    "risk_of_collision": False,
                    "dcpa": float("inf"),
                    "tcpa": float("inf"),
                }
            else:
                encounter = self._resolve_colregs_pair(vessel1_sim, vessel2_sim)

            separation = math.hypot(vessel2_sim.x - vessel1_sim.x, vessel2_sim.y - vessel1_sim.y)
            min_separation = min(min_separation, separation)

            risk_now = bool(encounter.get("risk_of_collision", False))

            vessel1_dist = self._distance_from_start(vessel1_sim, vessel1_start)
            vessel2_dist = self._distance_from_start(vessel2_sim, vessel2_start)
            if (
                risk_now
                and encounter["vessel1_role"] == "give_way"
                and not vessel1_reached
                and vessel1_dist >= self.envp.rl_takeover_distance
            ):
                takeover_viable = True
            if (
                risk_now
                and encounter["vessel2_role"] == "give_way"
                and not vessel2_reached
                and vessel2_dist >= self.envp.rl_takeover_distance
            ):
                takeover_viable = True

            for _ in range(max(1, self.envp.substeps)):
                if not vessel1_reached:
                    d_vessel1 = math.hypot(vessel1_sim.goal_x - vessel1_sim.x, vessel1_sim.goal_y - vessel1_sim.y)
                    if d_vessel1 <= self.envp.goal_radius:
                        vessel1_reached = True
                        vessel1_sim.speed = 0.0
                    else:
                        travel = min(vessel1_sim.speed * h, d_vessel1)
                        vessel1_sim.x += math.cos(vessel1_sim.h) * travel
                        vessel1_sim.y += math.sin(vessel1_sim.h) * travel
                        if math.hypot(vessel1_sim.goal_x - vessel1_sim.x, vessel1_sim.goal_y - vessel1_sim.y) <= self.envp.goal_radius:
                            vessel1_reached = True
                            vessel1_sim.speed = 0.0

                if not vessel2_reached:
                    d_vessel2 = math.hypot(vessel2_sim.goal_x - vessel2_sim.x, vessel2_sim.goal_y - vessel2_sim.y)
                    if d_vessel2 <= self.envp.goal_radius:
                        vessel2_reached = True
                        vessel2_sim.speed = 0.0
                    else:
                        rudder_cmd = self._pure_pursuit_rudder_cmd(vessel2_sim, vessel2_sim.goal_x, vessel2_sim.goal_y)
                        self._integrate_rudder_heading(vessel2_sim, rudder_cmd, h)
                        vessel2_sim.speed = clamp(vessel2_sim.speed, self.envp.vessel2_min_speed, self.envp.vessel2_max_speed)
                        travel = min(vessel2_sim.speed * h, d_vessel2)
                        vessel2_sim.x += travel * math.cos(vessel2_sim.h)
                        vessel2_sim.y += travel * math.sin(vessel2_sim.h)
                        if math.hypot(vessel2_sim.goal_x - vessel2_sim.x, vessel2_sim.goal_y - vessel2_sim.y) <= self.envp.goal_radius:
                            vessel2_reached = True
                            vessel2_sim.speed = 0.0

            if (vessel1_reached and vessel2_reached) or self._outside(vessel1_sim) or self._outside(vessel2_sim):
                break

        unavoidable_hazard = min_separation <= self.envp.near_miss_distance
        qualifies = takeover_viable and unavoidable_hazard
        _restore_latch_state()
        return qualifies

    def _point_on_big_circle(self, ang: float) -> Tuple[float, float]:
        r = self.envp.vessel2_outer_radius
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

        if self._goal_distance(v) <= self.envp.goal_radius:
            setattr(self, reached_attr, True)
            v.speed = 0.0

    def _advance_hold_course_speed(self, v: Vessel, reached_attr: str, dt: float) -> None:
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

        if self._goal_distance(v) <= self.envp.goal_radius:
            setattr(self, reached_attr, True)
            v.speed = 0.0

    def _fallback_colregs_action(self, vessel_name: str, scenario: str, role: str, standon_escalated: bool) -> Tuple[float, float, str]:
        if scenario == "head_on":
            return self.envp.fallback_starboard_rudder_cmd, self.envp.fallback_headon_throttle_cmd, "starboard_avoid"

        if scenario == "crossing" and role == "give_way":
            return self.envp.fallback_starboard_rudder_cmd, self.envp.fallback_crossing_throttle_cmd, "starboard_avoid"

        if scenario == "crossing" and role == "stand_on":
            if standon_escalated:
                return self.envp.fallback_starboard_rudder_cmd, self.envp.fallback_crossing_throttle_cmd, "standon_escalation"
            return 0.0, 0.0, "hold_course_speed"

        if scenario == "overtaking" and role == "give_way":
            return self.envp.fallback_starboard_rudder_cmd, self.envp.fallback_crossing_throttle_cmd, "starboard_avoid"

        if scenario == "overtaking" and role == "stand_on":
            return 0.0, 0.0, "hold_course_speed"

        if vessel_name == "vessel2":
            return 0.0, 0.0, "pure_pursuit"
        return 0.0, 0.0, "straight"

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

        if self._goal_distance(v) <= self.envp.goal_radius:
            setattr(self, reached_attr, True)
            v.speed = 0.0

    def _build_vessel1_planned_path(self) -> None:
        if self.vessel1 is None:
            self.vessel1_planned_path = []
            return
        self.vessel1_planned_path = [(self.vessel1.x, self.vessel1.y), (self.vessel1.goal_x, self.vessel1.goal_y)]

    def _build_vessel2_planned_path(self, sx: float, sy: float, sh: float, speed: float, goal_x: float, goal_y: float) -> None:
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

        self.vessel2_planned_path = pts

    def _get_obs_for_perspective(self, own: Vessel, other: Vessel, own_is_agent: bool) -> np.ndarray:
        own_speed_den = self.envp.max_speed if own_is_agent else self.envp.vessel2_max_speed
        other_speed_den = self.envp.vessel2_max_speed if own_is_agent else self.envp.max_speed
        return np.asarray(
            [
                own.x / self.envp.world_w,
                own.y / self.envp.world_h,
                own.h / math.pi,
                own.speed / own_speed_den,
                own.goal_x / self.envp.world_w,
                own.goal_y / self.envp.world_h,
                other.x / self.envp.world_w,
                other.y / self.envp.world_h,
                other.h / math.pi,
                other.speed / other_speed_den,
                other.goal_x / self.envp.world_w,
                other.goal_y / self.envp.world_h,
            ],
            dtype=np.float32,
        )

    def _get_obs_vessel1_perspective(self) -> np.ndarray:
        return self._get_obs_for_perspective(self.vessel1, self.vessel2, own_is_agent=True)

    def _get_obs_vessel2_perspective(self) -> np.ndarray:
        return self._get_obs_for_perspective(self.vessel2, self.vessel1, own_is_agent=False)

    def get_obs(self) -> np.ndarray:
        return self._get_obs_vessel1_perspective()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)

        max_tries = max(1, int(self.envp.reset_viable_episode_max_tries))
        require_viable_path = bool(self.envp.require_reset_viable_takeover_path)
        sampled_vessel1: Vessel | None = None
        sampled_vessel2: Vessel | None = None
        for _ in range(max_tries):
            start_ang_1 = self.rng.uniform(0.0, 2.0 * math.pi)
            goal_ang_1 = self.rng.uniform(0.0, 2.0 * math.pi)
            sx1, sy1 = self._point_on_big_circle(start_ang_1)
            agx, agy = self._point_on_big_circle(goal_ang_1)
            ah = math.atan2(agy - sy1, agx - sx1)
            aspeed = self.rng.uniform(self.envp.min_speed, self.envp.max_speed)
            candidate_agent = Vessel(sx1, sy1, ah, aspeed, agx, agy)
            candidate_target = self._sample_vessel2_path()
            if (not require_viable_path) or self._reset_sample_triggers_takeover(candidate_agent, candidate_target):
                sampled_vessel1 = candidate_agent
                sampled_vessel2 = candidate_target
                break

        self.reset_has_takeover_path = sampled_vessel1 is not None and sampled_vessel2 is not None
        if sampled_vessel1 is None or sampled_vessel2 is None:
            sampled_vessel1 = candidate_agent
            sampled_vessel2 = candidate_target

        self.vessel1 = sampled_vessel1
        self.vessel2 = sampled_vessel2

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
        self.vessel1_standon_escalated = False
        self.vessel2_standon_escalated = False
        self.vessel1_standon_risk_steps = 0
        self.vessel2_standon_risk_steps = 0
        self.vessel1_control_source = "straight"
        self.vessel2_control_source = "pure_pursuit"
        self.vessel1_rl_latched = False
        self.vessel2_rl_latched = False
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
        self.vessel1_standon_hold_awarded = False
        self.vessel2_standon_hold_awarded = False
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

        # Episode termination is fixed-time or both-reached (whichever occurs first).
        self.max_steps = max(1, int(round(self.envp.episode_seconds / self.envp.dt)))

        self._build_vessel1_planned_path()
        self._build_vessel2_planned_path(sx2, sy2, sh2, sp2, gx2, gy2)
        self.last_inter_vessel_distance = self._inter_vessel_distance()

        return self.get_obs()

    def _select_rl_action_for_vessel(
        self, vessel_name: str, external_action: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float]], str]:
        if vessel_name == "vessel1" and not self.vessel1_rl_active:
            return None, ""
        if vessel_name == "vessel2" and not self.vessel2_rl_active:
            return None, ""
        a = np.asarray(external_action, dtype=np.float32).reshape(-1)
        if a.size < 2:
            return None, ""
        return (clamp(float(a[0]), -1.0, 1.0), clamp(float(a[1]), -1.0, 1.0)), "rl_external"

    def step(self, action: Union[np.ndarray, Tuple[float, float], list]) -> Tuple[np.ndarray, float, bool, Dict[str, float | str | int]]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size < 2:
            raise ValueError("Action must contain [rudder_cmd, throttle_cmd].")
        rudder_cmd = clamp(float(a[0]), -1.0, 1.0)
        throttle_cmd = clamp(float(a[1]), -1.0, 1.0)
        give_way_vessel = "vessel1" if self.vessel1_role == "give_way" else "vessel2" if self.vessel2_role == "give_way" else "none"
        stand_on_vessel = "vessel1" if self.vessel1_role == "stand_on" else "vessel2" if self.vessel2_role == "stand_on" else "none"
        stand_on_nominal_mode = "pure_pursuit" if stand_on_vessel == "vessel2" else "straight" if stand_on_vessel == "vessel1" else "none"

        if self.paused:
            info: Dict[str, float | str | int] = {
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
                "vessel1_rl_latched": int(self.vessel1_rl_latched),
                "vessel2_rl_latched": int(self.vessel2_rl_latched),
                "vessel1_distance_from_start": float(self._distance_from_start(self.vessel1, self.vessel1_start_pos)),
                "vessel2_distance_from_start": float(self._distance_from_start(self.vessel2, self.vessel2_start_pos)),
                "vessel1_relative_bearing_deg": float(self.vessel1_relative_bearing_deg),
                "vessel2_relative_bearing_deg": float(self.vessel2_relative_bearing_deg),
                "vessel1_control_source": self.vessel1_control_source,
                "vessel2_control_source": self.vessel2_control_source,
                "vessel1_standon_escalated": int(self.vessel1_standon_escalated),
                "vessel2_standon_escalated": int(self.vessel2_standon_escalated),
                "inter_vessel_distance": float(self._inter_vessel_distance()),
                "collision": 0,
                "near_miss": int(self._inter_vessel_distance() <= self.envp.near_miss_distance),
                "safe_pass_awarded": int(self.safe_pass_awarded),
            }
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

        # Stand-on escalation disabled for this contract: stand-on remains nominal path-following.
        self.vessel1_standon_risk_steps = 0
        self.vessel2_standon_risk_steps = 0
        self.vessel1_standon_escalated = False
        self.vessel2_standon_escalated = False

        encounter_active = bool(self.locked)
        allow_vessel1_rl = encounter_active and self.vessel1_role == "give_way" and (not self.vessel1_reached)
        allow_vessel2_rl = encounter_active and self.vessel2_role == "give_way" and (not self.vessel2_reached)

        # RL controls the give-way vessel(s) during locked encounters.
        self.vessel1_rl_active = allow_vessel1_rl
        self.vessel2_rl_active = allow_vessel2_rl
        self.vessel1_rl_latched = self.vessel1_rl_latched or self.vessel1_rl_active
        self.vessel2_rl_latched = self.vessel2_rl_latched or self.vessel2_rl_active
        if self.vessel1_rl_active and self.vessel2_rl_active:
            self.rl_controlled_vessel = "both"
        elif self.vessel1_rl_active:
            self.rl_controlled_vessel = "vessel1"
        elif self.vessel2_rl_active:
            self.rl_controlled_vessel = "vessel2"
        else:
            self.rl_controlled_vessel = "none"
        self.any_rl_ever_triggered = self.any_rl_ever_triggered or self.vessel1_rl_active or self.vessel2_rl_active
        self.rl_ever_triggered = self.any_rl_ever_triggered

        early_cutoff_steps = max(1, int(self.envp.no_takeover_early_done_steps))
        if bool(self.envp.enable_no_takeover_early_done) and (not self.reset_has_takeover_path) and (not self.any_rl_ever_triggered) and (self.step_idx + 1) >= early_cutoff_steps:
            self.step_idx += 1
            self.time += self.envp.dt
            d_vessel1 = self._goal_distance(self.vessel1)
            d_vessel2 = self._goal_distance(self.vessel2)
            reward = self.rewp.living_penalty
            reward += self.rewp.progress_weight * (self.prev_goal_d_vessel1 - d_vessel1)
            reward += self.rewp.progress_weight * (self.prev_goal_d_vessel2 - d_vessel2)
            self.prev_goal_d_vessel1 = d_vessel1
            self.prev_goal_d_vessel2 = d_vessel2
            info: Dict[str, float | str | int] = {
                "reason": "no_takeover_trigger",
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
                "vessel1_rl_active": int(self.vessel1_rl_active),
                "vessel2_rl_active": int(self.vessel2_rl_active),
                "vessel1_rl_latched": int(self.vessel1_rl_latched),
                "vessel2_rl_latched": int(self.vessel2_rl_latched),
                "vessel1_distance_from_start": float(vessel1_dist),
                "vessel2_distance_from_start": float(vessel2_dist),
                "vessel1_relative_bearing_deg": float(self.vessel1_relative_bearing_deg),
                "vessel2_relative_bearing_deg": float(self.vessel2_relative_bearing_deg),
                "vessel1_control_source": self.vessel1_control_source,
                "vessel2_control_source": self.vessel2_control_source,
                "vessel1_standon_escalated": int(self.vessel1_standon_escalated),
                "vessel2_standon_escalated": int(self.vessel2_standon_escalated),
            }
            self.prev_vessel1_rl_active = self.vessel1_rl_active
            self.prev_vessel2_rl_active = self.vessel2_rl_active
            return self.get_obs(), float(reward), True, info

        h = self.envp.dt / max(1, self.envp.substeps)
        was_vessel1_active = not self.vessel1_reached
        was_vessel2_active = not self.vessel2_reached
        for _ in range(max(1, self.envp.substeps)):
            if self.vessel1_rl_active:
                vessel1_rl_cmd, vessel1_rl_src = self._select_rl_action_for_vessel("vessel1", a)
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
                vessel2_rl_cmd, vessel2_rl_src = self._select_rl_action_for_vessel("vessel2", a)
                if vessel2_rl_cmd is not None:
                    self.vessel2_control_source = vessel2_rl_src
                    self._advance_controlled(self.vessel2, "vessel2_reached", vessel2_rl_cmd[0], vessel2_rl_cmd[1], h)
                else:
                    self.vessel2_control_source = "pure_pursuit"
                    self._advance_target(h)
            else:
                self.vessel2_control_source = "pure_pursuit"
                self._advance_target(h)

        if was_vessel1_active:
            self.vessel1_steps_taken += 1
        if was_vessel2_active:
            self.vessel2_steps_taken += 1

        inter_vessel_distance = self._inter_vessel_distance()
        collision = inter_vessel_distance <= self.envp.collision_distance
        near_miss = (not collision) and (inter_vessel_distance <= self.envp.near_miss_distance)

        self.time += self.envp.dt
        self.step_idx += 1

        done = False
        reason = ""
        if collision:
            done, reason = True, "collision"
        elif self._outside(self.vessel1) or self._outside(self.vessel2):
            done, reason = True, "out_of_bounds"
        elif self.step_idx >= self.max_steps:
            done, reason = True, "timeout"
        elif self.vessel1_reached and self.vessel2_reached:
            done, reason = True, "both_reached"

        reward = self.rewp.living_penalty
        d_vessel1 = self._goal_distance(self.vessel1)
        d_vessel2 = self._goal_distance(self.vessel2)
        reward += self.rewp.progress_weight * (self.prev_goal_d_vessel1 - d_vessel1)
        reward += self.rewp.progress_weight * (self.prev_goal_d_vessel2 - d_vessel2)

        if self.vessel1_reached and self.vessel2_reached and reason == "both_reached":
            reward += self.rewp.goal_bonus

        if reason == "out_of_bounds":
            reward += self.rewp.out_of_bounds_penalty
        if reason == "collision":
            reward += self.rewp.collision_penalty

        if near_miss:
            reward += self.rewp.near_miss_penalty

        if inter_vessel_distance < self.envp.safe_pass_distance:
            reward -= self.rewp.unsafe_proximity_penalty_weight * (self.envp.safe_pass_distance - inter_vessel_distance)

        if self.risk_of_collision:
            self.encounter_was_risky = True

        if self.encounter_was_risky and (not self.risk_of_collision) and (inter_vessel_distance > self.envp.safe_pass_distance) and (not self.safe_pass_awarded):
            reward += self.rewp.safe_pass_bonus
            self.safe_pass_awarded = True

        if self.colregs_scenario in {"crossing", "head_on", "overtaking"} and self.risk_of_collision:
            if self.vessel1_role == "give_way" and not self.vessel1_giveway_action_awarded:
                if self.vessel1_control_source in {"starboard_avoid", "rl_external", "rl_internal"} and abs(self.vessel1.rudder) > 0.1:
                    if tcpa > self.envp.standon_escalation_tcpa:
                        reward += self.rewp.give_way_early_action_bonus
                    else:
                        reward += self.rewp.late_action_penalty
                    self.vessel1_giveway_action_awarded = True
            if self.vessel2_role == "give_way" and not self.vessel2_giveway_action_awarded:
                if self.vessel2_control_source in {"starboard_avoid", "rl_external", "rl_internal"} and abs(self.vessel2.rudder) > 0.1:
                    if tcpa > self.envp.standon_escalation_tcpa:
                        reward += self.rewp.give_way_early_action_bonus
                    else:
                        reward += self.rewp.late_action_penalty
                    self.vessel2_giveway_action_awarded = True

            if self.vessel1_role == "stand_on" and not self.vessel1_standon_escalated and not self.vessel1_standon_hold_awarded:
                if self.vessel1_control_source == "hold_course_speed":
                    reward += self.rewp.stand_on_hold_bonus
                    self.vessel1_standon_hold_awarded = True
                elif abs(self.vessel1.rudder) > 0.1:
                    reward += self.rewp.stand_on_unnecessary_action_penalty
            if self.vessel2_role == "stand_on" and not self.vessel2_standon_escalated and not self.vessel2_standon_hold_awarded:
                if self.vessel2_control_source == "hold_course_speed":
                    reward += self.rewp.stand_on_hold_bonus
                    self.vessel2_standon_hold_awarded = True
                elif abs(self.vessel2.rudder) > 0.1:
                    reward += self.rewp.stand_on_unnecessary_action_penalty

            if self.colregs_scenario == "crossing":
                if self.vessel1_role == "give_way" and tcpa <= self.envp.standon_escalation_tcpa and abs(self.vessel1_relative_bearing_deg) < self.envp.colregs_crossing_starboard_max_deg:
                    reward += self.rewp.crossing_ahead_penalty
                if self.vessel2_role == "give_way" and tcpa <= self.envp.standon_escalation_tcpa and abs(self.vessel2_relative_bearing_deg) < self.envp.colregs_crossing_starboard_max_deg:
                    reward += self.rewp.crossing_ahead_penalty

        vessel1_rudder_sign = 1 if self.vessel1.rudder > 1e-3 else -1 if self.vessel1.rudder < -1e-3 else 0
        vessel2_rudder_sign = 1 if self.vessel2.rudder > 1e-3 else -1 if self.vessel2.rudder < -1e-3 else 0
        if self.prev_vessel1_rudder_sign != 0 and vessel1_rudder_sign != 0 and vessel1_rudder_sign != self.prev_vessel1_rudder_sign:
            reward -= self.rewp.oscillation_penalty_weight
        if self.prev_vessel2_rudder_sign != 0 and vessel2_rudder_sign != 0 and vessel2_rudder_sign != self.prev_vessel2_rudder_sign:
            reward -= self.rewp.oscillation_penalty_weight
        self.prev_vessel1_rudder_sign = vessel1_rudder_sign
        self.prev_vessel2_rudder_sign = vessel2_rudder_sign
        self.last_inter_vessel_distance = inter_vessel_distance

        self.prev_goal_d_vessel1 = d_vessel1
        self.prev_goal_d_vessel2 = d_vessel2

        info: Dict[str, float | str | int] = {
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
            "vessel1_rl_latched": int(self.vessel1_rl_latched),
            "vessel2_rl_latched": int(self.vessel2_rl_latched),
            "vessel1_distance_from_start": float(vessel1_dist),
            "vessel2_distance_from_start": float(vessel2_dist),
            "vessel1_relative_bearing_deg": float(self.vessel1_relative_bearing_deg),
            "vessel2_relative_bearing_deg": float(self.vessel2_relative_bearing_deg),
            "inter_vessel_distance": float(inter_vessel_distance),
            "collision": int(collision),
            "near_miss": int(near_miss),
            "safe_pass_awarded": int(self.safe_pass_awarded),
            "vessel1_control_source": self.vessel1_control_source,
            "vessel2_control_source": self.vessel2_control_source,
            "vessel1_standon_escalated": int(self.vessel1_standon_escalated),
            "vessel2_standon_escalated": int(self.vessel2_standon_escalated),
        }

        # Show the RL takeover overlay exactly once per episode, on the first step RL activates.
        if (
            self.render_enabled
            and not self.rl_overlay_shown
            and self.rl_ever_triggered
            and (self.vessel1_rl_active or self.vessel2_rl_active)
        ):
            self.rl_overlay_shown = True
            self.paused = True
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
                "vessel1_rl_latched": int(self.vessel1_rl_latched),
                "vessel2_rl_latched": int(self.vessel2_rl_latched),
                "vessel1_control_source": self.vessel1_control_source,
                "vessel2_control_source": self.vessel2_control_source,
                "vessel1_standon_escalated": int(self.vessel1_standon_escalated),
                "vessel2_standon_escalated": int(self.vessel2_standon_escalated),
                "vessel1_distance": float(vessel1_dist),
                "vessel2_distance": float(vessel2_dist),
                "takeover_distance": float(self.envp.rl_takeover_distance),
            }

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

        self._draw_goal(self.vessel1.goal_x, self.vessel1.goal_y, (250, 215, 60))
        self._draw_goal(self.vessel2.goal_x, self.vessel2.goal_y, (255, 140, 90))

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

        self._draw_vessel(self.vessel1, (95, 170, 255), "V1")
        self._draw_vessel(self.vessel2, (255, 120, 120), "V2")

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
