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
    """Two-vessel setup: vessel-1 straight center->circle goal, vessel-2 follows pure pursuit path."""

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
        self.agent_start_heading = 0.0
        self.target_start_heading = 0.0

        # Vessel-2 scripted path state

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
        self.overtaking_latched = False
        self.latched_scenario = "safe"
        self.latched_agent_role = "none"
        self.latched_target_role = "none"
        self.overtaking_clear_steps = 0
        self.encounter_latched = False
        self.latched_encounter_active = False
        self.latched_geometry = "none"
        self.encounter_clear_steps = 0
        self.designated_agent_role = "none"
        self.designated_target_role = "none"
        self.rl_controlled_vessel = "none"
        self.candidate_scenario = "safe"
        self.candidate_agent_role = "none"
        self.candidate_target_role = "none"
        self.candidate_steps = 0
        self.active_non_overtaking_scenario = "safe"
        self.active_non_overtaking_agent_role = "none"
        self.active_non_overtaking_target_role = "none"
        self.active_non_overtaking_exit_steps = 0
        self.geometry_scenario = "none"
        self.agent_standon_escalated = False
        self.target_standon_escalated = False
        self.agent_standon_risk_steps = 0
        self.target_standon_risk_steps = 0
        self.agent_control_source = "straight"
        self.target_control_source = "pure_pursuit"
        self.agent_rl_latched = False
        self.target_rl_latched = False
        self.any_rl_ever_triggered = False
        self.latched_encounter_active = False
        self.latched_geometry = "none"
        self.encounter_clear_steps = 0
        self.designated_agent_role = "none"
        self.designated_target_role = "none"
        self.rl_controlled_vessel = "none"
        self.secondary_policy_fn = None
        self.last_inter_vessel_distance = float("inf")
        self.encounter_was_risky = False
        self.safe_pass_awarded = False
        self.agent_giveway_action_awarded = False
        self.target_giveway_action_awarded = False
        self.agent_standon_hold_awarded = False
        self.target_standon_hold_awarded = False
        self.prev_agent_rudder_sign = 0
        self.prev_target_rudder_sign = 0
        self.candidate_scenario = "safe"
        self.candidate_agent_role = "none"
        self.candidate_target_role = "none"
        self.candidate_steps = 0
        self.active_non_overtaking_scenario = "safe"
        self.active_non_overtaking_agent_role = "none"
        self.active_non_overtaking_target_role = "none"
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
        return math.hypot(self.target.x - self.agent.x, self.target.y - self.agent.y)

    def _relative_bearing_deg(self, observer: Vessel, target: Vessel) -> float:
        """Signed relative bearing in degrees.

        Positive values mean the target is to starboard (right) of observer bow,
        negative values mean port (left), and 0 means dead ahead.
        """
        dx = target.x - observer.x
        dy = target.y - observer.y
        ch = math.cos(observer.h)
        sh = math.sin(observer.h)
        x_rel = ch * dx + sh * dy
        y_rel = -sh * dx + ch * dy
        # Signed from observer-forward axis with +starboard / -port convention.
        return math.degrees(math.atan2(-y_rel, x_rel))

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

    def _classify_pair_geometry(self, agent: Vessel, target: Vessel) -> Dict[str, float | str]:
        own_bearing = self._relative_bearing_deg(agent, target)
        tgt_bearing = self._relative_bearing_deg(target, agent)

        head_on_half = self.envp.colregs_head_on_half_angle_deg
        head_on_min = (360.0 - head_on_half) % 360.0
        crossing_max = self.envp.colregs_crossing_starboard_max_deg
        overtaking_max = self.envp.colregs_overtaking_aft_max_deg

        agent_overtaking = self._bearing_in_sector(
            tgt_bearing, crossing_max, overtaking_max, inclusive=False
        ) and self._is_closing(agent, target)
        target_overtaking = self._bearing_in_sector(
            own_bearing, crossing_max, overtaking_max, inclusive=False
        ) and self._is_closing(target, agent)
        if agent_overtaking and not target_overtaking:
            return {
                "geometry": "overtaking_agent_geom",
                "agent_bearing_deg": own_bearing,
                "target_bearing_deg": tgt_bearing,
            }
        if target_overtaking and not agent_overtaking:
            return {
                "geometry": "overtaking_target_geom",
                "agent_bearing_deg": own_bearing,
                "target_bearing_deg": tgt_bearing,
            }

        head_on = self._bearing_in_sector(own_bearing, head_on_min, head_on_half) and self._bearing_in_sector(
            tgt_bearing, head_on_min, head_on_half
        )
        if head_on:
            return {
                "geometry": "head_on_geom",
                "agent_bearing_deg": own_bearing,
                "target_bearing_deg": tgt_bearing,
            }

        if self._bearing_in_sector(own_bearing, head_on_half, crossing_max):
            return {
                "geometry": "crossing_agent_give_way_geom",
                "agent_bearing_deg": own_bearing,
                "target_bearing_deg": tgt_bearing,
            }

        if self._bearing_in_sector(own_bearing, 360.0 - crossing_max, head_on_min):
            return {
                "geometry": "crossing_agent_stand_on_geom",
                "agent_bearing_deg": own_bearing,
                "target_bearing_deg": tgt_bearing,
            }

        return {
            "geometry": "none",
            "agent_bearing_deg": own_bearing,
            "target_bearing_deg": tgt_bearing,
        }

    def _assess_pair_risk(self, agent: Vessel, target: Vessel) -> Dict[str, float | bool]:
        tcpa, dcpa = self._tcpa_dcpa(agent, target)
        risk_of_collision = (0.0 <= tcpa <= self.envp.tcpa_risk_threshold) and (dcpa <= self.envp.dcpa_risk_threshold)
        return {
            "tcpa": tcpa,
            "dcpa": dcpa,
            "risk_of_collision": risk_of_collision,
        }

    def _resolve_colregs_pair(self, agent: Vessel, target: Vessel) -> Dict[str, float | str | bool]:
        geom = self._classify_pair_geometry(agent, target)
        risk = self._assess_pair_risk(agent, target)
        geometry = str(geom["geometry"])
        risk_of_collision = bool(risk["risk_of_collision"])
        sep = math.hypot(target.x - agent.x, target.y - agent.y)

        raw_scenario = "safe"
        raw_agent_role = "none"
        raw_target_role = "none"

        if geometry == "head_on_geom" and risk_of_collision:
            raw_scenario = "head_on"
            if self.envp.simplified_head_on_single_giveway:
                raw_agent_role = "give_way"
                raw_target_role = "stand_on"
            else:
                raw_agent_role = "give_way"
                raw_target_role = "give_way"
        elif geometry == "crossing_agent_give_way_geom" and risk_of_collision:
            raw_scenario = "crossing"
            raw_agent_role = "give_way"
            raw_target_role = "stand_on"
        elif geometry == "crossing_agent_stand_on_geom" and risk_of_collision:
            raw_scenario = "crossing"
            raw_agent_role = "stand_on"
            raw_target_role = "give_way"
        elif geometry == "overtaking_agent_geom" and risk_of_collision:
            raw_scenario = "overtaking"
            raw_agent_role = "give_way"
            raw_target_role = "stand_on"
        elif geometry == "overtaking_target_geom" and risk_of_collision:
            raw_scenario = "overtaking"
            raw_agent_role = "stand_on"
            raw_target_role = "give_way"
        elif geometry in {
            "head_on_geom",
            "crossing_agent_give_way_geom",
            "crossing_agent_stand_on_geom",
            "overtaking_agent_geom",
            "overtaking_target_geom",
        }:
            raw_scenario = "no_risk"

        if not self.latched_encounter_active and raw_scenario in {"head_on", "crossing", "overtaking"}:
            self.latched_encounter_active = True
            self.encounter_latched = True
            self.latched_geometry = geometry
            self.latched_scenario = raw_scenario
            self.latched_agent_role = raw_agent_role
            self.latched_target_role = raw_target_role
            self.designated_agent_role = raw_agent_role
            self.designated_target_role = raw_target_role
            self.overtaking_latched = raw_scenario == "overtaking"
            self.encounter_clear_steps = 0
            self.safe_pass_awarded = False
            self.encounter_was_risky = True

        scenario = raw_scenario
        agent_role = raw_agent_role
        target_role = raw_target_role
        encounter_latched = False

        if self.latched_encounter_active:
            scenario = self.latched_scenario
            agent_role = self.latched_agent_role
            target_role = self.latched_target_role
            encounter_latched = True
            if (not risk_of_collision) and (sep > self.envp.safe_pass_distance):
                self.encounter_clear_steps += 1
            else:
                self.encounter_clear_steps = 0

            if self.encounter_clear_steps >= 3:
                self.latched_encounter_active = False
                self.encounter_latched = False
                self.overtaking_latched = False
                self.latched_geometry = "none"
                self.latched_scenario = "safe"
                self.latched_agent_role = "none"
                self.latched_target_role = "none"
                self.encounter_clear_steps = 0
                self.agent_giveway_action_awarded = False
                self.target_giveway_action_awarded = False
                self.agent_standon_hold_awarded = False
                self.target_standon_hold_awarded = False
                scenario = raw_scenario
                agent_role = raw_agent_role
                target_role = raw_target_role
                encounter_latched = False

        if self.encounter_was_risky and agent_role == "none" and target_role == "none":
            agent_role = self.designated_agent_role
            target_role = self.designated_target_role

        return {
            "geometry": geometry,
            "scenario": scenario,
            "agent_role": agent_role,
            "target_role": target_role,
            "agent_bearing_deg": float(geom["agent_bearing_deg"]),
            "target_bearing_deg": float(geom["target_bearing_deg"]),
            "tcpa": float(risk["tcpa"]),
            "dcpa": float(risk["dcpa"]),
            "risk_of_collision": risk_of_collision,
            "encounter_latched": encounter_latched,
            "overtaking_latched": self.overtaking_latched,
        }

    def _apply_non_overtaking_hysteresis(
        self, scenario: str, agent_role: str, target_role: str
    ) -> Tuple[str, str, str]:
        if scenario not in {"head_on", "crossing"}:
            self.candidate_scenario = "safe"
            self.candidate_agent_role = "none"
            self.candidate_target_role = "none"
            self.candidate_steps = 0
            if self.active_non_overtaking_scenario in {"head_on", "crossing"}:
                self.active_non_overtaking_exit_steps += 1
                if self.active_non_overtaking_exit_steps < max(1, int(self.envp.encounter_exit_persistence_steps)):
                    return (
                        self.active_non_overtaking_scenario,
                        self.active_non_overtaking_agent_role,
                        self.active_non_overtaking_target_role,
                    )
            self.active_non_overtaking_scenario = "safe"
            self.active_non_overtaking_agent_role = "none"
            self.active_non_overtaking_target_role = "none"
            self.active_non_overtaking_exit_steps = 0
            return scenario, agent_role, target_role

        self.active_non_overtaking_exit_steps = 0
        if (
            self.candidate_scenario == scenario
            and self.candidate_agent_role == agent_role
            and self.candidate_target_role == target_role
        ):
            self.candidate_steps += 1
        else:
            self.candidate_scenario = scenario
            self.candidate_agent_role = agent_role
            self.candidate_target_role = target_role
            self.candidate_steps = 1

        if self.candidate_steps >= max(1, int(self.envp.encounter_enter_persistence_steps)):
            self.active_non_overtaking_scenario = scenario
            self.active_non_overtaking_agent_role = agent_role
            self.active_non_overtaking_target_role = target_role
            return scenario, agent_role, target_role

        if self.active_non_overtaking_scenario in {"head_on", "crossing"}:
            return (
                self.active_non_overtaking_scenario,
                self.active_non_overtaking_agent_role,
                self.active_non_overtaking_target_role,
            )
        return "no_risk", "none", "none"

    def _classify_colregs(self) -> Dict[str, float | str | bool]:
        if self.agent_reached or self.target_reached:
            fallback_agent_role = self.designated_agent_role if self.encounter_was_risky else "none"
            fallback_target_role = self.designated_target_role if self.encounter_was_risky else "none"
            if self.rl_controlled_vessel == "agent":
                fallback_agent_role = "give_way"
                fallback_target_role = "stand_on" if fallback_target_role == "none" else fallback_target_role
            elif self.rl_controlled_vessel == "target":
                fallback_target_role = "give_way"
                fallback_agent_role = "stand_on" if fallback_agent_role == "none" else fallback_agent_role
            return {
                "geometry": "none",
                "scenario": "safe",
                "agent_role": fallback_agent_role,
                "target_role": fallback_target_role,
                "agent_bearing_deg": 0.0,
                "target_bearing_deg": 0.0,
                "tcpa": float("inf"),
                "dcpa": float("inf"),
                "risk_of_collision": False,
                "encounter_latched": self.encounter_latched,
                "overtaking_latched": self.overtaking_latched,
            }
        return self._resolve_colregs_pair(self.agent, self.target)

    def _reset_sample_triggers_takeover(self, agent: Vessel, target: Vessel) -> bool:
        saved_latch_state = (
            self.overtaking_latched,
            self.latched_scenario,
            self.latched_agent_role,
            self.latched_target_role,
            self.overtaking_clear_steps,
            self.encounter_latched,
            self.latched_encounter_active,
            self.latched_geometry,
            self.encounter_clear_steps,
            self.designated_agent_role,
            self.designated_target_role,
            self.rl_controlled_vessel,
            self.candidate_scenario,
            self.candidate_agent_role,
            self.candidate_target_role,
            self.candidate_steps,
            self.active_non_overtaking_scenario,
            self.active_non_overtaking_agent_role,
            self.active_non_overtaking_target_role,
            self.active_non_overtaking_exit_steps,
        )

        def _restore_latch_state() -> None:
            (
                self.overtaking_latched,
                self.latched_scenario,
                self.latched_agent_role,
                self.latched_target_role,
                self.overtaking_clear_steps,
                self.encounter_latched,
                self.latched_encounter_active,
                self.latched_geometry,
                self.encounter_clear_steps,
                self.designated_agent_role,
                self.designated_target_role,
                self.rl_controlled_vessel,
                self.candidate_scenario,
                self.candidate_agent_role,
                self.candidate_target_role,
                self.candidate_steps,
                self.active_non_overtaking_scenario,
                self.active_non_overtaking_agent_role,
                self.active_non_overtaking_target_role,
                self.active_non_overtaking_exit_steps,
            ) = saved_latch_state

        self.overtaking_latched = False
        self.latched_scenario = "safe"
        self.latched_agent_role = "none"
        self.latched_target_role = "none"
        self.overtaking_clear_steps = 0
        self.encounter_latched = False
        self.latched_encounter_active = False
        self.latched_geometry = "none"
        self.encounter_clear_steps = 0
        self.designated_agent_role = "none"
        self.designated_target_role = "none"
        self.rl_controlled_vessel = "none"

        agent_sim = Vessel(agent.x, agent.y, agent.h, agent.speed, agent.goal_x, agent.goal_y, agent.rudder, agent.throttle)
        target_sim = Vessel(target.x, target.y, target.h, target.speed, target.goal_x, target.goal_y, target.rudder, target.throttle)

        agent_reached = False
        target_reached = False
        agent_start = (agent_sim.x, agent_sim.y)
        target_start = (target_sim.x, target_sim.y)
        h = self.envp.dt / max(1, self.envp.substeps)

        takeover_viable = False
        min_separation = float("inf")

        for _ in range(self.max_steps):
            if agent_reached or target_reached:
                encounter = {
                    "agent_role": "none",
                    "target_role": "none",
                    "risk_of_collision": False,
                    "dcpa": float("inf"),
                    "tcpa": float("inf"),
                }
            else:
                encounter = self._resolve_colregs_pair(agent_sim, target_sim)

            separation = math.hypot(target_sim.x - agent_sim.x, target_sim.y - agent_sim.y)
            min_separation = min(min_separation, separation)

            risk_now = bool(encounter.get("risk_of_collision", False))

            agent_dist = self._distance_from_start(agent_sim, agent_start)
            target_dist = self._distance_from_start(target_sim, target_start)
            if (
                risk_now
                and encounter["agent_role"] == "give_way"
                and not agent_reached
                and agent_dist >= self.envp.rl_takeover_distance
            ):
                takeover_viable = True
            if (
                risk_now
                and encounter["target_role"] == "give_way"
                and not target_reached
                and target_dist >= self.envp.rl_takeover_distance
            ):
                takeover_viable = True

            for _ in range(max(1, self.envp.substeps)):
                if not agent_reached:
                    d_agent = math.hypot(agent_sim.goal_x - agent_sim.x, agent_sim.goal_y - agent_sim.y)
                    if d_agent <= self.envp.goal_radius:
                        agent_reached = True
                        agent_sim.speed = 0.0
                    else:
                        travel = min(agent_sim.speed * h, d_agent)
                        agent_sim.x += math.cos(agent_sim.h) * travel
                        agent_sim.y += math.sin(agent_sim.h) * travel
                        if math.hypot(agent_sim.goal_x - agent_sim.x, agent_sim.goal_y - agent_sim.y) <= self.envp.goal_radius:
                            agent_reached = True
                            agent_sim.speed = 0.0

                if not target_reached:
                    d_target = math.hypot(target_sim.goal_x - target_sim.x, target_sim.goal_y - target_sim.y)
                    if d_target <= self.envp.goal_radius:
                        target_reached = True
                        target_sim.speed = 0.0
                    else:
                        rudder_cmd = self._pure_pursuit_rudder_cmd(target_sim, target_sim.goal_x, target_sim.goal_y)
                        self._integrate_rudder_heading(target_sim, rudder_cmd, h)
                        target_sim.speed = clamp(target_sim.speed, self.envp.target_min_speed, self.envp.target_max_speed)
                        travel = min(target_sim.speed * h, d_target)
                        target_sim.x += travel * math.cos(target_sim.h)
                        target_sim.y += travel * math.sin(target_sim.h)
                        if math.hypot(target_sim.goal_x - target_sim.x, target_sim.goal_y - target_sim.y) <= self.envp.goal_radius:
                            target_reached = True
                            target_sim.speed = 0.0

            if (agent_reached and target_reached) or self._outside(agent_sim) or self._outside(target_sim):
                break

        unavoidable_hazard = min_separation <= self.envp.near_miss_distance
        qualifies = takeover_viable and unavoidable_hazard
        _restore_latch_state()
        return qualifies

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

    def _pure_pursuit_rudder_cmd(self, v: Vessel, goal_x: float, goal_y: float) -> float:
        """Pure-pursuit rudder command that steers vessel directly toward goal position."""
        turning_radius = v.speed / max(self.envp.rudder_max_yaw_rate_rad_s, 1e-6)
        lookahead_dist = max(1e-6, self.envp.pp_lookahead_factor * turning_radius)

        bearing_to_goal = math.atan2(goal_y - v.y, goal_x - v.x)
        target_x = v.x + lookahead_dist * math.cos(bearing_to_goal)
        target_y = v.y + lookahead_dist * math.sin(bearing_to_goal)

        bearing = math.atan2(target_y - v.y, target_x - v.x)
        heading_error = wrap_pi(bearing - v.h)
        return clamp(
            heading_error / math.radians(self.envp.pp_heading_gain_deg),
            -1.0,
            1.0,
        )

    def _sample_target_path(self) -> Vessel:
        # Vessel 2: random start/goal on big circle with inward-facing start heading.
        # Goal is sampled to remain beyond a minimum start->goal distance for local pure-pursuit behavior checks.
        start_ang_2 = self.rng.uniform(0.0, 2.0 * math.pi)
        goal_ang_2 = self.rng.uniform(0.0, 2.0 * math.pi)

        sx2, sy2 = self._point_on_big_circle(start_ang_2)
        gx2, gy2 = self._point_on_big_circle(goal_ang_2)

        sh2 = self._inward_facing_heading(sx2, sy2)
        sp2 = self.rng.uniform(self.envp.target_min_speed, self.envp.target_max_speed)

        tries = 0
        min_goal_chord_dist = max(0.0, float(self.envp.target_min_goal_arc_distance_from_start))
        # Backward compatibility: allow legacy parameter names to override when explicitly provided.
        if self.envp.target_max_goal_arc_distance_from_start is not None:
            min_goal_chord_dist = max(0.0, float(self.envp.target_max_goal_arc_distance_from_start))
        if self.envp.target_max_goal_distance_from_start is not None:
            min_goal_chord_dist = max(0.0, float(self.envp.target_max_goal_distance_from_start))

        if bool(self.envp.adaptive_target_min_goal_arc_from_speed):
            omega_max = max(1e-9, float(self.envp.rudder_max_yaw_rate_rad_s))
            dcrit_chord = max(0.0, float(self.envp.target_min_goal_dcrit_factor)) * (2.0 * sp2 / omega_max)
            min_goal_chord_dist = max(min_goal_chord_dist, dcrit_chord)

        while math.hypot(gx2 - sx2, gy2 - sy2) < min_goal_chord_dist and tries < 40:
            goal_ang_2 = self.rng.uniform(0.0, 2.0 * math.pi)
            gx2, gy2 = self._point_on_big_circle(goal_ang_2)
            tries += 1

        return Vessel(sx2, sy2, sh2, sp2, gx2, gy2)

    def _advance_target(self, dt: float) -> None:
        if self.target is None or self.target_reached:
            return

        d_goal = self._goal_distance(self.target)
        if d_goal <= self.envp.goal_radius:
            self.target_reached = True
            self.target.speed = 0.0
            return

        rudder_cmd = self._pure_pursuit_rudder_cmd(self.target, self.target.goal_x, self.target.goal_y)

        self._integrate_rudder_heading(self.target, rudder_cmd, dt)

        # hold constant scripted speed in nominal mode
        self.target.speed = clamp(self.target.speed, self.envp.target_min_speed, self.envp.target_max_speed)

        d_goal = self._goal_distance(self.target)
        travel = min(self.target.speed * dt, d_goal)
        self.target.x += travel * math.cos(self.target.h)
        self.target.y += travel * math.sin(self.target.h)

        if self._goal_distance(self.target) <= self.envp.goal_radius:
            self.target_reached = True
            self.target.speed = 0.0

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

        if vessel_name == "target":
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

    def _build_agent_planned_path(self) -> None:
        if self.agent is None:
            self.agent_planned_path = []
            return
        self.agent_planned_path = [(self.start_x, self.start_y), (self.agent.goal_x, self.agent.goal_y)]

    def _build_target_planned_path(self, sx: float, sy: float, sh: float, speed: float, goal_x: float, goal_y: float) -> None:
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

        self.target_planned_path = pts

    def _get_obs_for_perspective(self, own: Vessel, other: Vessel, own_is_agent: bool) -> np.ndarray:
        own_speed_den = self.envp.max_speed if own_is_agent else self.envp.target_max_speed
        other_speed_den = self.envp.target_max_speed if own_is_agent else self.envp.max_speed
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

    def _get_obs_agent_perspective(self) -> np.ndarray:
        return self._get_obs_for_perspective(self.agent, self.target, own_is_agent=True)

    def _get_obs_target_perspective(self) -> np.ndarray:
        return self._get_obs_for_perspective(self.target, self.agent, own_is_agent=False)

    def get_obs(self) -> np.ndarray:
        return self._get_obs_agent_perspective()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)

        max_tries = max(1, int(self.envp.reset_viable_episode_max_tries))
        require_viable_path = bool(self.envp.require_reset_viable_takeover_path)
        sampled_agent: Vessel | None = None
        sampled_target: Vessel | None = None
        for _ in range(max_tries):
            goal_ang_1 = self.rng.uniform(0.0, 2.0 * math.pi)
            agx, agy = self._point_on_big_circle(goal_ang_1)
            ah = math.atan2(agy - self.start_y, agx - self.start_x)
            aspeed = self.rng.uniform(self.envp.min_speed, self.envp.max_speed)
            candidate_agent = Vessel(self.start_x, self.start_y, ah, aspeed, agx, agy)
            candidate_target = self._sample_target_path()
            if (not require_viable_path) or self._reset_sample_triggers_takeover(candidate_agent, candidate_target):
                sampled_agent = candidate_agent
                sampled_target = candidate_target
                break

        self.reset_has_takeover_path = sampled_agent is not None and sampled_target is not None
        if sampled_agent is None or sampled_target is None:
            sampled_agent = candidate_agent
            sampled_target = candidate_target

        self.agent = sampled_agent
        self.target = sampled_target

        sx2, sy2 = self.target.x, self.target.y
        sh2 = self.target.h
        sp2 = self.target.speed
        gx2, gy2 = self.target.goal_x, self.target.goal_y

        self.agent_start_speed = self.agent.speed
        self.agent_start_pos = (self.agent.x, self.agent.y)
        self.agent_start_heading = self.agent.h
        self.target_start_speed = sp2
        self.target_start_pos = (self.target.x, self.target.y)
        self.target_start_heading = self.target.h

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
        self.overtaking_latched = False
        self.latched_scenario = "safe"
        self.latched_agent_role = "none"
        self.latched_target_role = "none"
        self.overtaking_clear_steps = 0
        self.encounter_latched = False
        self.geometry_scenario = "none"
        self.agent_standon_escalated = False
        self.target_standon_escalated = False
        self.agent_standon_risk_steps = 0
        self.target_standon_risk_steps = 0
        self.agent_control_source = "straight"
        self.target_control_source = "pure_pursuit"
        self.agent_rl_latched = False
        self.target_rl_latched = False
        self.any_rl_ever_triggered = False
        self.latched_encounter_active = False
        self.latched_geometry = "none"
        self.encounter_clear_steps = 0
        self.designated_agent_role = "none"
        self.designated_target_role = "none"
        self.rl_controlled_vessel = "none"
        self.last_inter_vessel_distance = float("inf")
        self.encounter_was_risky = False
        self.safe_pass_awarded = False
        self.agent_giveway_action_awarded = False
        self.target_giveway_action_awarded = False
        self.agent_standon_hold_awarded = False
        self.target_standon_hold_awarded = False
        self.prev_agent_rudder_sign = 0
        self.prev_target_rudder_sign = 0
        self.candidate_scenario = "safe"
        self.candidate_agent_role = "none"
        self.candidate_target_role = "none"
        self.candidate_steps = 0
        self.active_non_overtaking_scenario = "safe"
        self.active_non_overtaking_agent_role = "none"
        self.active_non_overtaking_target_role = "none"
        self.active_non_overtaking_exit_steps = 0

        # Episode termination is fixed-time or both-reached (whichever occurs first).
        self.max_steps = max(1, int(round(self.envp.episode_seconds / self.envp.dt)))

        self._build_agent_planned_path()
        self._build_target_planned_path(sx2, sy2, sh2, sp2, gx2, gy2)
        self.last_inter_vessel_distance = self._inter_vessel_distance()

        return self.get_obs()

    def _select_rl_action_for_vessel(
        self, vessel_name: str, external_action: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float]], str]:
        if vessel_name == "agent" and not self.agent_rl_active:
            return None, ""
        if vessel_name == "target" and not self.target_rl_active:
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
        give_way_vessel = "agent" if self.agent_role == "give_way" else "target" if self.target_role == "give_way" else "none"
        stand_on_vessel = "agent" if self.agent_role == "stand_on" else "target" if self.target_role == "stand_on" else "none"
        stand_on_nominal_mode = "pure_pursuit" if stand_on_vessel == "target" else "straight" if stand_on_vessel == "agent" else "none"

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
                "geometry_scenario": self.geometry_scenario,
                "encounter_latched": int(self.encounter_latched),
                "overtaking_latched": int(self.overtaking_latched),
                "latched_scenario": self.latched_scenario,
                "agent_role": self.agent_role,
                "target_role": self.target_role,
                "designated_give_way_vessel": give_way_vessel,
                "designated_stand_on_vessel": stand_on_vessel,
                "stand_on_nominal_mode": stand_on_nominal_mode,
                "agent_rl_active": int(self.agent_rl_active),
                "target_rl_active": int(self.target_rl_active),
                "agent_rl_latched": int(self.agent_rl_latched),
                "target_rl_latched": int(self.target_rl_latched),
                "agent_distance_from_start": float(self._distance_from_start(self.agent, self.agent_start_pos)),
                "target_distance_from_start": float(self._distance_from_start(self.target, self.target_start_pos)),
                "agent_relative_bearing_deg": float(self.agent_relative_bearing_deg),
                "target_relative_bearing_deg": float(self.target_relative_bearing_deg),
                "agent_control_source": self.agent_control_source,
                "target_control_source": self.target_control_source,
                "agent_standon_escalated": int(self.agent_standon_escalated),
                "target_standon_escalated": int(self.target_standon_escalated),
                "inter_vessel_distance": float(self._inter_vessel_distance()),
                "collision": 0,
                "near_miss": int(self._inter_vessel_distance() <= self.envp.near_miss_distance),
                "safe_pass_awarded": int(self.safe_pass_awarded),
            }
            return self.get_obs(), 0.0, False, info

        encounter = self._classify_colregs()
        self.colregs_scenario = str(encounter["scenario"])
        self.geometry_scenario = str(encounter["geometry"])
        self.agent_role = str(encounter["agent_role"])
        self.target_role = str(encounter["target_role"])
        self.overtaking_latched = bool(encounter["overtaking_latched"])
        self.encounter_latched = bool(encounter["encounter_latched"])
        self.agent_relative_bearing_deg = float(encounter["agent_bearing_deg"])
        self.target_relative_bearing_deg = float(encounter["target_bearing_deg"])
        self.last_tcpa = float(encounter["tcpa"])
        self.last_dcpa = float(encounter["dcpa"])
        self.risk_of_collision = bool(encounter["risk_of_collision"])
        tcpa = self.last_tcpa
        dcpa = self.last_dcpa
        give_way_vessel = "agent" if self.agent_role == "give_way" else "target" if self.target_role == "give_way" else "none"
        stand_on_vessel = "agent" if self.agent_role == "stand_on" else "target" if self.target_role == "stand_on" else "none"
        stand_on_nominal_mode = "pure_pursuit" if stand_on_vessel == "target" else "straight" if stand_on_vessel == "agent" else "none"

        agent_dist = self._distance_from_start(self.agent, self.agent_start_pos)
        target_dist = self._distance_from_start(self.target, self.target_start_pos)

        # Stand-on escalation disabled for this contract: stand-on remains nominal path-following.
        self.agent_standon_risk_steps = 0
        self.target_standon_risk_steps = 0
        self.agent_standon_escalated = False
        self.target_standon_escalated = False

        encounter_active = bool(self.latched_encounter_active)
        allow_agent_rl = encounter_active and self.agent_role == "give_way"
        allow_target_rl = encounter_active and self.target_role == "give_way"

        if self.rl_controlled_vessel == "none":
            if allow_agent_rl and self.risk_of_collision and agent_dist >= self.envp.rl_takeover_distance and not self.agent_reached:
                self.rl_controlled_vessel = "agent"
            elif allow_target_rl and self.risk_of_collision and target_dist >= self.envp.rl_takeover_distance and not self.target_reached:
                self.rl_controlled_vessel = "target"

        self.agent_rl_latched = self.agent_rl_latched or (self.rl_controlled_vessel == "agent")
        self.target_rl_latched = self.target_rl_latched or (self.rl_controlled_vessel == "target")
        self.agent_rl_active = (self.rl_controlled_vessel == "agent") and (not self.agent_reached)
        self.target_rl_active = (self.rl_controlled_vessel == "target") and (not self.target_reached)
        self.any_rl_ever_triggered = self.any_rl_ever_triggered or (self.rl_controlled_vessel != "none")
        self.rl_ever_triggered = self.any_rl_ever_triggered

        early_cutoff_steps = max(1, int(self.envp.no_takeover_early_done_steps))
        if bool(self.envp.enable_no_takeover_early_done) and (not self.reset_has_takeover_path) and (not self.any_rl_ever_triggered) and (self.step_idx + 1) >= early_cutoff_steps:
            self.step_idx += 1
            self.time += self.envp.dt
            d_agent = self._goal_distance(self.agent)
            d_target = self._goal_distance(self.target)
            reward = self.rewp.living_penalty
            reward += self.rewp.progress_weight * (self.prev_goal_d_agent - d_agent)
            reward += self.rewp.progress_weight * (self.prev_goal_d_target - d_target)
            self.prev_goal_d_agent = d_agent
            self.prev_goal_d_target = d_target
            info: Dict[str, float | str | int] = {
                "reason": "no_takeover_trigger",
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
                "geometry_scenario": self.geometry_scenario,
                "encounter_latched": int(self.encounter_latched),
                "overtaking_latched": int(self.overtaking_latched),
                "agent_role": self.agent_role,
                "target_role": self.target_role,
                "agent_rl_active": int(self.agent_rl_active),
                "target_rl_active": int(self.target_rl_active),
                "agent_rl_latched": int(self.agent_rl_latched),
                "target_rl_latched": int(self.target_rl_latched),
                "agent_distance_from_start": float(agent_dist),
                "target_distance_from_start": float(target_dist),
                "agent_relative_bearing_deg": float(self.agent_relative_bearing_deg),
                "target_relative_bearing_deg": float(self.target_relative_bearing_deg),
                "agent_control_source": self.agent_control_source,
                "target_control_source": self.target_control_source,
                "agent_standon_escalated": int(self.agent_standon_escalated),
                "target_standon_escalated": int(self.target_standon_escalated),
            }
            self.prev_agent_rl_active = self.agent_rl_active
            self.prev_target_rl_active = self.target_rl_active
            return self.get_obs(), float(reward), True, info

        h = self.envp.dt / max(1, self.envp.substeps)
        was_agent_active = not self.agent_reached
        was_target_active = not self.target_reached
        encounter_active = bool(self.latched_encounter_active)
        for _ in range(max(1, self.envp.substeps)):
            if self.rl_controlled_vessel == "agent":
                agent_rl_cmd, agent_rl_src = self._select_rl_action_for_vessel("agent", a)
                if agent_rl_cmd is not None:
                    self.agent_control_source = agent_rl_src
                    self._advance_controlled(self.agent, "agent_reached", agent_rl_cmd[0], agent_rl_cmd[1], h)
                else:
                    self.agent_control_source = "straight"
                    self._advance_straight(self.agent, "agent_reached", h)
                self.target_control_source = "pure_pursuit"
                self._advance_target(h)
            elif self.rl_controlled_vessel == "target":
                self.agent_control_source = "straight"
                self._advance_straight(self.agent, "agent_reached", h)
                target_rl_cmd, target_rl_src = self._select_rl_action_for_vessel("target", a)
                if target_rl_cmd is not None:
                    self.target_control_source = target_rl_src
                    self._advance_controlled(self.target, "target_reached", target_rl_cmd[0], target_rl_cmd[1], h)
                else:
                    self.target_control_source = "pure_pursuit"
                    self._advance_target(h)
            else:
                self.agent_control_source = "straight"
                self._advance_straight(self.agent, "agent_reached", h)
                self.target_control_source = "pure_pursuit"
                self._advance_target(h)

        if was_agent_active:
            self.agent_steps_taken += 1
        if was_target_active:
            self.target_steps_taken += 1

        inter_vessel_distance = self._inter_vessel_distance()
        collision = inter_vessel_distance <= self.envp.collision_distance
        near_miss = (not collision) and (inter_vessel_distance <= self.envp.near_miss_distance)

        self.time += self.envp.dt
        self.step_idx += 1

        done = False
        reason = ""
        if collision:
            done, reason = True, "collision"
        elif self._outside(self.agent) or self._outside(self.target):
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
            if self.agent_role == "give_way" and not self.agent_giveway_action_awarded:
                if self.agent_control_source in {"starboard_avoid", "rl_external", "rl_internal"} and abs(self.agent.rudder) > 0.1:
                    if tcpa > self.envp.standon_escalation_tcpa:
                        reward += self.rewp.give_way_early_action_bonus
                    else:
                        reward += self.rewp.late_action_penalty
                    self.agent_giveway_action_awarded = True
            if self.target_role == "give_way" and not self.target_giveway_action_awarded:
                if self.target_control_source in {"starboard_avoid", "rl_external", "rl_internal"} and abs(self.target.rudder) > 0.1:
                    if tcpa > self.envp.standon_escalation_tcpa:
                        reward += self.rewp.give_way_early_action_bonus
                    else:
                        reward += self.rewp.late_action_penalty
                    self.target_giveway_action_awarded = True

            if self.agent_role == "stand_on" and not self.agent_standon_escalated and not self.agent_standon_hold_awarded:
                if self.agent_control_source == "hold_course_speed":
                    reward += self.rewp.stand_on_hold_bonus
                    self.agent_standon_hold_awarded = True
                elif abs(self.agent.rudder) > 0.1:
                    reward += self.rewp.stand_on_unnecessary_action_penalty
            if self.target_role == "stand_on" and not self.target_standon_escalated and not self.target_standon_hold_awarded:
                if self.target_control_source == "hold_course_speed":
                    reward += self.rewp.stand_on_hold_bonus
                    self.target_standon_hold_awarded = True
                elif abs(self.target.rudder) > 0.1:
                    reward += self.rewp.stand_on_unnecessary_action_penalty

            if self.colregs_scenario == "crossing":
                if self.agent_role == "give_way" and tcpa <= self.envp.standon_escalation_tcpa and abs(self.agent_relative_bearing_deg) < self.envp.colregs_crossing_starboard_max_deg:
                    reward += self.rewp.crossing_ahead_penalty
                if self.target_role == "give_way" and tcpa <= self.envp.standon_escalation_tcpa and abs(self.target_relative_bearing_deg) < self.envp.colregs_crossing_starboard_max_deg:
                    reward += self.rewp.crossing_ahead_penalty

        agent_rudder_sign = 1 if self.agent.rudder > 1e-3 else -1 if self.agent.rudder < -1e-3 else 0
        target_rudder_sign = 1 if self.target.rudder > 1e-3 else -1 if self.target.rudder < -1e-3 else 0
        if self.prev_agent_rudder_sign != 0 and agent_rudder_sign != 0 and agent_rudder_sign != self.prev_agent_rudder_sign:
            reward -= self.rewp.oscillation_penalty_weight
        if self.prev_target_rudder_sign != 0 and target_rudder_sign != 0 and target_rudder_sign != self.prev_target_rudder_sign:
            reward -= self.rewp.oscillation_penalty_weight
        self.prev_agent_rudder_sign = agent_rudder_sign
        self.prev_target_rudder_sign = target_rudder_sign
        self.last_inter_vessel_distance = inter_vessel_distance

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
            "geometry_scenario": self.geometry_scenario,
            "encounter_latched": int(self.encounter_latched),
            "overtaking_latched": int(self.overtaking_latched),
            "agent_role": self.agent_role,
            "target_role": self.target_role,
            "latched_scenario": self.latched_scenario,
            "designated_give_way_vessel": give_way_vessel,
            "designated_stand_on_vessel": stand_on_vessel,
            "stand_on_nominal_mode": stand_on_nominal_mode,
            "agent_rl_active": int(self.agent_rl_active),
            "target_rl_active": int(self.target_rl_active),
            "agent_rl_latched": int(self.agent_rl_latched),
            "target_rl_latched": int(self.target_rl_latched),
            "agent_distance_from_start": float(agent_dist),
            "target_distance_from_start": float(target_dist),
            "agent_relative_bearing_deg": float(self.agent_relative_bearing_deg),
            "target_relative_bearing_deg": float(self.target_relative_bearing_deg),
            "inter_vessel_distance": float(inter_vessel_distance),
            "collision": int(collision),
            "near_miss": int(near_miss),
            "safe_pass_awarded": int(self.safe_pass_awarded),
            "agent_control_source": self.agent_control_source,
            "target_control_source": self.target_control_source,
            "agent_standon_escalated": int(self.agent_standon_escalated),
            "target_standon_escalated": int(self.target_standon_escalated),
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
                "geometry": self.geometry_scenario,
                "scenario": self.colregs_scenario,
                "risk_of_collision": int(self.risk_of_collision),
                "encounter_latched": int(self.encounter_latched),
                "overtaking_latched": int(self.overtaking_latched),
                "agent_role": self.agent_role,
                "target_role": self.target_role,
                "dcpa": float(self.last_dcpa),
                "tcpa": float(self.last_tcpa),
                "agent_bearing": float(self.agent_relative_bearing_deg),
                "target_bearing": float(self.target_relative_bearing_deg),
                "agent_rl_active": int(self.agent_rl_active),
                "target_rl_active": int(self.target_rl_active),
                "agent_rl_latched": int(self.agent_rl_latched),
                "target_rl_latched": int(self.target_rl_latched),
                "agent_control_source": self.agent_control_source,
                "target_control_source": self.target_control_source,
                "agent_standon_escalated": int(self.agent_standon_escalated),
                "target_standon_escalated": int(self.target_standon_escalated),
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
        tcpa = float(p.get("tcpa", float("inf")))
        tcpa_txt = "inf" if math.isinf(tcpa) else f"{tcpa:.1f}s"
        agent_active = int(p.get("agent_rl_active", 0))
        target_active = int(p.get("target_rl_active", 0))
        if agent_active and target_active:
            rl_summary = "RL control active on BOTH vessels"
        elif agent_active:
            rl_summary = "RL control active on V1(agent); V2 uses fallback"
        elif target_active:
            rl_summary = "RL control active on V2(target); V1 uses fallback"
        else:
            rl_summary = "No vessel currently under RL control"

        lines = [
            "⚠  RL TAKEOVER / ENCOUNTER STATUS",
            f"Step {int(p.get('step', self.step_idx))}   Sim time {float(p.get('time', self.time)):.1f}s",
            f"Scenario={str(p.get('scenario', self.colregs_scenario)).upper()}",
            f"V1 role={p.get('agent_role', self.agent_role)}",
            f"V2 role={p.get('target_role', self.target_role)}",
            f"DCPA={float(p.get('dcpa', self.last_dcpa)):.1f}m  TCPA={tcpa_txt}  V1→V2={float(p.get('agent_bearing', self.agent_relative_bearing_deg)):.1f}°  V2→V1={float(p.get('target_bearing', self.target_relative_bearing_deg)):.1f}°",
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
            f"step={self.step_idx} t={self.time:.1f}s",
            True, (255, 255, 255),
        )
        hud1 = self._font.render(
            f"DCPA={self.last_dcpa:.1f}m TCPA={tcpa_txt} BRG V1→V2={self.agent_relative_bearing_deg:.1f}° V2→V1={self.target_relative_bearing_deg:.1f}°",
            True, (255, 240, 170),
        )
        hud2 = self._font.render(
            f"V1 spd={self.agent.speed:.2f}",
            True, (170, 220, 255),
        )
        hud3 = self._font.render(
            f"V2 spd={self.target.speed:.2f}",
            True, (255, 190, 190),
        )
        hud4 = self._font.render(
            f"V1 start=({self.agent_start_pos[0]:.1f},{self.agent_start_pos[1]:.1f}) goal=({self.agent.goal_x:.1f},{self.agent.goal_y:.1f}) h0={math.degrees(self.agent_start_heading):.1f}° v0={self.agent_start_speed:.2f}",
            True, (170, 220, 255),
        )
        hud5 = self._font.render(
            f"V2 start=({self.target_start_pos[0]:.1f},{self.target_start_pos[1]:.1f}) goal=({self.target.goal_x:.1f},{self.target.goal_y:.1f}) h0={math.degrees(self.target_start_heading):.1f}° v0={self.target_start_speed:.2f}",
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
