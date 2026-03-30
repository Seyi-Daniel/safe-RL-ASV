"""Standalone per-step crossing-objective reward port for RL training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .crossing_objective_state import CrossingObjectiveState


@dataclass(frozen=True)
class CrossingObjectiveConstants:
    """Old crossing-objective constants, retained exactly for reward-port parity."""

    GOAL_PROGRESS_BONUS: float = -1.2
    HEADING_ALIGNMENT_THRESHOLD_DEG: float = 12.0
    WRONG_ACTION_PENALTY: float = 1.0
    STEP_COST: float = 1.0
    STEP_COUNT_COST: float = 1.0
    GOAL_BONUS: float = -40.0
    TIMEOUT_PENALTY: float = 0.0
    DISTANCE_COST: float = 1.5
    DISTANCE_NORMALISER: float = 250.0
    COLLISION_PENALTY: float = 200.0
    GOAL_TOLERANCE: float = 10.0


class CrossingObjectivePort:
    """Computes dense stepwise RL rewards for the old crossing objective ingredients."""

    def __init__(self, constants: CrossingObjectiveConstants | None = None) -> None:
        self.constants = constants or CrossingObjectiveConstants()
        self.state = CrossingObjectiveState()
        self._last_breakdown: dict[str, float] = {
            "step_index": 0.0,
            "living_step_reward": 0.0,
            "living_stepcount_reward": 0.0,
            "distance_penalty": 0.0,
            "progress_reward": 0.0,
            "wrong_action_reward": 0.0,
            "collision_terminal_reward": 0.0,
            "goal_terminal_reward": 0.0,
            "timeout_terminal_reward": 0.0,
            "total_step_reward": 0.0,
            "cumulative_reward": 0.0,
            "cumulative_progress_reward": 0.0,
            "cumulative_wrong_action_penalty": 0.0,
            "cumulative_living_penalty": 0.0,
            "cumulative_distance_penalty": 0.0,
            "cumulative_terminal_reward": 0.0,
        }

    def reset(self) -> None:
        """Reset episodic state and last-step reward breakdown."""
        self.state.reset()
        self._last_breakdown.update(
            {
                "step_index": 0.0,
                "living_step_reward": 0.0,
                "living_stepcount_reward": 0.0,
                "distance_penalty": 0.0,
                "progress_reward": 0.0,
                "wrong_action_reward": 0.0,
                "collision_terminal_reward": 0.0,
                "goal_terminal_reward": 0.0,
                "timeout_terminal_reward": 0.0,
                "total_step_reward": 0.0,
                "cumulative_reward": 0.0,
                "cumulative_progress_reward": 0.0,
                "cumulative_wrong_action_penalty": 0.0,
                "cumulative_living_penalty": 0.0,
                "cumulative_distance_penalty": 0.0,
                "cumulative_terminal_reward": 0.0,
            }
        )

    def compute_step_reward(
        self,
        info: dict,
        *,
        action: object | None = None,
        env: object | None = None,
    ) -> float:
        """Compute one step reward using old crossing ingredients in RL reward form."""
        current_goal_distance = self._resolve_goal_distance(info=info, env=env)
        current_heading_error_deg = self._resolve_goal_heading_error_deg(info=info, env=env)

        living_step_reward = -float(self.constants.STEP_COST)
        living_stepcount_reward = -(
            float(self.constants.STEP_COUNT_COST)
            / max(1.0, float(self.constants.DISTANCE_NORMALISER))
        )
        distance_penalty = -float(self.constants.DISTANCE_COST) * (
            current_goal_distance / float(self.constants.DISTANCE_NORMALISER)
        )

        progress_reward = self._compute_progress_reward(
            current_goal_distance=current_goal_distance,
            current_heading_error_deg=current_heading_error_deg,
        )

        wrong_action_reward = self._compute_wrong_action_reward(info=info, action=action)

        collision_terminal_reward, goal_terminal_reward, timeout_terminal_reward = (
            self._compute_terminal_terms(info=info)
        )

        total_step_reward = (
            living_step_reward
            + living_stepcount_reward
            + distance_penalty
            + progress_reward
            + wrong_action_reward
            + collision_terminal_reward
            + goal_terminal_reward
            + timeout_terminal_reward
        )

        self.state.step_index += 1
        self.state.cumulative_reward += total_step_reward
        self.state.cumulative_progress_reward += progress_reward
        self.state.cumulative_wrong_action_penalty += wrong_action_reward
        self.state.cumulative_living_penalty += living_step_reward + living_stepcount_reward
        self.state.cumulative_distance_penalty += distance_penalty
        self.state.cumulative_terminal_reward += (
            collision_terminal_reward + goal_terminal_reward + timeout_terminal_reward
        )
        self.state.prev_goal_distance = current_goal_distance
        self.state.prev_goal_heading_error_deg = current_heading_error_deg

        self._last_breakdown = {
            "step_index": float(self.state.step_index),
            "living_step_reward": living_step_reward,
            "living_stepcount_reward": living_stepcount_reward,
            "distance_penalty": distance_penalty,
            "progress_reward": progress_reward,
            "wrong_action_reward": wrong_action_reward,
            "collision_terminal_reward": collision_terminal_reward,
            "goal_terminal_reward": goal_terminal_reward,
            "timeout_terminal_reward": timeout_terminal_reward,
            "total_step_reward": total_step_reward,
            "cumulative_reward": self.state.cumulative_reward,
            "cumulative_progress_reward": self.state.cumulative_progress_reward,
            "cumulative_wrong_action_penalty": self.state.cumulative_wrong_action_penalty,
            "cumulative_living_penalty": self.state.cumulative_living_penalty,
            "cumulative_distance_penalty": self.state.cumulative_distance_penalty,
            "cumulative_terminal_reward": self.state.cumulative_terminal_reward,
        }

        return total_step_reward

    def get_reward_breakdown(self) -> dict[str, float]:
        """Return the most recent per-step and cumulative reward breakdown."""
        return dict(self._last_breakdown)

    def _compute_progress_reward(
        self,
        *,
        current_goal_distance: float,
        current_heading_error_deg: float,
    ) -> float:
        if (
            self.state.prev_goal_distance is None
            or self.state.prev_goal_heading_error_deg is None
        ):
            return 0.0

        prev_goal_distance = self.state.prev_goal_distance
        prev_goal_heading_error_deg = self.state.prev_goal_heading_error_deg

        distance_improved = current_goal_distance < prev_goal_distance
        distance_regressed = current_goal_distance > prev_goal_distance
        current_heading_good = (
            abs(current_heading_error_deg)
            <= float(self.constants.HEADING_ALIGNMENT_THRESHOLD_DEG)
        )
        prev_heading_good = (
            abs(prev_goal_heading_error_deg)
            <= float(self.constants.HEADING_ALIGNMENT_THRESHOLD_DEG)
        )
        heading_improved = abs(current_heading_error_deg) < abs(prev_goal_heading_error_deg)
        heading_regressed = abs(current_heading_error_deg) > abs(prev_goal_heading_error_deg)

        progress_magnitude = abs(float(self.constants.GOAL_PROGRESS_BONUS))

        if (
            distance_improved
            and (current_heading_good or heading_improved or prev_heading_good)
        ) or (heading_improved and not distance_regressed):
            return progress_magnitude

        if (
            distance_regressed
            and (
                heading_regressed
                or (not current_heading_good and not prev_heading_good)
            )
        ) or (heading_regressed and not distance_improved):
            return -progress_magnitude

        return 0.0

    def _compute_wrong_action_reward(self, *, info: dict, action: object | None) -> float:
        scenario = str(info.get("colregs_scenario", info.get("scenario", "unknown")))
        if scenario != "crossing":
            return 0.0

        designated_give_way = info.get("designated_give_way_vessel")
        if designated_give_way is not None and str(designated_give_way) != "vessel1":
            return 0.0

        rudder_value = self._resolve_rudder_value(info=info, action=action)
        if rudder_value is None:
            return 0.0

        # Project rudder convention: positive = starboard (correct crossing give-way turn).
        # Therefore wrong action is non-starboard (<= 0.0).
        if rudder_value <= 0.0:
            return -float(self.constants.WRONG_ACTION_PENALTY)
        return 0.0

    def _compute_terminal_terms(self, *, info: dict) -> tuple[float, float, float]:
        collision = bool(info.get("collision", False))

        if "vessel1_reached" in info and "vessel2_reached" in info:
            success = bool(info.get("vessel1_reached")) and bool(info.get("vessel2_reached"))
        else:
            success = bool(info.get("success", False))

        reason = str(info.get("reason", ""))

        collision_terminal_reward = (
            -float(self.constants.COLLISION_PENALTY) if collision else 0.0
        )
        goal_terminal_reward = abs(float(self.constants.GOAL_BONUS)) if success else 0.0

        timeout_terminal_reward = 0.0
        if not success and reason == "timeout":
            timeout_terminal_reward = -float(self.constants.TIMEOUT_PENALTY)

        return collision_terminal_reward, goal_terminal_reward, timeout_terminal_reward

    def _resolve_goal_distance(self, *, info: dict, env: object | None) -> float:
        if "vessel1_goal_distance" in info:
            return float(info["vessel1_goal_distance"])

        if "goal_distance" in info:
            return float(info["goal_distance"])

        if env is not None and hasattr(env, "vessel1"):
            vessel1 = getattr(env, "vessel1")
            vessel_x = self._get_attr_value(vessel1, ("x", "position_x", "pos_x"))
            vessel_y = self._get_attr_value(vessel1, ("y", "position_y", "pos_y"))
            goal_x = self._get_goal_coordinate(vessel1, "x")
            goal_y = self._get_goal_coordinate(vessel1, "y")
            return math.hypot(goal_x - vessel_x, goal_y - vessel_y)

        raise ValueError(
            "Unable to resolve goal distance: expected info['vessel1_goal_distance'], "
            "info['goal_distance'], or env.vessel1 with position and goal coordinates."
        )

    def _resolve_goal_heading_error_deg(self, *, info: dict, env: object | None) -> float:
        if "vessel1_goal_heading_error_deg" in info:
            return float(info["vessel1_goal_heading_error_deg"])

        if "goal_heading_error_deg" in info:
            return float(info["goal_heading_error_deg"])

        if env is not None and hasattr(env, "vessel1"):
            vessel1 = getattr(env, "vessel1")
            vessel_x = self._get_attr_value(vessel1, ("x", "position_x", "pos_x"))
            vessel_y = self._get_attr_value(vessel1, ("y", "position_y", "pos_y"))
            goal_x = self._get_goal_coordinate(vessel1, "x")
            goal_y = self._get_goal_coordinate(vessel1, "y")

            heading_deg = self._get_attr_value(
                vessel1,
                ("heading_deg", "heading", "course_deg", "yaw_deg"),
            )
            desired_heading_deg = math.degrees(math.atan2(goal_y - vessel_y, goal_x - vessel_x))
            return self._signed_angle_difference_deg(desired_heading_deg, heading_deg)

        raise ValueError(
            "Unable to resolve heading error: expected info['vessel1_goal_heading_error_deg'], "
            "info['goal_heading_error_deg'], or env.vessel1 with heading, position, and goal."
        )

    @staticmethod
    def _signed_angle_difference_deg(target_deg: float, source_deg: float) -> float:
        wrapped = (target_deg - source_deg + 180.0) % 360.0 - 180.0
        return float(wrapped)

    @staticmethod
    def _get_attr_value(source: Any, names: tuple[str, ...]) -> float:
        for name in names:
            if hasattr(source, name):
                return float(getattr(source, name))
        raise ValueError(
            f"Required attribute missing. Tried names={names!r} on object={type(source).__name__}."
        )

    def _get_goal_coordinate(self, vessel: Any, axis: str) -> float:
        axis = axis.lower()
        direct_candidates = (
            f"goal_{axis}",
            f"target_{axis}",
            f"destination_{axis}",
        )
        for candidate in direct_candidates:
            if hasattr(vessel, candidate):
                return float(getattr(vessel, candidate))

        goal_obj_candidates = ("goal", "target", "destination")
        goal_axis_candidates = (
            axis,
            f"pos_{axis}",
            f"position_{axis}",
        )
        for goal_name in goal_obj_candidates:
            if hasattr(vessel, goal_name):
                goal_obj = getattr(vessel, goal_name)
                for axis_name in goal_axis_candidates:
                    if hasattr(goal_obj, axis_name):
                        return float(getattr(goal_obj, axis_name))

        raise ValueError(
            f"Unable to resolve vessel goal {axis}-coordinate from vessel object {type(vessel).__name__}."
        )

    @staticmethod
    def _resolve_rudder_value(*, info: dict, action: object | None) -> float | None:
        if "vessel1_rudder_deg" in info:
            return float(info["vessel1_rudder_deg"])

        if "rudder_cmd" in info:
            return float(info["rudder_cmd"])

        if action is None:
            return None

        if isinstance(action, (list, tuple)):
            if len(action) == 0:
                return None
            return float(action[0])

        if hasattr(action, "size") and hasattr(action, "__getitem__"):
            try:
                if int(getattr(action, "size")) > 0:
                    return float(action[0])
            except (TypeError, ValueError):
                return None

        return None
