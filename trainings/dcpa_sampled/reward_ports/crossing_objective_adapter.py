"""Adapter for applying CrossingObjectivePort to the currently designated give-way vessel."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any

from trainings.dcpa_sampled.reward_ports.crossing_objective_port import CrossingObjectivePort


@dataclass
class AdaptedRewardResult:
    """Return object for one adapted crossing reward computation."""

    reward: float
    vessel_id: str | None
    breakdown: dict[str, float]
    replay_eligible: bool


class _Vessel2ProxyEnv:
    """Lightweight env proxy exposing vessel2 as vessel1 for geometry fallback."""

    def __init__(self, env: object):
        vessel2 = getattr(env, "vessel2", None)
        if vessel2 is None:
            raise ValueError("Cannot build vessel2 proxy env because env.vessel2 is missing.")

        heading_deg = math.degrees(float(getattr(vessel2, "h"))) if hasattr(vessel2, "h") else float(
            getattr(vessel2, "heading_deg")
        )

        self.vessel1 = SimpleNamespace(
            x=float(getattr(vessel2, "x")),
            y=float(getattr(vessel2, "y")),
            goal_x=float(getattr(vessel2, "goal_x")),
            goal_y=float(getattr(vessel2, "goal_y")),
            heading_deg=heading_deg,
        )


class CrossingObjectiveAdapter:
    """Bridge layer that applies CrossingObjectivePort to vessel1 or vessel2 as give-way."""

    def __init__(self, reward_port: CrossingObjectivePort | None = None) -> None:
        self.reward_port = reward_port or CrossingObjectivePort()
        self._last_result = AdaptedRewardResult(
            reward=0.0,
            vessel_id=None,
            breakdown=self.reward_port.get_reward_breakdown(),
            replay_eligible=False,
        )

    def reset(self) -> None:
        """Reset underlying reward-port episodic state."""
        self.reward_port.reset()
        self._last_result = AdaptedRewardResult(
            reward=0.0,
            vessel_id=None,
            breakdown=self.reward_port.get_reward_breakdown(),
            replay_eligible=False,
        )

    def compute_reward(
        self,
        info: dict[str, Any],
        *,
        action_by_vessel: dict[str, Any] | None = None,
        env: object | None = None,
    ) -> AdaptedRewardResult:
        """Compute adapted reward for the current designated give-way vessel only."""
        designated = self._resolve_designated_give_way_vessel(info)
        if designated not in {"vessel1", "vessel2"}:
            result = AdaptedRewardResult(
                reward=0.0,
                vessel_id=None,
                breakdown=self.reward_port.get_reward_breakdown(),
                replay_eligible=False,
            )
            self._last_result = result
            return result

        action_for_designated = None
        if action_by_vessel is not None and designated in action_by_vessel:
            action_for_designated = action_by_vessel[designated]

        adapted_info, adapted_env = self._adapt_for_vessel(
            designated_vessel=designated,
            info=info,
            action=action_for_designated,
            env=env,
        )
        reward = self.reward_port.compute_step_reward(
            adapted_info,
            action=action_for_designated,
            env=adapted_env,
        )
        result = AdaptedRewardResult(
            reward=float(reward),
            vessel_id=designated,
            breakdown=self.reward_port.get_reward_breakdown(),
            replay_eligible=True,
        )
        self._last_result = result
        return result

    @staticmethod
    def _resolve_designated_give_way_vessel(info: dict[str, Any]) -> str | None:
        raw = str(info.get("designated_give_way_vessel", "")).strip()
        if raw in {"vessel1", "vessel2"}:
            return raw
        return None

    def _adapt_for_vessel(
        self,
        *,
        designated_vessel: str,
        info: dict[str, Any],
        action: Any,
        env: object | None,
    ) -> tuple[dict[str, Any], object | None]:
        if designated_vessel == "vessel1":
            adapted_info = dict(info)
            adapted_info["designated_give_way_vessel"] = "vessel1"
            return adapted_info, env

        adapted_info = dict(info)
        adapted_info["designated_give_way_vessel"] = "vessel1"

        if "vessel2_goal_distance" in info:
            adapted_info["vessel1_goal_distance"] = float(info["vessel2_goal_distance"])

        if "vessel2_goal_heading_error_deg" in info:
            adapted_info["vessel1_goal_heading_error_deg"] = float(info["vessel2_goal_heading_error_deg"])

        if "vessel2_rudder_deg" in info:
            adapted_info["vessel1_rudder_deg"] = float(info["vessel2_rudder_deg"])

        if "vessel2_reached" in info:
            adapted_info["vessel1_reached"] = bool(info["vessel2_reached"])

        if "vessel1_reached" in info:
            adapted_info["vessel2_reached"] = bool(info["vessel1_reached"])

        adapted_env = None
        if env is not None:
            adapted_env = _Vessel2ProxyEnv(env)

        _ = action  # kept for explicit signature parity and future extension
        return adapted_info, adapted_env

    def get_last_result(self) -> AdaptedRewardResult:
        """Return last computed adapted reward result."""
        return self._last_result
