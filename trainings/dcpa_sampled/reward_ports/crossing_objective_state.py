"""State container for the standalone crossing-objective reward port."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrossingObjectiveState:
    """Tracks per-episode state for dense crossing reward computation."""

    prev_goal_distance: float | None = None
    prev_goal_heading_error_deg: float | None = None
    step_index: int = 0
    cumulative_reward: float = 0.0
    cumulative_progress_reward: float = 0.0
    cumulative_wrong_action_penalty: float = 0.0
    cumulative_living_penalty: float = 0.0
    cumulative_distance_penalty: float = 0.0
    cumulative_terminal_reward: float = 0.0

    def reset(self) -> None:
        """Reset all tracked state back to default values."""
        self.prev_goal_distance = None
        self.prev_goal_heading_error_deg = None
        self.step_index = 0
        self.cumulative_reward = 0.0
        self.cumulative_progress_reward = 0.0
        self.cumulative_wrong_action_penalty = 0.0
        self.cumulative_living_penalty = 0.0
        self.cumulative_distance_penalty = 0.0
        self.cumulative_terminal_reward = 0.0
