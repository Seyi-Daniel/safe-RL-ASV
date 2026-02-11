"""Centralised hyperparameter definitions for the NEAT crossing project."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Tuple


def _hp(
    default: Any,
    description: str,
    *,
    category: str,
    alias: str | None = None,
) -> Any:
    metadata = {"help": description, "category": category}
    if alias is not None:
        metadata["alias"] = alias
    return field(default=default, metadata=metadata)


@dataclass
class HyperParameters:
    """Container holding every tunable numeric value for the project."""

    # Vessel geometry and kinetics -------------------------------------------------
    boat_length: float = _hp(6.0, "Length of each vessel hull in metres.", category="boat")
    boat_width: float = _hp(2.2, "Beam of each vessel in metres.", category="boat")
    boat_max_speed: float = _hp(7.0, "Upper bound on achievable surge speed (m/s).", category="boat")
    boat_min_speed: float = _hp(0.0, "Lower bound on surge speed (m/s).", category="boat")
    boat_accel_rate: float = _hp(0.20, "Rate of positive longitudinal acceleration (m/s²).", category="boat")
    boat_decel_rate: float = _hp(0.05, "Rate of commanded deceleration (m/s²).", category="boat")

    # Helm control -----------------------------------------------------------------
    rudder_max_angle_deg: float = _hp(
        35.0,
        "Maximum absolute rudder angle (degrees).",
        category="rudder",
    )
    rudder_max_yaw_rate: float = _hp(
        0.25,
        "Peak yaw rate achieved at max rudder deflection (rad/s).",
        category="rudder",
    )
    rudder_max_rate_degps: float = _hp(
        40.0,
        "Maximum rudder slew rate (degrees per second).",
        category="rudder",
    )

    # Environment -------------------------------------------------------------------
    env_world_w: float = _hp(700.0, "Width of the continuous simulation arena (metres).", category="environment")
    env_world_h: float = _hp(440.0, "Height of the continuous simulation arena (metres).", category="environment")
    env_dt: float = _hp(0.05, "Primary integration time step (seconds).", category="environment")
    env_substeps: int = _hp(1, "Number of internal sub-steps used for integration.", category="environment")
    env_pixels_per_meter: float = _hp(2, "Rendering scale when pygame visualisation is enabled.", category="environment")

    # Scenario geometry -------------------------------------------------------------
    scenario_crossing_distance: float = _hp(
        220.0,
        "Longitudinal distance between each vessel and the nominal crossing point (metres).",
        category="scenario",
    )
    scenario_goal_extension: float = _hp(
        220.0,
        "Additional distance beyond the crossing point used when placing terminal goals (metres).",
        category="scenario",
    )
    scenario_crossing_agent_speed: float = _hp(
        7.0,
        "Initial surge speed for the give-way vessel during crossing encounters (m/s).",
        category="scenario",
    )
    scenario_crossing_stand_on_speed: float = _hp(
        7.0,
        "Initial surge speed for the stand-on vessel during crossing encounters (m/s).",
        category="scenario",
    )
    scenario_head_on_agent_speed: float = _hp(
        7.0,
        "Initial surge speed for the give-way vessel during head-on encounters (m/s).",
        category="scenario",
    )
    scenario_head_on_stand_on_speed: float = _hp(
        7.0,
        "Initial surge speed for the stand-on vessel during head-on encounters (m/s).",
        category="scenario",
    )
    scenario_overtaking_agent_speed: float = _hp(
        7.0,
        "Initial surge speed for the give-way vessel during overtaking encounters (m/s).",
        category="scenario",
    )
    scenario_overtaking_stand_on_speed: float = _hp(
        7.0,
        "Initial surge speed for the stand-on vessel during overtaking encounters (m/s).",
        category="scenario",
    )

    # Feature scaling ---------------------------------------------------------------
    feature_position_scale: float = _hp(
        300.0,
        "Normalisation constant applied to world-space positions before feeding NEAT.",
        category="features",
    )
    feature_speed_scale: float = _hp(
        20.0,
        "Normalisation constant applied to speeds in the observation vector.",
        category="features",
    )
    feature_heading_scale: float = _hp(
        3.141592653589793,
        "Normalisation constant applied to headings (radians).",
        category="features",
    )

    # Episode termination -----------------------------------------------------------
    max_steps: int = _hp(2000, "Maximum number of simulation steps per scenario evaluation.", category="evaluation")
    goal_tolerance: float = _hp(10.0, "Distance to the goal at which an episode is marked successful (metres).", category="evaluation")
    collision_distance: float = _hp(8.0, "Separation threshold treated as a collision (metres).", category="evaluation")

    # Cost shaping ------------------------------------------------------------------
    step_cost: float = _hp(1.0, "Base cost accrued each simulation step (minimisation objective).", category="cost")
    step_normaliser: float = _hp(
        1000.0,
        "Normalisation constant applied to the episode step count when adding a shaping cost.",
        category="cost",
    )
    step_count_cost: float = _hp(
        1.0,
        "Multiplier applied to the normalised episode step count (encourages shorter episodes).",
        category="cost",
    )
    goal_bonus: float = _hp(
        -40.0,
        "Additional cost contribution applied when the goal is reached (negative rewards faster arrivals).",
        category="cost",
    )
    collision_penalty: float = _hp(200.0, "Cost applied when a collision occurs.", category="cost")
    timeout_penalty: float = _hp(
        0.0,
        "Additional cost applied when the agent fails to reach the goal before max_steps.",
        category="cost",
    )
    distance_cost: float = _hp(
        1.5,
        "Multiplier applied to the normalised remaining distance when time expires.",
        category="cost",
    )
    distance_normaliser: float = _hp(
        250.0,
        "Normalisation constant used when converting remaining distance to a cost contribution.",
        category="cost",
    )
    goal_progress_bonus: float = _hp(
        -1.2,
        "Per-step shaping magnitude applied when an action improves goal distance/heading (negative) or moves away (positive).",
        category="cost",
    )
    heading_alignment_threshold_deg: float = _hp(
        12.0,
        "Heading error tolerance inside which no heading-based shaping is applied (degrees).",
        category="cost",
    )

    # COLREGs shaping ---------------------------------------------------------------
    tcpa_threshold: float = _hp(
        90.0,
        "Time-to-closest-point-of-approach window in which COLREGs penalties are evaluated (seconds).",
        category="colregs",
    )
    dcpa_threshold: float = _hp(
        30.0,
        "Closest-point-of-approach distance that activates COLREGs checks (metres).",
        category="colregs",
    )
    angle_threshold_deg: float = _hp(
        112.5,
        "Maximum starboard relative bearing considered a COLREGs crossing encounter (degrees).",
        category="colregs",
    )
    wrong_action_penalty: float = _hp(
        1.0,
        "Per-step cost added when the agent chooses a non-starboard helm within the COLREGs envelope.",
        category="colregs",
    )

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow dictionary of hyperparameter names and values."""

        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}

    def update_from_items(self, overrides: Dict[str, Any]) -> None:
        """Mutate this instance according to ``overrides`` (keys match attribute names)."""

        for name, value in overrides.items():
            if not hasattr(self, name):
                raise KeyError(f"Unknown hyperparameter '{name}'.")
            setattr(self, name, value)

    def iter_documentation(self) -> Iterable[Tuple[str, Any, str]]:
        """Yield ``(name, value, help)`` tuples describing each hyperparameter."""

        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            metadata = field.metadata or {}
            yield field.name, value, metadata.get("help", "")


def parse_hyperparameter_override(text: str) -> Tuple[str, Any]:
    """Parse ``NAME=VALUE`` pairs from the command line."""

    if "=" not in text:
        raise ValueError("Overrides must use the NAME=VALUE format.")
    name, raw_value = text.split("=", 1)
    name = name.strip()
    raw_value = raw_value.strip()
    if not name:
        raise ValueError("Hyperparameter name cannot be empty.")

    try:
        if raw_value.lower() in {"true", "false"}:
            value: Any = raw_value.lower() == "true"
        elif raw_value.startswith("0x"):
            value = int(raw_value, 16)
        else:
            value = int(raw_value)
    except ValueError:
        try:
            value = float(raw_value)
        except ValueError:
            value = raw_value
    return name, value


def apply_cli_overrides(hparams: HyperParameters, entries: Iterable[str]) -> None:
    """Apply command-line overrides encoded as ``NAME=VALUE`` strings."""

    updates: Dict[str, Any] = {}
    for item in entries:
        name, value = parse_hyperparameter_override(item)
        updates[name] = value
    if updates:
        hparams.update_from_items(updates)


__all__ = ["HyperParameters", "apply_cli_overrides"]

