"""Geometry helpers for deterministic COLREGs encounter scenarios."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterable, Iterator, Optional, Sequence, Tuple


class ScenarioKind(str, Enum):
    """Enumeration of supported encounter geometries."""

    CROSSING = "crossing"
    HEAD_ON = "head_on"
    OVERTAKING = "overtaking"


COLREGS_STARBOARD_MIN_DEG = 5.0
COLREGS_STARBOARD_MAX_DEG = 112.5
COLREGS_HEAD_ON_MIN_DEG = 355.0
COLREGS_HEAD_ON_MAX_DEG = 5.0
COLREGS_OVERTAKING_MIN_DEG = 112.5
COLREGS_OVERTAKING_MAX_DEG = 247.5


def _bearing_fraction(fraction: float) -> float:
    span = COLREGS_STARBOARD_MAX_DEG - COLREGS_STARBOARD_MIN_DEG
    return COLREGS_STARBOARD_MIN_DEG + span * fraction


def _wrapped_bearing_fraction(
    start_deg: float, end_deg: float, fraction: float
) -> float:
    """Linear interpolation that correctly handles wrap-around at 360°."""

    start = start_deg % 360.0
    end = end_deg % 360.0
    if math.isclose(start, end):
        return start

    if start < end:
        span = end - start
        return (start + span * fraction) % 360.0

    # Wrap-around (e.g. 355° → 5°)
    span = (360.0 - start) + end
    value = start + span * fraction
    return value % 360.0


CROSSING_BEARINGS_DEG: Tuple[float, ...] = (
    COLREGS_STARBOARD_MIN_DEG,
    _bearing_fraction(0.25),
    _bearing_fraction(0.5),
    _bearing_fraction(0.75),
    COLREGS_STARBOARD_MAX_DEG,
)

HEAD_ON_BEARINGS_DEG: Tuple[float, ...] = tuple(
    _wrapped_bearing_fraction(
        COLREGS_HEAD_ON_MIN_DEG, COLREGS_HEAD_ON_MAX_DEG, fraction
    )
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
)

OVERTAKING_BEARINGS_DEG: Tuple[float, ...] = tuple(
    COLREGS_OVERTAKING_MIN_DEG
    + (COLREGS_OVERTAKING_MAX_DEG - COLREGS_OVERTAKING_MIN_DEG) * fraction
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
)

# Backwards compatibility with earlier code that imported this name directly.
STAND_ON_BEARINGS_DEG = CROSSING_BEARINGS_DEG


@dataclass(frozen=True)
class VesselState:
    """Description of an individual vessel participating in the encounter."""

    name: str
    x: float
    y: float
    heading_deg: float
    speed: float
    goal: Optional[Tuple[float, float]] = None

    def bearing_to(self, other: "VesselState") -> float:
        """Clockwise (starboard) relative bearing to ``other`` in degrees."""

        dx = other.x - self.x
        dy = other.y - self.y
        ch = math.cos(math.radians(self.heading_deg))
        sh = math.sin(math.radians(self.heading_deg))
        x_rel = ch * dx + sh * dy
        y_rel = -sh * dx + ch * dy
        rel_port = math.degrees(math.atan2(y_rel, x_rel))
        rel_port = (rel_port + 360.0) % 360.0
        return (360.0 - rel_port) % 360.0


@dataclass(frozen=True)
class EncounterScenario:
    """Container for the initial geometry of an encounter scenario."""

    kind: ScenarioKind
    agent: VesselState
    stand_on: VesselState
    crossing_point: Tuple[float, float]
    requested_bearing: float
    bearing_frame: str

    def describe(self) -> str:
        if self.kind is ScenarioKind.OVERTAKING:
            bearing = self.stand_on.bearing_to(self.agent)
            bearing_label = "Agent bearing from stand-on"
        else:
            bearing = self.agent.bearing_to(self.stand_on)
            bearing_label = "Stand-on bearing from agent"
        requested_label = f"{bearing_label} (requested)"
        realised_label = f"{bearing_label} (realised)"
        return (
            f"Scenario type                : {self.kind.value}\n"
            f"{requested_label:<30}: {self.requested_bearing:6.2f}°\n"
            f"{realised_label:<30}: {bearing:6.2f}°\n"
            f"Agent position               : ({self.agent.x:7.2f}, {self.agent.y:7.2f}) m\n"
            f"Stand-on position            : ({self.stand_on.x:7.2f}, {self.stand_on.y:7.2f}) m\n"
            f"Agent heading                : {self.agent.heading_deg:6.2f}°\n"
            f"Stand-on heading             : {self.stand_on.heading_deg:6.2f}°\n"
            f"Agent speed                  : {self.agent.speed:6.2f} m/s\n"
            f"Stand-on speed               : {self.stand_on.speed:6.2f} m/s"
        )


@dataclass(frozen=True)
class ScenarioRequest:
    """User-controllable parameters for all supported encounters."""

    crossing_distance: float = 220.0
    goal_extension: float = 220.0
    crossing_agent_speed: float = 7.0
    crossing_stand_on_speed: float = 7.0
    head_on_agent_speed: float = 7.0
    head_on_stand_on_speed: float = 7.0
    overtaking_agent_speed: float = 7.0
    overtaking_stand_on_speed: float = 7.0

    def speeds_for(self, kind: ScenarioKind) -> Tuple[float, float]:
        lookup: Dict[ScenarioKind, Tuple[float, float]] = {
            ScenarioKind.CROSSING: (
                self.crossing_agent_speed,
                self.crossing_stand_on_speed,
            ),
            ScenarioKind.HEAD_ON: (
                self.head_on_agent_speed,
                self.head_on_stand_on_speed,
            ),
            ScenarioKind.OVERTAKING: (
                self.overtaking_agent_speed,
                self.overtaking_stand_on_speed,
            ),
        }
        return lookup[kind]


def compute_crossing_geometry(
    angle_deg: float, request: ScenarioRequest
) -> EncounterScenario:
    """Create the crossing encounter for a single bearing value."""

    crossing_point = (0.0, 0.0)
    approach = request.crossing_distance

    goal_offset = request.goal_extension

    agent_speed, stand_on_speed = request.speeds_for(ScenarioKind.CROSSING)

    agent = VesselState(
        name="give_way",
        x=-approach,
        y=0.0,
        heading_deg=0.0,
        speed=agent_speed,
        goal=(crossing_point[0] + goal_offset, crossing_point[1]),
    )

    bearing_rad = math.radians(angle_deg)
    dir_x = math.cos(bearing_rad)
    dir_y = -math.sin(bearing_rad)

    stand_x = agent.x + approach * dir_x
    stand_y = agent.y + approach * dir_y

    distance_to_crossing = math.hypot(
        stand_x - crossing_point[0], stand_y - crossing_point[1]
    )
    min_cross_distance = approach
    if distance_to_crossing < min_cross_distance:
        sin_bearing = math.sin(bearing_rad)
        term = min_cross_distance**2 - (approach * sin_bearing) ** 2
        if term < 0.0:
            term = 0.0
        adjusted_r = approach * math.cos(bearing_rad) + math.sqrt(term)
        if adjusted_r > approach:
            stand_x = agent.x + adjusted_r * dir_x
            stand_y = agent.y + adjusted_r * dir_y

    heading_rad = math.atan2(crossing_point[1] - stand_y, crossing_point[0] - stand_x)
    heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0
    goal_x = crossing_point[0] + goal_offset * math.cos(heading_rad)
    goal_y = crossing_point[1] + goal_offset * math.sin(heading_rad)

    # Move the stand-on vessel closer to the crossing point (without exceeding
    # the requested speed) when its path would otherwise be too long to collide
    # with a straight-running agent. This keeps the realised bearing close to
    # the requested value while ensuring equal ETAs even for wide starboard
    # angles such as 85.62° and 112.5°.
    eta_agent = approach / agent_speed if agent_speed > 0.0 else float("inf")
    distance_to_crossing = math.hypot(stand_x - crossing_point[0], stand_y - crossing_point[1])
    desired_distance = stand_on_speed * eta_agent if math.isfinite(eta_agent) else distance_to_crossing
    if desired_distance < distance_to_crossing:
        heading_unit_x = math.cos(heading_rad)
        heading_unit_y = math.sin(heading_rad)
        stand_x = crossing_point[0] - desired_distance * heading_unit_x
        stand_y = crossing_point[1] - desired_distance * heading_unit_y

    stand_on = VesselState(
        name="stand_on",
        x=stand_x,
        y=stand_y,
        heading_deg=heading_deg,
        speed=stand_on_speed,
        goal=(goal_x, goal_y),
    )

    return EncounterScenario(
        kind=ScenarioKind.CROSSING,
        agent=agent,
        stand_on=stand_on,
        crossing_point=crossing_point,
        requested_bearing=angle_deg,
        bearing_frame="agent",
    )


def compute_head_on_geometry(angle_deg: float, request: ScenarioRequest) -> EncounterScenario:
    """Create a head-on encounter where both vessels approach the crossing point."""

    crossing_point = (0.0, 0.0)
    approach = request.crossing_distance
    goal_offset = request.goal_extension
    agent_speed, stand_on_speed = request.speeds_for(ScenarioKind.HEAD_ON)

    agent = VesselState(
        name="give_way",
        x=-approach,
        y=0.0,
        heading_deg=0.0,
        speed=agent_speed,
        goal=(crossing_point[0] + goal_offset, crossing_point[1]),
    )

    bearing_rad = math.radians(angle_deg)
    stand_x = crossing_point[0] + approach * math.cos(bearing_rad)
    stand_y = crossing_point[1] + approach * math.sin(bearing_rad)

    heading_rad = math.atan2(crossing_point[1] - stand_y, crossing_point[0] - stand_x)
    heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0
    goal_x = crossing_point[0] + goal_offset * math.cos(heading_rad)
    goal_y = crossing_point[1] + goal_offset * math.sin(heading_rad)

    stand_on = VesselState(
        name="stand_on",
        x=stand_x,
        y=stand_y,
        heading_deg=heading_deg,
        speed=stand_on_speed,
        goal=(goal_x, goal_y),
    )

    return EncounterScenario(
        kind=ScenarioKind.HEAD_ON,
        agent=agent,
        stand_on=stand_on,
        crossing_point=crossing_point,
        requested_bearing=angle_deg,
        bearing_frame="agent",
    )


def compute_overtaking_geometry(
    angle_deg: float, request: ScenarioRequest
) -> EncounterScenario:
    """Create an overtaking encounter with the give-way vessel closing from astern."""

    crossing_point = (0.0, 0.0)
    approach = request.crossing_distance
    goal_offset = request.goal_extension
    agent_speed, stand_on_speed = request.speeds_for(ScenarioKind.OVERTAKING)

    # Reduce the initial longitudinal separation so the faster give-way vessel
    # closes the gap (and collides when holding course) within a single
    # evaluation episode.
    separation = 0.5 * approach

    agent = VesselState(
        name="give_way",
        x=-approach,
        y=0.0,
        heading_deg=0.0,
        speed=agent_speed,
        goal=(crossing_point[0] + goal_offset, crossing_point[1]),
    )

    rel_port_deg = (360.0 - angle_deg) % 360.0
    rel_rad = math.radians(rel_port_deg)
    x_rel = math.cos(rel_rad) * separation
    y_rel = math.sin(rel_rad) * separation

    stand_x = agent.x - x_rel
    stand_y = agent.y - y_rel

    stand_on = VesselState(
        name="stand_on",
        x=stand_x,
        y=stand_y,
        heading_deg=0.0,
        speed=stand_on_speed,
        goal=(crossing_point[0] + goal_offset, crossing_point[1]),
    )

    return EncounterScenario(
        kind=ScenarioKind.OVERTAKING,
        agent=agent,
        stand_on=stand_on,
        crossing_point=crossing_point,
        requested_bearing=angle_deg,
        bearing_frame="stand_on",
    )


_SCENARIO_BUILDERS: Dict[
    ScenarioKind,
    Tuple[Tuple[float, ...], Callable[[float, ScenarioRequest], EncounterScenario]],
] = {
    ScenarioKind.CROSSING: (CROSSING_BEARINGS_DEG, compute_crossing_geometry),
    ScenarioKind.HEAD_ON: (HEAD_ON_BEARINGS_DEG, compute_head_on_geometry),
    ScenarioKind.OVERTAKING: (OVERTAKING_BEARINGS_DEG, compute_overtaking_geometry),
}


def iter_scenarios(
    angles: Iterable[float],
    request: ScenarioRequest,
    *,
    kind: ScenarioKind = ScenarioKind.CROSSING,
) -> Iterator[EncounterScenario]:
    """Yield scenarios for each provided bearing value and encounter type."""

    builder = {
        ScenarioKind.CROSSING: compute_crossing_geometry,
        ScenarioKind.HEAD_ON: compute_head_on_geometry,
        ScenarioKind.OVERTAKING: compute_overtaking_geometry,
    }[kind]

    for ang in angles:
        yield builder(float(ang), request)


def default_scenarios(request: ScenarioRequest) -> Iterator[EncounterScenario]:
    """Iterate over the predefined bearing sets for every encounter type."""

    for kind, (angles, builder) in _SCENARIO_BUILDERS.items():
        for angle in angles:
            yield builder(angle, request)


def scenario_states_for_env(
    env, scenario: EncounterScenario
) -> Tuple[Sequence[dict], dict]:
    """Convert the dataclass description into environment-specific state dictionaries."""

    cx = env.world_w / 2.0
    cy = env.world_h / 2.0
    cross_x = cx + scenario.crossing_point[0]
    cross_y = cy + scenario.crossing_point[1]

    def convert(vessel: VesselState) -> dict:
        data = {
            "x": cross_x + vessel.x,
            "y": cross_y + vessel.y,
            "heading": math.radians(vessel.heading_deg),
            "speed": vessel.speed,
        }
        if vessel.goal is not None:
            gx, gy = vessel.goal
            data["goal_x"] = cross_x + gx
            data["goal_y"] = cross_y + gy
        return data

    states: Sequence[dict] = [convert(scenario.agent), convert(scenario.stand_on)]
    meta = {
        "bearing": scenario.requested_bearing,
        "bearing_frame": scenario.bearing_frame,
        "scenario_kind": scenario.kind.value,
        "cross_x": cross_x,
        "cross_y": cross_y,
    }
    return states, meta


__all__ = [
    "ScenarioKind",
    "STAND_ON_BEARINGS_DEG",
    "CROSSING_BEARINGS_DEG",
    "HEAD_ON_BEARINGS_DEG",
    "OVERTAKING_BEARINGS_DEG",
    "VesselState",
    "EncounterScenario",
    "ScenarioRequest",
    "compute_crossing_geometry",
    "compute_head_on_geometry",
    "compute_overtaking_geometry",
    "iter_scenarios",
    "default_scenarios",
    "scenario_states_for_env",
]

