"""Shared helpers for the repository's command-line entry points."""
from __future__ import annotations

from typing import Iterable, List

import math

import neat

from .config import BoatParams, EnvConfig, RudderParams
from .env import CrossingScenarioEnv
from .hyperparameters import HyperParameters
from .neat_training import episode_cost, simulate_episode
from .scenario import EncounterScenario, ScenarioKind, ScenarioRequest


SCENARIO_KIND_CHOICES: tuple[str, ...] = (
    "all",
    ScenarioKind.CROSSING.value,
    ScenarioKind.HEAD_ON.value,
    ScenarioKind.OVERTAKING.value,
)


def build_boat_params(hparams: HyperParameters) -> BoatParams:
    """Construct :class:`BoatParams` from the provided hyperparameters."""

    return BoatParams(
        length=hparams.boat_length,
        width=hparams.boat_width,
        max_speed=hparams.boat_max_speed,
        min_speed=hparams.boat_min_speed,
        accel_rate=hparams.boat_accel_rate,
        decel_rate=hparams.boat_decel_rate,
    )


def build_rudder_config(hparams: HyperParameters) -> RudderParams:
    """Construct :class:`RudderParams` according to ``hparams``."""

    return RudderParams(
        max_rudder=math.radians(hparams.rudder_max_angle_deg),
        max_yaw_rate=hparams.rudder_max_yaw_rate,
        max_rudder_rate=math.radians(hparams.rudder_max_rate_degps),
    )


def build_env_config(hparams: HyperParameters, *, render: bool) -> EnvConfig:
    """Return environment configuration with the requested render toggle."""

    return EnvConfig(
        world_w=hparams.env_world_w,
        world_h=hparams.env_world_h,
        dt=hparams.env_dt,
        substeps=hparams.env_substeps,
        render=render,
        pixels_per_meter=hparams.env_pixels_per_meter,
        show_grid=False,
        show_trails=True,
        show_hud=True,
    )


def build_scenario_request(hparams: HyperParameters) -> ScenarioRequest:
    """Convert hyperparameters into a :class:`ScenarioRequest`."""

    return ScenarioRequest(
        crossing_distance=hparams.scenario_crossing_distance,
        goal_extension=hparams.scenario_goal_extension,
        crossing_agent_speed=hparams.scenario_crossing_agent_speed,
        crossing_stand_on_speed=hparams.scenario_crossing_stand_on_speed,
        head_on_agent_speed=hparams.scenario_head_on_agent_speed,
        head_on_stand_on_speed=hparams.scenario_head_on_stand_on_speed,
        overtaking_agent_speed=hparams.scenario_overtaking_agent_speed,
        overtaking_stand_on_speed=hparams.scenario_overtaking_stand_on_speed,
    )


def filter_scenarios_by_kind(
    scenarios: Iterable[EncounterScenario], selection: str
) -> List[EncounterScenario]:
    """Return the subset of scenarios matching ``selection``.

    ``selection`` should be one of :data:`SCENARIO_KIND_CHOICES`.
    """

    scenario_list = list(scenarios)
    if selection == "all":
        return scenario_list

    desired = ScenarioKind(selection)
    return [scenario for scenario in scenario_list if scenario.kind is desired]


def summarise_genome(
    genome,
    neat_config,
    scenarios: Iterable[EncounterScenario],
    hparams: HyperParameters,
    boat_params: BoatParams,
    rudder_cfg: RudderParams,
    env_cfg: EnvConfig,
    *,
    render: bool = False,
) -> None:
    """Evaluate ``genome`` across ``scenarios`` and print metrics.

    Rendering is optional and controlled via ``render``.
    """

    scenario_list = list(scenarios)
    if not scenario_list:
        print("No scenarios available for evaluation.")
        return

    network = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    print("\nWinner evaluation summary:")

    def _scenario_prefix(idx: int, scenario: EncounterScenario) -> str:
        frame = "stand-on" if scenario.bearing_frame == "agent" else "agent (stand-on frame)"
        return (
            f"Scenario {idx} [{scenario.kind.value}, "
            f"bearing {scenario.requested_bearing:6.2f}° ({frame})]"
        )

    if render:
        env = CrossingScenarioEnv(cfg=env_cfg, kin=boat_params, rudder_cfg=rudder_cfg)
        try:
            env.enable_render()
            total_cost = 0.0
            for idx, scenario in enumerate(scenario_list, start=1):
                metrics = simulate_episode(env, scenario, network, hparams, render=True)
                cost = episode_cost(metrics, hparams)
                total_cost += cost
                status = (
                    "goal"
                    if metrics.reached_goal
                    else "collision" if metrics.collided else "timeout"
                )
                prefix = _scenario_prefix(idx, scenario)
                print(
                    f"  {prefix}: steps={metrics.steps:4d} status={status:8s} "
                    f"min_sep={metrics.min_separation:6.2f}m colregs={metrics.wrong_action_cost:6.2f} "
                    f"cost={cost:7.2f}"
                )
            print(f"Average cost: {total_cost / len(scenario_list):.2f}")
        finally:
            env.close()
        return

    def _evaluate(idx_scenario: int, scenario: EncounterScenario):
        local_env = CrossingScenarioEnv(cfg=env_cfg, kin=boat_params, rudder_cfg=rudder_cfg)
        try:
            metrics = simulate_episode(local_env, scenario, network, hparams, render=False)
        finally:
            local_env.close()
        cost = episode_cost(metrics, hparams)
        status = (
            "goal"
            if metrics.reached_goal
            else "collision" if metrics.collided else "timeout"
        )
        return idx_scenario, metrics, cost, status

    results = []
    for idx, scenario in enumerate(scenario_list, start=1):
        results.append(_evaluate(idx, scenario))

    total_cost = 0.0
    for idx, metrics, cost, status in results:
        total_cost += cost
        scenario = scenario_list[idx - 1]
        prefix = _scenario_prefix(idx, scenario)
        print(
            f"  {prefix}: steps={metrics.steps:4d} status={status:8s} "
            f"min_sep={metrics.min_separation:6.2f}m colregs={metrics.wrong_action_cost:6.2f} "
            f"cost={cost:7.2f}"
        )
    print(f"Average cost: {total_cost / len(results):.2f}")


__all__ = [
    "SCENARIO_KIND_CHOICES",
    "build_boat_params",
    "build_rudder_config",
    "build_env_config",
    "build_scenario_request",
    "filter_scenarios_by_kind",
    "summarise_genome",
]

