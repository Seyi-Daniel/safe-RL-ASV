#!/usr/bin/env python3
"""Inspect the deterministic COLREGs encounter scenarios used for training."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asv_neat import (  # noqa: E402
    BoatParams,
    CrossingScenarioEnv,
    EncounterScenario,
    EnvConfig,
    HyperParameters,
    ScenarioKind,
    ScenarioRequest,
    apply_cli_overrides,
    build_scenarios,
    scenario_states_for_env,
)


class RandomGiveWayPolicy:
    """Placeholder controller emitting random helm/throttle combinations."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_action(self) -> tuple[float, int]:
        rudder_cmd = self._rng.uniform(-1.0, 1.0)
        throttle = self._rng.randrange(3)
        return rudder_cmd, throttle


def parse_args(hparams: HyperParameters) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable pygame visualisation for each scenario.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=35.0,
        help="Number of seconds to simulate when rendering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random give-way controller.",
    )
    parser.add_argument(
        "--scenario",
        choices=["all", "crossing", "head_on", "overtaking"],
        default="all",
        help="Select which encounter set to preview (default: all).",
    )
    parser.add_argument(
        "--hp",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Optional hyperparameter overrides (same format as the training script).",
    )
    return parser.parse_args()


def build_env_config(hparams: HyperParameters, *, render: bool) -> EnvConfig:
    return EnvConfig(
        world_w=hparams.env_world_w,
        world_h=hparams.env_world_h,
        dt=hparams.env_dt,
        substeps=hparams.env_substeps,
        render=render,
        pixels_per_meter=hparams.env_pixels_per_meter,
        show_grid=False,
        show_trails=False,
        show_hud=False,
    )


def print_scenario_descriptions(
    scenarios: Iterable[EncounterScenario], header: str
) -> None:
    scenario_list = list(scenarios)
    print(header)
    print("=" * len(header))
    if not scenario_list:
        print("\nNo scenarios match the current selection.")
        return

    grouped: dict[ScenarioKind, list[EncounterScenario]] = {}
    for scenario in scenario_list:
        grouped.setdefault(scenario.kind, []).append(scenario)

    for kind in ScenarioKind:
        items = grouped.get(kind)
        if not items:
            continue
        kind_title = kind.value.replace("_", " ").title()
        for idx, scenario in enumerate(items, start=1):
            print(f"\n{kind_title} scenario {idx}:")
            print(scenario.describe())


def format_scenario_heading(idx: int, scenario: EncounterScenario) -> str:
    frame = "stand-on" if scenario.bearing_frame == "agent" else "agent (stand-on frame)"
    kind_title = scenario.kind.value.replace("_", " ").title()
    return (
        f"Scenario {idx} [{kind_title}, "
        f"bearing {scenario.requested_bearing:6.2f}° ({frame})]"
    )


def main() -> None:
    hparams = HyperParameters()
    args = parse_args(hparams)

    try:
        apply_cli_overrides(hparams, args.hp)
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc))

    scenario_request = ScenarioRequest(
        crossing_distance=hparams.scenario_crossing_distance,
        goal_extension=hparams.scenario_goal_extension,
        crossing_agent_speed=hparams.scenario_crossing_agent_speed,
        crossing_stand_on_speed=hparams.scenario_crossing_stand_on_speed,
        head_on_agent_speed=hparams.scenario_head_on_agent_speed,
        head_on_stand_on_speed=hparams.scenario_head_on_stand_on_speed,
        overtaking_agent_speed=hparams.scenario_overtaking_agent_speed,
        overtaking_stand_on_speed=hparams.scenario_overtaking_stand_on_speed,
    )
    scenarios = build_scenarios(scenario_request)

    if args.scenario == "all":
        header = "Deterministic encounter scenarios"
    else:
        selected_kind = ScenarioKind(args.scenario)
        header = f"Deterministic {selected_kind.value} scenarios"
        scenarios = [sc for sc in scenarios if sc.kind is selected_kind]

    print_scenario_descriptions(scenarios, header)

    boat_params = BoatParams(
        length=hparams.boat_length,
        width=hparams.boat_width,
        max_speed=hparams.boat_max_speed,
        min_speed=hparams.boat_min_speed,
        accel_rate=hparams.boat_accel_rate,
        decel_rate=hparams.boat_decel_rate,
    )
    env_cfg = build_env_config(hparams, render=args.render)
    env = CrossingScenarioEnv(cfg=env_cfg, kin=boat_params)
    try:
        if args.render:
            env.enable_render()
            controller = RandomGiveWayPolicy(seed=args.seed)
            steps = max(1, int(round(args.duration / env_cfg.dt)))
            for idx, scenario in enumerate(scenarios, start=1):
                heading = format_scenario_heading(idx, scenario)
                print(f"\nRendering {heading}")
                states, meta = scenario_states_for_env(env, scenario)
                env.reset_from_states(states, meta=meta)
                for _ in range(steps):
                    env.step([controller.choose_action(), None])
                    env.render()
        else:
            print("\nRendering disabled; use --render to visualise the deterministic encounters.")
    finally:
        env.close()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

