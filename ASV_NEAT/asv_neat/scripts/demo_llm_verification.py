#!/usr/bin/env python3
"""Render a side-by-side replay from an LLM control verification JSON file."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asv_neat import (  # noqa: E402
    HyperParameters,
    ScenarioKind,
    apply_cli_overrides,
    build_scenarios,
    scenario_states_for_env,
)
from asv_neat.cli_helpers import (  # noqa: E402
    build_boat_params,
    build_env_config,
    build_rudder_config,
    build_scenario_request,
)
from asv_neat.env import CrossingScenarioEnv, HAS_PYGAME  # noqa: E402

SCENARIO_KIND_CHOICES = ("auto", "crossing", "head_on", "overtaking")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verification",
        type=Path,
        default=PROJECT_ROOT / "llm_control_verification.json",
        help="Path to llm_control_verification.json.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable pygame visualisation.",
    )
    parser.add_argument(
        "--scenario-kind",
        choices=SCENARIO_KIND_CHOICES,
        default="auto",
        help="Select a deterministic encounter kind (default: auto from features).",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=None,
        help="1-based index for the selected encounter kind.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.05,
        help="Seconds to sleep between frames when rendering.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=None,
        help="First step index to include (inclusive).",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=None,
        help="Last step index to include (inclusive).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps to replay.",
    )
    parser.add_argument(
        "--hp",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override hyperparameters to mirror the training run (repeatable).",
    )
    return parser


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_steps(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        steps = payload.get("steps", [])
    else:
        steps = payload
    if not isinstance(steps, list):
        raise ValueError("Expected verification JSON to contain a list of steps.")
    return [item for item in steps if isinstance(item, dict)]


def _select_steps(
    steps: List[Dict[str, Any]],
    *,
    start_step: Optional[int],
    end_step: Optional[int],
    max_steps: Optional[int],
) -> List[Dict[str, Any]]:
    filtered = []
    for item in steps:
        step_id = int(item.get("step", -1))
        if start_step is not None and step_id < start_step:
            continue
        if end_step is not None and step_id > end_step:
            continue
        filtered.append(item)
    if max_steps is not None:
        filtered = filtered[:max_steps]
    return filtered


def _denormalise(value: float, scale: float) -> float:
    if scale <= 0.0:
        return value
    return value * scale


def _has_prefix_features(features: Dict[str, float], prefix: str) -> bool:
    keys = (f"x_{prefix}", f"y_{prefix}", f"heading_{prefix}", f"speed_{prefix}")
    return any(key in features for key in keys)


def _state_from_features(
    features: Dict[str, float],
    prefix: str,
    params: HyperParameters,
) -> Dict[str, float]:
    pos_scale = params.feature_position_scale
    speed_scale = params.feature_speed_scale
    heading_scale = params.feature_heading_scale

    x_val = float(features.get(f"x_{prefix}", 0.0))
    y_val = float(features.get(f"y_{prefix}", 0.0))
    heading_val = float(features.get(f"heading_{prefix}", 0.0))
    speed_val = float(features.get(f"speed_{prefix}", 0.0))
    goal_x_val = float(features.get(f"x_goal_{prefix}", x_val))
    goal_y_val = float(features.get(f"y_goal_{prefix}", y_val))

    return {
        "x": _denormalise(x_val, pos_scale),
        "y": _denormalise(y_val, pos_scale),
        "heading": _denormalise(heading_val, heading_scale),
        "speed": _denormalise(speed_val, speed_scale),
        "goal_x": _denormalise(goal_x_val, pos_scale),
        "goal_y": _denormalise(goal_y_val, pos_scale),
    }


def _build_llm_feature_sequence(steps: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    if not steps:
        return []
    initial = steps[0].get("features", {})
    if not isinstance(initial, dict):
        initial = {}
    sequence = [initial]
    for item in steps[:-1]:
        next_features = item.get("simulation", {}).get("next_features")
        if isinstance(next_features, dict):
            sequence.append(next_features)
        else:
            sequence.append(sequence[-1])
    return sequence


def _apply_state(boat, state: Dict[str, float]) -> None:
    boat.x = float(state.get("x", boat.x))
    boat.y = float(state.get("y", boat.y))
    boat.h = float(state.get("heading", boat.h))
    boat.u = float(state.get("speed", boat.u))
    goal_x = state.get("goal_x")
    goal_y = state.get("goal_y")
    if goal_x is not None:
        boat.goal_x = float(goal_x)
    if goal_y is not None:
        boat.goal_y = float(goal_y)


def main() -> None:
    args = _build_parser().parse_args()

    payload = _load_json(args.verification)
    steps = _select_steps(
        _extract_steps(payload),
        start_step=args.start_step,
        end_step=args.end_step,
        max_steps=args.max_steps,
    )
    if not steps:
        raise SystemExit("No verification steps found for the requested range.")

    if args.render and not HAS_PYGAME:
        raise SystemExit("pygame is required for rendering but is not installed.")

    params = HyperParameters()
    try:
        apply_cli_overrides(params, args.hp)
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc))
    boat_params = build_boat_params(params)
    rudder_cfg = build_rudder_config(params)
    env_cfg = build_env_config(params, render=args.render)

    model_features = steps[0].get("features", {})
    if not isinstance(model_features, dict):
        model_features = {}

    llm_sequence = _build_llm_feature_sequence(steps)
    llm_features = llm_sequence[0] if llm_sequence else model_features

    states: List[Dict[str, float]]
    meta: Optional[dict] = None
    if args.scenario_kind != "auto":
        if args.scenario_index is None:
            raise SystemExit("--scenario-index is required when --scenario-kind is set.")
        scenario_request = build_scenario_request(params)
        scenarios = build_scenarios(scenario_request)
        selected_kind = ScenarioKind(args.scenario_kind)
        scenarios = [sc for sc in scenarios if sc.kind is selected_kind]
        if not scenarios:
            raise SystemExit("No scenarios available for the selected encounter kind.")
        if args.scenario_index < 1 or args.scenario_index > len(scenarios):
            raise SystemExit(
                f"Scenario index must be between 1 and {len(scenarios)} for {args.scenario_kind}."
            )
        selected_scenario = scenarios[args.scenario_index - 1]
        env = CrossingScenarioEnv(cfg=env_cfg, kin=boat_params, rudder_cfg=rudder_cfg)
        base_states, meta = scenario_states_for_env(env, selected_scenario)
        model_state = dict(base_states[0])
        llm_state = dict(base_states[0])
        model_state["name"] = "Model ASV"
        llm_state["name"] = "LLM Shadow"
        states = [model_state, llm_state]
        if len(base_states) > 1:
            tv_state = dict(base_states[1])
            tv_state["name"] = "Target Vessel"
            states.append(tv_state)
    else:
        model_state = _state_from_features(model_features, "ASV", params)
        llm_state = _state_from_features(llm_features, "ASV", params)
        model_state["name"] = "Model ASV"
        llm_state["name"] = "LLM Shadow"
        states = [model_state, llm_state]

        has_target = _has_prefix_features(model_features, "TV")
        if has_target:
            tv_state = _state_from_features(model_features, "TV", params)
            tv_state["name"] = "Target Vessel"
            states.append(tv_state)

        env = CrossingScenarioEnv(cfg=env_cfg, kin=boat_params, rudder_cfg=rudder_cfg)
    try:
        env.reset_from_states(states, meta=meta)

        for idx, item in enumerate(steps):
            step_id = int(item.get("step", idx))
            model_features = item.get("features", {})
            if not isinstance(model_features, dict):
                model_features = {}
            llm_features = (
                llm_sequence[idx] if idx < len(llm_sequence) else model_features
            )

            model_state = _state_from_features(model_features, "ASV", params)
            llm_state = _state_from_features(llm_features, "ASV", params)
            _apply_state(env.ships[0], model_state)
            _apply_state(env.ships[1], llm_state)

            if len(env.ships) > 2:
                tv_state = _state_from_features(model_features, "TV", params)
                _apply_state(env.ships[2], tv_state)

            model_rudder = item.get("model", {}).get("rudder_cmd")
            model_thr = item.get("model", {}).get("throttle")
            llm_rudder = item.get("llm", {}).get("rudder_cmd")
            llm_thr = item.get("llm", {}).get("throttle")

            env.ships[0].last_rudder_cmd = float(model_rudder or 0.0)
            env.ships[0].last_thr = int(model_thr or 0)
            env.ships[1].last_rudder_cmd = float(llm_rudder or 0.0)
            env.ships[1].last_thr = int(llm_thr or 0)
            if len(env.ships) > 2:
                env.ships[2].last_rudder_cmd = 0.0
                env.ships[2].last_thr = 0

            env.step_index = step_id
            env.time = step_id * env.cfg.dt

            if args.render:
                env.set_debug_overlay(
                    {
                        "step": step_id,
                        "rudder_cmd_for_arrow": float(model_rudder or 0.0),
                    }
                )
                env.render()
                env.set_debug_overlay(None)
                if args.step_delay:
                    time.sleep(args.step_delay)
    finally:
        env.close()

    if not args.render:
        print(f"Prepared {len(steps)} steps from {args.verification} (render disabled).")


if __name__ == "__main__":
    main()
