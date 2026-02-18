#!/usr/bin/env python3
"""Capture simulation traces and frames for later explanation."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import pygame
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("The 'pygame' package is required to capture render frames.") from exc

try:  # pragma: no cover - optional dependency
    import neat
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("neat-python must be installed to run the capture script.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asv_neat import (  # noqa: E402
    HyperParameters,
    apply_cli_overrides,
    build_scenarios,
    episode_cost,
    simulate_episode,
)
from asv_neat.cli_helpers import (  # noqa: E402
    SCENARIO_KIND_CHOICES,
    build_boat_params,
    build_env_config,
    build_scenario_request,
    build_rudder_config,
    filter_scenarios_by_kind,
)
from asv_neat.config import BoatParams, EnvConfig, RudderParams  # noqa: E402
from asv_neat.env import CrossingScenarioEnv  # noqa: E402
from asv_neat.neat_training import TraceCallback  # noqa: E402
from asv_neat.paths import default_winner_path  # noqa: E402
from asv_neat.scenario import EncounterScenario  # noqa: E402

FRAME_DIRNAME = "frames"


def _serialise_vessel(state) -> dict:
    data = asdict(state)
    goal = data.get("goal")
    if goal is not None:
        data["goal"] = list(goal)
    return data


def _build_parser(hparams: HyperParameters) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "neat_crossing.cfg",
        help="Path to the neat-python configuration file that matches the saved genome.",
    )
    parser.add_argument(
        "--winner",
        type=Path,
        default=None,
        help="Path to the pickled winning genome. Defaults to winners/<scenario>_winner.pkl.",
    )
    parser.add_argument(
        "--scenario-kind",
        choices=SCENARIO_KIND_CHOICES,
        default="all",
        help="Select which encounter family should be captured.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "captured_episodes",
        help="Directory where traces and frames will be written.",
    )
    parser.add_argument(
        "--hp",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a hyperparameter (repeatable).",
    )
    return parser


def _trace_recorder(container: List[dict]) -> TraceCallback:
    def _record(payload: dict) -> None:
        if "features" not in payload and "obs" in payload:
            payload = {**payload, "features": list(payload["obs"])}
        container.append(payload)

    return _record


def _scenario_metadata(
    scenario: EncounterScenario,
    metrics,
    cost: float,
) -> dict:
    metadata = {
        "scenario_kind": scenario.kind.value,
        "requested_bearing_deg": scenario.requested_bearing,
        "bearing_frame": scenario.bearing_frame,
        "agent": _serialise_vessel(scenario.agent),
        "stand_on": _serialise_vessel(scenario.stand_on),
        "metrics": asdict(metrics),
        "episode_cost": cost,
    }
    return metadata


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _save_frame(path: Path, surface) -> None:
    if surface is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(path))


def _load_winner(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def capture_scenarios(
    *,
    winner_path: Path,
    config_path: Path,
    scenarios: Iterable[EncounterScenario],
    hparams: HyperParameters,
    boat_params: BoatParams,
    rudder_cfg: RudderParams,
    env_cfg: EnvConfig,
    output_dir: Path,
) -> None:
    winner = _load_winner(winner_path)
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    network = neat.nn.FeedForwardNetwork.create(winner, neat_config)

    for idx, scenario in enumerate(scenarios, start=1):
        scenario_dir = output_dir / f"{idx:02d}_{scenario.kind.value}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        frame_dir = scenario_dir / FRAME_DIRNAME
        env = CrossingScenarioEnv(cfg=env_cfg, kin=boat_params, rudder_cfg=rudder_cfg)
        trace: List[dict] = []
        try:
            recorder = _trace_recorder(trace)

            def frame_recorder(step: int, surf) -> None:
                _save_frame(frame_dir / f"frame_{step:03d}.png", surf)

            metrics = simulate_episode(
                env,
                scenario,
                network,
                hparams,
                render=True,
                trace_callback=recorder,
                frame_callback=frame_recorder,
            )
        finally:
            env.close()
        cost = episode_cost(metrics, hparams)
        metadata = _scenario_metadata(scenario, metrics, cost)
        _write_json(scenario_dir / "metadata.json", metadata)
        _write_json(scenario_dir / "trace.json", trace)
        print(
            f"Captured scenario {idx:02d} [{scenario.kind.value}] steps={metrics.steps:4d} "
            f"cost={cost:7.2f} outputs saved to {scenario_dir}"
        )


def main(argv: Optional[list[str]] = None) -> None:
    hparams = HyperParameters()
    parser = _build_parser(hparams)
    args = parser.parse_args(argv)

    try:
        apply_cli_overrides(hparams, args.hp)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))

    scenario_request = build_scenario_request(hparams)
    scenarios = filter_scenarios_by_kind(
        build_scenarios(scenario_request), args.scenario_kind
    )
    if not scenarios:
        parser.error("No scenarios available for the requested encounter kind.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    winner_path = args.winner
    if winner_path is None:
        winner_path = default_winner_path(args.scenario_kind)
    if not winner_path.exists():
        parser.error(f"Winner file '{winner_path}' does not exist.")

    boat_params = build_boat_params(hparams)
    rudder_cfg = build_rudder_config(hparams)
    env_cfg = build_env_config(hparams, render=True)

    capture_scenarios(
        winner_path=winner_path,
        config_path=args.config,
        scenarios=scenarios,
        hparams=hparams,
        boat_params=boat_params,
        rudder_cfg=rudder_cfg,
        env_cfg=env_cfg,
        output_dir=output_dir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
