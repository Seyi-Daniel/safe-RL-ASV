#!/usr/bin/env python3
"""Render a saved NEAT genome on the deterministic COLREGs encounters."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    import neat
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("neat-python must be installed to run the demo script.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asv_neat import HyperParameters, apply_cli_overrides, build_scenarios  # noqa: E402
from asv_neat.cli_helpers import (  # noqa: E402
    SCENARIO_KIND_CHOICES,
    build_boat_params,
    build_env_config,
    build_scenario_request,
    build_rudder_config,
    filter_scenarios_by_kind,
    summarise_genome,
)
from asv_neat.paths import default_winner_path  # noqa: E402


def build_parser(hparams: HyperParameters) -> argparse.ArgumentParser:
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
        help="Select which encounter family should be replayed.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable pygame visualisation for each selected scenario.",
    )
    parser.add_argument(
        "--list-hyperparameters",
        action="store_true",
        help="List available hyperparameters and exit without replaying.",
    )
    parser.add_argument(
        "--hp",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a hyperparameter (repeatable).",
    )
    return parser


def print_hyperparameters(hparams: HyperParameters) -> None:
    print("Available hyperparameters (NAME = default | description):")
    for name, value, help_text in hparams.iter_documentation():
        description = help_text or ""
        print(f"  {name} = {value!r}\n      {description}")


def load_winner(path: Path) -> neat.DefaultGenome:
    with path.open("rb") as fh:
        return pickle.load(fh)


def main(argv: Optional[list[str]] = None) -> None:
    hparams = HyperParameters()
    parser = build_parser(hparams)
    args = parser.parse_args(argv)

    if args.list_hyperparameters:
        print_hyperparameters(hparams)
        return

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

    winner_path = args.winner
    if winner_path is None:
        winner_path = default_winner_path(args.scenario_kind)

    if not winner_path.exists():
        parser.error(f"Winner file '{winner_path}' does not exist.")

    winner = load_winner(winner_path)
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(args.config),
    )

    boat_params = build_boat_params(hparams)
    rudder_cfg = build_rudder_config(hparams)
    env_cfg = build_env_config(hparams, render=args.render)

    summarise_genome(
        winner,
        neat_config,
        scenarios,
        hparams,
        boat_params,
        rudder_cfg,
        env_cfg,
        render=args.render,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

