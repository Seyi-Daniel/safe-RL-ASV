#!/usr/bin/env python3
"""Train the give-way vessel controller across the deterministic COLREGs encounters."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    import neat
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("neat-python must be installed to run the training script.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asv_neat import (  # noqa: E402
    HyperParameters,
    apply_cli_overrides,
    build_scenarios,
    train_population,
)
from asv_neat.cli_helpers import (  # noqa: E402
    SCENARIO_KIND_CHOICES,
    build_boat_params,
    build_env_config,
    build_scenario_request,
    build_rudder_config,
    filter_scenarios_by_kind,
    summarise_genome,
)
from asv_neat.paths import (  # noqa: E402
    default_species_archive,
    default_winner_path,
    winner_directory,
)


def build_parser(hparams: HyperParameters) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "neat_crossing.cfg",
        help="Path to the neat-python configuration file.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of evolutionary generations to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible runs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory in which neat-python checkpoints should be stored.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Number of generations between checkpoint saves.",
    )
    parser.add_argument(
        "--save-winner",
        type=Path,
        default=None,
        help="Optional path for pickling the winning genome after training.",
    )
    parser.add_argument(
        "--species-archive",
        type=str,
        default=str(default_species_archive()),
        help=(
            "Directory to archive the top genomes for each species per generation. "
            "Use an empty string to skip archiving."
        ),
    )
    parser.add_argument(
        "--species-top-n",
        type=int,
        default=3,
        help="Number of genomes per species to preserve each generation.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable pygame visualisation while summarising the winning genome.",
    )
    parser.add_argument(
        "--scenario-kind",
        choices=SCENARIO_KIND_CHOICES,
        default="all",
        help="Select which encounter family should be used for training.",
    )
    parser.add_argument(
        "--list-hyperparameters",
        action="store_true",
        help="List available hyperparameters and exit without training.",
    )
    parser.add_argument(
        "--hp",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a hyperparameter (repeatable). See --list-hyperparameters for names.",
    )
    return parser


def print_hyperparameters(hparams: HyperParameters) -> None:
    print("Available hyperparameters (NAME = default | description):")
    for name, value, help_text in hparams.iter_documentation():
        description = help_text or ""
        print(f"  {name} = {value!r}\n      {description}")


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

    boat_params = build_boat_params(hparams)
    rudder_cfg = build_rudder_config(hparams)
    env_cfg = build_env_config(hparams, render=False)

    species_archive = Path(args.species_archive) if args.species_archive else None

    result = train_population(
        config_path=args.config,
        scenarios=scenarios,
        env_cfg=env_cfg,
        boat_params=boat_params,
        rudder_cfg=rudder_cfg,
        params=hparams,
        generations=args.generations,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        species_archive_dir=species_archive,
        species_top_n=args.species_top_n,
    )

    winner_path = args.save_winner
    auto_generated = False
    if winner_path is None:
        auto_generated = True
        winner_path = default_winner_path(args.scenario_kind)
    winner_directory().mkdir(parents=True, exist_ok=True)

    if winner_path is not None:
        winner_path.parent.mkdir(parents=True, exist_ok=True)
        with winner_path.open("wb") as fh:
            pickle.dump(result.winner, fh)
        if auto_generated:
            print(f"Saved winning genome to {winner_path} (auto-generated path)")
        else:
            print(f"Saved winning genome to {winner_path}")

    render_cfg = build_env_config(hparams, render=args.render)
    summarise_genome(
        result.winner,
        result.config,
        scenarios,
        hparams,
        boat_params,
        rudder_cfg,
        render_cfg,
        render=args.render,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

