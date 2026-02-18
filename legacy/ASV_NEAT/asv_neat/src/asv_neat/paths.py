"""Shared filesystem locations for CLI utilities and reports."""
from __future__ import annotations

from pathlib import Path


# The repository root sits three levels above this file:
#   asv_neat/src/asv_neat/paths.py -> asv_neat/src -> asv_neat -> <repo>
PROJECT_ROOT = Path(__file__).resolve().parents[3]

WINNERS_DIRNAME = "winners"


def winner_directory() -> Path:
    """Return the repository-level directory where genomes are stored."""

    return PROJECT_ROOT / WINNERS_DIRNAME


def default_winner_path(scenario_kind: str) -> Path:
    """Return the canonical path for a scenario-specific winning genome."""

    return winner_directory() / f"{scenario_kind}_winner.pkl"


def default_species_archive() -> Path:
    """Return the directory used to archive per-species elites."""

    return winner_directory() / "species_archive"


__all__ = [
    "PROJECT_ROOT",
    "WINNERS_DIRNAME",
    "default_species_archive",
    "default_winner_path",
    "winner_directory",
]

