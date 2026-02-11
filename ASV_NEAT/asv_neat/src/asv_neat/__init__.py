"""Convenience exports for the NEAT-controlled COLREGs crossing project."""

from .boat import Boat
from .config import BoatParams, EnvConfig, RudderParams
from .env import CrossingScenarioEnv
from .hyperparameters import HyperParameters, apply_cli_overrides
from .neat_training import (
    EpisodeMetrics,
    SpeciesElitesReporter,
    TrainingResult,
    build_scenarios,
    episode_cost,
    evaluate_population,
    simulate_episode,
    train_population,
)
from .scenario import (
    ScenarioKind,
    STAND_ON_BEARINGS_DEG,
    CROSSING_BEARINGS_DEG,
    HEAD_ON_BEARINGS_DEG,
    OVERTAKING_BEARINGS_DEG,
    EncounterScenario,
    ScenarioRequest,
    compute_crossing_geometry,
    compute_head_on_geometry,
    compute_overtaking_geometry,
    iter_scenarios,
    default_scenarios,
    scenario_states_for_env,
)

__all__ = [
    "Boat",
    "BoatParams",
    "EnvConfig",
    "RudderParams",
    "CrossingScenarioEnv",
    "HyperParameters",
    "apply_cli_overrides",
    "EpisodeMetrics",
    "SpeciesElitesReporter",
    "TrainingResult",
    "build_scenarios",
    "episode_cost",
    "evaluate_population",
    "simulate_episode",
    "train_population",
    "ScenarioKind",
    "STAND_ON_BEARINGS_DEG",
    "CROSSING_BEARINGS_DEG",
    "HEAD_ON_BEARINGS_DEG",
    "OVERTAKING_BEARINGS_DEG",
    "EncounterScenario",
    "ScenarioRequest",
    "compute_crossing_geometry",
    "compute_head_on_geometry",
    "compute_overtaking_geometry",
    "iter_scenarios",
    "default_scenarios",
    "scenario_states_for_env",
]
