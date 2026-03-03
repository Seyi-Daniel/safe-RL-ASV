from __future__ import annotations

from dataclasses import dataclass

from simulations.common.environment import SingleVessel2FeatureEnv
from simulations.common.hyperparameters import EnvParams, RewardParams


@dataclass(frozen=True)
class SimulationRuntimeConfig:
    """Simulation-only runtime config, intentionally isolated from train.py CLI wiring."""

    episode_seconds: float
    seed: int
    dcrit_factor: float
    step_risk_logs: bool


def build_sampling_env_params(cfg: SimulationRuntimeConfig) -> EnvParams:
    return EnvParams(
        episode_seconds=cfg.episode_seconds,
        seed=cfg.seed,
        vessel2_min_goal_arc_distance_from_start=0.0,
        adaptive_vessel2_min_goal_arc_from_speed=True,
        vessel2_min_goal_dcrit_factor=cfg.dcrit_factor,
        require_reset_viable_takeover_path=False,
        enable_no_takeover_early_done=False,
        enable_step_risk_logging=False,
    )


def build_playback_env_params(cfg: SimulationRuntimeConfig) -> EnvParams:
    return EnvParams(
        episode_seconds=cfg.episode_seconds,
        seed=cfg.seed,
        vessel2_min_goal_arc_distance_from_start=0.0,
        adaptive_vessel2_min_goal_arc_from_speed=True,
        vessel2_min_goal_dcrit_factor=cfg.dcrit_factor,
        require_reset_viable_takeover_path=False,
        enable_no_takeover_early_done=False,
        enable_step_risk_logging=cfg.step_risk_logs,
    )


def create_sim_env(envp: EnvParams, render: bool) -> SingleVessel2FeatureEnv:
    """Factory boundary for simulation environments.

    Keeping environment creation behind this module makes simulation-only evolution
    independent from train.py plumbing.
    """
    return SingleVessel2FeatureEnv(envp, RewardParams(), render=render)
