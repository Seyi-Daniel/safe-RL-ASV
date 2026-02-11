"""Central numeric configuration for unified feature-based RL ASV project."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EnvParams:
    # world / simulation
    world_w: float = 500.0
    world_h: float = 500.0
    dt: float = 0.10
    substeps: int = 2
    episode_seconds: float = 120.0
    seed: int | None = 7

    # rendering
    pixels_per_meter: float = 2.0
    render_fps: int = 60
    show_grid: bool = True
    show_spawn_rings: bool = True

    # vessel dynamics
    max_speed: float = 12.0
    min_speed: float = 0.0
    accel_rate: float = 1.0
    decel_rate: float = 1.2
    turn_rate_rad_s: float = math.radians(20.0)

    # spawn & goals
    spawn_margin: float = 20.0
    goal_radius: float = 10.0
    spawn_ring_radius: float = 35.0
    goal_ring_radius: float = 180.0


@dataclass
class RewardParams:
    # core objective
    living_penalty: float = -0.002
    progress_weight: float = 0.03
    goal_bonus: float = 8.0

    # terminal safety
    out_of_bounds_penalty: float = -8.0


@dataclass
class TrainParams:
    # DDQN training
    episodes: int = 600
    batch_size: int = 256
    replay_size: int = 200_000
    min_replay: int = 10_000
    gamma: float = 0.995
    learning_rate: float = 2e-4
    target_update: int = 4000

    # epsilon schedule (linear in global environment steps)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 300_000

    # network architecture: 6 -> hidden -> hidden -> 9 actions
    hidden_dim: int = 256

    # reproducibility / checkpoints
    seed: int = 7
    save_every: int = 20
    out_dir: str = "unified-feature-rl/runs"
