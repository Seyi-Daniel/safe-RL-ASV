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

    # vessel dynamics (ASV_NEAT-style rudder-limited yaw + continuous throttle)
    max_speed: float = 7.0
    min_speed: float = 0.0
    accel_rate: float = 0.20
    decel_rate: float = 0.05
    brake_rate: float = 0.20

    rudder_max_angle_rad: float = math.radians(35.0)
    rudder_max_yaw_rate_rad_s: float = 0.25
    rudder_max_rate_rad_s: float = math.radians(40.0)

    throttle_slew_rate: float = 0.4
    throttle_deadband: float = 0.02

    # spawn & goals
    spawn_margin: float = 20.0
    goal_radius: float = 10.0
    spawn_ring_radius: float = 35.0
    goal_ring_radius: float = 180.0
    vessel_outline_radius: float = 4.0

    # moving target vessel on outer-circle to outer-circle arc
    target_outer_radius: float = 180.0
    target_min_speed: float = 0.5
    target_max_speed: float = 7.0
    target_arc_min_deg: float = 20.0
    target_arc_max_deg: float = 110.0


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

    # network architecture: 10 -> hidden -> hidden -> 9 actions
    hidden_dim: int = 256

    # reproducibility / checkpoints
    seed: int = 7
    save_every: int = 20
    out_dir: str = "unified-feature-rl/runs"
