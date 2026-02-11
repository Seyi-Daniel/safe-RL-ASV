"""Central numeric configuration for unified feature-based RL ASV project.

Edit values here to tune environment dynamics, reward shaping, and training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EnvParams:
    # world / simulation
    world_w: float = 300.0
    world_h: float = 300.0
    dt: float = 0.10
    substeps: int = 2
    episode_seconds: float = 120.0
    seed: int | None = 7

    # rendering
    pixels_per_meter: float = 3.0
    render_fps: int = 60
    show_grid: bool = True
    show_sectors: bool = True

    # vessel dynamics
    max_speed: float = 12.0
    min_speed: float = 0.0
    accel_rate: float = 1.0
    decel_rate: float = 1.2
    turn_rate_rad_s: float = math.radians(20.0)

    # spawn & goals
    spawn_margin: float = 20.0
    min_start_separation: float = 40.0
    goal_radius: float = 10.0
    goal_min_distance: float = 100.0

    # sensing
    sensor_range: float = 140.0
    sectors: int = 12


@dataclass
class RewardParams:
    # core objective
    living_penalty: float = -0.002
    progress_weight: float = 0.03
    goal_bonus: float = 8.0

    # safety/terminal
    collision_radius: float = 7.0
    collision_penalty: float = -15.0
    out_of_bounds_penalty: float = -8.0

    # CPA shaping
    cpa_horizon: float = 60.0
    tcpa_decay: float = 20.0
    dcpa_scale: float = 120.0
    risk_weight: float = 0.12

    # COLREGs shaping
    colregs_window_deg: float = 112.5
    colregs_correct_bonus: float = 0.08
    colregs_wrong_penalty: float = 0.08

    # DCPA "super action"
    dcpa_improve_weight: float = 0.25
    dcpa_worse_weight: float = 0.12


@dataclass
class TrainParams:
    # CEM policy search (continuous actions)
    episodes: int = 200
    elite_frac: float = 0.2
    population: int = 32
    eval_rollouts: int = 1

    # policy architecture: 10 -> hidden -> 2
    hidden_dim: int = 32

    # sampling distribution over parameters
    init_std: float = 0.5
    min_std: float = 0.03
    std_decay: float = 0.995

    # reproducibility / checkpoints
    seed: int = 7
    save_every: int = 10
    out_dir: str = "unified-feature-rl/runs"
