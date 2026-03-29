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
    num_vessels: int = 2

    # rendering
    pixels_per_meter: float = 2.0
    render_fps: int = 60
    show_grid: bool = True
    show_spawn_rings: bool = True
    enable_step_risk_logging: bool = False
    debug_multi_vessel_status: bool = False

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
    goal_radius: float = 2.0
    goal_ring_radius: float = 180.0
    vessel_outline_radius: float = 4.0

    # shared big circle geometry (both vessel goals lie on this circumference)
    vessel2_outer_radius: float = 180.0
    vessel2_min_speed: float = 0.5
    vessel2_max_speed: float = 7.0
    vessel2_min_goal_arc_distance_from_start: float = 80.0  # Interpreted as straight-line (chord) distance
    adaptive_vessel2_min_goal_arc_from_speed: bool = True
    vessel2_min_goal_dcrit_factor: float = 1.1
    # Legacy alias from max-arc naming; keep for backward compatibility.
    vessel2_max_goal_arc_distance_from_start: float | None = None
    # Legacy alias (chord-distance naming); keep for backward compatibility.
    vessel2_max_goal_distance_from_start: float | None = None
    # pure pursuit controller parameters for vessel 2 scripted path
    pp_lookahead_factor: float = 2.0      # lookahead distance = factor × turning_radius
    pp_heading_gain_deg: float = 25.0     # proportional gain divisor for heading error -> rudder cmd

    # additive extra-vessel traffic configuration (kept vessel2-like by default)
    extra_vessel_spawn_mode: str = "perimeter"
    extra_vessel_min_speed: float = 0.5
    extra_vessel_max_speed: float = 7.0

    # COLREGS risk/takeover gating
    colregs_head_on_half_angle_deg: float = 10.0  # Engineering approximation for simulation stability.
    colregs_crossing_starboard_max_deg: float = 112.5
    colregs_overtaking_aft_max_deg: float = 247.5
    colregs_speed_eps: float = 0.2
    simplified_head_on_single_giveway: bool = True

    dcpa_risk_threshold: float = 20.0
    tcpa_risk_threshold: float = 20.0
    overtaking_clear_distance: float = 40.0
    overtaking_clear_steps_required: int = 5

    fallback_starboard_rudder_cmd: float = 0.6
    fallback_headon_throttle_cmd: float = -0.15
    fallback_crossing_throttle_cmd: float = -0.05
    standon_escalation_tcpa: float = 20.0
    standon_escalation_dcpa: float = 12.0
    standon_escalation_persistence_steps: int = 3
    lock_enter_persistence_steps: int = 2
    collision_distance: float = 8.0
    near_miss_distance: float = 15.0
    safe_pass_distance: float = 25.0
    sensor_range: float = 140.0


@dataclass
class RewardParams:
    """Reward coefficients for the current 2-vessel training setup.

    Notes on compatibility fields:
    - Some legacy shaping coefficients are intentionally kept so older configs
      can still deserialize.
    - Deprecated fields marked below are inactive in current reward code.
    """

    # core objective
    living_penalty: float = -0.002
    progress_weight: float = 0.03
    goal_bonus: float = 8.0

    # active shared/global safety terms
    collision_penalty: float = -20.0
    near_miss_penalty: float = -2.5
    unsafe_proximity_penalty_weight: float = 0.05
    safe_pass_bonus: float = 0.5
    oscillation_penalty_weight: float = 0.02

    # active scenario-shaping thresholds (post-takeover interpretation)
    starboard_min_rudder: float = 0.1
    port_max_rudder: float = -0.1
    safe_dcpa_threshold: float = 20.0
    danger_dcpa_threshold: float = 10.0

    # Deprecated / inactive legacy compatibility fields.
    # These are currently not read by active reward computation.
    give_way_early_action_bonus: float = 0.2
    late_action_penalty: float = -0.4
    crossing_ahead_penalty: float = -0.5
    early_action_tcpa_threshold: float = 25.0
    late_action_tcpa_threshold: float = 10.0
    out_of_bounds_penalty: float = -8.0
    stand_on_hold_bonus: float = 0.1
    stand_on_unnecessary_action_penalty: float = -0.2


@dataclass
class TrainParams:
    # DDPG-style continuous-control training
    episodes: int = 600
    batch_size: int = 256
    replay_size: int = 200_000
    min_replay: int = 10_000
    gamma: float = 0.995
    learning_rate: float = 2e-4
    target_update: int = 4000

    # epsilon-greedy exploration schedule (linear in global environment steps)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 300_000

    # network backbone: obs_dim(=96 in current radar observation design) -> hidden -> hidden -> 2 continuous actions
    hidden_dim: int = 256

    # reproducibility / checkpoints
    seed: int = 7
    save_every: int = 20
    out_dir: str = "runs"

    # training-only episode sampling thresholds (None -> inherit env CLI thresholds)
    sampling_dcpa_threshold: float | None = None
    sampling_tcpa_threshold: float | None = None
    # optional scripted seed-screening horizon caps (None -> disabled/unlimited)
    sampling_screen_max_steps: int | None = None
    sampling_screen_max_seconds: float | None = None
