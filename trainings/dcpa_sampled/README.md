# `trainings/dcpa_sampled`

## What this folder is

`trainings/dcpa_sampled` contains a self-contained reinforcement-learning training setup for COLREGS-aware collision-avoidance experiments using a sampled two-vessel encounter generator.

The current implementation priority is **high-quality 2-vessel behavior** (`vessel1` + `vessel2`). Multi-vessel hooks exist (vessel3+), but this folder is currently optimized around the foundational two-vessel case.

At a high level, training teaches a shared continuous-control policy to take over give-way vessels during risky encounters and produce safer maneuvers while still making progress toward goals.

---

## High-level environment behavior

The active setup is built around the following pattern:

- **Vessel 1 starts at the world center** and is treated as the main learner-facing vessel in compatibility paths.
- **Vessel 2 starts on a perimeter "big circle"** around the center and also has a goal on that circle.
- Initial speeds, headings, and goal angles are sampled so episodes cover different geometries (head-on, crossing, overtaking).
- During rollout, the environment computes TCPA/DCPA risk and COLREGS roles from current geometry.
- When a risky encounter is active and a vessel is designated give-way, the model can take over that vessel.
- Current engineering focus is to make this 2-vessel takeover behavior robust and stable.

---

## File-by-file overview

### `environment.py`

Implements `SingleVessel2FeatureEnv`, including:

- world state and vessel dynamics,
- sampled reset logic (center vessel + perimeter vessel),
- COLREGS geometry/risk classification,
- RL takeover gating and per-vessel control-latch logic,
- reward calculation,
- 96-dimensional vessel-centric observations,
- optional rendering/debug telemetry.

### `hyperparameters.py`

Defines dataclass config groups:

- `EnvParams`: simulation geometry/dynamics, risk thresholds, takeover gating, sensor settings.
- `RewardParams`: reward coefficients and compatibility placeholders.
- `TrainParams`: DDPG-style training, exploration, replay, checkpoint/output settings.

### `policy.py`

Defines the neural networks used by training:

- `ContinuousActor`: maps observation to 2 normalized actions (`[rudder_cmd, throttle_cmd]` in `[-1, 1]`).
- `ContinuousCritic`: estimates Q-value from concatenated state-action input.

### `train.py`

Implements the full training/evaluation entrypoint:

- argument parsing and run configuration,
- DDPG-style agent and replay buffer,
- seed sampling that prefers episodes meeting DCPA/TCPA criteria,
- per-step action collection for RL-active vessels,
- per-vessel replay insertion and optimization,
- eval-only scenario reporting,
- checkpoint + training history persistence.

---

## Observation design (96-dim radar format)

Each controlled vessel receives a **vessel-centric 96D observation**:

- **Radar block: 90 dims** = `9 sectors x 10 features`
- **Own-ship block: 6 dims**
- Total = **96**

### 9 radar sectors

The relative-bearing sectors are:

1. `[350 deg, 10 deg)` (ahead wrap sector)
2. `[10 deg, 40 deg)`
3. `[40 deg, 75 deg)`
4. `[75 deg, 112.5 deg)`
5. `[112.5 deg, 180 deg)`
6. `[180 deg, 247.5 deg)`
7. `[247.5 deg, 285 deg)`
8. `[285 deg, 320 deg)`
9. `[320 deg, 350 deg)`

Only the nearest in-range contact per sector is kept. Empty sectors are zero-filled.

### 10 features per sector

Per occupied sector:

1. occupied flag
2. normalized distance
3. bearing sin
4. bearing cos
5. relative-heading sin
6. relative-heading cos
7. normalized target speed
8. normalized closing speed (tanh-scaled signed value)
9. normalized TCPA term
10. normalized DCPA term

### 6 own-vessel features

1. normalized own speed
2. normalized distance to own goal
3. sin(goal bearing in vessel frame)
4. cos(goal bearing in vessel frame)
5. normalized rudder state
6. normalized throttle state

### Bearing convention

Relative bearing is encoded as:

- `0 deg` = dead ahead,
- `90 deg` = port,
- `270 deg` = starboard,
- values in `[0, 360)`.

### Sensor range and normalization summary

- Contact inclusion is limited by `sensor_range` (default 140 m).
- Distance-like quantities are clipped to `[0, 1]` with scale factors from sensor/risk settings.
- Angular values are represented with sin/cos pairs.
- Closing speed is bounded with `tanh`.

---

## Control and training design

The training loop uses a **shared policy** and applies it vessel-wise:

- At each step, the same actor network is queried separately for each currently RL-active vessel.
- Observations are per-vessel (`get_obs_for_vessel(vessel_id)`).
- Actions are then applied in the same environment step via a vessel-id to action map.
- The environment computes **per-vessel rewards** and returns them in `info["reward_by_vessel"]` (plus compatibility fields `reward_v1`/`reward_v2`).
- Replay is populated with **one transition per controlled vessel** per step.
- Model takeover is controlled by a **persistent latch** per vessel: once latched for a designated give-way vessel, control remains with the model until that vessel reaches goal (or episode reset).

---

## Reward design

Current reward is a combination of local progress and shared safety terms.

### 1) Progress reward with heading-aware shaping

For each vessel, progress reward is based on:

- change in goal distance (primary term), and
- a small heading-improvement shaping term.

This is active for all vessels.

### 2) Shared/global safety terms

A shared component is added to each vessel reward, including:

- living penalty,
- collision penalty,
- near-miss penalty,
- unsafe-proximity penalty based on separation vs safe-pass distance,
- safe-pass bonus after a risky encounter clears safely,
- rudder oscillation penalty (local to each vessel but safety-oriented).

### 3) Simplified scenario-specific shaping

When a vessel is RL-active, additional scenario shaping is applied:

- **head-on / crossing:** rudder-direction preference (starboard encouraged, port discouraged),
- **overtaking:** DCPA-threshold shaping (reward safer clearance, penalize dangerous clearance).

### 4) Legacy/deprecated compatibility fields

`RewardParams` keeps several older coefficients for config compatibility, but active reward logic does not currently use them:

- `give_way_early_action_bonus`
- `late_action_penalty`
- `crossing_ahead_penalty`
- `early_action_tcpa_threshold`
- `late_action_tcpa_threshold`
- `out_of_bounds_penalty`
- `stand_on_hold_bonus`
- `stand_on_unnecessary_action_penalty`

---

## Important parameters (plain-English guide)

## `EnvParams` (environment + scenario generation)

Key groups to tune most often:

- **World/simulation timing** (`world_w`, `world_h`, `dt`, `substeps`, `episode_seconds`): controls integration resolution and episode horizon.
- **Vessel motion envelope** (`max_speed`, accel/decel/brake rates, rudder/yaw/rudder-rate limits): sets what maneuvers are physically possible.
- **Spawn/goal geometry** (`goal_ring_radius`, `vessel2_outer_radius`, `goal_radius`): determines where vessels/goals are placed and what counts as goal completion.
- **Vessel-2 path behavior** (`vessel2_min_speed`, `vessel2_max_speed`, pure-pursuit lookahead/gain): governs scripted nominal motion for vessel2 when not RL-controlled.
- **Risk/takeover thresholds** (`dcpa_risk_threshold`, `tcpa_risk_threshold`, `rl_takeover_distance`): controls when encounters are considered risky and when takeover becomes viable.
- **Early termination controls** (`enable_no_takeover_early_done`, `no_takeover_early_done_steps`): can stop non-takeover episodes early (disabled in training script defaults).
- **Safety distance bands** (`collision_distance`, `near_miss_distance`, `safe_pass_distance`): used by termination and reward safety shaping.
- **Sensor model** (`sensor_range`): radar cutoff and normalization scale anchor.
- **Multi-vessel extension** (`num_vessels`, `extra_vessel_*`): optional additive traffic beyond 2 vessels.

## `RewardParams` (objective coefficients)

Most impactful active terms:

- **`progress_weight`**: scales distance-to-goal improvement reward.
- **`goal_bonus`**: large terminal encouragement for reaching goals.
- **`living_penalty`**: slight per-step cost to discourage unnecessary delay.
- **`collision_penalty` / `near_miss_penalty`**: global safety punishment.
- **`unsafe_proximity_penalty_weight`**: continuous penalty when vessels are too close.
- **`safe_pass_bonus`**: reward for safely clearing a previously risky encounter.
- **`oscillation_penalty_weight`**: discourages frequent rudder sign flips.
- **Scenario shaping thresholds** (`starboard_min_rudder`, `port_max_rudder`, `safe_dcpa_threshold`, `danger_dcpa_threshold`): define simple behavior preferences for give-way control.

Compatibility-only deprecated fields remain in the dataclass so old configs still load, but they are currently inactive in reward computation.

## `TrainParams` (DDPG-style optimization)

- **Data/optimization volume** (`episodes`, `batch_size`, `replay_size`, `min_replay`): controls how much experience is gathered and when learning starts.
- **Temporal learning** (`gamma`): future-return weighting.
- **Optimizer** (`learning_rate`): actor/critic learning step size.
- **Exploration schedule** (`eps_start`, `eps_end`, `eps_decay_steps`): epsilon-greedy random action probability over environment steps.
- **Model capacity** (`hidden_dim`): backbone width for actor/critic MLPs.
- **Reproducibility/output** (`seed`, `save_every`, `out_dir`): deterministic seeding and checkpoint cadence/location.
- **Training seed-sampling thresholds** (`sampling_dcpa_threshold`, `sampling_tcpa_threshold`): optional overrides for which episodes are accepted during sampling.

---

## Training and evaluation usage

Run from repository root.

### Basic training

```bash
python trainings/dcpa_sampled/train.py
```

### Training with common options

```bash
python trainings/dcpa_sampled/train.py \
  --episodes 800 \
  --batch-size 256 \
  --replay-size 200000 \
  --min-replay 10000 \
  --gamma 0.995 \
  --learning-rate 2e-4 \
  --eps-start 1.0 \
  --eps-end 0.05 \
  --eps-decay-steps 300000 \
  --hidden-dim 256 \
  --seed 7 \
  --episode-seconds 500 \
  --num-vessels 2 \
  --dcpa-threshold 10 \
  --tcpa-threshold 10 \
  --sampling-dcpa-threshold 10 \
  --sampling-tcpa-threshold 10 \
  --dcpa-sample-max-tries 0 \
  --max-sampling-steps-per-attempt 0 \
  --sampling-logs \
  --save-every 20 \
  --out-dir runs/dcpa_sampled \
  --render
```

### Evaluation-only mode

```bash
python trainings/dcpa_sampled/train.py \
  --eval-only \
  --out-dir runs/dcpa_sampled \
  --eval-checkpoint runs/dcpa_sampled/ddqn_policy.pt \
  --eval-scenario all \
  --eval-episodes 30 \
  --eval-max-tries-per-episode 200 \
  --seed 7
```

### CLI flags supported by `train.py`

- Core training: `--episodes`, `--batch-size`, `--replay-size`, `--min-replay`, `--gamma`, `--learning-rate`, `--target-update`, `--hidden-dim`
- Exploration: `--eps-start`, `--eps-end`, `--eps-decay-steps`
- Environment/run setup: `--seed`, `--episode-seconds`, `--num-vessels`, `--dcpa-threshold`, `--tcpa-threshold`
- Seed sampling controls: `--sampling-dcpa-threshold`, `--sampling-tcpa-threshold`, `--dcpa-sample-max-tries`, `--max-sampling-steps-per-attempt`, `--sampling-logs`, `--no-sampling-logs`
- Output/render: `--save-every`, `--out-dir`, `--render`, `--no-render`
- Evaluation: `--eval-only`, `--eval-episodes`, `--eval-scenario {head_on,crossing,overtaking,all}`, `--eval-max-tries-per-episode`, `--eval-checkpoint`

---

## Current limitations and near-term work

- **Primary focus remains 2-vessel quality.** The code has additive support for extra vessels, but the main design/metrics in this folder are centered on vessel1-vessel2 behavior.
- **Extra-vessel path is available but secondary.** It currently mirrors vessel2-like perimeter/path-following defaults unless configured otherwise.
- **Some compatibility fields remain intentionally.** Deprecated reward and alias parameters are retained so older configs/checkpoints do not break.
- **Practical future work areas:** richer multi-vessel evaluation metrics, stronger scenario-balancing/sampling diagnostics, and incremental reward/behavior refinements once 2-vessel stability targets are consistently met.
