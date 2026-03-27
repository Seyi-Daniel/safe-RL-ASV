# `trainings/dcpa_sampled`

This folder contains the sampled DCPA/TCPA training pipeline for the current **2-vessel** reinforcement-learning setup.

## Overview

The implementation is intentionally centered on **vessel1 + vessel2** encounters so COLREGS give-way behavior can be learned and measured in a controlled setting before relying on larger traffic scenes.

Extra-vessel support (`num_vessels > 2`) exists as an additive extension path, but it is not the current optimization target.

## Design philosophy

### Why 2-vessel first

The present training/evaluation loop is built to stabilize foundational encounter handling (head-on, crossing, overtaking) in the two-vessel case first.

Reasons this is intentional:

- It keeps scenario interpretation and metrics unambiguous while policy behavior is still being tuned.
- It makes takeover logic and reward attribution easier to inspect.
- It avoids conflating core give-way behavior quality with multi-actor coordination complexity.

## Environment setup (current behavior)

The base reset geometry uses a shared “big circle”:

- **vessel1** starts at world center.
- **vessel2** starts on the perimeter (`vessel2_outer_radius`) and receives a perimeter goal.
- Initial speed/heading/goal conditions are sampled each reset.
- Risk develops from live geometry (DCPA/TCPA + COLREGS role assignment during rollout).
- RL takeover is conditional and role-based (give-way vessels under risk), with per-vessel latch behavior once takeover is triggered.

## Control logic (scripted vs RL)

Vessels are **not** globally RL-controlled at all times.

Default behavior:

- Scripted/path-following control is used when takeover conditions are not active.

RL behavior:

- RL actions are applied only to vessels currently marked RL-active by the environment logic.
- A vessel must be designated give-way in a risky interaction for takeover to start.
- Once model-control latch is set for a vessel, that vessel remains model-controlled until goal completion/reset.

This keeps intervention targeted to safety-critical segments rather than replacing all baseline navigation.

## Observation design (96D)

Each vessel observation is a 96-dimensional vector:

- **Radar block: 90 dims** = `9 sectors × 10 features`
- **Own-vessel block: 6 dims**
- Total: **96 dims**

### Radar sectors (bearing bins)

Relative bearing is mapped to 9 sectors:

1. `[350°, 10°)`
2. `[10°, 40°)`
3. `[40°, 75°)`
4. `[75°, 112.5°)`
5. `[112.5°, 180°)`
6. `[180°, 247.5°)`
7. `[247.5°, 285°)`
8. `[285°, 320°)`
9. `[320°, 350°)`

Only the nearest in-range contact per sector is retained; empty sectors are zero-filled.

### Why this sectorized layout

- Fixed-size input regardless of nearby contact count.
- Keeps directional structure explicit for policy learning.
- Prioritizes the nearest threat in each bearing region.

### Per-sector features (10)

1. occupied flag
2. normalized distance
3. bearing sin
4. bearing cos
5. relative-heading sin
6. relative-heading cos
7. normalized target speed
8. normalized closing speed (`tanh`-bounded, signed)
9. normalized TCPA term
10. normalized DCPA term

### Why include TCPA / DCPA / closing speed

These are direct risk indicators:

- TCPA captures time proximity of potential closest approach.
- DCPA captures separation at closest approach.
- Closing speed captures convergence/divergence trend.

### Own-vessel features (6)

1. normalized own speed
2. normalized distance-to-goal
3. sin(goal bearing in own frame)
4. cos(goal bearing in own frame)
5. normalized rudder
6. normalized throttle

### Bearing convention and normalization summary

- Relative bearing convention is `[0°, 360)` with:
  - `0°` = ahead,
  - `90°` = port,
  - `270°` = starboard.
- Contacts beyond `sensor_range` are excluded from radar sectors.
- Distance-like values are clipped/scaled to `[0, 1]`; angles use sin/cos; closing speed uses `tanh` normalization.

## Reward design

Reward combines progress, shared safety terms, and lightweight scenario shaping.

### Reward philosophy

The active objective is safety-first while preserving navigation progress:

- Progress shaping drives movement toward goal.
- Strong global penalties/bonuses shape safe separation outcomes.
- Scenario-local shaping provides direction preferences for RL-active give-way behavior.

### Active components

1. **Progress + heading-aware shaping**
   - distance-to-goal improvement term
   - small heading-improvement shaping term

2. **Shared/global safety terms**
   - living penalty
   - collision/near-miss penalties
   - unsafe-proximity penalty (continuous)
   - safe-pass bonus after risky encounter clears safely
   - rudder oscillation penalty

3. **Simplified scenario shaping for RL-active vessels**
   - head-on/crossing: starboard rudder preference
   - overtaking: DCPA-threshold-based shaping

### Legacy compatibility parameters (inactive)

The following coefficients remain in `RewardParams` for compatibility, but active reward logic does not currently use them:

- `give_way_early_action_bonus`
- `late_action_penalty`
- `crossing_ahead_penalty`
- `early_action_tcpa_threshold`
- `late_action_tcpa_threshold`
- `out_of_bounds_penalty`
- `stand_on_hold_bonus`
- `stand_on_unnecessary_action_penalty`

## Training and sampling design

### Environment thresholds vs sampling thresholds

Two threshold groups are intentionally separate:

- **Environment thresholds** (`--dcpa-threshold`, `--tcpa-threshold`) drive runtime risk/takeover logic.
- **Sampling thresholds** (`--sampling-dcpa-threshold`, `--sampling-tcpa-threshold`) are used only to screen candidate training seeds.

Sampling thresholds do not replace environment risk/takeover thresholds.

### Candidate-seed screening pipeline (training)

For each training episode:

1. Sample candidate seed.
2. Run scripted screening rollout.
3. Accept/reject by sampling DCPA/TCPA thresholds.
4. If accepted, perform a fresh reset and run the real RL training episode.

Why this exists: screening filters out low-value candidates so learning concentrates on encounters that actually express relevant collision risk geometry.

### Screening behavior details

- Screening is policy-independent (no RL action injection during screening).
- Screening horizon is unlimited by default.
- Optional screening-only horizon controls:
  - `--sampling-screen-max-steps`
  - `--sampling-screen-max-seconds`
- Legacy alias:
  - `--max-sampling-steps-per-attempt` (maps to step cap when new step flag is not explicitly set)

## Evaluation mode

`train.py` supports deterministic evaluation-only execution via:

- `--eval-only`
- `--eval-scenario {head_on,crossing,overtaking,all}`
- `--eval-episodes`
- `--eval-max-tries-per-episode`
- optional checkpoint override via `--eval-checkpoint` (default: `<out-dir>/ddqn_policy.pt`)

Evaluation output is grouped by requested scenario.

## Additional quality metrics currently reported

Evaluation (and related training logs/history fields) include:

- collision count/rate
- goal completion count/rate (both vessel1 and vessel2 reach goals)
- safe-pass count/rate
- scenario step counts (`head_on`, `crossing`, `overtaking`)
- starboard-compliance opportunities/count/rate
- takeover count
- both-controlled step count (both vessels RL-active on the same step)

## CLI usage examples

Run commands from repository root.

### 1) Normal training

```bash
python trainings/dcpa_sampled/train.py \
  --out-dir runs/dcpa_sampled
```

### 2) Evaluation mode

```bash
python trainings/dcpa_sampled/train.py \
  --eval-only \
  --out-dir runs/dcpa_sampled \
  --eval-scenario all \
  --eval-episodes 30 \
  --eval-max-tries-per-episode 200
```

### 3) Training with custom sampling thresholds

```bash
python trainings/dcpa_sampled/train.py \
  --dcpa-threshold 20 \
  --tcpa-threshold 20 \
  --sampling-dcpa-threshold 12 \
  --sampling-tcpa-threshold 12 \
  --dcpa-sample-max-tries 200
```

### 4) Training with optional screening horizons

```bash
python trainings/dcpa_sampled/train.py \
  --sampling-screen-max-steps 600 \
  --sampling-screen-max-seconds 120 \
  --dcpa-sample-max-tries 300
```

## Current limitations

- Primary tuning/evaluation emphasis remains the 2-vessel setting.
- Multi-vessel coordination behavior is not the primary validated target in this folder.
- This setup does not claim fully solved COLREGS compliance across all timing/traffic edge cases.

## Scope note

This README describes currently implemented behavior in `trainings/dcpa_sampled` and avoids presenting future ambitions as already implemented.
