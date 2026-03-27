# `trainings/dcpa_sampled`

This folder contains the sampled DCPA/TCPA training pipeline for the **current 2-vessel RL setup**.

## Current focus

- The primary target is robust behavior for **vessel1 + vessel2** encounters.
- Optional extra-vessel support (`num_vessels > 2`) is present, but it is an additive path and **not the current optimization priority**.

## Environment setup (current behavior)

The base scenario is sampled around a shared “big circle” geometry:

- **vessel1** starts at the world center.
- **vessel2** starts on the perimeter (`vessel2_outer_radius`) and also receives a perimeter goal.
- Initial headings/speeds/goals are sampled per reset.
- During rollout, COLREGS geometry and DCPA/TCPA risk are recomputed from the live state.
- RL takeover is allowed for vessels currently designated **give-way** (with latch behavior once takeover starts).

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

### Own-vessel features (6)

1. normalized own speed
2. normalized distance-to-goal
3. sin(goal bearing in own frame)
4. cos(goal bearing in own frame)
5. normalized rudder
6. normalized throttle

### Bearing convention and sensor range

- Relative bearing convention is `[0°, 360)` with:
  - `0°` = ahead,
  - `90°` = port,
  - `270°` = starboard.
- Contacts beyond `sensor_range` are excluded from radar sectors.
- Distance-like values are clipped/scaled to `[0, 1]`; angles use sin/cos; closing speed uses `tanh` normalization.

## Reward design (current)

Reward is composed of local progress + shared safety + lightweight scenario shaping.

1. **Progress reward with heading-aware shaping**
   - distance-to-goal improvement term
   - small heading-improvement shaping term

2. **Shared/global safety terms**
   - living penalty
   - collision/near-miss penalties
   - unsafe-proximity penalty (continuous)
   - safe-pass bonus when a risky encounter clears safely
   - rudder oscillation penalty

3. **Simplified scenario shaping for RL-active vessels**
   - head-on/crossing: starboard rudder preference
   - overtaking: DCPA-threshold-based shaping

4. **Legacy compatibility params retained but inactive**
   - `give_way_early_action_bonus`
   - `late_action_penalty`
   - `crossing_ahead_penalty`
   - `early_action_tcpa_threshold`
   - `late_action_tcpa_threshold`
   - `out_of_bounds_penalty`
   - `stand_on_hold_bonus`
   - `stand_on_unnecessary_action_penalty`

## Training and sampling design

### Important separation: env risk/takeover vs training sampling

The environment’s runtime risk/takeover logic uses environment thresholds (`--dcpa-threshold`, `--tcpa-threshold`), while training episode selection can use separate sampling thresholds (`--sampling-dcpa-threshold`, `--sampling-tcpa-threshold`).

### Screening pipeline used for each training episode

For each training episode, the script uses a dedicated candidate-seed screening flow:

1. **Sample candidate seed**
2. **Run scripted screening rollout** (policy-independent; no RL action injection)
3. **Accept/reject** using sampling thresholds (DCPA/TCPA)
4. If accepted, **fresh reset** with that seed for the real training episode

Notes:

- Screening is intentionally **policy-independent**.
- Screening horizon is **unlimited by default**.
- Optional screening-only horizon controls:
  - `--sampling-screen-max-steps`
  - `--sampling-screen-max-seconds`
- Legacy alias `--max-sampling-steps-per-attempt` maps to step horizon when the new flag is not set.

## Evaluation mode

`train.py` supports deterministic evaluation-only execution via:

- `--eval-only`
- `--eval-scenario {head_on,crossing,overtaking,all}`
- `--eval-episodes`
- `--eval-max-tries-per-episode`
- optional checkpoint override via `--eval-checkpoint` (default: `<out-dir>/ddqn_policy.pt`)

Evaluation reports metrics grouped per requested scenario.

## Additional quality metrics currently reported

In evaluation output (and partly in training history/logging), the pipeline tracks:

- collision count/rate
- goal completion count/rate (both vessels reach goals)
- safe-pass count/rate
- scenario step counts (`head_on`, `crossing`, `overtaking`)
- starboard-compliance opportunities/count/rate
- takeover count
- both-controlled step count (both vessel1 and vessel2 RL-active in same step)

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

## Notes on scope

This README describes the **implemented behavior in this folder today**. It intentionally does not present future multi-vessel ambitions as current behavior.
