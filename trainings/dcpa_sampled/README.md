# `trainings/dcpa_sampled`

This folder contains the sampled DCPA/TCPA training pipeline for the current **2-vessel** reinforcement-learning setup.

## Overview

The implementation is intentionally centered on **vessel1 + vessel2** encounters so COLREGS give-way behavior can be learned and measured in a controlled setting before relying on larger traffic scenes.

Extra-vessel support (`num_vessels > 2`) exists as an additive extension path, but it is not the current optimization target.

## Environment setup (current behavior)

The base reset geometry uses a shared “big circle”:

- **vessel1** starts at world center.
- **vessel2** starts on the perimeter (`vessel2_outer_radius`) and receives a perimeter goal.
- Initial speed/heading/goal conditions are sampled each reset.
- Risk develops from live geometry (DCPA/TCPA + COLREGS role assignment during rollout).
- RL takeover is conditional and role-based (give-way vessels under risk), with per-vessel latch behavior once takeover is triggered.

## Control logic (scripted vs RL)

Vessels are **not** globally RL-controlled at all times.

- Scripted/path-following control is used when takeover conditions are not active.
- RL actions are applied only to vessels currently marked RL-active by environment logic.
- A vessel must be designated give-way in a risky interaction for takeover to start.
- Once model-control latch is set for a vessel, that vessel remains model-controlled until goal completion/reset.

## Observation design (96D)

Each vessel observation is a 96-dimensional vector:

- **Radar block: 90 dims** = `9 sectors × 10 features`
- **Own-vessel block: 6 dims**
- Total: **96 dims**

## Epsilon / exploration behavior (current)

Training uses **episode-based multiplicative epsilon decay**:

- Epsilon is mutable training state on the agent.
- Epsilon is used directly for exploration decisions (`random() < epsilon`).
- Epsilon stays constant for all steps within an episode.
- Epsilon is updated **once at episode end** using:
  - `epsilon = max(eps_end, epsilon * epsilon_decay)`

Default behavior derives `epsilon_decay` from an episode horizon:

- `epsilon_decay = (eps_end / eps_start) ** (1.0 / epsilon_decay_episodes)`
- default `epsilon_decay_episodes = 1000`

Precedence:

1. If `--epsilon-decay` is explicitly provided, that multiplier is used.
2. Otherwise, decay is derived from `--eps-start`, `--eps-end`, and `--epsilon-decay-episodes`.

## Training-side seed screening pipeline (authoritative)

For each training episode:

1. Generate candidate seed(s).
2. Run scripted-only screening rollout (no policy action injection).
3. Accept/reject by sampling thresholds.
4. If accepted, run a fresh reset on that accepted seed for the real training rollout.

Important design notes:

- This screening pipeline is the authoritative training episode selection gate.
- Reset itself has **no hidden viability filtering**.
- Sampling thresholds are separate from environment risk/takeover thresholds.

## Runtime flags (`train.py`)

Run from repository root:

```bash
python trainings/dcpa_sampled/train.py [flags]
```

### 1) Training control

- `--episodes`: number of training episodes.
- `--batch-size`: replay batch size for updates.
- `--replay-size`: replay buffer capacity.
- `--min-replay`: minimum replay size before updates can start (with takeover condition).
- `--save-every`: checkpoint/history save period (episodes).
- `--out-dir`: output folder for checkpoint/history.
- `--seed`: random seed for Python/NumPy/PyTorch and seed generation base.

### 2) Epsilon / exploration

- `--eps-start`: episode-level epsilon at training start.
- `--eps-end`: epsilon floor.
- `--epsilon-decay`: explicit episode-based multiplicative decay override.
- `--epsilon-decay-episodes`: episode horizon used to derive decay when `--epsilon-decay` is not provided (default `1000`).

### 3) Sampling / screening

- `--sampling-dcpa-threshold`: DCPA threshold used only for candidate-seed acceptance.
- `--sampling-tcpa-threshold`: TCPA threshold used only for candidate-seed acceptance.
- `--dcpa-sample-max-tries`: max candidate-seed attempts per training episode (`0` = unlimited).
- `--sampling-screen-max-steps`: optional screening-only step cap per candidate attempt.
- `--sampling-screen-max-seconds`: optional screening-only simulated-seconds cap per candidate attempt.
- `--max-sampling-steps-per-attempt`: legacy alias for screening step cap.
- `--sampling-logs` / `--no-sampling-logs`: enable/disable per-attempt sampling logs.

### 4) Environment / risk thresholds

- `--episode-seconds`: per-episode simulated time budget.
- `--dcpa-threshold`: environment risk/takeover DCPA threshold.
- `--tcpa-threshold`: environment risk/takeover TCPA threshold.
- `--num-vessels`: total vessel count. Current validated training focus remains **2-vessel quality**.

### 5) Rendering / visualization

- `--render` / `--no-render`: training-time rendering on/off.
- `--show-risk-overlay` / `--hide-risk-overlay`: takeover/risk HUD overlay on/off.
- `--auto-show-risk-sector-overlay` / `--no-auto-show-risk-sector-overlay`: whether takeover HUD pause automatically shows sector rays.

### 6) Evaluation mode

- `--eval-only`: run deterministic evaluation without training updates.
- `--eval-episodes`: requested accepted episodes per scenario.
- `--eval-scenario`: scenario selection (`head_on`, `crossing`, `overtaking`, `all`).
- `--eval-max-tries-per-episode`: max seed attempts to match requested eval scenario per episode.
- `--eval-checkpoint`: checkpoint path override (default `<out-dir>/ddqn_policy.pt`).

## Demo playback flags (`demo_model.py`)

`demo_model.py` supports render/HUD controls consistent with training:

- `--show-risk-overlay` / `--hide-risk-overlay`
- `--auto-show-risk-sector-overlay` / `--no-auto-show-risk-sector-overlay`

## CLI usage examples

### Normal training

```bash
python trainings/dcpa_sampled/train.py --out-dir runs/dcpa_sampled
```

### Training with explicit epsilon multiplier override

```bash
python trainings/dcpa_sampled/train.py \
  --eps-start 1.0 \
  --eps-end 0.05 \
  --epsilon-decay 0.997
```

### Training with derived epsilon decay over 1000 episodes

```bash
python trainings/dcpa_sampled/train.py \
  --eps-start 1.0 \
  --eps-end 0.05 \
  --epsilon-decay-episodes 1000
```

### Evaluation mode

```bash
python trainings/dcpa_sampled/train.py \
  --eval-only \
  --out-dir runs/dcpa_sampled \
  --eval-scenario all \
  --eval-episodes 30 \
  --eval-max-tries-per-episode 200
```

## Current limitations

- Primary tuning/evaluation emphasis remains the 2-vessel setting.
- Multi-vessel coordination behavior is not the primary validated target in this folder.
- This setup does not claim fully solved COLREGS compliance across all timing/traffic edge cases.
