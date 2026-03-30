# `trainings/dcpa_sampled`

This folder contains the sampled DCPA/TCPA training pipeline for the current **2-vessel** reinforcement-learning setup.

## Overview / project purpose

This pipeline is intentionally optimized for **2-vessel training quality first** (`vessel1` + `vessel2`) so COLREGS-aligned give-way behavior can be learned and measured in a controlled setting.

Support for `num_vessels > 2` exists as an additive extension path, but it is **not** the current optimization or validation target for this training package.

## Key defaults (current)

- **Runtime environment risk/takeover thresholds**
  - `dcpa_risk_threshold = 20.0`
  - `tcpa_risk_threshold = 20.0`
  - CLI compatibility flags: `--dcpa-threshold`, `--tcpa-threshold` (these are runtime risk/takeover thresholds).
- **Training seed-screening thresholds**
  - `sampling_dcpa_threshold = 20.0`
  - `sampling_tcpa_threshold = 20.0`
  - CLI flags: `--sampling-dcpa-threshold`, `--sampling-tcpa-threshold`.
- **Vessel 2 min speed default**
  - `vessel2_min_speed = 0.0`.
- **Replay capacity default**
  - `replay_size = 1_000_000`.
- **Checkpoint cadence default**
  - `save_every = 100` episodes.

## Threshold families and separation

The threshold families are intentionally independent:

1. **Runtime risk/takeover thresholds** (`dcpa_risk_threshold`, `tcpa_risk_threshold`) are used by environment risk assessment and RL takeover gating during rollout.
2. **Training sampling thresholds** (`sampling_dcpa_threshold`, `sampling_tcpa_threshold`) are used only by scripted candidate-seed screening/acceptance before a training rollout starts.

Sampling thresholds do not drive runtime takeover logic.

## Environment setup (current behavior)

The base reset geometry uses a shared outer ring / big-circle layout:

- **vessel1** starts at world center.
- **vessel2** starts on the outer circle (`vessel2_outer_radius`) and receives a perimeter goal.
- Initial headings, speeds, and goals are sampled per reset.
- DCPA/TCPA risk and COLREGS role assignments emerge from the live rollout geometry.

## Control logic (scripted vs RL)

Vessels are not globally RL-driven at all times.

- Scripted/path-following control is used when takeover is inactive.
- RL actions are applied only for vessels currently marked RL-active by environment logic.
- RL activation is role-aware (give-way emphasis in risky encounters).
- Once a vessel enters model-control latch, control remains latched for that vessel until goal completion/reset.

## Observation design (96D)

Each controlled vessel gets a fixed-size 96-dimensional observation:

- **Radar block (90 dims):** `9 sectors × 10 features`.
- **Own-state block (6 dims):** per-vessel kinematics/goal-relative features.
- Total: **96 dims**.

Radar uses a fixed-size sectorized representation so policy input size does not change with contact count. Per sector, the encoding tracks nearest-contact style geometry/features to summarize local traffic compactly.

## Policy / value network architecture

Actor and critic both use a 3-layer MLP backbone with hidden widths:

- `512 -> 256 -> 128`

Shapes:

- **Actor:** `obs(96) -> 512 -> 256 -> 128 -> action(2)` (Tanh output).
- **Critic:** `(obs(96)+action(2)) -> 512 -> 256 -> 128 -> Q(1)`.

This update does **not** redesign optimizer/loss/target-update strategy; Adam + existing objectives and update flow are unchanged.

## Epsilon / exploration behavior

Training uses episode-based multiplicative epsilon decay:

- Epsilon is an episode-level exploration parameter.
- Epsilon remains constant during each episode.
- Epsilon decays once at episode end.
- Explicit `--epsilon-decay` overrides derived decay behavior.
- If explicit decay is not provided, decay is derived from `--eps-start`, `--eps-end`, and `--epsilon-decay-episodes` (default horizon: `1000` episodes).

## Training seed-screening pipeline

For each training episode:

1. Generate candidate seed(s).
2. Run scripted-only screening rollout (no policy action injection).
3. Accept/reject candidates using sampling thresholds.
4. If accepted, perform a fresh reset on that accepted seed and run the real training rollout.

Important notes:

- Screening is the authoritative episode-selection gate.
- Environment reset itself has no hidden viability filtering.
- When a candidate first satisfies screening thresholds (`DCPA <= sampling_dcpa_threshold` and `0 < TCPA <= sampling_tcpa_threshold`), that exact step is treated as the acceptance/risk-trigger moment.
- If `--sampling-scenario` is provided, scenario filtering is applied to the scenario observed at that acceptance moment (not just reset-time geometry).

## Output layout

Training outputs are saved under `trainings/dcpa_sampled/runs/`.

Each training launch creates a unique timestamped run subfolder using:

- `mmdd_hrmn`
- example: `0330_1323`

Artifacts for that run (checkpoints + history) are saved into that run folder.

## Checkpoint naming

Per-save checkpoints use episode-appended names:

- `policy_<episode>.pt`
- examples: `policy_100.pt`, `policy_15000.pt`

A convenience rolling file is also written:

- `policy_latest.pt`

Training history is written to:

- `train_history.json`

## Runtime flags (`train.py`)

Run from repository root:

```bash
python trainings/dcpa_sampled/train.py [flags]
```

### 1) Training control

- `--episodes`: number of training episodes.
- `--batch-size`: replay batch size for updates.
- `--replay-size`: replay buffer capacity (default `1000000`).
- `--min-replay`: minimum replay size before updates can start (with takeover condition).
- `--save-every`: checkpoint/history save period in episodes (default `100`).
- `--out-dir`: base output folder (default `runs`; training creates a timestamped run subfolder).
- `--seed`: random seed for Python/NumPy/PyTorch and seed generation base.
- `--cuda`: CUDA GPU index (for example `--cuda 0`, `--cuda 1`). If CUDA is available and this flag is omitted, runtime defaults to `cuda:0`. If CUDA is unavailable, runtime falls back to CPU.
- `--hidden-dim-1`, `--hidden-dim-2`, `--hidden-dim-3`: policy/critic hidden widths.

### 2) Epsilon / exploration

- `--eps-start`: episode-level epsilon at training start.
- `--eps-end`: epsilon floor.
- `--epsilon-decay`: explicit episode-based multiplicative decay override.
- `--epsilon-decay-episodes`: episode horizon used to derive decay when `--epsilon-decay` is not provided.

### 3) Sampling / screening thresholds

- `--sampling-dcpa-threshold`: training-only candidate-seed DCPA acceptance threshold (default `20.0`).
- `--sampling-tcpa-threshold`: training-only candidate-seed TCPA acceptance threshold (default `20.0`).
- `--sampling-scenario`: optional training-only scenario filter for candidate-seed acceptance (`head_on`, `crossing`, `overtaking`). If omitted, all scenarios are eligible and mixed-scenario training behavior is preserved.
- `--dcpa-sample-max-tries`: max candidate-seed attempts per training episode (`0` = unlimited).
- `--sampling-screen-max-steps`: optional screening-only step cap per candidate attempt.
- `--sampling-screen-max-seconds`: optional screening-only simulated-seconds cap per candidate attempt.
- `--max-sampling-steps-per-attempt`: legacy alias for screening step cap.

`--sampling-scenario` applies **only** to training candidate-seed screening/acceptance. It filters by the scenario observed at the scripted screening acceptance moment (the step where threshold acceptance is first met), not merely the initial/reset scenario label. It does **not** change runtime environment risk/takeover logic and does **not** change reward logic.

### 4) Environment / risk thresholds

- `--dcpa-threshold`: runtime environment risk/takeover DCPA threshold (default `20.0`).
- `--tcpa-threshold`: runtime environment risk/takeover TCPA threshold (default `20.0`).
- `--episode-seconds`: per-episode simulated time budget.
- `--num-vessels`: total vessel count.

### 5) Rendering / visualization

- `--render` / `--no-render`: training-time rendering on/off.
- `--show-risk-overlay` / `--hide-risk-overlay`: takeover/risk HUD overlay on/off.
- `--auto-show-risk-sector-overlay` / `--no-auto-show-risk-sector-overlay`: whether takeover HUD pause auto-shows sector rays.

### 6) Evaluation mode

- `--eval-only`: run deterministic evaluation without training updates.
- `--eval-episodes`: requested accepted episodes per scenario.
- `--eval-scenario`: scenario selection (`head_on`, `crossing`, `overtaking`, `all`).
- `--eval-max-tries-per-episode`: max seed attempts to match requested eval scenario per episode.
- `--eval-checkpoint`: checkpoint override. If omitted, eval first checks `<out-dir>/policy_latest.pt`, otherwise auto-resolves the newest timestamped run under `<out-dir>` and uses that run's `policy_latest.pt`.

## Demo vs eval

- `demo_model.py`: visual/qualitative playback for inspecting behavior and trajectories.
- `train.py --eval-only`: quantitative evaluation path that reports aggregate metrics over requested scenarios/episodes.

## Demo playback (`demo_model.py`)

Use a checkpoint from a run subfolder, for example:

```bash
python trainings/dcpa_sampled/demo_model.py --checkpoint trainings/dcpa_sampled/runs/0330_1323/policy_100.pt
```

or:

```bash
python trainings/dcpa_sampled/demo_model.py --checkpoint trainings/dcpa_sampled/runs/0330_1323/policy_latest.pt
```

### Normal demo mode (existing simple flow)

By default, demo playback keeps the simple seed-selection behavior:

- If `--scenario` is omitted (or `all`), it plays accepted reset seeds directly.
- If `--scenario` is set (`head_on`, `crossing`, `overtaking`), it searches for a seed whose **initial** geometry matches that scenario.
- This mode is useful for quick visual inspection, but it may still include episodes where takeover never activates.

### Screened demo mode (training-style seed acceptance)

Enable screened demo selection with `--screened-demo` to mirror training-style scripted screening before playback:

- Candidate seeds are rolled out with **scripted-only control** (no policy actions injected during screening).
- Acceptance requires meeting screening thresholds:
  - `DCPA <= --screening-dcpa-threshold` (default `20.0`)
  - `0 < TCPA <= --screening-tcpa-threshold` (default `20.0`)
- Optional screening horizons:
  - `--screening-max-steps`
  - `--screening-max-seconds`
- Candidate search budget:
  - `--screening-max-tries` (`0` means unlimited)

This better matches the training seed-selection philosophy and reduces “safe/no-risk” demo episodes where the model had nothing to do.

### Optional takeover requirement

Use `--require-takeover` **together with** `--screened-demo` when you want only takeover-relevant demo episodes:

- A candidate seed is accepted only if scripted screening observed at least one RL-active/takeover step.
- If omitted, screened mode still enforces DCPA/TCPA acceptance, but does not force takeover occurrence.

### Combining scenario filtering + screening

You can combine `--scenario` with `--screened-demo` (and optionally `--require-takeover`):

- Scenario filter must match (`head_on`, `crossing`, `overtaking`) **and**
- Scripted screening acceptance must pass threshold checks (and takeover requirement if enabled).

### Playback takeover discipline

`demo_model.py` enforces the same takeover discipline expected during training:

- Before takeover: actor is **not queried**, observations are **not built** for model use, and no model actions are produced.
- After takeover activates: actor is queried only for RL-active vessel IDs, and only those vessels receive action entries.

Per episode, demo logs include at least:

- accepted seed
- scenario label
- whether screening was used
- whether takeover was observed during screening
- whether takeover occurred during actual playback
- number of playback steps with at least one RL-active vessel
- collision / success / terminal reason

### Recommended screened demo command

```bash
python trainings/dcpa_sampled/demo_model.py \
  --checkpoint trainings/dcpa_sampled/runs/0330_1323/policy_latest.pt \
  --seed 7 \
  --episodes 5 \
  --screened-demo \
  --scenario crossing \
  --require-takeover \
  --screening-dcpa-threshold 20.0 \
  --screening-tcpa-threshold 20.0 \
  --screening-max-tries 500
```

Device selection behavior for both `train.py` and `demo_model.py`:

- If CUDA is available and `--cuda` is not provided, device defaults to `cuda:0`.
- If CUDA is available and `--cuda N` is provided, device `cuda:N` is used (validated against `torch.cuda.device_count()`).
- If CUDA is unavailable, runtime falls back to `cpu` cleanly.

## Current limitations / scope

- Primary validated emphasis remains 2-vessel training quality.
- Multi-vessel support is additive and available, but not the current validated optimization focus in this folder.
