# unified-feature-rl

New standalone project folder for your corrected feature-based RL implementation.

## What this implements

- One **agent/ASV** vessel + one **target** vessel.
- Both vessels have their own goals and move each episode.
- Random initial positions, headings, speeds, and goals (within square world bounds).
- Internal 12-sector encounter awareness.
- **10 input features** for the agent policy (target goal removed):
  1. agent x
  2. agent y
  3. agent heading
  4. agent speed
  5. agent goal x
  6. agent goal y
  7. target heading relative to agent
  8. target speed
  9. relative bearing to target
  10. relative distance to target
- **2 continuous outputs** from the policy:
  - rudder command in `[-1, 1]`
  - throttle command in `[-1, 1]`

## Single place to tweak numerics/hyperparameters

Edit:
- `unified-feature-rl/hyperparameters.py`

This file contains environment dynamics, reward weights, and training hyperparameters.
The values are imported and used by:
- `unified-feature-rl/environment.py`
- `unified-feature-rl/train.py`
- `unified-feature-rl/run_episode.py`

## Files

- `hyperparameters.py` — all main numerics/hyperparameters.
- `environment.py` — environment dynamics, observation build, reward terms, COLREGs + DCPA shaping, pygame render.
- `policy.py` — small continuous policy MLP utilities.
- `train.py` — CEM-based RL training (continuous outputs, no torch dependency).
- `run_episode.py` — run episodes with random policy or a saved trained policy.

## Run commands

### 1) Run visualization / episodes

```bash
python unified-feature-rl/run_episode.py --render
```

### 2) Train policy

```bash
python unified-feature-rl/train.py --episodes 200 --population 32
```

### 3) Run with trained policy

```bash
python unified-feature-rl/run_episode.py --render --policy unified-feature-rl/runs/policy_mean.npy
```

## Complete command options

### `run_episode.py`

```bash
python unified-feature-rl/run_episode.py [options]
```

Options:
- `--episodes <int>` number of episodes (default: 5)
- `--seed <int>` base random seed (default: 7)
- `--policy <path.npy>` load trained policy (`policy_mean.npy`)
- `--hidden-dim <int>` policy hidden size (default: 32)
- `--render` enable pygame visualization (**visibility ON**)
- `--no-render` disable pygame visualization (**visibility OFF**)
- `--show-grid` draw map grid
- `--hide-grid` hide map grid
- `--show-sectors` draw 12 sector rays
- `--hide-sectors` hide 12 sector rays
- `--episode-seconds <float>` episode duration
- `--world-w <float>` world width
- `--world-h <float>` world height
- `--sensor-range <float>` sensing range used in features / COLREG logic window
- `--pixels-per-meter <float>` render scale
- `--save-log <path.json>` save episode summaries

### `train.py`

```bash
python unified-feature-rl/train.py [options]
```

Options:
- `--episodes <int>` number of CEM iterations
- `--population <int>` candidate policies per iteration
- `--elite-frac <float>` top fraction kept as elites
- `--eval-rollouts <int>` rollouts per candidate
- `--hidden-dim <int>` policy hidden size
- `--init-std <float>` initial parameter sampling std
- `--min-std <float>` lower bound on std
- `--std-decay <float>` std decay factor
- `--seed <int>` random seed
- `--save-every <int>` checkpoint frequency
- `--out-dir <path>` directory for outputs
- `--render` enable pygame visualization during training
- `--no-render` disable pygame visualization during training
