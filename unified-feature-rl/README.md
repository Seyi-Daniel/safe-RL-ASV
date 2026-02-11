# unified-feature-rl

Single-agent ASV training sandbox using **DDQN** (feature-based state + discrete control actions).

## What this now implements

- One learning ASV vessel (ego) plus one moving target vessel.
- Agent always starts at the world center.
- Agent heading is randomized each episode.
- Agent goal is sampled on a random angle along a configurable outer ring around the start point.
- Target vessel behavior:
  - spawns on a large outer circle (`target_outer_radius`)
  - picks random left/right traversal and random arc angle
  - follows a smooth inner arc and stops at an endpoint on the same outer circle
- Optional dotted rings rendered around the start location:
  - `spawn_ring_radius` (inner ring)
  - `goal_ring_radius` (outer goal ring)
  - `vessel_outline_radius` (agent marker ring size in render)

## State, action, and algorithm

### Observation/state (10 features)
1. normalized agent x
2. normalized agent y
3. normalized agent heading
4. normalized agent speed
5. normalized goal x
6. normalized goal y
7. normalized target x
8. normalized target y
9. normalized target heading
10. normalized target speed

### Action space (9 discrete actions)
Actions are a Cartesian product of:
- steer: `{none, starboard, port}`
- throttle: `{coast, accelerate, decelerate}`

Each DDQN action index `[0..8]` is decoded into continuous control commands for the simulator (`rudder`, `throttle` in `{-1,0,1}`).

### Training algorithm
Training is now **Double DQN** (not CEM):
- online and target Q networks
- replay buffer
- epsilon-greedy exploration with linear decay by environment steps
- Double-DQN target selection (`argmax` from online net, value from target net)
- smooth L1 (Huber) loss and gradient clipping

## Why this differs from the previous CEM version

The RL_ASV and feature-RL-ASV subprojects are value-based DDQN-style implementations. This subproject now follows that same family so hyperparameters and behavior match expected DDQN workflows (episodes, replay, target updates, epsilon schedule) instead of population/elites.

## Suggested improvements included

Compared with a minimal DDQN baseline, this setup includes:
- gradient clipping (`max_norm=5.0`) for stability
- Huber loss (robust to outliers)
- checkpoint metadata in `ddqn_policy.pt` (`obs_dim`, `hidden_dim`, `n_actions`)

If you want, next improvements could be:
- prioritized replay
- dueling DDQN head
- n-step returns
- reward normalization

## Main files

- `hyperparameters.py` — environment/reward/DDQN hyperparameters.
- `environment.py` — ego + moving-target environment and rendering.
- `policy.py` — DDQN Q-network and action decoding.
- `train.py` — DDQN training loop.
- `run_episode.py` — evaluate with random or saved DDQN policy.

## Commands

### Train (DDQN)

```bash
python unified-feature-rl/train.py --episodes 600 --render
# or headless
python unified-feature-rl/train.py --episodes 600 --no-render
```

### Run episodes (evaluation/visualization)

```bash
python unified-feature-rl/run_episode.py --render
python unified-feature-rl/run_episode.py --render --policy unified-feature-rl/runs/ddqn_policy.pt
```

## Key train options

- `--episodes`
- `--batch-size`
- `--replay-size`
- `--min-replay`
- `--gamma`
- `--learning-rate`
- `--target-update`
- `--eps-start`
- `--eps-end`
- `--eps-decay-steps`
- `--hidden-dim`
- `--save-every`
- `--out-dir`
- `--render` / `--no-render`
