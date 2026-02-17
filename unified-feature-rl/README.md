# unified-feature-rl

Two-vessel ASV simulation/training sandbox with continuous rudder/throttle I/O.

## What this now implements

- Two vessels on a shared big circle centered at the world center.
- Vessel 1 (agent slot):
  - starts exactly at world center
  - gets a random goal on the big-circle circumference
  - travels straight from center to that goal at constant randomized speed.
- Vessel 2 (target slot):
  - starts at a random point on the same circumference
  - gets a random goal on the same circumference
  - starts with randomized heading and randomized constant speed
  - follows an exact Dubins path family candidate (LSL/RSR/LSR/RSL/RLR/LRL) selected by episode objective and tracked with smooth rudder dynamics.
- Vessel-2 end-heading candidates include:
  - clockwise/counter-clockwise tangents at goal
  - heading toward center
  - heading along start-goal chord
  - ± angular sweeps around each base heading.
- Per-episode planner objective for vessel 2 is randomly sampled from:
  - shortest path length
  - minimum steering effort
  - minimum curvature change
  - closest to straight line.
- Episode terminates when both vessels reach their goals (or safety timeout/oob fallback), with continuous arrival motion (no goal snap teleport).
## State, action, and algorithm

### Observation/state (12 features)
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
11. normalized target goal x
12. normalized target goal y

### Action interface (continuous)
The environment step input remains continuous `[rudder, throttle]` where each channel is expected in `[-1, 1]`.

At the moment, vessel control is scripted by the path planners (as requested), so model outputs are still produced but are not used to steer either boat during training rollouts.

### Training algorithm
Training is now **Double DQN** (not CEM):
- online and target Q networks
- replay buffer
- epsilon-greedy exploration with linear decay by environment steps
- Double-DQN target selection (`argmax` from online net, value from target net)
- smooth L1 (Huber) loss and gradient clipping

## Why this differs from the previous CEM version

The RL_ASV and feature-RL-ASV subprojects are value-based DDQN-style implementations. This subproject keeps that model/replay scaffolding, while boat motion is currently scripted to validate scenario geometry/path planning before RL control is re-enabled.

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
