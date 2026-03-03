# safe-RL-ASV

Project code is split into independent tracks:

- `training/` — model training, checkpoints, and trained-model demos.
- `simulations/` — standalone simulation system with decoupled views.

## Entry points

### Training
- `training/train.py` — DDPG-style training loop.
- `demo_model.py` — visualize/test a trained checkpoint from `training/train.py`.

### Simulations
- `simulations/test_simulation.py` — simulation sandbox runner for all available views.

Current views (same package level, decoupled modules):
- `dcpa-sampled-episode` (`simulations/sim_views/dcpa_sampled_episode.py`) — vessel-1 starts at center.
- `perimeter-start-dcpa-sampled-episode` (`simulations/sim_views/perimeter_start_episode.py`) — vessel-1 starts on the big-circle circumference.

## Quick start

```bash
python training/train.py --no-render
python demo_model.py --policy runs/ddqn_policy.pt --render
python simulations/test_simulation.py --view dcpa-sampled-episode --render
python simulations/test_simulation.py --view perimeter-start-dcpa-sampled-episode --render
```

## Repository layout

- `training/` contains its own `environment.py`, `hyperparameters.py`, `policy.py`, and `train.py`.
- `simulations/` contains simulation environments/configs and decoupled view modules under `simulations/sim_views/`.
- `legacy/` keeps archived older subprojects for reference.
