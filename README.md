# safe-RL-ASV

Project code is split into two independent tracks:

- `training/` — model training, checkpoints, and trained-model demos.
- `simulations/` — standalone simulation views and simulation-focused tests.

## Entry points

### Training
- `training/train.py` — DDPG-style training loop.
- `demo_model.py` — visualize/test a trained checkpoint from `training/train.py`.

### Simulations
- `simulations/test_simulation.py` — simulation sandbox runner (view-based).

## Quick start

```bash
python training/train.py --no-render
python demo_model.py --policy runs/ddqn_policy.pt --render
python simulations/test_simulation.py --view dcpa-sampled-episode --render
```

## Repository layout

- `training/` contains its own `environment.py`, `hyperparameters.py`, `policy.py`, and `train.py`.
- `simulations/` contains its own `environment.py`, `hyperparameters.py`, `sim_views/`, tests, and simulation runner.
- `legacy/` keeps archived older subprojects for reference.
