# safe-RL-ASV

Project code is split into independent tracks:

- `training/` — model training, checkpoints, and trained-model demos.
- `simulations/` — standalone simulation stacks and simulation-focused tests.

## Entry points

### Training
- `training/train.py` — DDPG-style training loop.
- `demo_model.py` — visualize/test a trained checkpoint from `training/train.py`.

### Simulations (standalone stacks)
- `simulations/test_simulation.py` — original DCPA sampled episode flow (vessel-1 starts at center).
- `simulations/perimeter_start/test_simulation.py` — perimeter-start variant (vessel-1 starts on big-circle circumference).

## Quick start

```bash
python training/train.py --no-render
python demo_model.py --policy runs/ddqn_policy.pt --render
python simulations/test_simulation.py --view dcpa-sampled-episode --render
python simulations/perimeter_start/test_simulation.py --view dcpa-sampled-episode --render
```

## Repository layout

- `training/` contains its own `environment.py`, `hyperparameters.py`, `policy.py`, and `train.py`.
- `simulations/` contains two decoupled simulation stacks:
  - root simulation stack (`simulations/environment.py`, `simulations/hyperparameters.py`, `simulations/sim_views/`)
  - perimeter-start simulation stack (`simulations/perimeter_start/environment.py`, `simulations/perimeter_start/hyperparameters.py`, `simulations/perimeter_start/sim_views/`)
- `legacy/` keeps archived older subprojects for reference.
