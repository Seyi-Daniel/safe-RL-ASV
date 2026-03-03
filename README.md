# safe-RL-ASV

Project code is split into independent tracks:

- `trainings/` — fully standalone RL training stacks for both scenario families.
- `simulations/` — fully standalone simulation stacks/views for both scenario families.

## Entry points

### Trainings
- `trainings/dcpa_sampled/train.py` — training pipeline for the center-start (DCPA sampled) scenario.
- `trainings/perimeter_start/train.py` — training pipeline for the perimeter-start scenario.
- `demo_model.py` — visualize/test a trained checkpoint from either standalone training stack.

### Simulations
- `simulations/test_simulation.py` — simulation sandbox runner for all available views.

Current views (same package level, decoupled modules):
- `dcpa-sampled-episode` (`simulations/sim_views/dcpa_sampled_episode.py`) — vessel-1 starts at center.
- `perimeter-start-dcpa-sampled-episode` (`simulations/sim_views/perimeter_start_episode.py`) — vessel-1 starts on the big-circle circumference.

## Quick start

```bash
python trainings/dcpa_sampled/train.py --no-render
python trainings/perimeter_start/train.py --no-render
python demo_model.py --scenario dcpa_sampled --policy runs/dcpa_sampled/ddqn_policy.pt --render
python demo_model.py --scenario perimeter_start --policy runs/perimeter_start/ddqn_policy.pt --render
python simulations/test_simulation.py --view dcpa-sampled-episode --render
python simulations/test_simulation.py --view perimeter-start-dcpa-sampled-episode --render
```

## Repository layout

- `trainings/dcpa_sampled/` and `trainings/perimeter_start/` are decoupled stacks (each has its own `environment.py`, `hyperparameters.py`, `policy.py`, and `train.py`).
- `simulations/` contains decoupled simulation environments/configs and view modules under `simulations/sim_views/`.
- `legacy/` keeps archived older subprojects for reference.
