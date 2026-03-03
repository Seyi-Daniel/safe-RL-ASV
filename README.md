# safe-RL-ASV

Project code is split into independent tracks:

- `trainings/` — RL training pipelines (DDPG-style) for both scenario families.
- `simulations/` — standalone simulation system with decoupled views.

## Entry points

### Trainings
- `trainings/train_dcpa_sampled.py` — training pipeline for the center-start (DCPA sampled) scenario.
- `trainings/train_perimeter_start.py` — training pipeline for the perimeter-start scenario.
- `demo_model.py` — visualize/test a trained checkpoint from either training pipeline.

### Simulations
- `simulations/test_simulation.py` — simulation sandbox runner for all available views.

Current views (same package level, decoupled modules):
- `dcpa-sampled-episode` (`simulations/sim_views/dcpa_sampled_episode.py`) — vessel-1 starts at center.
- `perimeter-start-dcpa-sampled-episode` (`simulations/sim_views/perimeter_start_episode.py`) — vessel-1 starts on the big-circle circumference.

## Quick start

```bash
python trainings/train_dcpa_sampled.py --no-render
python trainings/train_perimeter_start.py --no-render
python demo_model.py --scenario dcpa_sampled --policy runs/dcpa_sampled/ddqn_policy.pt --render
python demo_model.py --scenario perimeter_start --policy runs/perimeter_start/ddqn_policy.pt --render
python simulations/test_simulation.py --view dcpa-sampled-episode --render
python simulations/test_simulation.py --view perimeter-start-dcpa-sampled-episode --render
```

## Repository layout

- `trainings/` contains two scenario-specific training scripts and shared training network/config modules.
- `simulations/` contains simulation environments/configs and decoupled view modules under `simulations/sim_views/`.
- `legacy/` keeps archived older subprojects for reference.
