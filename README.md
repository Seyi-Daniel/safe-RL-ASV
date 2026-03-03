# safe-RL-ASV

This repository is now intentionally organized into exactly three active parts (plus `legacy/`, which is preserved untouched):

1. `simulations/` → simulation-only experiments (no RL training).
2. `trainings/` → RL training pipelines built on top of chosen simulations.
3. `demo_saved_model.py` → single demo runner for trained models.

`legacy/` is archival and was not modified.

---

## 1) Simulations folder (no RL)

`simulations/` is where you prototype and test scenario behavior.

### Current layout

- `simulations/common/`
  - `environment.py` (shared simulator + COLREGS logic)
  - `hyperparameters.py` (shared env/reward/training dataclasses)
- `simulations/scenarios/`
  - `dcpa_sampling/`
    - `simulation.py` (the actual simulation-only scenario runner)
    - `runtime.py` (scenario-specific runtime config)
- `simulations/testbed.py`
  - top-level sandbox runner that selects scenario modules from a registry.

### Add a new simulation

Create a new folder under `simulations/scenarios/<your_sim>/` with its own `simulation.py` (and optional `runtime.py`).
This keeps each simulation decoupled while still allowing shared utilities where needed.

Run simulation testbed:

```bash
python -m simulations.testbed --view dcpa-sampled-episode --episodes 3 --render
```

---

## 2) Trainings folder (full RL)

`trainings/` contains reinforcement-learning training pipelines that are tied to specific simulation setups.

### Current layout

- `trainings/common/`
  - `policy.py` (shared RL actor/critic models)
- `trainings/dcpa_sampling/`
  - `train.py` (RL training for the DCPA sampling simulation family)

Training output defaults to:

- models: `artifacts/models/`
- logs: `artifacts/training_logs/`

Run training:

```bash
python -m trainings.dcpa_sampling.train --no-render
```

---

## 3) Single demo file for saved models

- `demo_saved_model.py`

This file is the one place for loading saved checkpoints and running demo/evaluation episodes.

Example:

```bash
python demo_saved_model.py --policy artifacts/models/dcpa_sampling/model_ep020.pt --render
```

---

## Artifacts

- `artifacts/models/` → trained model checkpoints
- `artifacts/training_logs/` → training/evaluation logs
- `artifacts/simulation_logs/` → simulation-only outputs

---

## How this changed from before

Previously, the active code path was spread across package/workflow wrappers and compatibility layers.
Now the active project is flattened into:

- simulation-only work in `simulations/`
- RL training work in `trainings/`
- demo execution in `demo_saved_model.py`

This was done to make the repository easier to read and maintain as multiple simulation types and matching trainings are added.
