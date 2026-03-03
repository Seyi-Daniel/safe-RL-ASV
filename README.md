# safe-RL-ASV (main project)

The **default/main project is now the root-level unified feature RL stack**.

## Main entry points (root)

- `train.py` — DDQN training loop
- `demo_model.py` — demo/visualization runner for trained checkpoints
- `environment.py` — simulation environment and rendering
- `hyperparameters.py` — tunable config dataclasses
- `policy.py` — DDQN network definition

## Quick start

```bash
python train.py --no-render
python demo_model.py --policy runs/ddqn_policy.pt --render
```

## Repository layout

- Root files above = active/main code path.
- `legacy/` = archived older subprojects kept for reference:
  - `legacy/RL_ASV`
  - `legacy/feature-RL-ASV`
  - `legacy/ASV_NEAT`

## Notes

- The active development target is the root-level files.
