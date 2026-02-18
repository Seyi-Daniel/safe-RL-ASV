# safe-RL-ASV (main project)

The **default/main project is now the root-level unified feature RL stack**.

## Main entry points (root)

- `train.py` — DDQN training loop
- `run_episode.py` — evaluation/visualization runner
- `environment.py` — simulation environment and rendering
- `hyperparameters.py` — tunable config dataclasses
- `policy.py` — DDQN network definition

## Quick start

```bash
python train.py --no-render
python run_episode.py --render
```

## Repository layout

- Root files above = active/main code path.
- `legacy/` = archived older subprojects kept for reference and compatibility:
  - `legacy/RL_ASV`
  - `legacy/feature-RL-ASV`
  - `legacy/ASV_NEAT`
- `unified-feature-rl/` now contains compatibility wrappers that forward to root files.

## Notes

- Existing commands using `python unified-feature-rl/train.py` and
  `python unified-feature-rl/run_episode.py` still work via wrappers.
- The active development target should now be the root-level files.
