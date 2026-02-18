# unified-feature-rl (compatibility layer)

This folder is now a **compatibility shim**.

The active project moved to repository root:

- `train.py`
- `run_episode.py`
- `environment.py`
- `hyperparameters.py`
- `policy.py`

You can still run legacy commands from this folder path:

```bash
python unified-feature-rl/train.py
python unified-feature-rl/run_episode.py
```

These wrappers forward to the root-level main implementation.
