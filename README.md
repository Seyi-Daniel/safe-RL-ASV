# safe-RL-ASV

Training bootstrap now supports the requested initialization behavior:

- World size defaults to `500 x 500`.
- Pixels-per-meter defaults to `2.0`.
- Vessel starts at world center.
- Vessel heading is randomized at reset.
- A dotted inner circle is drawn around the vessel (`--spawn-ring-radius` hyperparameter).
- A larger outer circle is drawn (`--goal-ring-radius` hyperparameter).
- Goal is sampled randomly on that outer circle.

## Run training

Without rendering:

```bash
python train.py
```

With initialization preview rendering:

```bash
python train.py --render
```

Adjustable hyperparameters example:

```bash
python train.py --render --world-width 500 --world-height 500 --pixels-per-meter 2.0 --spawn-ring-radius 60 --goal-ring-radius 180 --seed 7
```

> Press `r` in the render window to resample heading + goal on reset.
