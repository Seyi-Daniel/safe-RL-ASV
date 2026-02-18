# RL_ASV

## New feature-based COLREGs environment

`colregs_random_feature_env.py` provides a standalone reinforcement-learning environment with:

- one controllable ASV (ego) and one random target vessel,
- random episode initialisation inside a square world,
- 12-sector internal encounter awareness within a configurable radius,
- 10-feature observation vector (ego state + goal + relative target features),
- COLREGs-inspired reward shaping (head-on, crossing, overtaking),
- DCPA "super action" reward (positive reward when DCPA increases).

### Observation (10 features)

1. Ego `x`
2. Ego `y`
3. Ego heading
4. Ego speed
5. Ego goal `x`
6. Ego goal `y`
7. Target relative heading
8. Target speed
9. Relative bearing to target
10. Relative distance to target (clamped by encounter radius)

### Actions

Discrete `0..8`, decoded as `(steer, throttle)` using 3x3 combinations:

- steer: `straight`, `right`, `left`
- throttle: `coast`, `accelerate`, `decelerate`

### Quick run

```bash
python RL_ASV/colregs_random_feature_env.py
```

### Train (DQN)

```bash
python RL_ASV/train_colregs_feature.py --episodes 200
```
