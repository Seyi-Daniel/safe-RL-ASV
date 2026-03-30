# Parallel Crossing-Port Training Path

This document describes a **fully parallel** DDPG training/eval/demo path under `trainings/dcpa_sampled/`.

## Key guarantee

- Existing project files are untouched.
- Environment dynamics, observations, takeover logic, and risk logic come from the existing environment implementation.
- Learning reward for this path comes from `CrossingObjectivePort` through a new adapter layer.

## New files

- `trainings/dcpa_sampled/train_crossing_port.py`
- `trainings/dcpa_sampled/demo_crossing_port.py`
- `trainings/dcpa_sampled/reward_ports/crossing_objective_adapter.py`
- `trainings/dcpa_sampled/README_crossing_port_training.md`

## What differs from existing `train.py`

- Environment stepping is unchanged (`SingleVessel2FeatureEnv`), but **replay reward is not taken from env scalar reward**.
- For each step, this path computes `port_reward` via adapter + `CrossingObjectivePort`.
- Optimization is driven by inserted replay transitions that use `port_reward`.
- Episode logging records both:
  - `env_return` (for inspection)
  - `port_return` (crossing-objective reward accumulation)

## Adapter behavior and vessel2 support

The reward port is vessel1-shaped. The new adapter handles both give-way cases:

- If designated give-way vessel is `vessel1`: pass-through `info`/`env`/action.
- If designated give-way vessel is `vessel2`: map vessel2 fields into vessel1-shaped keys expected by the reward port:
  - `vessel2_goal_distance -> vessel1_goal_distance`
  - `vessel2_goal_heading_error_deg -> vessel1_goal_heading_error_deg`
  - `vessel2_rudder_deg -> vessel1_rudder_deg`
  - success fields remapped in local adapted info (`vessel2_reached -> vessel1_reached`, and mirrored counterpart)
  - `designated_give_way_vessel` rewritten to `vessel1` in adapted info
- If env geometry fallback is needed while adapting vessel2, a lightweight proxy exposes `env.vessel2` as a vessel1-equivalent object.

Rudder-sign note for this project:

- Positive rudder is treated as **starboard**.
- Crossing wrong-action penalty is therefore applied to **non-starboard** actions (rudder `<= 0.0`), not starboard actions.

## Replay insertion policy (required crossing semantics)

Replay entries are inserted only when all conditions hold at the step:

1. A designated give-way vessel is resolvable (`vessel1` or `vessel2`).
2. That same vessel is RL-controlled in the current step (pre-step control list).
3. Adapter successfully computes crossing port reward for that vessel.

Then exactly one replay transition is inserted for that acting designated give-way vessel.

This means:
- non-give-way RL-controlled vessels do **not** receive mirrored crossing reward entries,
- if give-way vessel cannot be resolved, no crossing-port replay insertion occurs for that step.

## Outputs and folder layout

Default output base folder for this parallel path:

- `trainings/dcpa_sampled/runs_crossing_port/`

Each run creates a timestamped subfolder `mmdd_HHMM` containing:

- `policy_<episode>.pt`
- `policy_latest.pt`
- `train_history.json`

## History fields

`train_history.json` includes at least:

- `episode`
- `env_return`
- `port_return`
- `steps`
- `epsilon`
- `mean_actor_loss`
- `mean_critic_loss`
- `collision`
- `goal_completion`
- `safe_pass`
- `takeovers`
- `both_controlled_steps`
- `starboard_opportunities`
- `starboard_compliant`
- `starboard_compliance_rate`
- `sample_attempt`
- `sample_steps`
- `sample_scenario`
- `sampling_scenario_filter`
- `port_reward_vessel_role_count`
- `port_replay_entries_inserted`

## Train

Example:

```bash
python trainings/dcpa_sampled/train_crossing_port.py \
  --episodes 600 \
  --out-dir runs_crossing_port \
  --sampling-scenario crossing
```

## Eval-only

Example:

```bash
python trainings/dcpa_sampled/train_crossing_port.py \
  --eval-only \
  --out-dir runs_crossing_port \
  --eval-scenario all \
  --eval-episodes 30
```

Checkpoint resolution order in eval-only:
1. `--eval-checkpoint` if provided
2. `<out-dir>/policy_latest.pt` if it exists
3. newest timestamped run under `<out-dir>` that has `policy_latest.pt`
4. error if none found

## Demo

Example:

```bash
python trainings/dcpa_sampled/demo_crossing_port.py \
  --checkpoint trainings/dcpa_sampled/runs_crossing_port/<mmdd_HHMM>/policy_latest.pt \
  --episodes 5
```

Demo summary prints include:
- env total reward
- port total reward
- steps
- collision
- success
- reason
- scenario
- per-episode vessel counts receiving port reward

## Limitations

- This path is crossing-objective centered because `CrossingObjectivePort` is crossing-focused.
- Replay insertion is intentionally restricted to the acting designated give-way vessel.
- This is not a full symmetric replacement for the existing project reward system.
