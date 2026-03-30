# Crossing Objective Reward Port (Standalone)

## What this is

This directory introduces a **new, standalone reward port** for crossing behavior:

- `crossing_objective_state.py`
- `crossing_objective_port.py`

It does **not** alter any existing project file and does **not** replace existing reward logic. It is designed for later integration by direct import and call from any training loop that has access to `info` and optional `env` / `action` values.

## Scope and non-goals

- This is **not** a port of old NEAT control flow.
- This is **not** an episode-fitness transplant.
- This **is** a per-step RL reward conversion of the old crossing objective ingredients, using the same constants and explicit cost-to-reward sign conversion.

## Constants used (exact)

```python
GOAL_PROGRESS_BONUS = -1.2
HEADING_ALIGNMENT_THRESHOLD_DEG = 12.0
WRONG_ACTION_PENALTY = 1.0
STEP_COST = 1.0
STEP_COUNT_COST = 1.0
GOAL_BONUS = -40.0
TIMEOUT_PENALTY = 0.0
DISTANCE_COST = 1.5
DISTANCE_NORMALISER = 250.0
COLLISION_PENALTY = 200.0
GOAL_TOLERANCE = 10.0
```

## Cost-to-reward sign conversion

The old objective was cost-based. This port returns RL rewards, so signs are converted explicitly:

- Positive old costs -> negative rewards
- Negative old costs/bonuses -> positive rewards

Concretely:

- step cost: `-STEP_COST`
- wrong action: `-WRONG_ACTION_PENALTY`
- collision: `-COLLISION_PENALTY`
- timeout: `-TIMEOUT_PENALTY`
- distance cost: `-DISTANCE_COST * (distance / DISTANCE_NORMALISER)`
- goal bonus: `+abs(GOAL_BONUS)`
- progress shaping from `GOAL_PROGRESS_BONUS = -1.2`:
  - positive progress: `+abs(GOAL_PROGRESS_BONUS)` = `+1.2`
  - negative progress: `-abs(GOAL_PROGRESS_BONUS)` = `-1.2`

## Dense living penalty (every step)

Every call applies both:

1. `living_step_reward = -STEP_COST`
2. `living_stepcount_reward = -(STEP_COUNT_COST / max(1.0, DISTANCE_NORMALISER))`

The second term uses `DISTANCE_NORMALISER` as required.

## Dense distance penalty (every step)

Each step computes goal distance and applies:

`distance_penalty = -DISTANCE_COST * (current_goal_distance / DISTANCE_NORMALISER)`

Goal distance resolution order:

1. `info["vessel1_goal_distance"]`
2. `info["goal_distance"]`
3. computed from `env.vessel1` position and goal coordinates
4. otherwise raises `ValueError`

## Progress shaping rule (every step)

Progress shaping is skipped only on first step (no previous values yet). Otherwise compare current vs previous distance and heading error, then apply exactly:

- positive `+abs(GOAL_PROGRESS_BONUS)` if:
  - `distance_improved AND (current_heading_good OR heading_improved OR prev_heading_good)`
  - OR `heading_improved AND NOT distance_regressed`
- negative `-abs(GOAL_PROGRESS_BONUS)` if:
  - `distance_regressed AND (heading_regressed OR (not current_heading_good and not prev_heading_good))`
  - OR `heading_regressed AND NOT distance_improved`
- else `0.0`

Heading-good threshold uses:

`abs(goal_heading_error_deg) <= HEADING_ALIGNMENT_THRESHOLD_DEG`

Heading error resolution order:

1. `info["vessel1_goal_heading_error_deg"]`
2. `info["goal_heading_error_deg"]`
3. computed from `env.vessel1` heading and bearing-to-goal
4. otherwise raises `ValueError`

## Wrong-action penalty rule (crossing-only)

Wrong-action penalty is evaluated only when scenario resolves to crossing:

Scenario resolution:

1. `info["colregs_scenario"]`
2. `info["scenario"]`
3. fallback `"unknown"`

Apply wrong-action logic only for `"crossing"`.

If `info["designated_give_way_vessel"]` exists and is not `"vessel1"`, skip wrong-action penalty.

Rudder/action resolution order:

1. `info["vessel1_rudder_deg"]`
2. `info["rudder_cmd"]`
3. `action[0]` for list/tuple or array-like action
4. else no wrong-action penalty

Ported semantics:

- starboard turn is correct
- non-starboard is wrong
- if resolved rudder/action value is `<= 0.0`: `wrong_action_reward = -WRONG_ACTION_PENALTY`
- else `wrong_action_reward = 0.0`

## Terminal terms (applied when signaled)

Signals:

- `collision = info.get("collision", False)`
- success:
  - if both keys present: `info["vessel1_reached"] and info["vessel2_reached"]`
  - else `info.get("success", False)`
- `reason = info.get("reason", "")`

Terminal rewards:

1. collision -> `-COLLISION_PENALTY`
2. success -> `+abs(GOAL_BONUS)`
3. timeout (`reason == "timeout"`) -> `-TIMEOUT_PENALTY`

Success and timeout are not double-counted simultaneously.

## Breakdown API

`CrossingObjectivePort.get_reward_breakdown()` returns per-step terms and cumulative totals, including:

- `step_index`
- `living_step_reward`
- `living_stepcount_reward`
- `distance_penalty`
- `progress_reward`
- `wrong_action_reward`
- `collision_terminal_reward`
- `goal_terminal_reward`
- `timeout_terminal_reward`
- `total_step_reward`
- `cumulative_reward`
- `cumulative_progress_reward`
- `cumulative_wrong_action_penalty`
- `cumulative_living_penalty`
- `cumulative_distance_penalty`
- `cumulative_terminal_reward`

## Assumptions and fallbacks

- `info` is assumed to be a dictionary-like step metadata object.
- If needed fields are absent from `info`, optional `env.vessel1` is used for derived geometry.
- For environment-based geometry, the code attempts common attribute names for vessel position, heading, and goal coordinates.
- Missing resolvable geometry fields raise explicit `ValueError` messages so integration issues fail fast and clearly.
