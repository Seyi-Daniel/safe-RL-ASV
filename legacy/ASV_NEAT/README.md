# ASV NEAT — COLREGs Crossing Scenario

This repository hosts an experimental NEAT-Python implementation for training a
give-way autonomous surface vessel (ASV) to resolve COLREGs-compliant
encounters. The codebase has been consolidated into the `asv_neat/` project,
which cleanly separates three concerns:

* **Scenario generation** – deterministic construction of fifteen COLREGs
  encounters (five each for crossing, head-on and overtaking) so the stand-on
  vessel always lies within the rule-defined sector and the speed profiles can
  be tuned per situation.
* **Environment simulation** – a lightweight pygame-compatible integrator with
  continuous rudder dynamics (limited angle/rate) and simple throttle while
  tracking both vessels’ goals.
* **Evolutionary training** – NEAT-Python integration that evaluates every
  genome across the fifteen scenarios _in parallel_, aggregates the cost
  metrics and feeds the resulting **minimisation** objective back to NEAT.

The give-way vessel receives a 12-dimensional observation vector (position,
heading, speed and goal coordinates for itself and the stand-on craft) and
outputs two values: a continuous rudder command in ``[-1, 1]`` (scaled by the
rudder limits) and a throttle selection (coast/accelerate/decelerate). Reward
shaping encourages quick, collision-free arrivals while penalising COLREGs
violations within a configurable TCPA/DCPA envelope.

---

## Repository layout

```
asv_neat/
├── configs/                # neat-python configuration (input/output counts etc.)
├── scripts/                # CLI entry points for training and scenario previewing
└── src/asv_neat/           # Python package with env, scenarios, NEAT utilities
```

Key modules:

* `src/asv_neat/scenario.py` — deterministic geometry helpers and scenario
  dataclasses (including goal placement offsets).
* `src/asv_neat/env.py` — pygame-ready simulation core that advances both boats
  according to helm/throttle commands while exposing snapshots for NEAT.
* `src/asv_neat/neat_training.py` — parallel evaluation loop, cost function and
  training harness wrapping neat-python.
* `src/asv_neat/hyperparameters.py` — a single source of truth for every
  numeric tunable used by the project, with a CLI-friendly override mechanism.

---

## Hyperparameters

All numeric values live in `HyperParameters` and can be adjusted at runtime. To
inspect them:

```bash
python asv_neat/scripts/train.py --list-hyperparameters
```

Each entry can be overridden on the command line with `--hp NAME=value`. For
example, to test a slower acceleration profile and a tighter COLREGs window:

```bash
python asv_neat/scripts/train.py --hp boat_accel_rate=1.2 --hp tcpa_threshold=60
```

Important groups include:

| Name prefix | Purpose |
|-------------|---------|
| `boat_*`    | Hull geometry and surge acceleration/deceleration limits. |
| `rudder_*`  | Continuous rudder limits (max angle, yaw rate, and rudder slew rate). |
| `env_*`     | Integration settings and render scaling. |
| `scenario_*`| Encounter layout (distance/goal extension) and per-scenario speed profiles. |
| `feature_*` | Normalisation constants applied to the 12-element observation vector. |
| `max_steps`, `step_cost`, `goal_bonus`, `collision_penalty`, `timeout_penalty`, `distance_cost`, `distance_normaliser` | Cost shaping terms for the minimisation objective. |
| `tcpa_threshold`, `dcpa_threshold`, `angle_threshold_deg`, `wrong_action_penalty` | Continuous COLREGs violation penalties (per-step costs when the wrong turn is taken inside the window). |

Invalid overrides raise a friendly error so experiments remain reproducible.

---

## Training workflow

1. Install the core dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Launch training (optionally selecting a specific encounter family):

   ```bash
   # Crossing-only run with a custom reward shaping term.
   python asv_neat/scripts/train.py --scenario-kind crossing --generations 100 --hp goal_bonus=-50
   ```

   * Every genome is evaluated on all fifteen scenarios using a thread pool so the
     environment dynamics remain independent.
   * Fitness values are set to the **negative** average cost, meaning lower cost
     solutions yield higher NEAT fitness while respecting the minimisation
     framing.
   * The default NEAT config (`configs/neat_crossing.cfg`) already expects 12
     inputs and two outputs (rudder command and throttle selector), matching the
     observation/action definitions.

3. The winning genome is saved by default to `winners/<scenario>_winner.pkl` so
   it can be replayed immediately. Override the location with `--save-winner`
   if you prefer a custom path. The demo (`scripts/demo_winner.py`) and LIME
   explainer (`scripts/lime_explain.py`) both look in the same directory when no
   `--winner` path is supplied.
4. After evolution the script prints a scenario-by-scenario summary (steps,
   COLREGs penalty, collision status, average cost) and, if `--render` is used,
   replays each encounter via pygame. Without `--render` the fifteen summaries are
   gathered in parallel so the evaluations finish together before printing.

Checkpoints can be enabled with `--checkpoint-dir` and
`--checkpoint-interval` to capture intermediate states during longer runs. You
can also archive the top genomes for every species and generation by pointing
`--species-archive` at a directory (defaults to `winners/species_archive`). The
archive stores up to `--species-top-n` pickled genomes (default 3) per species,
along with the NEAT config used to train them.

---

## Scenario visualisation

Use the helper script to inspect the deterministic encounters and render random
helm/throttle sequences. By default every crossing, head-on and overtaking case
is replayed in series, but you can focus on a specific family via
`--scenario`:

```bash
python asv_neat/scripts/crossing_scenario.py --render --duration 40 --seed 42
```

CLI flags accepted by the preview script:

| Flag | Description |
|------|-------------|
| `--render` | Enable the pygame viewer so that each scenario is visualised in sequence. |
| `--duration <seconds>` | Number of simulated seconds to replay when `--render` is enabled (default `35`). |
| `--seed <int>` | Seed for the random give-way policy that drives the placeholder helm/throttle inputs. |
| `--scenario {all,crossing,head_on,overtaking}` | Limit the preview to a single encounter family (default `all`). |
| `--hp NAME=VALUE` | Repeatable overrides for any hyperparameter exposed by `HyperParameters`. |

Hyperparameters such as the encounter distances or the six per-scenario speed
settings can be adjusted
here too via `--hp` overrides, ensuring the preview matches the training
configuration. During rendering the give-way and stand-on destinations are now
drawn as colour-coded markers so you can confirm that each vessel has its own
goal beyond the shared crossing point.

**HUD overlay:** when the pygame window is open you can press `H` to toggle the
right-aligned HUD that lists each vessel’s position, heading and speed along
with the TCPA/DCPA estimates. The preview script starts with the HUD hidden so
`H` is the quickest way to bring it into view during a render session.

### Replaying saved winners

Use `scripts/demo_winner.py` to visualise a previously trained genome. By
default it looks for `winners/<scenario>_winner.pkl` so the file emitted by a
scenario-specific `train.py --render` session can be replayed immediately:

```bash
python asv_neat/scripts/demo_winner.py --scenario-kind crossing --render
```

All hyperparameter overrides accepted by the training CLI are available here as
well, ensuring the demo uses the exact same environment conditions as the run
that produced the saved genome.

---

## LIME explanation workflow

The project ships with an end-to-end LIME pipeline (`scripts/lime_explain.py`)
for analysing the controller’s per-step decisions. Explanations now **reuse
captured traces and render frames** from an earlier run instead of re-simulating
each scenario, keeping the explanation phase fast and repeatable.

### Prerequisites

1. Install the explainer dependencies (included in `requirements.txt`):

   ```bash
   pip install -r requirements.txt
   ```

2. Produce a `winner.pkl` via `scripts/train.py --save-winner` or grab the
   auto-saved file emitted by a rendered training/demo session. Ensure the
   matching NEAT config (defaults to `configs/neat_crossing.cfg`) is available so
   the genome can be re-instantiated.

### Capturing traces and frames for reuse

Run the capture utility once to save traces, metadata and render frames for each
deterministic scenario. The resulting folders are consumed by both the LIME and
SHAP explainers:

```bash
python asv_neat/scripts/capture_episode_data.py \
  --scenario-kind crossing \
  --winner winners/crossing_winner.pkl \
  --output-dir captured_episodes \
  --hp max_steps=400  # optional overrides to mirror the training run
```

This writes per-scenario subdirectories (e.g. `captured_episodes/01_crossing/`)
containing `metadata.json`, `trace.json`, and a `frames/` folder with
`frame_000.png`, … files recorded during the replay.

### Running the explainer

```
python asv_neat/scripts/lime_explain.py \
  --scenario-kind crossing \
  --winner winners/crossing_winner.pkl \
  --data-dir captured_episodes \
  --output-dir lime_reports \
  --hp max_steps=400  # optional overrides so the replay matches training
```

For every deterministic scenario of the selected encounter family the script:

1. Rebuilds the NEAT network from the pickle/config pair.
2. Loads the previously captured `trace.json` and `frames/` for each scenario
   from `--data-dir` instead of re-running the simulation.
3. Computes summary metrics (steps, collision status, goal distance, COLREGs
   penalties, etc.) and their aggregate cost using the captured metadata.
4. Feeds the captured feature matrix through `lime.LimeTabularExplainer` using
   the built-in feature names and action labels, yielding a local explanation for
   every time-step.

### Output artefacts

Results are grouped per scenario inside the requested `--output-dir`:

```
lime_reports/
└── 01_crossing/
    ├── metadata.json      # scenario geometry + per-episode metrics
    ├── trace.json         # ordered list of recorded features/states/actions
    ├── lime_step_000.json # individual per-step explanations (one file each)
    └── lime_summary.json  # array of all per-step explanations in order
```

Each explanation records the rudder output, the discretised throttle choice
with its derived probabilities, the normalised feature values presented to the
controller, and the LIME-attributed weights for those features. This makes it
straightforward to line up a specific command choice with its causal inputs,
repeatable across all steps (`max_steps`) and across five canonical encounters
per scenario type.

### SHAP explanation workflow

For a global/stepwise view of feature importance, a parallel SHAP pipeline is
available via `scripts/shap_explain.py`. It consumes the captured traces and
frames from `capture_episode_data.py`, then feeds the feature matrix through
`shap.KernelExplainer` so every timestep receives SHAP values, plots and a
stitched animation alongside the environment render frames.

#### Prerequisites

Install the SHAP dependencies (included in `requirements.txt`) in the same
environment used for training/demo:

```bash
pip install -r requirements.txt
```

#### Running the explainer

```bash
python asv_neat/scripts/shap_explain.py \
  --scenario-kind crossing \
  --winner winners/crossing_winner.pkl \
  --data-dir captured_episodes \
  --output-dir shap_reports \
  --hp max_steps=400  # optional overrides so the replay matches training
```

Artefacts mirror the LIME layout with SHAP-specific filenames. Each scenario is
loaded from the matching folder in `--data-dir` so explanations complete without
running the simulator again:

```
shap_reports/
└── 01_crossing/
    ├── metadata.json       # scenario geometry + per-episode metrics
    ├── trace.json          # ordered list of recorded features/states/actions
    ├── shap_step_000.json  # per-step SHAP attributions (one file each)
    ├── shap_summary.json   # array of all SHAP explanations in order
    └── explanation_animation.gif  # combined render + bar plot per step
```

### LLM interpretation workflow

You can generate step-by-step natural language interpretations by combining the
LIME and SHAP summaries with an LLM. The helper script reads both
`lime_summary.json` and `shap_summary.json`, builds a per-step prompt with the
feature values and attributions, and stores the resulting explanations in a
matching `llm_reports/` directory. The default system prompt enforces a strict
JSON-only response: one object per step with the feature attributions, linked
COLREGs rule(s), and a succinct justification.

```bash
export LLM_API_KEY="your_provider_key"
python asv_neat/scripts/llm_explain.py \
  --lime-summary lime_reports/01_crossing/lime_summary.json \
  --shap-summary shap_reports/01_crossing/shap_summary.json \
  --output-dir llm_reports/01_crossing \
  --metadata-file captured_episodes/01_crossing/metadata.json \
  --frames-dir captured_episodes/01_crossing/frames \
  --model gpt-4o-mini \
  --max-features 8 \
  --max-steps 50
```

To give the LLM additional simulation context (vessel dimensions, speeds,
rudder limits, etc.), include the default hyperparameters in the prompt and
optionally override any values with `--hp` so they match the run that generated
the traces:

```bash
python asv_neat/scripts/llm_explain.py \
  --lime-summary lime_reports/01_crossing/lime_summary.json \
  --shap-summary shap_reports/01_crossing/shap_summary.json \
  --output-dir llm_reports/01_crossing \
  --metadata-file captured_episodes/01_crossing/metadata.json \
  --frames-dir captured_episodes/01_crossing/frames \
  --include-hyperparameters \
  --hp boat_max_speed=6.5 \
  --hp rudder_max_angle_deg=30
```

The script writes a `llm_summary.json` file plus one JSON file per step:

```
llm_reports/
└── 01_crossing/
    ├── llm_step_000.json
    ├── llm_step_001.json
    └── llm_summary.json
```

### Sensitivity testing from LIME + SHAP top features

After generating `lime_summary.json` and `shap_summary.json`, you can run a
feature-perturbation sensitivity test that validates whether the top-attributed
features actually drive the model outputs.

The helper script `asv_neat/scripts/explanation_sensitivity.py`:

1. Loads matching scenario folders from `--lime-dir` and `--shap-dir`.
2. For every common step, extracts the top-`k` (default 3) most influential
   features separately for:
   * LIME rudder attributions,
   * LIME throttle attributions,
   * SHAP rudder attributions,
   * SHAP throttle attributions.
3. Perturbs each selected feature by ±1%, ±2%, and ±3% (configurable).
4. Re-runs the NEAT model for each perturbation and records:
   * rudder output deltas and magnitude changes,
   * rudder inferred helm label changes (`turn_port`, `turn_starboard`,
     `keep_straight`),
   * throttle raw output deltas,
   * throttle discrete command/label changes.

Example:

```bash
python asv_neat/scripts/explanation_sensitivity.py \
  --winner winners/crossing_winner.pkl \
  --config asv_neat/configs/neat_crossing.cfg \
  --lime-dir lime_reports \
  --shap-dir shap_reports \
  --output-dir sensitivity_reports
```

Optional overrides:

```bash
python asv_neat/scripts/explanation_sensitivity.py \
  --winner winners/crossing_winner.pkl \
  --lime-dir lime_reports \
  --shap-dir shap_reports \
  --output-dir sensitivity_reports \
  --top-k 3 \
  --perturbation-pcts 1 2 3
```

Output layout:

```
sensitivity_reports/
├── sensitivity_index.json
└── 01_crossing/
    └── sensitivity_summary.json
```

Each `sensitivity_summary.json` contains per-step baseline features/outputs and
per-feature perturbation rows for both explainers, including directional
changes (+/-) and output deltas so the influence can be compared directly.

To make this report easier to review in spreadsheet form, convert it into a
flat table (CSV and optional XLSX with merged grouping cells) using:

```bash
python asv_neat/scripts/sensitivity_table.py \
  --input sensitivity_reports/01_crossing/sensitivity_summary.json
```

This creates:

* `sensitivity_summary_table.csv` (portable table output), and
* `sensitivity_summary_table.xlsx` (grouped/merged cells for repeated `Step`,
  `Explanation`, `Output Channel`, and `Feature`, with centered alignment and
  borders for readability), including both `Label Change` and `New Label`
  columns so changed outputs are explicit.

If you only need CSV, add `--skip-xlsx`.

### Building a combined LIME+SHAP animation

After running the individual LIME and SHAP explainers you can stitch their
outputs into a side-by-side animation that pairs every simulation frame with a
2×2 grid of rudder/throttle attribution plots. The helper script automatically
discovers the relevant subdirectories (frames and plots) inside each run so you
do **not** need to hard-code names like `frames/` or `plots/`:

```bash
python asv_neat/scripts/combine_lime_shap_animation.py \
  --lime-dir lime_reports/01_crossing \
  --shap-dir shap_reports/01_crossing \
  --output-dir combined_reports/01_crossing \
  --fps 10
```

The script:

1. Searches the provided LIME/SHAP scenario folders for the frame files
   (`frame_000.png`, …) and the explanation plots
   (`explanation_rudder_000.png` / `explanation_throttle_000.png`). The search
   relies on existing repository constants/layout rather than fixed directory
   names.
2. Identifies step indices available in the frame set and keeps only those
   where **all** four plots (LIME/SHAP × rudder/throttle) exist.
3. Builds composite frames in `combined_frames/` under the chosen output
   directory, named `combined_000.png`, `combined_001.png`, …
4. Emits a final GIF at
   `combined_reports/01_crossing/lime_shap_explanation_animation.gif`, with
   frame rate controlled by `--fps` (default `8`).

Each composite frame keeps the simulation render on the left and positions a
height-matched 2×2 grid of plots on the right (LIME/SHAP rudder on the top row,
LIME/SHAP throttle on the bottom row). The script normalises plot heights so the
stack matches the scene, padding with whitespace to preserve readability instead
of distorting images.

---

## LLM control verification

To validate whether the NEAT controller’s per-step outputs look reasonable, you
can re-query a language model with the same normalized input features stored in
the LIME/SHAP summary JSON files. The `llm_verify_controls.py` script sends the
feature values to an OpenAI-compatible endpoint and compares the LLM’s returned
rudder/throttle against the model’s original output.

### Prerequisites

* A LIME or SHAP summary JSON file (e.g., `lime_summary.json`).
* An OpenAI-compatible API endpoint and API key (defaults use
  `LLM_API_URL`/`LLM_API_KEY`).

### Example usage

```bash
export LLM_API_KEY="your-key-here"
python asv_neat/scripts/llm_verify_controls.py \
  --summary lime_summary.json \
  --output llm_control_verification.json \
  --model gpt-4o-mini \
  --rudder-tolerance 0.1 \
  --max-steps 50
```

### Output format

The script writes a JSON report with an overall summary and a per-step record:

* `summary` — includes the source summary path, model name, total steps,
  matched steps, match rate, and the rudder tolerance used.
* `steps` — for each step, includes the input features, the original model
  outputs, the LLM outputs (plus raw response), and a comparison block with
  rudder/throttle match flags.

If you want to limit the verification window, use `--start-step` / `--end-step`
or `--max-steps`. To tune the matching threshold for rudder outputs, adjust
`--rudder-tolerance`. The script still writes a structured report even when an
LLM call fails (the `error` field will be populated).

### Visualising LLM verification replays

To compare the original model trajectory against the LLM-driven “shadow”
trajectory, use the verification replay demo. The script reads the
`llm_control_verification.json` output and replays both vessels in a single
render window so their positions and commands can be visually compared.

```bash
python asv_neat/scripts/demo_llm_verification.py \
  --verification llm_control_verification.json \
  --render \
  --step-delay 0.05
```

Use `--start-step`, `--end-step`, or `--max-steps` to restrict the replay range.
When rendering, the model vessel is labelled “Model ASV” and the LLM vessel is
labelled “LLM Shadow”. If the verification file includes target-vessel features,
the target is shown alongside both trajectories.

To mirror the deterministic training setups (15 total scenarios across the three
encounter types), pass the scenario kind and 1-based index along with any
hyperparameter overrides that were used during training:

```bash
python asv_neat/scripts/demo_llm_verification.py \
  --verification llm_control_verification.json \
  --scenario-kind crossing \
  --scenario-index 1 \
  --hp env_dt=0.2 \
  --hp env_pixels_per_meter=2 \
  --render
```

When `--scenario-kind` is provided, the replay uses the matching deterministic
scenario geometry for the initial state and HUD metadata, keeping the starting
positions, goals, and heading/speed scales aligned with the training setup.

---

## Cost function overview

The minimisation objective combines several components:

* **Step cost** (`step_cost × steps`) guarantees that slower solutions accrue
  higher penalties (1,000 steps → cost 1,000 by default).
* **Goal bonus** (negative value) rewards fast arrivals, while a configurable
  timeout penalty and distance term apply when the goal is missed.
* **Collision penalty** adds a fixed cost when separation falls below the
  collision distance.
* **COLREGs penalty** accrues `wrong_action_penalty` each step the agent chooses
  a non-starboard helm while the encounter falls within the specified TCPA,
  DCPA and bearing thresholds.

These terms can all be tuned through the shared hyperparameter interface to
support different research experiments without touching the core code.
