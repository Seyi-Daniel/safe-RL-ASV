"""NEAT integration for the deterministic COLREGs crossing scenarios."""
from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
import pickle
import shutil

import neat
import matplotlib.pyplot as plt
from neat.reporting import BaseReporter
from .config import BoatParams, EnvConfig, RudderParams
from .env import CrossingScenarioEnv
from .hyperparameters import HyperParameters
from .scenario import (
    EncounterScenario,
    ScenarioRequest,
    default_scenarios,
    scenario_states_for_env,
)
from .utils import (
    euclidean_distance,
    goal_distance,
    heading_error_deg,
    helm_label_from_rudder_cmd,
    relative_bearing_deg,
    clamp,
    tcpa_dcpa,
)


@dataclass
class EpisodeMetrics:
    """Summary statistics produced by a single simulated episode."""

    steps: int
    reached_goal: bool
    collided: bool
    final_distance: float
    min_separation: float
    wrong_action_cost: float
    goal_progress_bonus: float


def _normalise(value: float, scale: float) -> float:
    if scale <= 0.0:
        return value
    return max(-1.0, min(1.0, value / scale))


def observation_vector(
    agent: dict,
    stand_on: dict,
    params: HyperParameters,
) -> List[float]:
    """Return the 12-element feature vector consumed by the controller."""

    def pack(state: dict) -> List[float]:
        x = _normalise(float(state["x"]), params.feature_position_scale)
        y = _normalise(float(state["y"]), params.feature_position_scale)
        heading = _normalise(float(state["heading"]), params.feature_heading_scale)
        speed = _normalise(float(state.get("speed", 0.0)), params.feature_speed_scale)
        goal_x = _normalise(float(state.get("goal_x", state["x"])), params.feature_position_scale)
        goal_y = _normalise(float(state.get("goal_y", state["y"])), params.feature_position_scale)
        return [x, y, heading, speed, goal_x, goal_y]

    return pack(agent) + pack(stand_on)


TraceCallback = Callable[[dict], None]


def simulate_episode(
    env: CrossingScenarioEnv,
    scenario: EncounterScenario,
    network: neat.nn.FeedForwardNetwork,
    params: HyperParameters,
    *,
    render: bool = False,
    trace_callback: Optional[TraceCallback] = None,
    frame_callback: Optional[Callable[[int, Any], None]] = None,
) -> EpisodeMetrics:
    """Roll a network-controlled episode within ``env`` for ``scenario``."""

    states, meta = scenario_states_for_env(env, scenario)
    env.reset_from_states(states, meta=meta)

    if render:
        env.enable_render()

    min_sep = float("inf")
    wrong_action_cost = 0.0
    goal_progress_bonus = 0.0
    steps = 0

    for step_idx in range(params.max_steps):
        snapshot = env.snapshot()
        if not snapshot:
            break

        agent_state = snapshot[0]
        stand_on_state = snapshot[1] if len(snapshot) > 1 else snapshot[0]
        previous_distance = goal_distance(agent_state)
        previous_heading_error = heading_error_deg(agent_state)

        features = observation_vector(agent_state, stand_on_state, params)
        outputs = network.activate(features)
        rudder_cmd_raw = outputs[0] if outputs else 0.0
        throttle_raw = outputs[1] if len(outputs) > 1 else 0.0

        rudder_cmd = clamp(float(rudder_cmd_raw), -1.0, 1.0)
        throttle_i = clamp(int(round(float(throttle_raw) * 2.0)), 0, 2)

        helm_label = helm_label_from_rudder_cmd(rudder_cmd)
        action = (rudder_cmd, throttle_i)
        actions = [action, None]

        if trace_callback is not None:
            trace_callback(
                {
                    "step": step_idx,
                    "obs": list(features),
                    "features": list(features),
                    "outputs": list(outputs),
                    "action": action,
                    "rudder_cmd": float(rudder_cmd),
                    "throttle": int(throttle_i),
                    "helm_label": helm_label,
                    "agent_state": dict(agent_state),
                    "stand_on_state": dict(stand_on_state)
                    if stand_on_state is not None
                    else None,
                }
            )

        if render:
            env.apply_actions(actions)
            env.set_debug_overlay(
                {
                    "step": step_idx,
                    "rudder_cmd_for_arrow": float(rudder_cmd),
                    "rudder_cmd_raw": float(rudder_cmd_raw),
                }
            )
            env.render()
            if frame_callback is not None:
                frame_callback(step_idx, getattr(env, "_screen", None))
            env.advance_applied_actions()
            env.set_debug_overlay(None)
        else:
            env.step(actions)
        steps = step_idx + 1

        snapshot = env.snapshot()
        if not snapshot:
            break
        agent_state = snapshot[0]
        stand_on_state = snapshot[1] if len(snapshot) > 1 else None

        distance = goal_distance(agent_state)
        heading_error = heading_error_deg(agent_state)

        distance_improved = distance < (previous_distance - 1e-6)
        distance_regressed = distance >= (previous_distance + 1e-6)

        prev_outside = previous_heading_error > params.heading_alignment_threshold_deg
        new_outside = heading_error > params.heading_alignment_threshold_deg
        heading_delta = heading_error - previous_heading_error
        heading_improved = heading_delta < 0.0 and (prev_outside or new_outside)
        heading_regressed = heading_delta > 0.0 and new_outside

        reward_step = False
        penalty_step = False

        if distance_improved and (heading_delta <= 0.0 or not new_outside or heading_improved):
            reward_step = True
        elif distance_regressed and (heading_regressed or new_outside):
            penalty_step = True

        if not reward_step and heading_improved:
            reward_step = True
        if not penalty_step and heading_regressed and not distance_improved:
            penalty_step = True

        if reward_step:
            goal_progress_bonus += params.goal_progress_bonus
        elif penalty_step:
            goal_progress_bonus -= params.goal_progress_bonus

        if stand_on_state is not None:
            sep = euclidean_distance(
                float(agent_state["x"]),
                float(agent_state["y"]),
                float(stand_on_state["x"]),
                float(stand_on_state["y"]),
            )
            min_sep = min(min_sep, sep)

            if sep <= params.collision_distance:
                return EpisodeMetrics(
                    steps=steps,
                    reached_goal=False,
                    collided=True,
                    final_distance=distance,
                    min_separation=min_sep,
                    wrong_action_cost=wrong_action_cost,
                    goal_progress_bonus=goal_progress_bonus,
                )

            tcpa, dcpa = tcpa_dcpa(agent_state, stand_on_state)
            bearing = relative_bearing_deg(agent_state, stand_on_state)
            if (
                0.0 <= tcpa <= params.tcpa_threshold
                and dcpa <= params.dcpa_threshold
                and bearing <= params.angle_threshold_deg
            ):
                if rudder_cmd >= 0.0:
                    wrong_action_cost += params.wrong_action_penalty

        if distance <= params.goal_tolerance:
            return EpisodeMetrics(
                steps=steps,
                reached_goal=True,
                collided=False,
                final_distance=distance,
                min_separation=min_sep,
                wrong_action_cost=wrong_action_cost,
                goal_progress_bonus=goal_progress_bonus,
            )

    snapshot = env.snapshot()
    if snapshot:
        agent_state = snapshot[0]
        distance = goal_distance(agent_state)
        if len(snapshot) > 1:
            min_sep = min(
                min_sep,
                euclidean_distance(
                    float(agent_state["x"]),
                    float(agent_state["y"]),
                    float(snapshot[1]["x"]),
                    float(snapshot[1]["y"]),
                ),
            )
    else:
        distance = 0.0

    return EpisodeMetrics(
        steps=max(steps, params.max_steps),
        reached_goal=False,
        collided=False,
        final_distance=distance,
        min_separation=min_sep,
        wrong_action_cost=wrong_action_cost,
        goal_progress_bonus=goal_progress_bonus,
    )


def episode_cost(metrics: EpisodeMetrics, params: HyperParameters) -> float:
    """Convert ``metrics`` into a scalar cost value (lower is better)."""

    normalised_steps = metrics.steps / max(1.0, params.step_normaliser)

    cost = params.step_cost * metrics.steps
    cost += params.step_count_cost * normalised_steps

    if metrics.reached_goal:
        cost += params.goal_bonus
    else:
        cost += params.timeout_penalty
        normaliser = max(1.0, params.distance_normaliser)
        cost += params.distance_cost * (metrics.final_distance / normaliser)

    if metrics.collided:
        cost += params.collision_penalty

    cost += metrics.wrong_action_cost
    cost += metrics.goal_progress_bonus
    return cost


def _make_env(cfg: EnvConfig, kin: BoatParams, rudder: RudderParams) -> CrossingScenarioEnv:
    return CrossingScenarioEnv(cfg=cfg, kin=kin, rudder_cfg=rudder)


def evaluate_individual(
    genome,
    config,
    scenarios: Sequence[EncounterScenario],
    env_cfg: EnvConfig,
    boat_params: BoatParams,
    rudder_cfg: RudderParams,
    params: HyperParameters,
) -> float:
    """Return the average cost accrued by ``genome`` over all scenarios."""

    def run_single(scenario: EncounterScenario) -> EpisodeMetrics:
        local_network = neat.nn.FeedForwardNetwork.create(genome, config)
        env = _make_env(env_cfg, boat_params, rudder_cfg)
        try:
            return simulate_episode(env, scenario, local_network, params)
        finally:
            env.close()

    with ThreadPoolExecutor(max_workers=len(scenarios)) as executor:
        metrics = list(executor.map(run_single, scenarios))

    total_cost = sum(episode_cost(item, params) for item in metrics)
    return total_cost / len(metrics)


def evaluate_population(
    genomes,
    config,
    scenarios: Sequence[EncounterScenario],
    env_cfg: EnvConfig,
    boat_params: BoatParams,
    rudder_cfg: RudderParams,
    params: HyperParameters,
) -> None:
    """Assign NEAT fitness to each genome in ``genomes`` using the minimisation cost."""

    for _, genome in genomes:
        average_cost = evaluate_individual(
            genome,
            config,
            scenarios,
            env_cfg,
            boat_params,
            rudder_cfg,
            params,
        )
        genome.fitness = -average_cost


class SpeciesElitesReporter(BaseReporter):
    """Persist the top-N genomes per species after each generation."""

    def __init__(
        self,
        output_dir: Path,
        *,
        top_n: int = 3,
        config_path: Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_n = max(1, top_n)
        self._generation = 0
        self._config_path = Path(config_path) if config_path is not None else None
        self._config_copied = False

    def start_generation(self, generation: int) -> None:  # pragma: no cover - reporter hook
        self._generation = generation

    def post_evaluate(
        self,
        config,
        population,
        species_set,
        best_genome,
    ) -> None:  # pragma: no cover - reporter hook
        self._copy_config_once()

        for species_id, species in species_set.species.items():
            members = list(species.members.items())
            elite_candidates = [
                (genome_id, genome)
                for genome_id, genome in members
                if getattr(genome, "fitness", None) is not None
            ]

            if not elite_candidates:
                continue

            elite_candidates.sort(key=lambda item: item[1].fitness, reverse=True)

            species_dir = self.output_dir / f"species_{species_id:03d}"
            species_dir.mkdir(parents=True, exist_ok=True)

            for rank, (genome_id, genome) in enumerate(
                elite_candidates[: self.top_n], start=1
            ):
                metadata = {
                    "generation": self._generation,
                    "species_id": species_id,
                    "genome_id": genome_id,
                    "fitness": genome.fitness,
                }
                filename = (
                    species_dir
                    / f"gen_{self._generation:04d}_rank{rank}_gid{genome_id}.pkl"
                )
                with filename.open("wb") as fh:
                    pickle.dump({"genome": genome, "metadata": metadata}, fh)

    def _copy_config_once(self) -> None:
        if self._config_copied or self._config_path is None:
            return

        destination = self.output_dir / self._config_path.name
        if not destination.exists():
            shutil.copyfile(self._config_path, destination)
        self._config_copied = True


@dataclass
class TrainingResult:
    """Return value from :func:`train_population`."""

    winner: neat.DefaultGenome
    config: neat.Config
    statistics: neat.StatisticsReporter


def train_population(
    config_path: Path,
    scenarios: Sequence[EncounterScenario],
    env_cfg: EnvConfig,
    boat_params: BoatParams,
    rudder_cfg: RudderParams,
    params: HyperParameters,
    generations: int,
    seed: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_interval: int = 5,
    species_archive_dir: Optional[Path] = None,
    species_top_n: int = 3,
) -> TrainingResult:
    """Run NEAT evolution configured for the COLREGs crossing experiments."""

    if seed is not None:
        random.seed(seed)

    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(Path(config_path)),
    )

    population = neat.Population(neat_config)
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.StdOutReporter(True))

    if species_archive_dir is not None:
        species_reporter = SpeciesElitesReporter(
            species_archive_dir,
            top_n=species_top_n,
            config_path=config_path,
        )
        population.add_reporter(species_reporter)

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prefix = os.path.join(str(checkpoint_dir), "neat-checkpoint-")
        population.add_reporter(
            neat.Checkpointer(
                generation_interval=max(1, checkpoint_interval),
                filename_prefix=prefix,
            )
        )

    def _eval(genomes, neat_config):
        evaluate_population(
            genomes, neat_config, scenarios, env_cfg, boat_params, rudder_cfg, params
        )

    winner = population.run(_eval, generations)

    return TrainingResult(winner=winner, config=neat_config, statistics=stats)


def build_scenarios(request: ScenarioRequest) -> List[EncounterScenario]:
    """Generate the fifteen deterministic scenarios (crossing, head-on, overtaking)."""

    return list(default_scenarios(request))


def extract_training_curves(statistics: neat.StatisticsReporter) -> Dict[str, List[float]]:
    """
    Extract fitness and cost statistics from a ``neat.StatisticsReporter``.

    Returns
    -------
    Dict[str, List[float]]
        Keys include ``best_fitness``, ``mean_fitness``, ``stdev_fitness``, and the
        corresponding ``best_cost``/``mean_cost`` values (cost is simply
        ``-fitness`` because genomes maximise fitness while episodes minimise cost).
    """

    best_fitness = statistics.get_fitness_stat(max)
    mean_fitness = statistics.get_fitness_mean()
    stdev_fitness = statistics.get_fitness_stdev()

    best_cost = [-f for f in best_fitness]
    mean_cost = [-m for m in mean_fitness]

    return {
        "best_fitness": best_fitness,
        "mean_fitness": mean_fitness,
        "stdev_fitness": stdev_fitness,
        "best_cost": best_cost,
        "mean_cost": mean_cost,
    }


def plot_training_curves(
    statistics: neat.StatisticsReporter,
    *,
    output_dir: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Plot per-generation fitness and cost curves using matplotlib.

    When ``output_dir`` is provided the plots are also written as PNGs named
    ``fitness_over_generations.png`` and ``cost_over_generations.png``. Setting
    ``show=False`` is useful for headless environments while still saving files.
    """

    curves = extract_training_curves(statistics)

    best_fitness = curves["best_fitness"]
    mean_fitness = curves["mean_fitness"]
    stdev_fitness = curves["stdev_fitness"]
    best_cost = curves["best_cost"]
    mean_cost = curves["mean_cost"]

    generations = list(range(len(best_fitness)))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Fitness plot ------------------------------------------------------------
    plt.figure()
    plt.plot(generations, best_fitness, label="Best fitness")
    plt.plot(generations, mean_fitness, label="Mean fitness")
    lower_f = [m - s for m, s in zip(mean_fitness, stdev_fitness)]
    upper_f = [m + s for m, s in zip(mean_fitness, stdev_fitness)]
    plt.fill_between(generations, lower_f, upper_f, alpha=0.2, label="Mean ± std")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (higher is better)")
    plt.title("Fitness over generations")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "fitness_over_generations.png", dpi=150)

    if show:
        plt.show()
    else:
        plt.close()

    # Cost plot ---------------------------------------------------------------
    plt.figure()
    plt.plot(generations, best_cost, label="Best cost")
    plt.plot(generations, mean_cost, label="Mean cost")
    lower_c = [m - s for m, s in zip(mean_cost, stdev_fitness)]
    upper_c = [m + s for m, s in zip(mean_cost, stdev_fitness)]
    plt.fill_between(generations, lower_c, upper_c, alpha=0.2, label="Mean ± std (approx)")
    plt.xlabel("Generation")
    plt.ylabel("Cost (lower is better)")
    plt.title("Cost over generations")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "cost_over_generations.png", dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


__all__ = [
    "EpisodeMetrics",
    "SpeciesElitesReporter",
    "TrainingResult",
    "build_scenarios",
    "episode_cost",
    "evaluate_population",
    "simulate_episode",
    "train_population",
    "extract_training_curves",
    "plot_training_curves",
]

