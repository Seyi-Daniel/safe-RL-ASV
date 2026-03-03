from __future__ import annotations

import argparse

import numpy as np

from simulations.environment_perimeter_start import HAS_PYGAME, SingleVessel2FeatureEnv
from simulations.hyperparameters_perimeter_start import EnvParams, RewardParams

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationRuntimeConfig:
    """Simulation-only runtime config, intentionally isolated from train.py CLI wiring."""

    episode_seconds: float
    seed: int
    dcrit_factor: float
    step_risk_logs: bool


def build_sampling_env_params(cfg: SimulationRuntimeConfig) -> EnvParams:
    return EnvParams(
        episode_seconds=cfg.episode_seconds,
        seed=cfg.seed,
        vessel2_min_goal_arc_distance_from_start=0.0,
        adaptive_vessel2_min_goal_arc_from_speed=True,
        vessel2_min_goal_dcrit_factor=cfg.dcrit_factor,
        require_reset_viable_takeover_path=False,
        enable_no_takeover_early_done=False,
        enable_step_risk_logging=False,
    )


def build_playback_env_params(cfg: SimulationRuntimeConfig) -> EnvParams:
    return EnvParams(
        episode_seconds=cfg.episode_seconds,
        seed=cfg.seed,
        vessel2_min_goal_arc_distance_from_start=0.0,
        adaptive_vessel2_min_goal_arc_from_speed=True,
        vessel2_min_goal_dcrit_factor=cfg.dcrit_factor,
        require_reset_viable_takeover_path=False,
        enable_no_takeover_early_done=False,
        enable_step_risk_logging=cfg.step_risk_logs,
    )


def create_sim_env(envp: EnvParams, render: bool) -> SingleVessel2FeatureEnv:
    return SingleVessel2FeatureEnv(envp, RewardParams(), render=render)


if HAS_PYGAME:
    import pygame


def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--render", action="store_true", help="enable pygame visualization")
    p.add_argument("--episode-seconds", type=float, default=60.0)

    p.add_argument("--dcrit-factor", type=float, default=1.1, help="Multiplier on dcrit for minimum goal arc distance")

    p.add_argument("--dcpa-threshold", type=float, default=20.0, help="Accept sampled episodes only if DCPA reaches this threshold or lower")
    p.add_argument("--tcpa-threshold", type=float, default=90.0, help="Accept sampled episodes only if TCPA is within [0, threshold] when DCPA condition is met")
    p.add_argument("--dcpa-sample-max-tries", type=int, default=0, help="Max resampling attempts per accepted episode (0 = unlimited)")
    p.add_argument("--debug-sampling", action="store_true", help="Enable detailed DCPA/TCPA sampling debug logs")
    p.add_argument("--sampling-logs", dest="sampling_logs", action="store_true", help="Print per-attempt sampling summaries")
    p.add_argument("--no-sampling-logs", dest="sampling_logs", action="store_false", help="Silence per-attempt sampling summaries")
    p.set_defaults(sampling_logs=True)
    p.add_argument("--step-risk-logs", dest="step_risk_logs", action="store_true", help="Enable [RISK TRACE] logging for every step")
    p.add_argument("--no-step-risk-logs", dest="step_risk_logs", action="store_false", help="Disable [RISK TRACE] per-step logging")
    p.set_defaults(step_risk_logs=False)
    p.add_argument("--debug-sampling-step-log-every", type=int, default=100, help="When debug is on, print per-attempt status every N steps")
    p.add_argument("--max-sampling-steps-per-attempt", type=int, default=0, help="Safety cap for candidate sampling steps (0 = automatic cap of 2x episode max steps)")


def _episode_hits_dcpa_threshold(
    env: SingleVessel2FeatureEnv,
    seed: int,
    dcpa_threshold: float,
    tcpa_threshold: float,
    debug_sampling: bool,
    debug_step_log_every: int,
    max_sampling_steps_per_attempt: int,
) -> tuple[bool, float, float, int, dict[str, float | str | int], str]:
    _ = env.reset(seed=seed)
    effective_step_cap = int(max_sampling_steps_per_attempt) if int(max_sampling_steps_per_attempt) > 0 else max(1, 2 * int(env.max_steps))
    if debug_sampling:
        print(
            f"[sample-debug] start seed={seed} max_steps={env.max_steps} "
            f"episode_seconds={env.envp.episode_seconds} effective_step_cap={effective_step_cap}"
        )

    best_dcpa = float("inf")
    best_tcpa = float("inf")
    done = False
    steps = 0
    fail_reason = "terminated_without_threshold"
    final_info: dict[str, float | str | int] = {"reason": "unknown"}

    while not done:
        if env.vessel1_reached or env.vessel2_reached:
            fail_reason = "reached_goal_before_threshold"
            break
        _, _, done, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        steps += 1
        final_info = info

        dcpa = float(info.get("dcpa", float("inf")))
        tcpa = float(info.get("tcpa", float("inf")))
        best_dcpa = min(best_dcpa, dcpa)
        if tcpa > 0.0:
            best_tcpa = min(best_tcpa, tcpa)

        if debug_sampling and (steps % max(1, int(debug_step_log_every)) == 0):
            print(
                f"[sample-debug] seed={seed} step={steps} dcpa={dcpa:.2f} tcpa={tcpa:.2f} "
                f"best_dcpa={best_dcpa:.2f} best_tcpa={best_tcpa:.2f} "
                f"vessel1_reached={int(env.vessel1_reached)} vessel2_reached={int(env.vessel2_reached)} done={int(done)}"
            )

        if steps >= effective_step_cap:
            fail_reason = "max_sampling_steps_per_attempt_guard"
            break

        if (dcpa <= dcpa_threshold) and (0.0 < tcpa <= tcpa_threshold):
            if debug_sampling:
                print(
                    f"[sample-debug] seed={seed} accepted at step={steps} "
                    f"dcpa={dcpa:.2f} tcpa={tcpa:.2f}"
                )
            return True, best_dcpa, best_tcpa, steps, final_info, "accepted"

    if done and fail_reason == "terminated_without_threshold":
        fail_reason = str(final_info.get("reason", "done_without_threshold"))
    if debug_sampling:
        print(
            f"[sample-debug] seed={seed} fail_reason={fail_reason} steps={steps} "
            f"best_dcpa={best_dcpa:.2f} best_tcpa={best_tcpa:.2f}"
        )
    return False, best_dcpa, best_tcpa, steps, final_info, fail_reason


def run(args: argparse.Namespace) -> None:
    cfg = SimulationRuntimeConfig(
        episode_seconds=args.episode_seconds,
        seed=args.seed,
        dcrit_factor=args.dcrit_factor,
        step_risk_logs=args.step_risk_logs,
    )

    sample_env = create_sim_env(build_sampling_env_params(cfg), render=False)
    playback_env = create_sim_env(build_playback_env_params(cfg), render=True) if args.render else None

    try:
        for ep in range(1, args.episodes + 1):
            accepted_seed = None
            accepted_best_dcpa = float("inf")
            accepted_best_tcpa = float("inf")
            accepted_attempt = -1
            accepted_sample_steps = 0

            max_tries = int(args.dcpa_sample_max_tries)
            attempt = 0
            while True:
                if max_tries > 0 and attempt >= max_tries:
                    break
                candidate_seed = args.seed + ep * 100_000 + attempt
                ok, best_dcpa, best_tcpa, sample_steps, _, fail_reason = _episode_hits_dcpa_threshold(
                    sample_env,
                    candidate_seed,
                    args.dcpa_threshold,
                    args.tcpa_threshold,
                    args.debug_sampling,
                    args.debug_sampling_step_log_every,
                    args.max_sampling_steps_per_attempt,
                )
                if ok:
                    accepted_seed = candidate_seed
                    accepted_best_dcpa = best_dcpa
                    accepted_best_tcpa = best_tcpa
                    accepted_attempt = attempt
                    accepted_sample_steps = sample_steps
                    break

                if args.sampling_logs:
                    print(
                        f"Episode {ep}: failed attempt={attempt} seed={candidate_seed} steps={sample_steps} reason={fail_reason} "
                        f"(best_dcpa={best_dcpa:.2f}, best_tcpa={best_tcpa:.2f}; "
                        f"need dcpa <= {args.dcpa_threshold:.2f} and tcpa <= {args.tcpa_threshold:.2f})"
                    )
                attempt += 1

            if accepted_seed is None:
                print(
                    f"Episode {ep}: no sample found with dcpa <= {args.dcpa_threshold:.2f} and tcpa <= {args.tcpa_threshold:.2f} "
                    f"within {max_tries} tries"
                )
                continue

            run_env = playback_env if playback_env is not None else sample_env
            run_env.envp.enable_step_risk_logging = bool(args.step_risk_logs)
            _ = run_env.reset(seed=accepted_seed)
            done = False
            total = 0.0
            run_best_dcpa = float("inf")
            run_best_tcpa = float("inf")
            threshold_hit_before_goal = False
            info: dict[str, float | str | int] = {"reason": "unknown", "dcpa": float("inf"), "tcpa": float("inf")}
            while not done:
                reached_before_step = run_env.vessel1_reached or run_env.vessel2_reached
                _, reward, done, info = run_env.step(np.array([0.0, 0.0], dtype=np.float32))
                total += reward
                dcpa = float(info.get("dcpa", float("inf")))
                tcpa = float(info.get("tcpa", float("inf")))
                run_best_dcpa = min(run_best_dcpa, dcpa)
                if tcpa > 0.0:
                    run_best_tcpa = min(run_best_tcpa, tcpa)
                if (not reached_before_step) and (dcpa <= args.dcpa_threshold) and (0.0 < tcpa <= args.tcpa_threshold):
                    threshold_hit_before_goal = True
                if args.render and playback_env is not None:
                    run_env.render()
                    if HAS_PYGAME:
                        pygame.event.pump()

            if not threshold_hit_before_goal:
                print(
                    f"Episode {ep}: rejected post-check for seed={accepted_seed} "
                    f"(best_dcpa={run_best_dcpa:.2f}, best_tcpa={run_best_tcpa:.2f}; "
                    f"thresholds dcpa<={args.dcpa_threshold:.2f}, tcpa<={args.tcpa_threshold:.2f})"
                )
                continue

            print(
                f"Episode {ep}: accepted_seed={accepted_seed} attempt={accepted_attempt} sample_steps={accepted_sample_steps} "
                f"sample_best_dcpa={accepted_best_dcpa:.2f} sample_best_tcpa={accepted_best_tcpa:.2f} "
                f"run_best_dcpa={run_best_dcpa:.2f} run_best_tcpa={run_best_tcpa:.2f} "
                f"(thresholds: dcpa<={args.dcpa_threshold:.2f}, tcpa<={args.tcpa_threshold:.2f})"
            )
            print(
                f"  end: reason={info.get('reason', 'unknown')} steps={run_env.step_idx} return={float(total):.3f} "
                f"final_dcpa={float(info.get('dcpa', float('inf'))):.2f}"
            )
    finally:
        sample_env.close()
        if playback_env is not None:
            playback_env.close()
