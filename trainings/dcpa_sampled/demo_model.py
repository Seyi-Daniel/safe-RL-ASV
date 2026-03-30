#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np
import torch

from trainings.dcpa_sampled.environment import SingleVessel2FeatureEnv
from trainings.dcpa_sampled.hyperparameters import EnvParams, RewardParams, TrainParams
from trainings.dcpa_sampled.policy import ACTION_DIM, ContinuousActor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual playback for a trained dcpa_sampled policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/<mmdd_hrmn>/policy_latest.pt",
        help="checkpoint file path (for example: runs/0330_1323/policy_100.pt)",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--cuda",
        type=int,
        default=None,
        help=(
            "CUDA device index to use (for example: --cuda 0, --cuda 1). "
            "When CUDA is available and this flag is omitted, defaults to GPU 0."
        ),
    )
    parser.add_argument("--scenario", choices=["head_on", "crossing", "overtaking", "all"], default=None)
    parser.add_argument("--num-vessels", type=int, default=2)
    parser.add_argument("--show-risk-overlay", dest="show_risk_overlay", action="store_true",
                        help="show RL takeover/risk HUD overlay during render mode")
    parser.add_argument("--hide-risk-overlay", dest="show_risk_overlay", action="store_false",
                        help="hide RL takeover/risk HUD overlay during render mode")
    parser.set_defaults(show_risk_overlay=EnvParams().show_risk_overlay)
    parser.add_argument(
        "--auto-show-risk-sector-overlay",
        dest="auto_show_risk_sector_overlay",
        action="store_true",
        help="when takeover HUD pause triggers, also auto-show radar sector rays until continue",
    )
    parser.add_argument(
        "--no-auto-show-risk-sector-overlay",
        dest="auto_show_risk_sector_overlay",
        action="store_false",
        help="do not auto-show radar sector rays at takeover HUD pause",
    )
    parser.set_defaults(auto_show_risk_sector_overlay=EnvParams().auto_show_risk_sector_overlay)
    parser.add_argument(
        "--screened-demo",
        action="store_true",
        help=(
            "enable scripted-only seed screening before playback; accepted seeds must pass "
            "DCPA/TCPA thresholds (and any optional scenario/takeover requirements)"
        ),
    )
    parser.add_argument(
        "--screening-dcpa-threshold",
        type=float,
        default=TrainParams().sampling_dcpa_threshold,
        help="scripted screening acceptance threshold: require DCPA <= this value (default: 20.0)",
    )
    parser.add_argument(
        "--screening-tcpa-threshold",
        type=float,
        default=TrainParams().sampling_tcpa_threshold,
        help="scripted screening acceptance threshold: require 0 < TCPA <= this value (default: 20.0)",
    )
    parser.add_argument(
        "--screening-max-tries",
        type=int,
        default=200,
        help="max candidate seeds tried per demo episode when --screened-demo is enabled (0 = unlimited)",
    )
    parser.add_argument(
        "--screening-max-steps",
        type=int,
        default=None,
        help="optional scripted screening-only step cap per candidate seed (default: disabled/unlimited)",
    )
    parser.add_argument(
        "--screening-max-seconds",
        type=float,
        default=None,
        help="optional scripted screening-only simulated-seconds cap per candidate seed (default: disabled/unlimited)",
    )
    parser.add_argument(
        "--require-takeover",
        action="store_true",
        help=(
            "with --screened-demo, accept only seeds whose scripted screening rollout observed "
            "at least one RL-active (takeover) step"
        ),
    )
    return parser.parse_args()


def select_runtime_device(cuda_index: int | None) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        if cuda_index is not None:
            print("CUDA was requested, but no CUDA device is available. Falling back to CPU.")
        return torch.device("cpu")

    device_count = torch.cuda.device_count()
    selected_index = 0 if cuda_index is None else int(cuda_index)
    if selected_index < 0 or selected_index >= device_count:
        raise ValueError(
            f"Invalid CUDA device index {selected_index}. "
            f"Available CUDA devices: 0 to {device_count - 1}."
        )
    return torch.device(f"cuda:{selected_index}")


def classify_initial_scenario(env: SingleVessel2FeatureEnv) -> str:
    if env.vessel1 is None or env.vessel2 is None:
        return "safe"
    scenario, _, _ = env.classify_geometry(env.vessel1, env.vessel2)
    return str(scenario)


def find_matching_seed(
    env: SingleVessel2FeatureEnv,
    target_scenario: str,
    base_seed: int | None,
    episode_index: int,
    max_tries: int = 200,
) -> int | None:
    """Find a seed whose reset geometry matches the requested scenario."""
    for attempt in range(max_tries):
        if base_seed is not None:
            candidate_seed = int(base_seed) + episode_index * 100_000 + attempt
            _ = env.reset(seed=candidate_seed)
        else:
            candidate_seed = None
            _ = env.reset()

        if classify_initial_scenario(env) == target_scenario:
            return candidate_seed
    return None


def _screen_candidate_episode(
    env: SingleVessel2FeatureEnv,
    seed: int,
    screening_dcpa_threshold: float,
    screening_tcpa_threshold: float,
    screening_max_steps: int | None,
    screening_max_seconds: float | None,
    require_takeover: bool,
) -> tuple[bool, float, float, int, str, str, bool]:
    """Mirror train.py scripted-only seed screening semantics for demo selection."""
    _ = env.reset(seed=seed)
    initial_scenario = classify_initial_scenario(env)

    done = False
    steps = 0
    best_dcpa = float("inf")
    best_tcpa = float("inf")
    fail_reason = "terminated_without_threshold"
    takeover_observed = False

    step_cap = int(screening_max_steps) if screening_max_steps is not None and int(screening_max_steps) > 0 else None
    seconds_cap = (
        float(screening_max_seconds)
        if screening_max_seconds is not None and float(screening_max_seconds) > 0.0
        else None
    )
    screen_start_time = float(env.time)

    while not done:
        if env.vessel1_reached or env.vessel2_reached:
            fail_reason = "reached_goal_before_threshold"
            break

        # Scripted/default rollout only (strictly no actor query/action injection during screening).
        _, _, done, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        steps += 1

        takeover_observed = takeover_observed or bool(env.get_rl_controlled_vessel_ids())

        dcpa = float(info.get("dcpa", float("inf")))
        tcpa = float(info.get("tcpa", float("inf")))
        best_dcpa = min(best_dcpa, dcpa)
        if tcpa > 0.0:
            best_tcpa = min(best_tcpa, tcpa)

        risk_threshold_met = (dcpa <= screening_dcpa_threshold) and (0.0 < tcpa <= screening_tcpa_threshold)
        if risk_threshold_met:
            if require_takeover and not takeover_observed:
                fail_reason = "threshold_met_without_takeover"
            else:
                return True, best_dcpa, best_tcpa, steps, "accepted", initial_scenario, takeover_observed

        reached_step_cap = step_cap is not None and steps >= step_cap
        elapsed_seconds = float(env.time) - screen_start_time
        reached_seconds_cap = seconds_cap is not None and elapsed_seconds >= seconds_cap
        if (not done) and (reached_step_cap or reached_seconds_cap):
            if reached_step_cap and reached_seconds_cap:
                fail_reason = "screen_horizon_steps_and_seconds"
            elif reached_step_cap:
                fail_reason = "screen_horizon_steps"
            else:
                fail_reason = "screen_horizon_seconds"
            break

    if done and fail_reason == "terminated_without_threshold":
        fail_reason = str(info.get("reason", "done_without_threshold"))
    if require_takeover and not takeover_observed and fail_reason == "terminated_without_threshold":
        fail_reason = "no_takeover_observed"
    return False, best_dcpa, best_tcpa, steps, fail_reason, initial_scenario, takeover_observed


def _find_screened_demo_seed(
    env: SingleVessel2FeatureEnv,
    *,
    episode_index: int,
    base_seed: int,
    target_scenario: str | None,
    screening_dcpa_threshold: float,
    screening_tcpa_threshold: float,
    screening_max_tries: int,
    screening_max_steps: int | None,
    screening_max_seconds: float | None,
    require_takeover: bool,
) -> tuple[int | None, dict[str, object]]:
    """Search candidate seeds with scripted screening + optional scenario/takeover filters."""
    max_tries = int(screening_max_tries)
    attempt = 0

    while True:
        if max_tries > 0 and attempt >= max_tries:
            return None, {}

        candidate_seed = base_seed + episode_index * 100_000 + attempt
        ok, best_dcpa, best_tcpa, screen_steps, status, scenario, takeover_observed = _screen_candidate_episode(
            env=env,
            seed=candidate_seed,
            screening_dcpa_threshold=screening_dcpa_threshold,
            screening_tcpa_threshold=screening_tcpa_threshold,
            screening_max_steps=screening_max_steps,
            screening_max_seconds=screening_max_seconds,
            require_takeover=require_takeover,
        )

        scenario_match = target_scenario is None or scenario == target_scenario
        accepted = ok and scenario_match

        if accepted:
            return candidate_seed, {
                "screened": True,
                "screen_status": status,
                "screen_best_dcpa": best_dcpa,
                "screen_best_tcpa": best_tcpa,
                "screen_steps": screen_steps,
                "screen_takeover_observed": takeover_observed,
                "screen_scenario": scenario,
                "scenario_matched": scenario_match,
                "attempt": attempt,
            }

        attempt += 1


def load_actor(checkpoint_path: Path, obs_dim: int, device: torch.device) -> ContinuousActor:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor = ContinuousActor(in_dim=obs_dim, action_dim=ACTION_DIM).to(device)

    if isinstance(checkpoint, dict) and "actor_state_dict" in checkpoint:
        state_dict = checkpoint["actor_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Fallback: assume checkpoint itself is a pure state dict.
        state_dict = checkpoint

    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def select_action(actor: ContinuousActor, obs: np.ndarray, device: torch.device) -> np.ndarray:
    """Deterministic (greedy) action from the policy network."""
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        action = actor(obs_t).squeeze(0).cpu().numpy()
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def run_demo(args: argparse.Namespace) -> None:
    env_params = EnvParams(
        num_vessels=int(args.num_vessels),
        show_risk_overlay=bool(args.show_risk_overlay),
        auto_show_risk_sector_overlay=bool(args.auto_show_risk_sector_overlay),
    )
    reward_params = RewardParams()
    env = SingleVessel2FeatureEnv(env_params=env_params, reward_params=reward_params, render=True)

    try:
        device = select_runtime_device(args.cuda)
    except ValueError as exc:
        raise SystemExit(f"Device selection error: {exc}") from exc

    if device.type == "cuda":
        print(f"Using device: {device} (CUDA devices available: {torch.cuda.device_count()})")
    else:
        print("Using device: cpu")

    try:
        initial_obs = env.reset(seed=args.seed)
        actor = load_actor(Path(args.checkpoint), obs_dim=int(initial_obs.shape[0]), device=device)

        for ep in range(max(1, int(args.episodes))):
            matched_seed = args.seed
            target_scenario = None if args.scenario in {None, "all"} else args.scenario
            episode_selection: dict[str, object] = {
                "screened": False,
                "scenario_matched": target_scenario is None,
                "screen_takeover_observed": False,
            }

            if args.screened_demo:
                if args.seed is None:
                    raise SystemExit("--screened-demo requires --seed so candidate seeds are reproducible.")
                matched_seed, episode_selection = _find_screened_demo_seed(
                    env=env,
                    episode_index=ep,
                    base_seed=int(args.seed),
                    target_scenario=target_scenario,
                    screening_dcpa_threshold=float(args.screening_dcpa_threshold),
                    screening_tcpa_threshold=float(args.screening_tcpa_threshold),
                    screening_max_tries=int(args.screening_max_tries),
                    screening_max_steps=args.screening_max_steps,
                    screening_max_seconds=args.screening_max_seconds,
                    require_takeover=bool(args.require_takeover),
                )
                if matched_seed is None:
                    print(
                        f"Episode {ep + 1}: skipped (no screened candidate accepted within "
                        f"{int(args.screening_max_tries)} tries; scenario={target_scenario or 'any'}, "
                        f"require_takeover={bool(args.require_takeover)})."
                    )
                    continue
            elif target_scenario is not None:
                matched_seed = find_matching_seed(
                    env=env,
                    target_scenario=target_scenario,
                    base_seed=args.seed,
                    episode_index=ep,
                )
                if matched_seed is None:
                    print(
                        f"Episode {ep + 1}: skipped (no '{target_scenario}' scenario found within search limit)."
                    )
                    continue
                episode_selection["scenario_matched"] = True

            _ = env.reset(seed=matched_seed)
            done = False
            total_reward = 0.0
            steps = 0
            rl_active_step_count = 0
            takeover_observed_in_playback = False
            last_info: dict[str, object] = {}

            while not done:
                if getattr(env, "paused", False):
                    env.render()
                    continue

                rl_vessel_ids = env.get_rl_controlled_vessel_ids()
                action_dict: dict[str, np.ndarray] = {}

                # Strict takeover discipline: actor is not queried until takeover is active.
                if rl_vessel_ids:
                    takeover_observed_in_playback = True
                    rl_active_step_count += 1
                    for vessel_id in rl_vessel_ids:
                        vessel_obs = env.get_obs_for_vessel(vessel_id)
                        action_dict[vessel_id] = select_action(actor, vessel_obs, device)

                _, reward, done, info = env.step(action_dict)
                env.render()

                total_reward += float(reward)
                steps += 1
                last_info = info

            collision = bool(int(last_info.get("collision", 0)))
            reason = str(last_info.get("reason", ""))
            success = reason == "both_reached"
            scenario_name = str(last_info.get("colregs_scenario", "unknown"))

            print(
                f"Episode {ep + 1}/{args.episodes} | seed={matched_seed} | scenario={scenario_name} | "
                f"screened={bool(episode_selection.get('screened', False))} | "
                f"scenario_matched={bool(episode_selection.get('scenario_matched', False))} | "
                f"screen_takeover_observed={bool(episode_selection.get('screen_takeover_observed', False))} | "
                f"playback_takeover_observed={takeover_observed_in_playback} | "
                f"playback_rl_active_steps={rl_active_step_count} | total_reward={total_reward:.3f} | "
                f"steps={steps} | collision={collision} | success={success} | reason={reason}"
            )

            if bool(episode_selection.get("screened", False)):
                print(
                    f"  screening_details: attempt={int(episode_selection.get('attempt', -1))} "
                    f"status={episode_selection.get('screen_status', 'unknown')} "
                    f"screen_steps={int(episode_selection.get('screen_steps', 0))} "
                    f"best_dcpa={float(episode_selection.get('screen_best_dcpa', float('inf'))):.2f} "
                    f"best_tcpa={float(episode_selection.get('screen_best_tcpa', float('inf'))):.2f}"
                )

    finally:
        env.close()


def main() -> None:
    args = parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
