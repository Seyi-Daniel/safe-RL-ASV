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
from trainings.dcpa_sampled.reward_ports.crossing_objective_adapter import CrossingObjectiveAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo playback for the crossing-port parallel policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs_crossing_port/<mmdd_HHMM>/policy_latest.pt",
        help="checkpoint file path for crossing-port run",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=TrainParams().seed)
    parser.add_argument("--cuda", type=int, default=None)
    parser.add_argument("--scenario", choices=["head_on", "crossing", "overtaking", "all"], default=None)
    parser.add_argument("--num-vessels", type=int, default=2)
    parser.add_argument("--episode-seconds", type=float, default=500.0)
    parser.add_argument("--show-risk-overlay", dest="show_risk_overlay", action="store_true")
    parser.add_argument("--hide-risk-overlay", dest="show_risk_overlay", action="store_false")
    parser.set_defaults(show_risk_overlay=EnvParams().show_risk_overlay)
    parser.add_argument("--auto-show-risk-sector-overlay", dest="auto_show_risk_sector_overlay", action="store_true")
    parser.add_argument("--no-auto-show-risk-sector-overlay", dest="auto_show_risk_sector_overlay", action="store_false")
    parser.set_defaults(auto_show_risk_sector_overlay=EnvParams().auto_show_risk_sector_overlay)
    return parser.parse_args()


def _select_runtime_device(cuda_index: int | None) -> torch.device:
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


def _classify_two_vessel_scenario(env: SingleVessel2FeatureEnv) -> str:
    if env.vessel1 is None or env.vessel2 is None:
        return "safe"
    scenario, _, _ = env.classify_geometry(env.vessel1, env.vessel2)
    return str(scenario)


def _find_matching_seed(
    env: SingleVessel2FeatureEnv,
    target_scenario: str,
    base_seed: int,
    episode_index: int,
    max_tries: int = 200,
) -> int | None:
    for attempt in range(max_tries):
        candidate_seed = base_seed + episode_index * 100_000 + attempt
        _ = env.reset(seed=candidate_seed)
        if _classify_two_vessel_scenario(env) == target_scenario:
            return candidate_seed
    return None


def main() -> None:
    args = parse_args()
    try:
        device = _select_runtime_device(args.cuda)
    except ValueError as exc:
        raise SystemExit(f"Device selection error: {exc}") from exc

    envp = EnvParams(
        seed=args.seed,
        num_vessels=max(2, int(args.num_vessels)),
        episode_seconds=float(args.episode_seconds),
        show_risk_overlay=bool(args.show_risk_overlay),
        auto_show_risk_sector_overlay=bool(args.auto_show_risk_sector_overlay),
    )
    env = SingleVessel2FeatureEnv(envp, RewardParams(), render=True)
    reward_adapter = CrossingObjectiveAdapter()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Use --checkpoint runs_crossing_port/<mmdd_HHMM>/policy_latest.pt"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    obs_dim = int(checkpoint.get("obs_dim", env.reset(seed=args.seed).shape[0]))
    hidden_dims = checkpoint.get("hidden_dims", [512, 256, 128])
    actor = ContinuousActor(
        in_dim=obs_dim,
        hidden_dim_1=int(hidden_dims[0]),
        hidden_dim_2=int(hidden_dims[1]),
        hidden_dim_3=int(hidden_dims[2]),
        action_dim=ACTION_DIM,
    ).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    for ep in range(args.episodes):
        if args.scenario and args.scenario != "all":
            matched_seed = _find_matching_seed(env, args.scenario, int(args.seed), ep)
            if matched_seed is None:
                print(f"ep={ep + 1:03d} skipped: no seed matched scenario={args.scenario}")
                continue
            _ = env.reset(seed=matched_seed)
        else:
            _ = env.reset(seed=int(args.seed) + ep)

        reward_adapter.reset()
        done = False
        env_return = 0.0
        port_return = 0.0
        rewarded_vessels: dict[str, int] = {"vessel1": 0, "vessel2": 0, "none": 0}
        last_info: dict[str, object] = {}

        while not done:
            action_by_vessel: dict[str, np.ndarray] = {}
            for vessel_id in env.get_rl_controlled_vessel_ids():
                obs = env.get_obs_for_vessel(vessel_id)
                with torch.no_grad():
                    state_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    action = actor(state_t).squeeze(0).cpu().numpy()
                action_by_vessel[vessel_id] = np.clip(action, -1.0, 1.0).astype(np.float32)

            step_action = action_by_vessel if action_by_vessel else np.array([0.0, 0.0], dtype=np.float32)
            _, env_reward, done, info = env.step(step_action)
            adapted = reward_adapter.compute_reward(info, action_by_vessel=action_by_vessel, env=env)

            env_return += float(env_reward)
            port_return += float(adapted.reward)
            key = adapted.vessel_id if adapted.vessel_id in {"vessel1", "vessel2"} else "none"
            rewarded_vessels[key] += 1
            last_info = info
            env.render()

        print(
            f"ep={ep + 1:03d} env_total_reward={env_return:.3f} port_total_reward={port_return:.3f} "
            f"steps={int(env.step_idx)} collision={int(last_info.get('collision', 0))} "
            f"success={int(bool(last_info.get('vessel1_reached', 0)) and bool(last_info.get('vessel2_reached', 0)))} "
            f"reason={last_info.get('reason', '')} scenario={last_info.get('colregs_scenario', 'unknown')} "
            f"port_reward_vessels={rewarded_vessels}"
        )

    env.close()


if __name__ == "__main__":
    main()
