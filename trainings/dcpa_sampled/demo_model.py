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
from trainings.dcpa_sampled.hyperparameters import EnvParams, RewardParams
from trainings.dcpa_sampled.policy import ACTION_DIM, ContinuousActor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual playback for a trained dcpa_sampled policy")
    parser.add_argument("--checkpoint", type=str, default="runs/dcpa_sampled/ddqn_policy.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenario", choices=["head_on", "crossing", "overtaking", "all"], default=None)
    parser.add_argument("--num-vessels", type=int, default=2)
    return parser.parse_args()


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
    env_params = EnvParams(num_vessels=int(args.num_vessels))
    reward_params = RewardParams()
    env = SingleVessel2FeatureEnv(env_params=env_params, reward_params=reward_params, render=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        initial_obs = env.reset(seed=args.seed)
        actor = load_actor(Path(args.checkpoint), obs_dim=int(initial_obs.shape[0]), device=device)

        for ep in range(max(1, int(args.episodes))):
            matched_seed = args.seed
            if args.scenario and args.scenario != "all":
                matched_seed = find_matching_seed(
                    env=env,
                    target_scenario=args.scenario,
                    base_seed=args.seed,
                    episode_index=ep,
                )
                if matched_seed is None:
                    print(
                        f"Episode {ep + 1}: skipped (no '{args.scenario}' scenario found within search limit)."
                    )
                    continue

            _ = env.reset(seed=matched_seed)
            done = False
            total_reward = 0.0
            steps = 0
            last_info: dict[str, object] = {}

            while not done:
                if getattr(env, "paused", False):
                    env.render()
                    continue

                rl_vessel_ids = env.get_rl_controlled_vessel_ids()
                action_dict: dict[str, np.ndarray] = {}
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
                f"Episode {ep + 1}/{args.episodes} | seed={matched_seed} | "
                f"scenario={scenario_name} | total_reward={total_reward:.3f} | "
                f"steps={steps} | collision={collision} | success={success} | reason={reason}"
            )

    finally:
        env.close()


def main() -> None:
    args = parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
