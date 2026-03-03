#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo/visualize a trained continuous-control policy")
    p.add_argument("--policy", type=str, required=True, help=".pt checkpoint produced by trainings scripts")
    p.add_argument("--scenario", choices=("dcpa_sampled", "perimeter_start"), default="dcpa_sampled",
                   help="environment scenario matching the checkpoint training track")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--hidden-dim", type=int, default=256)

    p.add_argument("--render", action="store_true", help="enable pygame visualization")
    p.add_argument("--no-render", dest="render", action="store_false", help="disable pygame visualization")
    p.set_defaults(render=True)

    p.add_argument("--show-grid", action="store_true")
    p.add_argument("--hide-grid", dest="show_grid", action="store_false")
    p.set_defaults(show_grid=True)

    p.add_argument("--episode-seconds", type=float, default=120.0)
    p.add_argument("--world-w", type=float, default=500.0)
    p.add_argument("--world-h", type=float, default=500.0)
    p.add_argument("--pixels-per-meter", type=float, default=2.0)
    p.add_argument("--save-log", type=str, default="", help="optional json file for episode summaries")
    return p.parse_args()


def _build_stack(scenario: str, args: argparse.Namespace):
    if scenario == "dcpa_sampled":
        from trainings.dcpa_sampled.environment import SingleVessel2FeatureEnv
        from trainings.dcpa_sampled.hyperparameters import EnvParams, RewardParams
        from trainings.dcpa_sampled.policy import ACTION_DIM, DEFAULT_OBS_DIM, ContinuousActor
    else:
        from trainings.perimeter_start.environment import SingleVessel2FeatureEnv
        from trainings.perimeter_start.hyperparameters import EnvParams, RewardParams
        from trainings.perimeter_start.policy import ACTION_DIM, DEFAULT_OBS_DIM, ContinuousActor

    envp = EnvParams(
        world_w=args.world_w,
        world_h=args.world_h,
        episode_seconds=args.episode_seconds,
        pixels_per_meter=args.pixels_per_meter,
        show_grid=args.show_grid,
        seed=args.seed,
    )
    env = SingleVessel2FeatureEnv(envp, RewardParams(), render=args.render)
    return env, ContinuousActor, ACTION_DIM, DEFAULT_OBS_DIM


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env, ActorCls, ACTION_DIM, DEFAULT_OBS_DIM = _build_stack(args.scenario, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.policy, map_location=device)
    obs_dim = int(ckpt.get("obs_dim", DEFAULT_OBS_DIM))
    hidden_dim = int(ckpt.get("hidden_dim", args.hidden_dim))
    action_dim = int(ckpt.get("action_dim", ACTION_DIM))
    if action_dim != ACTION_DIM:
        raise ValueError(f"unsupported action count in checkpoint: {action_dim}")

    policy = ActorCls(in_dim=obs_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device)
    state = ckpt.get("actor_state_dict") or ckpt.get("online_state_dict")
    if state is None:
        raise ValueError("checkpoint missing actor/online state dict")
    policy.load_state_dict(state)
    policy.eval()

    summaries = []
    for ep in range(1, args.episodes + 1):
        obs = env.reset(seed=args.seed + ep)
        total = 0.0
        done = False
        info = {"reason": ""}
        while not done:
            if args.render and getattr(env, "paused", False):
                env.render()
                continue
            with torch.no_grad():
                s = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                action = policy(s).squeeze(0).detach().cpu().numpy().astype(np.float32)
            obs, r, done, info = env.step(action)
            total += r
            if args.render:
                env.render()

        summary = {
            "episode": ep,
            "return": float(total),
            "steps": env.step_idx,
            "reason": info["reason"],
            "final_vessel1_goal_distance": float(info["vessel1_goal_distance"]),
        }
        summaries.append(summary)
        print(summary)

    if args.save_log:
        p = Path(args.save_log)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)

    env.close()


if __name__ == "__main__":
    main()
