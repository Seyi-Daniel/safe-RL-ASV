#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from environment import SingleTargetFeatureEnv
from hyperparameters import EnvParams, RewardParams
from policy import DDQNQNet, N_ACTIONS, decode_action_idx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run episodes with DDQN policy")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--policy", type=str, default="", help=".pt DDQN checkpoint from training")
    p.add_argument("--hidden-dim", type=int, default=256)

    p.add_argument("--render", action="store_true", help="enable pygame visualization")
    p.add_argument("--no-render", dest="render", action="store_false", help="disable pygame visualization")
    p.set_defaults(render=True)

    p.add_argument("--show-grid", action="store_true")
    p.add_argument("--hide-grid", dest="show_grid", action="store_false")
    p.set_defaults(show_grid=True)

    p.add_argument("--episode-seconds", type=float, default=EnvParams().episode_seconds)
    p.add_argument("--world-w", type=float, default=EnvParams().world_w)
    p.add_argument("--world-h", type=float, default=EnvParams().world_h)
    p.add_argument("--pixels-per-meter", type=float, default=EnvParams().pixels_per_meter)
    p.add_argument("--save-log", type=str, default="", help="optional json file for episode summaries")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    envp = EnvParams(
        world_w=args.world_w,
        world_h=args.world_h,
        episode_seconds=args.episode_seconds,
        pixels_per_meter=args.pixels_per_meter,
        show_grid=args.show_grid,
        seed=args.seed,
    )
    env = SingleTargetFeatureEnv(envp, RewardParams(), render=args.render)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = None

    if args.policy:
        ckpt = torch.load(args.policy, map_location=device)
        obs_dim = int(ckpt.get("obs_dim", 6))
        hidden_dim = int(ckpt.get("hidden_dim", args.hidden_dim))
        n_actions = int(ckpt.get("n_actions", N_ACTIONS))
        if n_actions != N_ACTIONS:
            raise ValueError(f"unsupported action count in checkpoint: {n_actions}")

        policy = DDQNQNet(in_dim=obs_dim, hidden_dim=hidden_dim, n_actions=n_actions).to(device)
        policy.load_state_dict(ckpt["online_state_dict"])
        policy.eval()

    summaries = []
    for ep in range(1, args.episodes + 1):
        obs = env.reset(seed=args.seed + ep)
        total = 0.0
        done = False
        info = {"reason": ""}
        while not done:
            if policy is None:
                a_idx = np.random.randint(0, N_ACTIONS)
            else:
                with torch.no_grad():
                    s = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    a_idx = int(torch.argmax(policy(s), dim=1).item())

            cmd = decode_action_idx(a_idx)
            action = np.asarray([cmd.rudder, cmd.throttle], dtype=np.float32)
            obs, r, done, info = env.step(action)
            total += r
            if args.render:
                env.render()

        summary = {
            "episode": ep,
            "return": float(total),
            "steps": env.step_idx,
            "reason": info["reason"],
            "final_agent_goal_distance": float(info["agent_goal_distance"]),
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
