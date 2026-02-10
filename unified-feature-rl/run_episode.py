#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from environment import SingleTargetFeatureEnv
from hyperparameters import EnvParams, RewardParams
from policy import policy_action, theta_dim


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run episodes with render/no-render")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--policy", type=str, default="", help=".npy policy file from training")
    p.add_argument("--hidden-dim", type=int, default=32)

    p.add_argument("--render", action="store_true", help="enable pygame visualization")
    p.add_argument("--no-render", dest="render", action="store_false", help="disable pygame visualization")
    p.set_defaults(render=True)

    p.add_argument("--show-grid", action="store_true")
    p.add_argument("--hide-grid", dest="show_grid", action="store_false")
    p.set_defaults(show_grid=True)

    p.add_argument("--show-sectors", action="store_true")
    p.add_argument("--hide-sectors", dest="show_sectors", action="store_false")
    p.set_defaults(show_sectors=True)

    p.add_argument("--episode-seconds", type=float, default=EnvParams().episode_seconds)
    p.add_argument("--world-w", type=float, default=EnvParams().world_w)
    p.add_argument("--world-h", type=float, default=EnvParams().world_h)
    p.add_argument("--sensor-range", type=float, default=EnvParams().sensor_range)
    p.add_argument("--pixels-per-meter", type=float, default=EnvParams().pixels_per_meter)
    p.add_argument("--save-log", type=str, default="", help="optional json file for episode summaries")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    envp = EnvParams(
        world_w=args.world_w,
        world_h=args.world_h,
        episode_seconds=args.episode_seconds,
        sensor_range=args.sensor_range,
        pixels_per_meter=args.pixels_per_meter,
        show_grid=args.show_grid,
        show_sectors=args.show_sectors,
        seed=args.seed,
    )
    env = SingleTargetFeatureEnv(envp, RewardParams(), render=args.render)

    theta = None
    if args.policy:
        theta = np.load(args.policy)
        expected = theta_dim(hidden_dim=args.hidden_dim)
        if theta.shape[0] != expected:
            raise ValueError(f"policy size mismatch: got {theta.shape[0]}, expected {expected}")

    summaries = []
    for ep in range(1, args.episodes + 1):
        obs = env.reset(seed=args.seed + ep)
        total = 0.0
        done = False
        info = {"reason": ""}
        while not done:
            if theta is None:
                action = np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
            else:
                action = policy_action(obs, theta, hidden_dim=args.hidden_dim)
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
            "final_target_goal_distance": float(info["target_goal_distance"]),
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
