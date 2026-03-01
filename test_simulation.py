#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

import numpy as np

from environment import SingleTargetFeatureEnv
from hyperparameters import EnvParams, RewardParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lightweight simulation sandbox for manual behavior checks")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--render", action="store_true", help="enable pygame visualization")
    p.add_argument("--episode-seconds", type=float, default=60.0)
    p.add_argument(
        "--target-max-goal-distance",
        type=float,
        default=EnvParams().target_max_goal_distance_from_start,
        help="Maximum allowed vessel-2 start->goal distance",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    envp = EnvParams(
        episode_seconds=args.episode_seconds,
        seed=args.seed,
        target_max_goal_distance_from_start=args.target_max_goal_distance,
    )
    env = SingleTargetFeatureEnv(envp, RewardParams(), render=args.render)

    try:
        for ep in range(1, args.episodes + 1):
            obs = env.reset(seed=args.seed + ep)
            _ = obs
            v2_start = env.target_start_pos
            v2_goal = (env.target.goal_x, env.target.goal_y)
            v2_goal_dist = float(np.hypot(v2_goal[0] - v2_start[0], v2_goal[1] - v2_start[1]))
            print(
                f"Episode {ep}: vessel2 start={v2_start}, goal={v2_goal}, "
                f"distance={v2_goal_dist:.2f} m (max={args.target_max_goal_distance:.2f} m)"
            )

            done = False
            total = 0.0
            while not done:
                # Keep vessel-1 neutral so vessel-2 pure pursuit behavior is easy to observe.
                action = np.array([0.0, 0.0], dtype=np.float32)
                _, reward, done, info = env.step(action)
                total += reward
                if args.render:
                    env.render()

            print(
                f"  end: reason={info.get('reason', 'unknown')} steps={env.step_idx} "
                f"return={float(total):.3f} target_goal_distance={float(info.get('target_goal_distance', -1.0)):.3f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
