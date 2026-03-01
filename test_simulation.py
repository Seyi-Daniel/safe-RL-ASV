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
        "--target-max-goal-arc-distance",
        "--target-max-goal-distance",
        dest="target_max_goal_arc_distance",
        type=float,
        default=EnvParams().target_max_goal_arc_distance_from_start,
        help="Maximum allowed vessel-2 start->goal arc distance along the big circle",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    envp = EnvParams(
        episode_seconds=args.episode_seconds,
        seed=args.seed,
        target_max_goal_arc_distance_from_start=args.target_max_goal_arc_distance,
    )
    env = SingleTargetFeatureEnv(envp, RewardParams(), render=args.render)

    try:
        for ep in range(1, args.episodes + 1):
            obs = env.reset(seed=args.seed + ep)
            _ = obs
            v2_start = env.target_start_pos
            v2_goal = (env.target.goal_x, env.target.goal_y)
            start_angle = float(np.arctan2(v2_start[1] - env.start_y, v2_start[0] - env.start_x))
            goal_angle = float(np.arctan2(v2_goal[1] - env.start_y, v2_goal[0] - env.start_x))
            v2_goal_arc_dist = float(env.envp.target_outer_radius * env._arc_gap(start_angle, goal_angle))
            print(
                f"Episode {ep}: vessel2 start={v2_start}, goal={v2_goal}, "
                f"arc_distance={v2_goal_arc_dist:.2f} m (max={args.target_max_goal_arc_distance:.2f} m)"
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
