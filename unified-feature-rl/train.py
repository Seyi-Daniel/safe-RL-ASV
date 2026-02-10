#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from environment import SingleTargetFeatureEnv
from hyperparameters import EnvParams, RewardParams, TrainParams
from policy import policy_action, theta_dim


def rollout(env: SingleTargetFeatureEnv, theta: np.ndarray, hidden_dim: int, render: bool = False) -> float:
    obs = env.reset()
    total = 0.0
    done = False
    while not done:
        action = policy_action(obs, theta, hidden_dim=hidden_dim)
        obs, r, done, _ = env.step(action)
        total += r
        if render:
            env.render()
    return float(total)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train continuous 2-output policy with CEM")
    p.add_argument("--episodes", type=int, default=TrainParams().episodes)
    p.add_argument("--population", type=int, default=TrainParams().population)
    p.add_argument("--elite-frac", type=float, default=TrainParams().elite_frac)
    p.add_argument("--eval-rollouts", type=int, default=TrainParams().eval_rollouts)
    p.add_argument("--hidden-dim", type=int, default=TrainParams().hidden_dim)
    p.add_argument("--init-std", type=float, default=TrainParams().init_std)
    p.add_argument("--min-std", type=float, default=TrainParams().min_std)
    p.add_argument("--std-decay", type=float, default=TrainParams().std_decay)
    p.add_argument("--seed", type=int, default=TrainParams().seed)
    p.add_argument("--save-every", type=int, default=TrainParams().save_every)
    p.add_argument("--out-dir", type=str, default=TrainParams().out_dir)
    p.add_argument("--render", action="store_true", help="render during training")
    p.add_argument("--no-render", dest="render", action="store_false")
    p.set_defaults(render=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)

    envp = EnvParams(seed=args.seed)
    rewp = RewardParams()
    env = SingleTargetFeatureEnv(envp, rewp, render=args.render)

    dim = theta_dim(hidden_dim=args.hidden_dim)
    mean = np.zeros(dim, dtype=np.float64)
    std = np.ones(dim, dtype=np.float64) * args.init_std

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.episodes + 1):
        population = np.random.randn(args.population, dim) * std + mean
        returns = np.zeros(args.population, dtype=np.float64)

        for i in range(args.population):
            rs = [rollout(env, population[i], hidden_dim=args.hidden_dim, render=args.render) for _ in range(args.eval_rollouts)]
            returns[i] = float(np.mean(rs))

        elite_n = max(1, int(round(args.population * args.elite_frac)))
        elite_idx = np.argsort(returns)[-elite_n:]
        elites = population[elite_idx]

        mean = elites.mean(axis=0)
        std = np.maximum(args.min_std, elites.std(axis=0) * args.std_decay)

        best_i = int(np.argmax(returns))
        print(f"iter={ep:04d} best={returns[best_i]:8.3f} mean={returns.mean():8.3f} elite_n={elite_n}")

        if ep % args.save_every == 0 or ep == args.episodes:
            np.save(out_dir / "policy_mean.npy", mean)
            with open(out_dir / "train_meta.json", "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2)

    env.close()


if __name__ == "__main__":
    main()
