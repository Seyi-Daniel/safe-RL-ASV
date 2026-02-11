#!/usr/bin/env python3
"""Minimal DQN trainer for ColregsFeatureEnv (10-feature state)."""

from __future__ import annotations

import argparse
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from colregs_random_feature_env import ColregsFeatureEnv

Transition = namedtuple("Transition", ["s", "a", "r", "ns", "d"])


class QNet(nn.Module):
    def __init__(self, in_dim: int = 10, n_actions: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = ColregsFeatureEnv(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = QNet().to(device)
    tq = QNet().to(device)
    tq.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=args.lr)
    mem: deque[Transition] = deque(maxlen=args.replay_size)

    eps = args.eps_start
    total_steps = 0

    for ep in range(1, args.episodes + 1):
        s = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            total_steps += 1
            if random.random() < eps:
                a = random.randint(0, 8)
            else:
                with torch.no_grad():
                    a = int(q(torch.from_numpy(s).float().unsqueeze(0).to(device)).argmax(dim=1).item())

            ns, r, done, info = env.step(a)
            ep_ret += r
            mem.append(Transition(s, a, r, ns, float(done)))
            s = ns

            if len(mem) >= args.batch:
                batch = random.sample(mem, args.batch)
                bs = torch.tensor(np.array([t.s for t in batch]), dtype=torch.float32, device=device)
                ba = torch.tensor([t.a for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
                br = torch.tensor([t.r for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
                bns = torch.tensor(np.array([t.ns for t in batch]), dtype=torch.float32, device=device)
                bd = torch.tensor([t.d for t in batch], dtype=torch.float32, device=device).unsqueeze(1)

                qsa = q(bs).gather(1, ba)
                with torch.no_grad():
                    na = q(bns).argmax(dim=1, keepdim=True)
                    nq = tq(bns).gather(1, na)
                    y = br + args.gamma * nq * (1.0 - bd)

                loss = nn.functional.mse_loss(qsa, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if total_steps % args.target_update == 0:
                    tq.load_state_dict(q.state_dict())

        eps = max(args.eps_end, eps * args.eps_decay)
        print(f"ep={ep:04d} return={ep_ret:8.3f} steps={env.step_count:4d} eps={eps:.3f} reason={info['reason']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--replay-size", type=int, default=50_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target-update", type=int, default=200)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=float, default=0.995)
    p.add_argument("--seed", type=int, default=7)
    train(p.parse_args())
