#!/usr/bin/env python3
"""
Train a 5-layer FC Double-DQN on TwoBoatSectorsEnv (tabular 100-d input).

This version logs EVERY action at EVERY step for EVERY episode to a single CSV:
    results/<timestamp>/actions_all.csv

CSV columns:
episode, step, boat_id, action, steer, throttle, was_random, epsilon, reward,
ship_x, ship_y, ship_speed, ship_heading_deg, goal_x, goal_y, goal_dist, reached, reason
"""

from __future__ import annotations
import argparse
import csv
import math
import os
import random
from dataclasses import dataclass, asdict
from collections import deque, namedtuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from new_environment import TwoBoatSectorsEnv, EnvConfig

# =============================
# Hyperparameters
# =============================
@dataclass
class HParams:
    episodes: int = 1500
    steps_per_episode: int = 2000
    gamma: float = 0.995

    lr: float = 2e-4
    batch_size: int = 256
    replay_size: int = 200_000
    min_replay: int = 10_000
    target_update: int = 4000

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 300_000  # linear decay over agent.global_step

    save_every: int = 50
    seed: int = 0
    render: bool = False

    # --- ACTION LOGGING (always-on per step across ALL episodes) ---
    log_actions: bool = True
    actions_filename: str = "actions_all.csv"   # one file for all episodes
    print_action_log: bool = False              # echo per-step lines to stdout

# =============================
# Replay
# =============================
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class Replay:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buf)

# =============================
# Network
# =============================
class QNet(nn.Module):
    def __init__(self, in_dim: int = 100, n_actions: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256),    nn.ReLU(inplace=True),
            nn.Linear(256, 128),    nn.ReLU(inplace=True),
            nn.Linear(128, 128),    nn.ReLU(inplace=True),
            nn.Linear(128, n_actions),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =============================
# Agent (Double DQN)
# =============================
class Agent:
    def __init__(self, in_dim: int, n_actions: int, hp: HParams, device: torch.device):
        self.q = QNet(in_dim, n_actions).to(device)
        self.t = QNet(in_dim, n_actions).to(device)
        self.t.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=hp.lr)

        self.n_actions = n_actions
        self.gamma = hp.gamma
        self.device = device

        self.eps_start = hp.eps_start
        self.eps_end   = hp.eps_end
        self.eps_decay_steps = hp.eps_decay_steps
        self.global_step = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.global_step / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state: np.ndarray) -> tuple[int, bool, float]:
        """Returns (action, was_random, epsilon_used)."""
        eps = self.epsilon()
        if random.random() < eps:
            return random.randrange(self.n_actions), True, eps
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            qv = self.q(s)[0]
            return int(torch.argmax(qv).item()), False, eps

    def update(self, replay: Replay, batch_size: int):
        if len(replay) < batch_size:
            return None
        batch = replay.sample(batch_size)
        s  = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        a  = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        r  = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
        d  = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            next_best = torch.argmax(self.q(ns), dim=1, keepdim=True)
            t_q = self.t(ns).gather(1, next_best)
            target = r + (1.0 - d) * self.gamma * t_q

        loss = F.smooth_l1_loss(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.optim.step()
        return float(loss.item())

    def hard_update(self):
        self.t.load_state_dict(self.q.state_dict())

# =============================
# Helpers
# =============================
def decode_action_idx(a: int) -> tuple[int, int]:
    """Return (steer, throttle) with steer∈{0 none,1 right,2 left}, throttle∈{0 coast,1 accel,2 decel}"""
    return a // 3, a % 3

# =============================
# Training
# =============================
def train(hp: HParams):
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = EnvConfig(
        render=hp.render,
        show_sectors=False,
        show_grid=True,
        dt=0.05,
        physics_substeps=4,
        seed=hp.seed,
        goal_ahead_distance=450.0,
        goal_radius=10.0,  # meters, must be tight now that units are real
    )
    env = TwoBoatSectorsEnv(cfg)

    replay = Replay(hp.replay_size)
    agent  = Agent(in_dim=100, n_actions=9, hp=hp, device=device)

    # outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", ts)
    os.makedirs(out_dir, exist_ok=True)

    # --- open a single CSV for ALL episodes ---
    writer = None
    fhandle = None
    if hp.log_actions:
        log_path = os.path.join(out_dir, hp.actions_filename)
        fhandle = open(log_path, "w", newline="")
        writer = csv.writer(fhandle)
        writer.writerow([
            "episode","step","boat_id",
            "action","steer","throttle","was_random","epsilon",
            "reward",
            "ship_x","ship_y","ship_speed","ship_heading_deg",
            "goal_x","goal_y","goal_dist","reached","reason"
        ])

    returns = []
    ema_loss = 0.0

    for ep in range(1, hp.episodes + 1):
        obs0, obs1 = env.reset()
        ep_ret = 0.0
        reason = ""

        for step in range(hp.steps_per_episode):
            if hp.render:
                env.render()

            # act for both ships
            a0, rand0, eps0 = agent.act(obs0)
            a1, rand1, eps1 = agent.act(obs1)

            done, info = env.step((a0, a1))
            r0, r1 = info["rewards"]
            reason = info.get("reason", "")
            reached0, reached1 = info.get("reached", (False, False))

            next0 = env.get_obs(0)
            next1 = env.get_obs(1)

            # store transitions (two per step)
            agent.global_step += 1
            replay.push(obs0, a0, r0, next0, float(done))
            agent.global_step += 1
            replay.push(obs1, a1, r1, next1, float(done))

            # updates
            loss0 = agent.update(replay, hp.batch_size)
            loss1 = agent.update(replay, hp.batch_size)
            if loss0 is not None:
                ema_loss = 0.99 * ema_loss + 0.01 * ((loss0 or 0.0) + (loss1 or 0.0))

            ep_ret += (r0 + r1)

            # target net sync
            if agent.global_step % hp.target_update == 0:
                agent.hard_update()

            # --- ACTION LOGGING ---
            if writer is not None:
                # ship 0
                sh0 = env.ships[0]; gx0, gy0 = env.goals[0]
                d0 = math.hypot(gx0 - sh0.x, gy0 - sh0.y)
                s0, t0 = decode_action_idx(a0)
                writer.writerow([
                    ep, step+1, 0, a0, s0, t0, bool(rand0), float(eps0), float(r0),
                    float(sh0.x), float(sh0.y), float(sh0.u), float(sh0.h*180.0/math.pi),
                    float(gx0), float(gy0), float(d0), bool(reached0), reason
                ])
                # ship 1
                sh1 = env.ships[1]; gx1, gy1 = env.goals[1]
                d1 = math.hypot(gx1 - sh1.x, gy1 - sh1.y)
                s1, t1 = decode_action_idx(a1)
                writer.writerow([
                    ep, step+1, 1, a1, s1, t1, bool(rand1), float(eps1), float(r1),
                    float(sh1.x), float(sh1.y), float(sh1.u), float(sh1.h*180.0/math.pi),
                    float(gx1), float(gy1), float(d1), bool(reached1), reason
                ])

            obs0, obs1 = next0, next1
            if done:
                if hp.render:
                    env.render()
                break

            # keep file fresh
            if fhandle is not None and (step % 200 == 0):
                fhandle.flush()

        returns.append(ep_ret)
        print(f"Ep {ep:04d} | steps {step+1:4d} | return {ep_ret:7.3f} | eps {agent.epsilon():.3f} | "
              f"replay {len(replay):6d} | loss_ema {ema_loss:.5f} | reason={reason}")

        if ep % hp.save_every == 0 or ep == hp.episodes:
            torch.save(agent.q.state_dict(), os.path.join(out_dir, f"q_ep{ep}.pth"))
            np.save(os.path.join(out_dir, "returns.npy"), np.array(returns, dtype=np.float32))
            if fhandle is not None:
                fhandle.flush()

    if fhandle is not None:
        fhandle.close()

    print("Training finished. Results in:", out_dir)
    if hp.log_actions:
        print("Action log:", os.path.join(out_dir, hp.actions_filename))

# =============================
# CLI
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k, v in asdict(HParams()).items():
        if isinstance(v, bool):
            if v: parser.add_argument(f"--no-{k}", dest=k, action="store_false", help=f"Disable {k}")
            else: parser.add_argument(f"--{k}", dest=k, action="store_true", help=f"Enable {k}")
        else:
            parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    hp = HParams(**vars(args))
    train(hp)
