#!/usr/bin/env python3
"""
Double-DQN trainer for Multi-Boat Sectors Env (feature-based, 100-d input)
Logs EVERY action for EVERY boat at EVERY step to results/<ts>/actions_all.csv
"""

from __future__ import annotations
import argparse, csv, math, os, random
from dataclasses import dataclass, asdict
from collections import deque, namedtuple
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from multi_boat_turn_session_env import (
    MultiBoatSectorsEnv, EnvConfig, BoatParams, TurnSessionConfig, SpawnConfig
)

# ---------------- Hyperparams ----------------
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
    eps_decay_steps: int = 300_000  # linear decay over global steps

    save_every: int = 50
    seed: int = 0
    render: bool = False

    # action logging
    log_actions: bool = True
    actions_filename: str = "actions_all.csv"
    print_action_log: bool = False

    # fleet
    n_boats: int = 2

# ---------------- Replay ----------------
Transition = namedtuple("Transition", ("state","action","reward","next_state","done"))

class Replay:
    def __init__(self, capacity: int): self.buf = deque(maxlen=capacity)
    def push(self, *args):             self.buf.append(Transition(*args))
    def sample(self, bs: int):
        batch = random.sample(self.buf, bs)
        return Transition(*zip(*batch))
    def __len__(self): return len(self.buf)

# ---------------- Network ----------------
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
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)

# ---------------- Agent (Double-DQN) ----------------
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
        eps = self.epsilon()
        if random.random() < eps:
            return random.randrange(self.n_actions), True, eps
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            qv = self.q(s)[0]
            return int(torch.argmax(qv).item()), False, eps

    def update(self, replay: Replay, batch_size: int):
        if len(replay) < max(batch_size, 1):
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

    def hard_update(self): self.t.load_state_dict(self.q.state_dict())

# ---------------- Helpers ----------------
def decode_action_idx(a: int) -> tuple[int, int]:
    """(steer, throttle) with steer∈{0 none,1 right,2 left}, throttle∈{0 coast,1 accel,2 decel}"""
    return a // 3, a % 3

# ---------------- Training Loop ----------------
def train(hp: HParams):
    random.seed(hp.seed); np.random.seed(hp.seed); torch.manual_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg   = EnvConfig(render=hp.render, seed=hp.seed)
    kin   = BoatParams()
    tcfg  = TurnSessionConfig(turn_deg=15.0, yaw_rate_degps=45.0, hysteresis_deg=2.0, allow_cancel=False)
    spawn = SpawnConfig(n_boats=hp.n_boats, start_speed=0.0)
    env   = MultiBoatSectorsEnv(cfg, kin, tcfg, spawn)

    replay = Replay(hp.replay_size)
    agent  = Agent(in_dim=100, n_actions=9, hp=hp, device=device)

    # outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", ts); os.makedirs(out_dir, exist_ok=True)

    writer = None; fhandle = None
    if hp.log_actions:
        path = os.path.join(out_dir, hp.actions_filename)
        fhandle = open(path, "w", newline="")
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
        obs_list = env.reset(seed=hp.seed + ep)  # shuffle starts per ep
        ep_ret = 0.0
        reason = ""
        steps  = 0

        for t in range(hp.steps_per_episode):
            if hp.render: env.render()

            actions = []
            was_rand = []
            eps_used = []
            for obs in obs_list:
                a, rnd, eps = agent.act(obs)
                actions.append(a); was_rand.append(rnd); eps_used.append(eps)

            next_obs_list, rewards, done, info = env.step(actions)
            reason = info.get("reason","")

            # push N transitions
            for i in range(hp.n_boats):
                agent.global_step += 1
                replay.push(obs_list[i], actions[i], rewards[i], next_obs_list[i], float(done))
                loss = agent.update(replay, hp.batch_size)
                if loss is not None:
                    ema_loss = 0.99 * ema_loss + 0.01 * loss

            ep_ret += float(sum(rewards))
            steps += 1

            # target sync
            if agent.global_step % hp.target_update == 0:
                agent.hard_update()

            # logging per boat
            if writer is not None:
                for i in range(hp.n_boats):
                    sh = env.ships[i]; gx, gy = env.goals[i]
                    dist = math.hypot(gx - sh.x, gy - sh.y)
                    s, th = decode_action_idx(actions[i])
                    writer.writerow([
                        ep, steps, i, actions[i], s, th, bool(was_rand[i]), float(eps_used[i]), float(rewards[i]),
                        float(sh.x), float(sh.y), float(sh.u), float(sh.h*180.0/math.pi),
                        float(gx), float(gy), float(dist), bool(env.reached[i]), reason
                    ])
            obs_list = next_obs_list
            if done:
                if hp.render: env.render()
                break

            if fhandle is not None and (steps % 200 == 0): fhandle.flush()

        returns.append(ep_ret)
        print(f"Ep {ep:04d} | steps {steps:4d} | return {ep_ret:8.3f} | eps {agent.epsilon():.3f} | "
              f"replay {len(replay):6d} | loss_ema {ema_loss:.5f} | reason={reason}")

        # save
        if ep % hp.save_every == 0 or ep == hp.episodes:
            torch.save(agent.q.state_dict(), os.path.join(out_dir, f"q_ep{ep}.pth"))
            np.save(os.path.join(out_dir, "returns.npy"), np.array(returns, dtype=np.float32))
            if fhandle is not None: fhandle.flush()

    if fhandle is not None: fhandle.close()
    print("Training finished. Results in:", out_dir)
    if hp.log_actions:
        print("Action log:", os.path.join(out_dir, hp.actions_filename))

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k, v in asdict(HParams()).items():
        if isinstance(v, bool):
            if v: parser.add_argument(f"--no-{k}", dest=k, action="store_false")
            else: parser.add_argument(f"--{k}", dest=k, action="store_true")
        else:
            parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    hp = HParams(**vars(args))
    train(hp)
