#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import deque, namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from environment import SingleTargetFeatureEnv
from hyperparameters import EnvParams, RewardParams, TrainParams
from policy import DDQNQNet, N_ACTIONS, decode_action_idx

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buf)


class DDQNAgent:
    def __init__(self, in_dim: int, hp: TrainParams, device: torch.device):
        self.online = DDQNQNet(in_dim=in_dim, hidden_dim=hp.hidden_dim, n_actions=N_ACTIONS).to(device)
        self.target = DDQNQNet(in_dim=in_dim, hidden_dim=hp.hidden_dim, n_actions=N_ACTIONS).to(device)
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = optim.Adam(self.online.parameters(), lr=hp.learning_rate)
        self.gamma = hp.gamma
        self.device = device
        self.eps_start = hp.eps_start
        self.eps_end = hp.eps_end
        self.eps_decay_steps = hp.eps_decay_steps

        self.global_step = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.global_step / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon():
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            s = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            return int(torch.argmax(self.online(s), dim=1).item())

    def update(self, replay: ReplayBuffer, batch_size: int) -> float | None:
        if len(replay) < batch_size:
            return None

        batch = replay.sample(batch_size)
        s = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        a = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
        d = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.online(s).gather(1, a)
        with torch.no_grad():
            next_online_a = torch.argmax(self.online(ns), dim=1, keepdim=True)
            next_target_q = self.target(ns).gather(1, next_online_a)
            target = r + (1.0 - d) * self.gamma * next_target_q

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DDQN policy (discrete 9-action control)")
    p.add_argument("--episodes", type=int, default=TrainParams().episodes)
    p.add_argument("--batch-size", type=int, default=TrainParams().batch_size)
    p.add_argument("--replay-size", type=int, default=TrainParams().replay_size)
    p.add_argument("--min-replay", type=int, default=TrainParams().min_replay)
    p.add_argument("--gamma", type=float, default=TrainParams().gamma)
    p.add_argument("--learning-rate", type=float, default=TrainParams().learning_rate)
    p.add_argument("--target-update", type=int, default=TrainParams().target_update)
    p.add_argument("--eps-start", type=float, default=TrainParams().eps_start)
    p.add_argument("--eps-end", type=float, default=TrainParams().eps_end)
    p.add_argument("--eps-decay-steps", type=int, default=TrainParams().eps_decay_steps)
    p.add_argument("--hidden-dim", type=int, default=TrainParams().hidden_dim)
    p.add_argument("--seed", type=int, default=TrainParams().seed)
    p.add_argument("--save-every", type=int, default=TrainParams().save_every)
    p.add_argument("--out-dir", type=str, default=TrainParams().out_dir)
    p.add_argument("--render", action="store_true", help="render during training")
    p.add_argument("--no-render", dest="render", action="store_false")
    p.set_defaults(render=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_hp = TrainParams(
        episodes=args.episodes,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        min_replay=args.min_replay,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        target_update=args.target_update,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        save_every=args.save_every,
        out_dir=args.out_dir,
    )

    env = SingleTargetFeatureEnv(EnvParams(seed=args.seed), RewardParams(), render=args.render)
    obs_dim = int(env.reset(seed=args.seed).shape[0])

    agent = DDQNAgent(in_dim=obs_dim, hp=train_hp, device=device)
    replay = ReplayBuffer(train_hp.replay_size)

    out_dir = Path(train_hp.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    for ep in range(1, train_hp.episodes + 1):
        obs = env.reset(seed=args.seed + ep)
        done = False
        ep_return = 0.0
        ep_loss = 0.0
        loss_count = 0

        while not done:
            action_idx = agent.act(obs)
            cmd = decode_action_idx(action_idx)
            next_obs, reward, done, _ = env.step(np.asarray([cmd.rudder, cmd.throttle], dtype=np.float32))

            replay.push(obs, action_idx, reward, next_obs, done)
            obs = next_obs
            ep_return += reward
            agent.global_step += 1

            if len(replay) >= train_hp.min_replay:
                loss = agent.update(replay, train_hp.batch_size)
                if loss is not None:
                    ep_loss += loss
                    loss_count += 1

            if agent.global_step % train_hp.target_update == 0:
                agent.sync_target()

            if args.render:
                env.render()

        mean_loss = ep_loss / max(1, loss_count)
        eps_now = agent.epsilon()
        history.append(
            {
                "episode": ep,
                "return": float(ep_return),
                "steps": env.step_idx,
                "epsilon": float(eps_now),
                "mean_loss": float(mean_loss),
            }
        )
        print(
            f"ep={ep:04d} return={ep_return:8.3f} steps={env.step_idx:4d} "
            f"eps={eps_now:0.3f} loss={mean_loss:0.4f} replay={len(replay)}"
        )

        if ep % train_hp.save_every == 0 or ep == train_hp.episodes:
            torch.save(
                {
                    "online_state_dict": agent.online.state_dict(),
                    "obs_dim": obs_dim,
                    "hidden_dim": train_hp.hidden_dim,
                    "n_actions": N_ACTIONS,
                    "train_args": vars(args),
                },
                out_dir / "ddqn_policy.pt",
            )
            with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    env.close()


if __name__ == "__main__":
    main()
