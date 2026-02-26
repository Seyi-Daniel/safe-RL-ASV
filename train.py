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
from policy import ACTION_DIM, ContinuousActor, ContinuousCritic

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


class DDPGAgent:
    def __init__(self, in_dim: int, hp: TrainParams, device: torch.device):
        self.actor = ContinuousActor(in_dim=in_dim, hidden_dim=hp.hidden_dim, action_dim=ACTION_DIM).to(device)
        self.actor_tgt = ContinuousActor(in_dim=in_dim, hidden_dim=hp.hidden_dim, action_dim=ACTION_DIM).to(device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())

        self.critic = ContinuousCritic(in_dim=in_dim, hidden_dim=hp.hidden_dim, action_dim=ACTION_DIM).to(device)
        self.critic_tgt = ContinuousCritic(in_dim=in_dim, hidden_dim=hp.hidden_dim, action_dim=ACTION_DIM).to(device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=hp.learning_rate)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=hp.learning_rate)

        self.gamma = hp.gamma
        self.device = device
        self.eps_start = hp.eps_start
        self.eps_end = hp.eps_end
        self.eps_decay_steps = hp.eps_decay_steps
        self.tau = 0.005
        self.global_step = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.global_step / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs: np.ndarray, greedy: bool = False) -> np.ndarray:
        with torch.no_grad():
            s = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            a = self.actor(s).squeeze(0).cpu().numpy()
        if not greedy:
            noise_scale = self.epsilon()
            a = a + noise_scale * np.random.normal(0.0, 0.25, size=(ACTION_DIM,)).astype(np.float32)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def _soft_update(self, src: torch.nn.Module, tgt: torch.nn.Module) -> None:
        with torch.no_grad():
            for p, tp in zip(src.parameters(), tgt.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * p.data)

    def update(self, replay: ReplayBuffer, batch_size: int) -> tuple[float, float] | None:
        if len(replay) < batch_size:
            return None

        batch = replay.sample(batch_size)
        s = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        a = torch.from_numpy(np.stack(batch.action)).float().to(self.device)
        r = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
        d = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_a = self.actor_tgt(ns)
            target_q = self.critic_tgt(ns, next_a)
            y = r + (1.0 - d) * self.gamma * target_q

        q = self.critic(s, a)
        critic_loss = F.smooth_l1_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_opt.step()

        pred_a = self.actor(s)
        actor_loss = -self.critic(s, pred_a).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()

        self._soft_update(self.actor, self.actor_tgt)
        self._soft_update(self.critic, self.critic_tgt)

        return float(actor_loss.item()), float(critic_loss.item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train continuous-control policy (DDPG-style)")
    p.add_argument("--episodes", type=int, default=TrainParams().episodes)
    p.add_argument("--batch-size", type=int, default=TrainParams().batch_size)
    p.add_argument("--replay-size", type=int, default=TrainParams().replay_size)
    p.add_argument("--min-replay", type=int, default=TrainParams().min_replay)
    p.add_argument("--gamma", type=float, default=TrainParams().gamma)
    p.add_argument("--learning-rate", type=float, default=TrainParams().learning_rate)
    p.add_argument("--target-update", type=int, default=TrainParams().target_update)
    p.add_argument("--eps-start", type=float, default=TrainParams().eps_start,
                   help="exploration noise scale at step 0 (continuous control)")
    p.add_argument("--eps-end", type=float, default=TrainParams().eps_end,
                   help="final exploration noise scale after decay (continuous control)")
    p.add_argument("--eps-decay-steps", type=int, default=TrainParams().eps_decay_steps,
                   help="steps to linearly decay exploration noise scale")
    p.add_argument("--hidden-dim", type=int, default=TrainParams().hidden_dim)
    p.add_argument("--seed", type=int, default=TrainParams().seed)
    p.add_argument("--save-every", type=int, default=TrainParams().save_every)
    p.add_argument("--out-dir", type=str, default=TrainParams().out_dir)
    p.add_argument("--render", action="store_true", help="render during training")
    p.add_argument("--shared-dual-control", action="store_true", help="reuse actor as secondary internal policy during training")
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
    # Single source of truth for observation dimension: infer directly from environment output.
    obs_dim = int(env.reset(seed=args.seed).shape[0])

    agent = DDPGAgent(in_dim=obs_dim, hp=train_hp, device=device)
    replay = ReplayBuffer(train_hp.replay_size)

    if args.shared_dual_control:
        def _secondary_policy(obs: np.ndarray) -> np.ndarray:
            return agent.act(obs, greedy=False)

        env.set_secondary_policy(_secondary_policy)

    out_dir = Path(train_hp.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    for ep in range(1, train_hp.episodes + 1):
        obs = env.reset(seed=args.seed + ep)
        done = False
        ep_return = 0.0
        ep_actor_loss = 0.0
        ep_critic_loss = 0.0
        loss_count = 0

        while not done:
            if args.render and getattr(env, "paused", False):
                env.render()
                continue

            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)

            replay.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_return += reward
            agent.global_step += 1

            if len(replay) >= train_hp.min_replay:
                losses = agent.update(replay, train_hp.batch_size)
                if losses is not None:
                    a_loss, c_loss = losses
                    ep_actor_loss += a_loss
                    ep_critic_loss += c_loss
                    loss_count += 1

            if args.render:
                env.render()

        mean_actor_loss = ep_actor_loss / max(1, loss_count)
        mean_critic_loss = ep_critic_loss / max(1, loss_count)
        eps_now = agent.epsilon()
        history.append(
            {
                "episode": ep,
                "return": float(ep_return),
                "steps": env.step_idx,
                "epsilon": float(eps_now),
                "mean_actor_loss": float(mean_actor_loss),
                "mean_critic_loss": float(mean_critic_loss),
            }
        )
        print(
            f"ep={ep:04d} return={ep_return:8.3f} steps={env.step_idx:4d} "
            f"epsilon={eps_now:0.3f} actor_loss={mean_actor_loss:0.4f} critic_loss={mean_critic_loss:0.4f} replay={len(replay)}"
        )

        if ep % train_hp.save_every == 0 or ep == train_hp.episodes:
            torch.save(
                {
                    "actor_state_dict": agent.actor.state_dict(),
                    "critic_state_dict": agent.critic.state_dict(),
                    "obs_dim": obs_dim,
                    "hidden_dim": train_hp.hidden_dim,
                    "action_dim": ACTION_DIM,
                    "algo": "ddpg_style",
                    "train_args": vars(args),
                },
                out_dir / "ddqn_policy.pt",
            )
            with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    env.close()


if __name__ == "__main__":
    main()
