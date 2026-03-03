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

from trainings.perimeter_start.environment import SingleVessel2FeatureEnv
from trainings.perimeter_start.hyperparameters import EnvParams, RewardParams, TrainParams
from trainings.perimeter_start.policy import ACTION_DIM, ContinuousActor, ContinuousCritic

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


def _episode_hits_dcpa_threshold(
    env: SingleVessel2FeatureEnv,
    seed: int,
    dcpa_threshold: float,
    tcpa_threshold: float,
    max_sampling_steps_per_attempt: int,
) -> tuple[bool, float, float, int, str]:
    _ = env.reset(seed=seed)
    done = False
    steps = 0
    best_dcpa = float("inf")
    best_tcpa = float("inf")
    fail_reason = "terminated_without_threshold"
    step_cap = int(max_sampling_steps_per_attempt) if int(max_sampling_steps_per_attempt) > 0 else max(1, 2 * int(env.max_steps))

    while not done:
        if env.vessel1_reached or env.vessel2_reached:
            fail_reason = "reached_goal_before_threshold"
            break

        _, _, done, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        steps += 1

        dcpa = float(info.get("dcpa", float("inf")))
        tcpa = float(info.get("tcpa", float("inf")))
        best_dcpa = min(best_dcpa, dcpa)
        if tcpa > 0.0:
            best_tcpa = min(best_tcpa, tcpa)

        if (dcpa <= dcpa_threshold) and (0.0 < tcpa <= tcpa_threshold):
            return True, best_dcpa, best_tcpa, steps, "accepted"

        if steps >= step_cap:
            fail_reason = "max_sampling_steps_per_attempt_guard"
            break

    if done and fail_reason == "terminated_without_threshold":
        fail_reason = str(info.get("reason", "done_without_threshold"))
    return False, best_dcpa, best_tcpa, steps, fail_reason


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train continuous-control policy (DDPG-style) for perimeter_start")
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
    p.add_argument("--episode-seconds", type=float, default=500.0)
    p.add_argument("--dcpa-threshold", type=float, default=10.0)
    p.add_argument("--tcpa-threshold", type=float, default=10.0)
    p.add_argument("--dcpa-sample-max-tries", type=int, default=0, help="max sampling tries per training episode (0=unlimited)")
    p.add_argument("--max-sampling-steps-per-attempt", type=int, default=0, help="sampling step cap per candidate seed (0=2x episode steps)")
    p.add_argument("--sampling-logs", dest="sampling_logs", action="store_true")
    p.add_argument("--no-sampling-logs", dest="sampling_logs", action="store_false")
    p.set_defaults(sampling_logs=True)
    p.add_argument("--save-every", type=int, default=TrainParams().save_every)
    p.add_argument("--out-dir", type=str, default="runs/perimeter_start")
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
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        save_every=args.save_every,
        out_dir=args.out_dir,
    )

    envp = EnvParams(
        seed=args.seed,
        episode_seconds=args.episode_seconds,
        dcpa_risk_threshold=args.dcpa_threshold,
        tcpa_risk_threshold=args.tcpa_threshold,
        require_reset_viable_takeover_path=False,
        enable_no_takeover_early_done=False,
    )
    sample_env = SingleVessel2FeatureEnv(envp, RewardParams(), render=False)
    env = SingleVessel2FeatureEnv(envp, RewardParams(), render=args.render)
    # Single source of truth for observation dimension: infer directly from environment output.
    obs_dim = int(env.reset(seed=args.seed).shape[0])

    agent = DDPGAgent(in_dim=obs_dim, hp=train_hp, device=device)
    replay = ReplayBuffer(train_hp.replay_size)

    out_dir = Path(train_hp.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    for ep in range(1, train_hp.episodes + 1):
        accepted_seed = None
        accepted_attempt = -1
        accepted_best_dcpa = float("inf")
        accepted_best_tcpa = float("inf")
        max_tries = int(args.dcpa_sample_max_tries)
        attempt = 0
        while True:
            if max_tries > 0 and attempt >= max_tries:
                break
            candidate_seed = args.seed + ep * 100_000 + attempt
            ok, best_dcpa, best_tcpa, sample_steps, fail_reason = _episode_hits_dcpa_threshold(
                sample_env,
                candidate_seed,
                args.dcpa_threshold,
                args.tcpa_threshold,
                args.max_sampling_steps_per_attempt,
            )
            if ok:
                accepted_seed = candidate_seed
                accepted_attempt = attempt
                accepted_best_dcpa = best_dcpa
                accepted_best_tcpa = best_tcpa
                if args.sampling_logs:
                    print(
                        f"ep={ep:04d} accepted_seed={accepted_seed} attempt={accepted_attempt} "
                        f"sample_steps={sample_steps} sample_best_dcpa={accepted_best_dcpa:.2f} "
                        f"sample_best_tcpa={accepted_best_tcpa:.2f}"
                    )
                break
            if args.sampling_logs:
                print(
                    f"ep={ep:04d} failed attempt={attempt} seed={candidate_seed} steps={sample_steps} "
                    f"reason={fail_reason} (best_dcpa={best_dcpa:.2f}, best_tcpa={best_tcpa:.2f}; "
                    f"need dcpa <= {args.dcpa_threshold:.2f} and tcpa <= {args.tcpa_threshold:.2f})"
                )
            attempt += 1

        if accepted_seed is None:
            print(
                f"ep={ep:04d} skipped: no sample found with dcpa <= {args.dcpa_threshold:.2f} "
                f"and tcpa <= {args.tcpa_threshold:.2f} within {max_tries} tries"
            )
            continue

        obs = env.reset(seed=accepted_seed)
        done = False
        ep_return = 0.0
        ep_actor_loss = 0.0
        ep_critic_loss = 0.0
        loss_count = 0
        takeover_triggered = False

        while not done:
            if args.render and getattr(env, "paused", False):
                env.render()
                continue

            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)

            replay.push(obs, action, reward, next_obs, done)
            takeover_triggered = takeover_triggered or bool(info.get("vessel1_rl_active", 0)) or bool(info.get("vessel2_rl_active", 0))
            obs = next_obs
            ep_return += reward
            agent.global_step += 1

            if takeover_triggered and len(replay) >= train_hp.min_replay:
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
        if not takeover_triggered:
            print(f"ep={ep:04d} skipped_learning=no_takeover return={ep_return:8.3f} steps={env.step_idx:4d} replay={len(replay)}")
        else:
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
    sample_env.close()


if __name__ == "__main__":
    main()
