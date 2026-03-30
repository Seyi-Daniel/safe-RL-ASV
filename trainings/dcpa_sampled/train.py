#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import random
import sys
from collections import deque, namedtuple
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from trainings.dcpa_sampled.environment import SingleVessel2FeatureEnv
from trainings.dcpa_sampled.hyperparameters import EnvParams, RewardParams, TrainParams
from trainings.dcpa_sampled.policy import ACTION_DIM, ContinuousActor, ContinuousCritic

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
        self.actor = ContinuousActor(
            in_dim=in_dim,
            hidden_dim_1=hp.hidden_dim_1,
            hidden_dim_2=hp.hidden_dim_2,
            hidden_dim_3=hp.hidden_dim_3,
            action_dim=ACTION_DIM,
        ).to(device)
        self.actor_tgt = ContinuousActor(
            in_dim=in_dim,
            hidden_dim_1=hp.hidden_dim_1,
            hidden_dim_2=hp.hidden_dim_2,
            hidden_dim_3=hp.hidden_dim_3,
            action_dim=ACTION_DIM,
        ).to(device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())

        self.critic = ContinuousCritic(
            in_dim=in_dim,
            hidden_dim_1=hp.hidden_dim_1,
            hidden_dim_2=hp.hidden_dim_2,
            hidden_dim_3=hp.hidden_dim_3,
            action_dim=ACTION_DIM,
        ).to(device)
        self.critic_tgt = ContinuousCritic(
            in_dim=in_dim,
            hidden_dim_1=hp.hidden_dim_1,
            hidden_dim_2=hp.hidden_dim_2,
            hidden_dim_3=hp.hidden_dim_3,
            action_dim=ACTION_DIM,
        ).to(device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=hp.learning_rate)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=hp.learning_rate)

        self.gamma = hp.gamma
        self.device = device
        self.eps_start = float(hp.eps_start)
        self.eps_end = float(hp.eps_end)
        self.epsilon_decay = float(hp.resolve_epsilon_decay())
        self.epsilon = self.eps_start
        self.tau = 0.005
        self.global_step = 0

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.eps_end, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def act(self, obs: np.ndarray, greedy: bool = False) -> np.ndarray:
        if not greedy and np.random.random() < self.epsilon:
            return np.random.uniform(-1.0, 1.0, size=(ACTION_DIM,)).astype(np.float32)

        with torch.no_grad():
            s = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            a = self.actor(s).squeeze(0).cpu().numpy()
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


def _screen_candidate_episode(
    env: SingleVessel2FeatureEnv,
    seed: int,
    sampling_dcpa_threshold: float,
    sampling_tcpa_threshold: float,
    sampling_screen_max_steps: int | None,
    sampling_screen_max_seconds: float | None,
) -> tuple[bool, float, float, int, str, str]:
    """Run deterministic scripted-only screening for a candidate training seed.

    Screening is strictly policy-independent and is the authoritative episode-
    selection gate for training seed acceptance.
    """
    _ = env.reset(seed=seed)
    initial_scenario = _classify_initial_two_vessel_scenario(env)
    done = False
    steps = 0
    best_dcpa = float("inf")
    best_tcpa = float("inf")
    fail_reason = "terminated_without_threshold"
    step_cap = int(sampling_screen_max_steps) if sampling_screen_max_steps is not None and int(sampling_screen_max_steps) > 0 else None
    seconds_cap = (
        float(sampling_screen_max_seconds)
        if sampling_screen_max_seconds is not None and float(sampling_screen_max_seconds) > 0.0
        else None
    )
    screen_start_time = float(env.time)

    while not done:
        if env.vessel1_reached or env.vessel2_reached:
            fail_reason = "reached_goal_before_threshold"
            break

        # Scripted/default rollout only (no policy action injection during screening).
        _, _, done, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        steps += 1

        dcpa = float(info.get("dcpa", float("inf")))
        tcpa = float(info.get("tcpa", float("inf")))
        best_dcpa = min(best_dcpa, dcpa)
        if tcpa > 0.0:
            best_tcpa = min(best_tcpa, tcpa)

        if (dcpa <= sampling_dcpa_threshold) and (0.0 < tcpa <= sampling_tcpa_threshold):
            return True, best_dcpa, best_tcpa, steps, "accepted", initial_scenario

        # Optional screening-only horizons; do not alter real training rollout limits.
        reached_step_cap = step_cap is not None and steps >= step_cap
        elapsed_seconds = float(env.time) - screen_start_time
        reached_seconds_cap = seconds_cap is not None and elapsed_seconds >= seconds_cap
        if (not done) and (reached_step_cap or reached_seconds_cap):
            if reached_step_cap and reached_seconds_cap:
                fail_reason = "screen_horizon_steps_and_seconds"
            elif reached_step_cap:
                fail_reason = "screen_horizon_steps"
            else:
                fail_reason = "screen_horizon_seconds"
            break

    if done and fail_reason == "terminated_without_threshold":
        fail_reason = str(info.get("reason", "done_without_threshold"))
    return False, best_dcpa, best_tcpa, steps, fail_reason, initial_scenario


def _find_accepted_seed(
    sample_env: SingleVessel2FeatureEnv,
    *,
    episode_index: int,
    base_seed: int,
    max_tries: int,
    sampling_dcpa_threshold: float,
    sampling_tcpa_threshold: float,
    sampling_screen_max_steps: int | None,
    sampling_screen_max_seconds: float | None,
    sampling_logs: bool,
) -> tuple[int | None, int, float, float, int, str]:
    """Search candidate seeds using scripted screening and return the first accepted seed."""
    accepted_seed = None
    accepted_attempt = -1
    accepted_best_dcpa = float("inf")
    accepted_best_tcpa = float("inf")
    accepted_sample_steps = 0
    accepted_scenario = "safe"

    attempt = 0
    while True:
        if max_tries > 0 and attempt >= max_tries:
            break

        candidate_seed = base_seed + episode_index * 100_000 + attempt
        ok, best_dcpa, best_tcpa, sample_steps, status, scenario = _screen_candidate_episode(
            sample_env,
            candidate_seed,
            sampling_dcpa_threshold,
            sampling_tcpa_threshold,
            sampling_screen_max_steps,
            sampling_screen_max_seconds,
        )

        if ok:
            accepted_seed = candidate_seed
            accepted_attempt = attempt
            accepted_best_dcpa = best_dcpa
            accepted_best_tcpa = best_tcpa
            accepted_sample_steps = sample_steps
            accepted_scenario = scenario
            if sampling_logs:
                print(
                    f"ep={episode_index:04d} accepted_seed={accepted_seed} attempt={accepted_attempt} "
                    f"sample_steps={accepted_sample_steps} sample_best_dcpa={accepted_best_dcpa:.2f} "
                    f"sample_best_tcpa={accepted_best_tcpa:.2f} sample_scenario={accepted_scenario}"
                )
            break

        if sampling_logs:
            horizon_suffix = " (stopped by screening horizon)" if status.startswith("screen_horizon_") else ""
            print(
                f"ep={episode_index:04d} failed attempt={attempt} seed={candidate_seed} steps={sample_steps} "
                f"reason={status}{horizon_suffix} sample_scenario={scenario} "
                f"(best_dcpa={best_dcpa:.2f}, best_tcpa={best_tcpa:.2f}; "
                f"need dcpa <= {sampling_dcpa_threshold:.2f} and tcpa <= {sampling_tcpa_threshold:.2f})"
            )
        attempt += 1

    return accepted_seed, accepted_attempt, accepted_best_dcpa, accepted_best_tcpa, accepted_sample_steps, accepted_scenario


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train continuous-control policy (DDPG-style) for dcpa_sampled")
    p.add_argument("--episodes", type=int, default=TrainParams().episodes)
    p.add_argument("--batch-size", type=int, default=TrainParams().batch_size)
    p.add_argument("--replay-size", type=int, default=TrainParams().replay_size)
    p.add_argument("--min-replay", type=int, default=TrainParams().min_replay)
    p.add_argument("--gamma", type=float, default=TrainParams().gamma)
    p.add_argument("--learning-rate", type=float, default=TrainParams().learning_rate)
    p.add_argument("--target-update", type=int, default=TrainParams().target_update)
    p.add_argument("--eps-start", type=float, default=TrainParams().eps_start,
                   help="episode-level epsilon at the start of training")
    p.add_argument("--eps-end", type=float, default=TrainParams().eps_end,
                   help="minimum episode-level epsilon floor")
    p.add_argument(
        "--epsilon-decay",
        type=float,
        default=None,
        help=(
            "episode-based multiplicative epsilon decay. "
            "If provided, overrides decay derived from --epsilon-decay-episodes"
        ),
    )
    p.add_argument(
        "--epsilon-decay-episodes",
        type=int,
        default=TrainParams().epsilon_decay_episodes,
        help=(
            "number of episodes used to derive decay when --epsilon-decay is not provided; "
            "default: 1000"
        ),
    )
    p.add_argument("--hidden-dim-1", type=int, default=TrainParams().hidden_dim_1)
    p.add_argument("--hidden-dim-2", type=int, default=TrainParams().hidden_dim_2)
    p.add_argument("--hidden-dim-3", type=int, default=TrainParams().hidden_dim_3)
    p.add_argument("--seed", type=int, default=TrainParams().seed)
    p.add_argument("--episode-seconds", type=float, default=500.0)
    p.add_argument("--num-vessels", type=int, default=EnvParams().num_vessels)
    p.add_argument(
        "--dcpa-threshold",
        type=float,
        default=EnvParams().dcpa_risk_threshold,
        help="environment runtime risk/takeover DCPA threshold (not sampling threshold)",
    )
    p.add_argument(
        "--tcpa-threshold",
        type=float,
        default=EnvParams().tcpa_risk_threshold,
        help="environment runtime risk/takeover TCPA threshold (not sampling threshold)",
    )
    p.add_argument(
        "--sampling-dcpa-threshold",
        type=float,
        default=TrainParams().sampling_dcpa_threshold,
        help="training-only seed-screening DCPA threshold (independent from runtime risk threshold)",
    )
    p.add_argument(
        "--sampling-tcpa-threshold",
        type=float,
        default=TrainParams().sampling_tcpa_threshold,
        help="training-only seed-screening TCPA threshold (independent from runtime risk threshold)",
    )
    p.add_argument("--dcpa-sample-max-tries", type=int, default=0, help="max sampling tries per training episode (0=unlimited)")
    p.add_argument(
        "--max-sampling-steps-per-attempt",
        type=int,
        default=0,
        help="legacy alias for --sampling-screen-max-steps (0=disabled/unlimited)",
    )
    p.add_argument(
        "--sampling-screen-max-steps",
        type=int,
        default=TrainParams().sampling_screen_max_steps,
        help="optional screening-only step cap per candidate seed (default: disabled/unlimited)",
    )
    p.add_argument(
        "--sampling-screen-max-seconds",
        type=float,
        default=TrainParams().sampling_screen_max_seconds,
        help="optional screening-only simulated-seconds cap per candidate seed (default: disabled/unlimited)",
    )
    p.add_argument("--sampling-logs", dest="sampling_logs", action="store_true")
    p.add_argument("--no-sampling-logs", dest="sampling_logs", action="store_false")
    p.set_defaults(sampling_logs=True)
    p.add_argument("--save-every", type=int, default=TrainParams().save_every)
    p.add_argument(
        "--out-dir",
        type=str,
        default=TrainParams().out_dir,
        help="base output folder; training creates a timestamped run subfolder under this path",
    )
    p.add_argument("--render", action="store_true", help="render during training")
    p.add_argument("--no-render", dest="render", action="store_false")
    p.set_defaults(render=False)
    p.add_argument("--show-risk-overlay", dest="show_risk_overlay", action="store_true",
                   help="show RL takeover/risk HUD overlay during render mode")
    p.add_argument("--hide-risk-overlay", dest="show_risk_overlay", action="store_false",
                   help="hide RL takeover/risk HUD overlay during render mode")
    p.set_defaults(show_risk_overlay=EnvParams().show_risk_overlay)
    p.add_argument(
        "--auto-show-risk-sector-overlay",
        dest="auto_show_risk_sector_overlay",
        action="store_true",
        help="when takeover HUD pause triggers, also auto-show radar sector rays until continue",
    )
    p.add_argument(
        "--no-auto-show-risk-sector-overlay",
        dest="auto_show_risk_sector_overlay",
        action="store_false",
        help="do not auto-show radar sector rays at takeover HUD pause",
    )
    p.set_defaults(auto_show_risk_sector_overlay=EnvParams().auto_show_risk_sector_overlay)
    p.add_argument("--eval-only", action="store_true", help="run deterministic evaluation only (no training updates)")
    p.add_argument("--eval-episodes", type=int, default=30, help="accepted evaluation episodes per scenario")
    p.add_argument(
        "--eval-scenario",
        type=str,
        default="all",
        choices=["head_on", "crossing", "overtaking", "all"],
        help="scenario to evaluate (or all)",
    )
    p.add_argument(
        "--eval-max-tries-per-episode",
        type=int,
        default=200,
        help="max seed attempts to find a matching scenario per requested eval episode",
    )
    p.add_argument(
        "--eval-checkpoint",
        type=str,
        default="",
        help=(
            "checkpoint path for eval-only mode. "
            "If omitted, resolve <out-dir>/policy_latest.pt or the newest timestamped run's policy_latest.pt"
        ),
    )
    return p.parse_args()


def _collect_rl_actions_for_step(
    env: SingleVessel2FeatureEnv,
    agent: DDPGAgent,
    greedy: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Collect per-vessel observations/actions from a single pre-step world state.

    A single shared policy is queried separately for each RL-active vessel.
    """
    obs_by_vessel: dict[str, np.ndarray] = {}
    action_by_vessel: dict[str, np.ndarray] = {}
    for vessel_id in env.get_rl_controlled_vessel_ids():
        obs = env.get_obs_for_vessel(vessel_id)
        obs_by_vessel[vessel_id] = obs
        action_by_vessel[vessel_id] = agent.act(obs, greedy=greedy)
    return obs_by_vessel, action_by_vessel


def _classify_initial_two_vessel_scenario(env: SingleVessel2FeatureEnv) -> str:
    if env.vessel1 is None or env.vessel2 is None:
        return "safe"
    scenario, _, _ = env.classify_geometry(env.vessel1, env.vessel2)
    return str(scenario)


def _update_starboard_compliance_metrics(
    info: dict[str, object],
    *,
    starboard_opportunities: int,
    starboard_compliant: int,
) -> tuple[int, int]:
    """Count simple COLREGS starboard opportunities/compliance for RL-controlled give-way vessel steps."""
    scenario = str(info.get("colregs_scenario", "safe"))
    if scenario not in {"head_on", "crossing"}:
        return starboard_opportunities, starboard_compliant

    give_way_vessel = str(info.get("designated_give_way_vessel", "none"))
    if give_way_vessel not in {"vessel1", "vessel2"}:
        return starboard_opportunities, starboard_compliant

    control_source = str(info.get(f"{give_way_vessel}_control_source", ""))
    if control_source != "rl_external":
        return starboard_opportunities, starboard_compliant

    rudder_key = f"{give_way_vessel}_rudder_deg"
    rudder_deg = float(info.get(rudder_key, 0.0))
    starboard_opportunities += 1
    if rudder_deg > 0.0:
        starboard_compliant += 1
    return starboard_opportunities, starboard_compliant


def _run_eval_only(
    args: argparse.Namespace,
    env: SingleVessel2FeatureEnv,
    agent: DDPGAgent,
) -> None:
    checkpoint_path = _resolve_eval_checkpoint_path(args.eval_checkpoint, Path(args.out_dir))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=agent.device)
    actor_state = ckpt.get("actor_state_dict")
    if actor_state is None:
        raise KeyError(f"Checkpoint missing actor_state_dict: {checkpoint_path}")
    agent.actor.load_state_dict(actor_state)
    agent.actor.eval()

    scenarios = ["head_on", "crossing", "overtaking"] if args.eval_scenario == "all" else [args.eval_scenario]
    base_seed = int(args.seed)
    eval_episodes = max(1, int(args.eval_episodes))
    max_tries = max(1, int(args.eval_max_tries_per_episode))

    for scenario_idx, target_scenario in enumerate(scenarios):
        requested_episodes = eval_episodes
        accepted_episodes = 0
        candidate_episodes = 0
        total_tries = 0
        collision_count = 0
        goal_completion_count = 0
        safe_pass_count = 0
        takeover_count = 0
        both_controlled_step_count = 0
        scenario_step_counts: dict[str, int] = {"head_on": 0, "crossing": 0, "overtaking": 0}
        starboard_opportunities = 0
        starboard_compliant = 0
        total_return = 0.0
        total_steps = 0

        print(f"\n[eval] scenario={target_scenario}")
        for ep in range(requested_episodes):
            matched_seed = None
            for attempt in range(max_tries):
                candidate_seed = base_seed + scenario_idx * 10_000_000 + ep * 10_000 + attempt
                candidate_episodes += 1
                total_tries += 1
                _ = env.reset(seed=candidate_seed)
                initial_scenario = _classify_initial_two_vessel_scenario(env)
                if initial_scenario == target_scenario:
                    matched_seed = candidate_seed
                    break

            if matched_seed is None:
                print(
                    f"[eval] scenario={target_scenario} episode={ep + 1:03d} skipped "
                    f"(no match within {max_tries} tries)"
                )
                continue

            _ = env.reset(seed=matched_seed)
            done = False
            ep_return = 0.0
            last_info: dict[str, object] = {}
            prev_v1_latched = 0
            prev_v2_latched = 0

            while not done:
                if args.render and getattr(env, "paused", False):
                    env.render()
                    continue

                _, action_by_vessel = _collect_rl_actions_for_step(env, agent, greedy=True)
                step_action = action_by_vessel if action_by_vessel else np.array([0.0, 0.0], dtype=np.float32)
                _, reward, done, info = env.step(step_action)
                ep_return += float(reward)
                last_info = info
                step_scenario = str(info.get("colregs_scenario", "safe"))
                if step_scenario in scenario_step_counts:
                    scenario_step_counts[step_scenario] += 1
                if int(info.get("vessel1_rl_active", 0)) and int(info.get("vessel2_rl_active", 0)):
                    both_controlled_step_count += 1
                starboard_opportunities, starboard_compliant = _update_starboard_compliance_metrics(
                    info,
                    starboard_opportunities=starboard_opportunities,
                    starboard_compliant=starboard_compliant,
                )
                v1_latched = int(info.get("vessel1_model_control_latched", info.get("vessel1_rl_latched", 0)))
                v2_latched = int(info.get("vessel2_model_control_latched", info.get("vessel2_rl_latched", 0)))
                if v1_latched and not prev_v1_latched:
                    takeover_count += 1
                if v2_latched and not prev_v2_latched:
                    takeover_count += 1
                prev_v1_latched = v1_latched
                prev_v2_latched = v2_latched

                if args.render:
                    env.render()

            accepted_episodes += 1
            total_return += ep_return
            total_steps += int(env.step_idx)
            collision_count += int(last_info.get("collision", 0))
            v1_reached = int(last_info.get("vessel1_reached", 0))
            v2_reached = int(last_info.get("vessel2_reached", 0))
            goal_completion_count += int(v1_reached and v2_reached)
            safe_pass_count += int(last_info.get("safe_pass_awarded", 0))

        collision_rate = (collision_count / accepted_episodes) if accepted_episodes else 0.0
        goal_rate = (goal_completion_count / accepted_episodes) if accepted_episodes else 0.0
        safe_pass_rate = (safe_pass_count / accepted_episodes) if accepted_episodes else 0.0
        starboard_rate = (starboard_compliant / starboard_opportunities) if starboard_opportunities else 0.0
        avg_return = (total_return / accepted_episodes) if accepted_episodes else 0.0
        avg_length = (total_steps / accepted_episodes) if accepted_episodes else 0.0

        print(f"[eval][{target_scenario}] requested_episode_count={requested_episodes}")
        print(f"[eval][{target_scenario}] candidate_episode_count={candidate_episodes}")
        print(f"[eval][{target_scenario}] accepted_episode_count={accepted_episodes}")
        print(f"[eval][{target_scenario}] total_seed_tries={total_tries}")
        print(f"[eval][{target_scenario}] collisions={collision_count} collision_rate={collision_rate:.3f}")
        print(f"[eval][{target_scenario}] goal_completion={goal_completion_count} goal_completion_rate={goal_rate:.3f}")
        print(f"[eval][{target_scenario}] safe_pass={safe_pass_count} safe_pass_rate={safe_pass_rate:.3f}")
        print(
            f"[eval][{target_scenario}] starboard_compliance={starboard_compliant}/{starboard_opportunities} "
            f"starboard_compliance_rate={starboard_rate:.3f}"
        )
        print(f"[eval][{target_scenario}] takeovers={takeover_count}")
        print(f"[eval][{target_scenario}] both_controlled_steps={both_controlled_step_count}")
        print(
            f"[eval][{target_scenario}] scenario_step_counts="
            f"head_on:{scenario_step_counts['head_on']} "
            f"crossing:{scenario_step_counts['crossing']} "
            f"overtaking:{scenario_step_counts['overtaking']}"
        )
        print(f"[eval][{target_scenario}] average_return={avg_return:.3f}")
        print(f"[eval][{target_scenario}] average_episode_length={avg_length:.2f}")


def _resolve_eval_checkpoint_path(eval_checkpoint: str, out_dir: Path) -> Path:
    """Resolve eval checkpoint from explicit path or latest timestamped run folder."""
    if eval_checkpoint:
        return Path(eval_checkpoint)

    direct_policy_latest = out_dir / "policy_latest.pt"
    if direct_policy_latest.exists():
        return direct_policy_latest

    candidate_runs = [
        run_dir for run_dir in out_dir.iterdir() if run_dir.is_dir() and (run_dir / "policy_latest.pt").exists()
    ] if out_dir.exists() else []
    if candidate_runs:
        latest_run = max(candidate_runs, key=lambda p: p.stat().st_mtime)
        return latest_run / "policy_latest.pt"

    raise FileNotFoundError(
        "No evaluation checkpoint found. Pass --eval-checkpoint explicitly, or set --out-dir to a run folder "
        "containing policy_latest.pt, or ensure at least one timestamped run exists under the base out-dir."
    )


def main() -> None:
    args = parse_args()
    # IMPORTANT:
    # Sampling thresholds are ONLY for training episode seed selection.
    # Environment risk/takeover logic MUST use env.dcpa_risk_threshold / tcpa_risk_threshold.
    sampling_dcpa_threshold = float(args.sampling_dcpa_threshold)
    sampling_tcpa_threshold = float(args.sampling_tcpa_threshold)
    sampling_screen_max_steps = (
        int(args.sampling_screen_max_steps)
        if args.sampling_screen_max_steps is not None and int(args.sampling_screen_max_steps) > 0
        else None
    )
    sampling_screen_max_seconds = (
        float(args.sampling_screen_max_seconds)
        if args.sampling_screen_max_seconds is not None and float(args.sampling_screen_max_seconds) > 0.0
        else None
    )
    # Backward-compatibility: allow legacy flag to act as screening-only step horizon
    # when the new explicit option is not provided.
    if sampling_screen_max_steps is None and int(args.max_sampling_steps_per_attempt) > 0:
        sampling_screen_max_steps = int(args.max_sampling_steps_per_attempt)

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
        epsilon_decay=args.epsilon_decay,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        hidden_dim_1=args.hidden_dim_1,
        hidden_dim_2=args.hidden_dim_2,
        hidden_dim_3=args.hidden_dim_3,
        seed=args.seed,
        save_every=args.save_every,
        out_dir=args.out_dir,
    )

    envp = EnvParams(
        seed=args.seed,
        num_vessels=max(2, int(args.num_vessels)),
        episode_seconds=args.episode_seconds,
        dcpa_risk_threshold=args.dcpa_threshold,
        tcpa_risk_threshold=args.tcpa_threshold,
        show_risk_overlay=bool(args.show_risk_overlay),
        auto_show_risk_sector_overlay=bool(args.auto_show_risk_sector_overlay),
    )
    sample_env = SingleVessel2FeatureEnv(envp, RewardParams(), render=False)
    env = SingleVessel2FeatureEnv(envp, RewardParams(), render=args.render)
    # Single source of truth for observation dimension: infer directly from environment output.
    obs_dim = int(env.reset(seed=args.seed).shape[0])

    agent = DDPGAgent(in_dim=obs_dim, hp=train_hp, device=device)
    replay = ReplayBuffer(train_hp.replay_size)

    script_dir = Path(__file__).resolve().parent
    base_out_dir = Path(train_hp.out_dir)
    if not base_out_dir.is_absolute():
        base_out_dir = script_dir / base_out_dir
    base_out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(base_out_dir)

    if args.eval_only:
        _run_eval_only(args, env, agent)
        env.close()
        sample_env.close()
        return

    run_stamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = base_out_dir / run_stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    for ep in range(1, train_hp.episodes + 1):
        max_tries = int(args.dcpa_sample_max_tries)
        (
            accepted_seed,
            accepted_attempt,
            accepted_best_dcpa,
            accepted_best_tcpa,
            accepted_sample_steps,
            accepted_scenario,
        ) = _find_accepted_seed(
            sample_env,
            episode_index=ep,
            base_seed=int(args.seed),
            max_tries=max_tries,
            sampling_dcpa_threshold=sampling_dcpa_threshold,
            sampling_tcpa_threshold=sampling_tcpa_threshold,
            sampling_screen_max_steps=sampling_screen_max_steps,
            sampling_screen_max_seconds=sampling_screen_max_seconds,
            sampling_logs=bool(args.sampling_logs),
        )

        if accepted_seed is None:
            print(
                f"ep={ep:04d} skipped: no sample found with dcpa <= {sampling_dcpa_threshold:.2f} "
                f"and tcpa <= {sampling_tcpa_threshold:.2f} within {max_tries} tries"
            )
            continue

        # Real training rollout starts from a fresh reset of the accepted seed (no hidden reset-side viability gate).
        _ = env.reset(seed=accepted_seed)
        done = False
        ep_return = 0.0
        ep_actor_loss = 0.0
        ep_critic_loss = 0.0
        loss_count = 0
        takeover_triggered = False
        ep_takeover_count = 0
        ep_both_controlled_steps = 0
        ep_starboard_opportunities = 0
        ep_starboard_compliant = 0
        prev_v1_latched = 0
        prev_v2_latched = 0

        while not done:
            if args.render and getattr(env, "paused", False):
                env.render()
                continue

            obs_by_vessel, action_by_vessel = _collect_rl_actions_for_step(env, agent, greedy=False)
            step_action = action_by_vessel if action_by_vessel else np.array([0.0, 0.0], dtype=np.float32)
            _, reward, done, info = env.step(step_action)
            if int(info.get("vessel1_rl_active", 0)) and int(info.get("vessel2_rl_active", 0)):
                ep_both_controlled_steps += 1
            ep_starboard_opportunities, ep_starboard_compliant = _update_starboard_compliance_metrics(
                info,
                starboard_opportunities=ep_starboard_opportunities,
                starboard_compliant=ep_starboard_compliant,
            )
            v1_latched = int(info.get("vessel1_model_control_latched", info.get("vessel1_rl_latched", 0)))
            v2_latched = int(info.get("vessel2_model_control_latched", info.get("vessel2_rl_latched", 0)))
            if v1_latched and not prev_v1_latched:
                ep_takeover_count += 1
            if v2_latched and not prev_v2_latched:
                ep_takeover_count += 1
            prev_v1_latched = v1_latched
            prev_v2_latched = v2_latched

            for vessel_id, vessel_obs in obs_by_vessel.items():
                vessel_next_obs = env.get_obs_for_vessel(vessel_id)
                reward_by_vessel = info.get("reward_by_vessel")
                if isinstance(reward_by_vessel, dict) and (vessel_id in reward_by_vessel):
                    vessel_reward = float(reward_by_vessel[vessel_id])
                else:
                    if vessel_id == "vessel1":
                        vessel_reward = float(info["reward_v1"])
                    elif vessel_id == "vessel2":
                        vessel_reward = float(info["reward_v2"])
                    else:
                        vessel_reward = float(reward)
                # Replay is built from per-vessel rewards, not the scalar compatibility reward.
                replay.push(vessel_obs, action_by_vessel[vessel_id], vessel_reward, vessel_next_obs, done)
            takeover_triggered = takeover_triggered or bool(env.get_rl_controlled_vessel_ids())
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
        eps_now = agent.epsilon
        history.append(
            {
                "episode": ep,
                "return": float(ep_return),
                "steps": env.step_idx,
                "epsilon": float(eps_now),
                "mean_actor_loss": float(mean_actor_loss),
                "mean_critic_loss": float(mean_critic_loss),
                "collision": int(info.get("collision", 0)),
                "goal_completion": int(bool(info.get("vessel1_reached", 0)) and bool(info.get("vessel2_reached", 0))),
                "safe_pass": int(info.get("safe_pass_awarded", 0)),
                "takeovers": int(ep_takeover_count),
                "both_controlled_steps": int(ep_both_controlled_steps),
                "starboard_opportunities": int(ep_starboard_opportunities),
                "starboard_compliant": int(ep_starboard_compliant),
                "starboard_compliance_rate": (
                    float(ep_starboard_compliant / ep_starboard_opportunities) if ep_starboard_opportunities else 0.0
                ),
            }
        )
        agent.decay_epsilon()

        if not takeover_triggered:
            print(
                f"ep={ep:04d} skipped_learning=no_takeover return={ep_return:8.3f} steps={env.step_idx:4d} "
                f"replay={len(replay)} collision={int(info.get('collision', 0))} "
                f"goals={int(bool(info.get('vessel1_reached', 0)) and bool(info.get('vessel2_reached', 0)))} "
                f"safe_pass={int(info.get('safe_pass_awarded', 0))} takeovers={ep_takeover_count} "
                f"both_ctrl_steps={ep_both_controlled_steps} sample_attempt={accepted_attempt} "
                f"sample_steps={accepted_sample_steps} sample_scenario={accepted_scenario}"
            )
        else:
            print(
                f"ep={ep:04d} return={ep_return:8.3f} steps={env.step_idx:4d} "
                f"epsilon={eps_now:0.3f} actor_loss={mean_actor_loss:0.4f} critic_loss={mean_critic_loss:0.4f} "
                f"replay={len(replay)} collision={int(info.get('collision', 0))} "
                f"goals={int(bool(info.get('vessel1_reached', 0)) and bool(info.get('vessel2_reached', 0)))} "
                f"safe_pass={int(info.get('safe_pass_awarded', 0))} takeovers={ep_takeover_count} "
                f"both_ctrl_steps={ep_both_controlled_steps} sample_attempt={accepted_attempt} "
                f"sample_steps={accepted_sample_steps} sample_scenario={accepted_scenario}"
            )

        if ep % train_hp.save_every == 0 or ep == train_hp.episodes:
            checkpoint_payload = {
                "actor_state_dict": agent.actor.state_dict(),
                "critic_state_dict": agent.critic.state_dict(),
                "obs_dim": obs_dim,
                "hidden_dims": [train_hp.hidden_dim_1, train_hp.hidden_dim_2, train_hp.hidden_dim_3],
                "action_dim": ACTION_DIM,
                "algo": "ddpg_style",
                "train_args": vars(args),
            }
            torch.save(
                checkpoint_payload,
                out_dir / f"policy_{ep}.pt",
            )
            torch.save(checkpoint_payload, out_dir / "policy_latest.pt")
            with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    env.close()
    sample_env.close()


if __name__ == "__main__":
    main()
