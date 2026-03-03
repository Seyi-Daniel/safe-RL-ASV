#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np

from environment import HAS_PYGAME, SingleTargetFeatureEnv, Vessel, clamp
from hyperparameters import EnvParams, RewardParams

if HAS_PYGAME:
    import pygame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lightweight simulation sandbox for manual behavior checks")
    p.add_argument("--view", choices=("episode", "v2-heading-range", "dcpa-sampled-episode"), default="episode")

    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--render", action="store_true", help="enable pygame visualization")
    p.add_argument("--episode-seconds", type=float, default=60.0)
    p.add_argument(
        "--target-min-goal-arc-distance",
        "--target-max-goal-arc-distance",
        "--target-max-goal-distance",
        dest="target_min_goal_arc_distance",
        type=float,
        default=EnvParams().target_min_goal_arc_distance_from_start,
        help="Minimum required vessel-2 start->goal straight-line (chord) distance",
    )

    p.add_argument("--adaptive-dcrit-min-arc", action="store_true", help="Use speed-based dcrit to raise minimum target goal arc distance")
    p.add_argument("--no-adaptive-dcrit-min-arc", dest="adaptive_dcrit_min_arc", action="store_false", help="Disable speed-based dcrit minimum arc adjustment")
    p.set_defaults(adaptive_dcrit_min_arc=EnvParams().adaptive_target_min_goal_arc_from_speed)
    p.add_argument("--dcrit-factor", type=float, default=EnvParams().target_min_goal_dcrit_factor, help="Multiplier on dcrit when adaptive dcrit min-arc is enabled")

    # dcpa-sampled-episode view options
    p.add_argument("--dcpa-threshold", type=float, default=20.0, help="Accept sampled episodes only if DCPA reaches this threshold or lower")
    p.add_argument("--tcpa-threshold", type=float, default=90.0, help="Accept sampled episodes only if TCPA is within [0, threshold] when DCPA condition is met")
    p.add_argument("--dcpa-sample-max-tries", type=int, default=0, help="Max resampling attempts per accepted episode (0 = unlimited)")
    p.add_argument("--debug-sampling", action="store_true", help="Enable detailed DCPA/TCPA sampling debug logs")
    p.add_argument("--debug-sampling-step-log-every", type=int, default=100, help="When debug is on, print per-attempt status every N steps")
    p.add_argument("--max-sampling-steps-per-attempt", type=int, default=0, help="Safety cap for candidate sampling steps (0 = automatic cap of 2x episode max steps)")

    # v2-heading-range view options
    p.add_argument("--v2-start-angle-deg", type=float, default=40.0, help="Single vessel-2 start point angle on the big circle")
    p.add_argument("--heading-line-length", type=float, default=28.0, help="Length (meters) of each heading-boundary line")
    p.add_argument("--heading-path-speed", type=float, default=-1.0, help="Speed used for planned-path rollout (-1 uses midpoint of target min/max speed)")
    p.add_argument("--auto-close-seconds", type=float, default=0.0, help="Auto-close window after N seconds (0 = wait until closed)")
    p.add_argument("--save-image", type=str, default="", help="Optional image path to save current heading-range view")
    return p.parse_args()


def _make_env_params(args: argparse.Namespace) -> EnvParams:
    return EnvParams(
        episode_seconds=args.episode_seconds,
        seed=args.seed,
        target_min_goal_arc_distance_from_start=args.target_min_goal_arc_distance,
        adaptive_target_min_goal_arc_from_speed=args.adaptive_dcrit_min_arc,
        target_min_goal_dcrit_factor=args.dcrit_factor,
    )


def run_episode_view(args: argparse.Namespace, envp: EnvParams) -> None:
    env = SingleTargetFeatureEnv(envp, RewardParams(), render=args.render)
    try:
        for ep in range(1, args.episodes + 1):
            _ = env.reset(seed=args.seed + ep)
            v2_start = env.target_start_pos
            v2_goal = (env.target.goal_x, env.target.goal_y)
            v2_goal_dist = float(np.hypot(v2_goal[0] - v2_start[0], v2_goal[1] - v2_start[1]))
            print(
                f"Episode {ep}: vessel2 start={v2_start}, goal={v2_goal}, "
                f"distance={v2_goal_dist:.2f} m (min={args.target_min_goal_arc_distance:.2f} m)"
            )

            done = False
            total = 0.0
            while not done:
                action = np.array([0.0, 0.0], dtype=np.float32)
                _, reward, done, info = env.step(action)
                total += reward
                if args.render:
                    env.render()

            print(
                f"  end: reason={info.get('reason', 'unknown')} steps={env.step_idx} "
                f"return={float(total):.3f} vessel2_goal_distance={float(info.get('vessel2_goal_distance', info.get('target_goal_distance', -1.0))):.3f}"
            )
    finally:
        env.close()


def _episode_hits_dcpa_threshold(
    env: SingleTargetFeatureEnv,
    seed: int,
    dcpa_threshold: float,
    tcpa_threshold: float,
    pump_events: bool = False,
    render_sampling: bool = False,
    debug_sampling: bool = False,
    debug_step_log_every: int = 100,
    max_sampling_steps_per_attempt: int = 0,
) -> tuple[bool, float, float, int, dict[str, float | str | int], str]:
    _ = env.reset(seed=seed)
    def _meets_sampling_thresholds(dcpa: float, tcpa: float) -> bool:
        return (dcpa <= dcpa_threshold) and (0.0 < tcpa <= tcpa_threshold)

    # Sampling is automated; disable modal overlay pause that otherwise freezes progression
    # when risk lock-in first triggers without a dismissal keypress.
    env.rl_overlay_shown = True
    env.risk_overlay_active = False
    env.paused = False

    effective_step_cap = int(max_sampling_steps_per_attempt) if int(max_sampling_steps_per_attempt) > 0 else max(1, 2 * int(env.max_steps))
    if debug_sampling:
        print(
            f"[sample-debug] start seed={seed} max_steps={env.max_steps} "
            f"episode_seconds={env.envp.episode_seconds} effective_step_cap={effective_step_cap}"
        )
    best_dcpa = float("inf")
    best_tcpa = float("inf")
    done = False
    steps = 0
    fail_reason = "terminated_without_threshold"
    final_info: dict[str, float | str | int] = {"reason": "unknown"}
    while not done:
        if pump_events and HAS_PYGAME:
            pygame.event.pump()
        # Requirement: threshold crossing must happen before either vessel reaches goal.
        if env.agent_reached or env.target_reached:
            fail_reason = "reached_goal_before_threshold"
            break
        _, _, done, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        steps += 1
        final_info = info
        if render_sampling:
            env.render()
        dcpa = float(info.get("dcpa", float("inf")))
        tcpa = float(info.get("tcpa", float("inf")))
        best_dcpa = min(best_dcpa, dcpa)
        if tcpa > 0.0:
            best_tcpa = min(best_tcpa, tcpa)

        if debug_sampling and (steps % max(1, int(debug_step_log_every)) == 0):
            print(
                f"[sample-debug] seed={seed} step={steps} dcpa={dcpa:.2f} tcpa={tcpa:.2f} "
                f"best_dcpa={best_dcpa:.2f} best_tcpa={best_tcpa:.2f} "
                f"vessel1_reached={int(env.agent_reached)} vessel2_reached={int(env.target_reached)} done={int(done)}"
            )

        if steps >= effective_step_cap:
            fail_reason = "max_sampling_steps_per_attempt_guard"
            break

        # Strict requirement: threshold crossing must occur while both vessels are still active.
        if _meets_sampling_thresholds(dcpa, tcpa) and (not env.agent_reached) and (not env.target_reached):
            if debug_sampling:
                print(
                    f"[sample-debug] seed={seed} accepted at step={steps} "
                    f"dcpa={dcpa:.2f} tcpa={tcpa:.2f}"
                )
            return True, best_dcpa, best_tcpa, steps, final_info, "accepted"

    if done and fail_reason == "terminated_without_threshold":
        fail_reason = str(final_info.get("reason", "done_without_threshold"))
    if debug_sampling:
        print(
            f"[sample-debug] seed={seed} fail_reason={fail_reason} steps={steps} "
            f"best_dcpa={best_dcpa:.2f} best_tcpa={best_tcpa:.2f}"
        )
    return False, best_dcpa, best_tcpa, steps, final_info, fail_reason


def run_dcpa_sampled_episode_view(args: argparse.Namespace) -> None:
    def _meets_sampling_thresholds(dcpa: float, tcpa: float) -> bool:
        return (dcpa <= args.dcpa_threshold) and (0.0 < tcpa <= args.tcpa_threshold)

    # In this view, keep min-goal baseline at 0 but retain adaptive dcrit-from-speed filtering,
    # while disabling reset/early-done gating so DCPA/TCPA criteria drive episode selection.
    envp = EnvParams(
        episode_seconds=args.episode_seconds,
        seed=args.seed,
        target_min_goal_arc_distance_from_start=0.0,
        adaptive_target_min_goal_arc_from_speed=True,
        target_min_goal_dcrit_factor=args.dcrit_factor,
        require_reset_viable_takeover_path=False,
        enable_no_takeover_early_done=False,
    )

    env = SingleTargetFeatureEnv(envp, RewardParams(), render=args.render)
    try:
        for ep in range(1, args.episodes + 1):
            accepted_seed = None
            accepted_best_dcpa = float("inf")
            accepted_best_tcpa = float("inf")
            accepted_attempt = -1
            max_tries = int(args.dcpa_sample_max_tries)
            attempt = 0
            while True:
                if max_tries > 0 and attempt >= max_tries:
                    break
                candidate_seed = args.seed + ep * 100_000 + attempt
                ok, best_dcpa, best_tcpa, sample_steps, _, fail_reason = _episode_hits_dcpa_threshold(
                    env,
                    candidate_seed,
                    args.dcpa_threshold,
                    args.tcpa_threshold,
                    pump_events=args.render,
                    render_sampling=args.render,
                    debug_sampling=args.debug_sampling,
                    debug_step_log_every=args.debug_sampling_step_log_every,
                    max_sampling_steps_per_attempt=args.max_sampling_steps_per_attempt,
                )
                if ok:
                    accepted_seed = candidate_seed
                    accepted_best_dcpa = best_dcpa
                    accepted_best_tcpa = best_tcpa
                    accepted_attempt = attempt
                    break
                print(
                    f"Episode {ep}: failed attempt={attempt} seed={candidate_seed} steps={sample_steps} reason={fail_reason} "
                    f"(best_dcpa={best_dcpa:.2f}, best_tcpa={best_tcpa:.2f}; "
                    f"need dcpa <= {args.dcpa_threshold:.2f} and tcpa <= {args.tcpa_threshold:.2f})"
                )
                attempt += 1

            if accepted_seed is None:
                print(
                    f"Episode {ep}: no sample found with dcpa <= {args.dcpa_threshold:.2f} and tcpa <= {args.tcpa_threshold:.2f} "
                    f"within {max_tries} tries"
                )
                continue

            # Re-run accepted seed for actual episode playback and strictly verify threshold again.
            _ = env.reset(seed=accepted_seed)
            done = False
            total = 0.0
            run_best_dcpa = float("inf")
            run_best_tcpa = float("inf")
            threshold_hit_before_goal = False
            info: dict[str, float | str | int] = {"reason": "unknown", "dcpa": float("inf"), "tcpa": float("inf")}
            while not done:
                reached_before_step = env.agent_reached or env.target_reached
                action = np.array([0.0, 0.0], dtype=np.float32)
                _, reward, done, info = env.step(action)
                total += reward
                dcpa = float(info.get("dcpa", float("inf")))
                tcpa = float(info.get("tcpa", float("inf")))
                run_best_dcpa = min(run_best_dcpa, dcpa)
                if tcpa > 0.0:
                    run_best_tcpa = min(run_best_tcpa, tcpa)
                # Mirror sampling acceptance rule exactly during post-check replay.
                if (not reached_before_step) and (not env.agent_reached) and (not env.target_reached) and _meets_sampling_thresholds(dcpa, tcpa):
                    threshold_hit_before_goal = True
                if args.render:
                    env.render()

            if not threshold_hit_before_goal:
                print(
                    f"Episode {ep}: rejected post-check for seed={accepted_seed} "
                    f"(best_dcpa={run_best_dcpa:.2f}, best_tcpa={run_best_tcpa:.2f}; "
                    f"thresholds dcpa<={args.dcpa_threshold:.2f}, tcpa<={args.tcpa_threshold:.2f})"
                )
                continue

            print(
                f"Episode {ep}: accepted_seed={accepted_seed} attempt={accepted_attempt} sample_steps={sample_steps} "
                f"sample_best_dcpa={accepted_best_dcpa:.2f} sample_best_tcpa={accepted_best_tcpa:.2f} "
                f"run_best_dcpa={run_best_dcpa:.2f} run_best_tcpa={run_best_tcpa:.2f} "
                f"(thresholds: dcpa<={args.dcpa_threshold:.2f}, tcpa<={args.tcpa_threshold:.2f})"
            )
            print(
                f"  end: reason={info.get('reason', 'unknown')} steps={env.step_idx} return={float(total):.3f} "
                f"final_dcpa={float(info.get('dcpa', float('inf'))):.2f}"
            )
    finally:
        env.close()


def _draw_heading_line(surface, sx: int, sy: int, heading: float, length_px: float, color: tuple[int, int, int]) -> None:
    ex = int(round(sx + length_px * math.cos(heading)))
    ey = int(round(sy + length_px * math.sin(heading)))
    pygame.draw.line(surface, color, (sx, sy), (ex, ey), 2)


def _simulate_target_path(env: SingleTargetFeatureEnv, sx: float, sy: float, heading: float, speed: float, goal_x: float, goal_y: float) -> list[tuple[float, float]]:
    sim = Vessel(sx, sy, heading, speed, goal_x, goal_y, rudder=0.0, throttle=0.0)
    pts: list[tuple[float, float]] = [(sim.x, sim.y)]
    h = env.envp.dt / max(1, env.envp.substeps)
    max_sim_steps = max(2000, int(2.0 * env.max_steps * max(1, env.envp.substeps)))

    for _ in range(max_sim_steps):
        d_goal = math.hypot(goal_x - sim.x, goal_y - sim.y)
        if d_goal <= env.envp.goal_radius:
            break
        rudder_cmd = env._pure_pursuit_rudder_cmd(sim, goal_x, goal_y)
        env._integrate_rudder_heading(sim, rudder_cmd, h)
        sim.speed = clamp(sim.speed, env.envp.target_min_speed, env.envp.target_max_speed)
        travel = min(sim.speed * h, d_goal)
        sim.x += travel * math.cos(sim.h)
        sim.y += travel * math.sin(sim.h)
        pts.append((sim.x, sim.y))
        if travel + 1e-9 >= d_goal:
            break
        if env._outside(sim):
            break
    return pts


def run_heading_range_view(args: argparse.Namespace, envp: EnvParams) -> None:
    if not HAS_PYGAME:
        raise RuntimeError("pygame is required for --view v2-heading-range")

    env = SingleTargetFeatureEnv(envp, RewardParams(), render=False)

    pygame.init()
    w = int(envp.world_w * envp.pixels_per_meter)
    h = int(envp.world_h * envp.pixels_per_meter)
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("V2 Initial Heading Range View")
    clock = pygame.time.Clock()

    cx = 0.5 * envp.world_w
    cy = 0.5 * envp.world_h

    def sx(x: float) -> int:
        return int(round(x * envp.pixels_per_meter))

    def sy(y: float) -> int:
        return int(round(y * envp.pixels_per_meter))

    line_len_px = args.heading_line_length * envp.pixels_per_meter

    start_ang = math.radians(float(args.v2_start_angle_deg))
    start_x = cx + envp.target_outer_radius * math.cos(start_ang)
    start_y = cy + envp.target_outer_radius * math.sin(start_ang)

    to_center = math.atan2(cy - start_y, cx - start_x)
    h_left = to_center - 0.5 * math.pi
    h_right = to_center + 0.5 * math.pi

    min_goal_dist = max(0.0, float(envp.target_min_goal_arc_distance_from_start))
    circle_r = max(1e-9, float(envp.target_outer_radius))
    min_goal_dist = min(min_goal_dist, 2.0 * circle_r)
    chord_delta = 2.0 * math.asin(min_goal_dist / (2.0 * circle_r))

    goal_left_ang = start_ang - chord_delta
    goal_right_ang = start_ang + chord_delta
    goal_left = (cx + envp.target_outer_radius * math.cos(goal_left_ang), cy + envp.target_outer_radius * math.sin(goal_left_ang))
    goal_right = (cx + envp.target_outer_radius * math.cos(goal_right_ang), cy + envp.target_outer_radius * math.sin(goal_right_ang))

    path_speed = float(args.heading_path_speed)
    if path_speed <= 0.0:
        path_speed = 0.5 * (envp.target_min_speed + envp.target_max_speed)

    paths = {
        "left_heading_to_left_goal": _simulate_target_path(env, start_x, start_y, h_left, path_speed, goal_left[0], goal_left[1]),
        "left_heading_to_right_goal": _simulate_target_path(env, start_x, start_y, h_left, path_speed, goal_right[0], goal_right[1]),
        "right_heading_to_left_goal": _simulate_target_path(env, start_x, start_y, h_right, path_speed, goal_left[0], goal_left[1]),
        "right_heading_to_right_goal": _simulate_target_path(env, start_x, start_y, h_right, path_speed, goal_right[0], goal_right[1]),
    }

    def draw_polyline(pts: list[tuple[float, float]], color: tuple[int, int, int]) -> None:
        if len(pts) < 2:
            return
        pix = [(sx(x), sy(y)) for x, y in pts]
        pygame.draw.lines(screen, color, False, pix, 2)

    def draw_frame() -> None:
        screen.fill((17, 58, 92))
        pygame.draw.rect(screen, (170, 170, 170), (0, 0, sx(envp.world_w), sy(envp.world_h)), 2)

        if envp.show_grid:
            step = 50
            for x in range(0, int(envp.world_w) + 1, step):
                pygame.draw.line(screen, (40, 80, 110), (sx(x), 0), (sx(x), sy(envp.world_h)))
            for y in range(0, int(envp.world_h) + 1, step):
                pygame.draw.line(screen, (40, 80, 110), (0, sy(y)), (sx(envp.world_w), sy(y)))

        pygame.draw.circle(screen, (255, 225, 120), (sx(cx), sy(cy)), int(round(envp.target_outer_radius * envp.pixels_per_meter)), 1)

        px, py = sx(start_x), sy(start_y)
        pygame.draw.circle(screen, (255, 160, 160), (px, py), 5)
        _draw_heading_line(screen, px, py, h_left, line_len_px, (100, 255, 120))
        _draw_heading_line(screen, px, py, h_right, line_len_px, (255, 130, 220))

        pygame.draw.circle(screen, (120, 255, 150), (sx(goal_left[0]), sy(goal_left[1])), 5)
        pygame.draw.circle(screen, (255, 160, 230), (sx(goal_right[0]), sy(goal_right[1])), 5)

        draw_polyline(paths["left_heading_to_left_goal"], (0, 220, 120))
        draw_polyline(paths["left_heading_to_right_goal"], (120, 255, 190))
        draw_polyline(paths["right_heading_to_left_goal"], (220, 120, 230))
        draw_polyline(paths["right_heading_to_right_goal"], (255, 170, 245))

        pygame.display.flip()

    draw_frame()

    if args.save_image:
        out = Path(args.save_image)
        out.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(screen, str(out))
        print(f"Saved heading-range image to {out}")

    elapsed = 0.0
    running = True
    while running:
        dt = clock.tick(envp.render_fps) / 1000.0
        elapsed += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        if args.auto_close_seconds > 0.0 and elapsed >= args.auto_close_seconds:
            running = False

    env.close()
    pygame.quit()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    envp = _make_env_params(args)
    if args.view == "episode":
        run_episode_view(args, envp)
    elif args.view == "v2-heading-range":
        run_heading_range_view(args, envp)
    elif args.view == "dcpa-sampled-episode":
        run_dcpa_sampled_episode_view(args)


if __name__ == "__main__":
    main()
