#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np

from environment import HAS_PYGAME, SingleTargetFeatureEnv
from hyperparameters import EnvParams, RewardParams

if HAS_PYGAME:
    import pygame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lightweight simulation sandbox for manual behavior checks")
    p.add_argument("--view", choices=("episode", "v2-heading-range"), default="episode")

    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--render", action="store_true", help="enable pygame visualization")
    p.add_argument("--episode-seconds", type=float, default=60.0)
    p.add_argument("--require-reset-viable-takeover-path", action="store_true", help="Require reset sampling to find a takeover-viable/risky path")
    p.add_argument("--allow-any-reset-path", dest="require_reset_viable_takeover_path", action="store_false", help="Disable reset-time viability filtering")
    p.set_defaults(require_reset_viable_takeover_path=EnvParams().require_reset_viable_takeover_path)

    p.add_argument("--enable-no-takeover-early-done", action="store_true", help="Terminate early when no takeover path is detected")
    p.add_argument("--disable-no-takeover-early-done", dest="enable_no_takeover_early_done", action="store_false", help="Disable early cutoff for no-takeover episodes")
    p.set_defaults(enable_no_takeover_early_done=EnvParams().enable_no_takeover_early_done)
    p.add_argument(
        "--target-min-goal-arc-distance",
        "--target-max-goal-arc-distance",
        "--target-max-goal-distance",
        dest="target_min_goal_arc_distance",
        type=float,
        default=EnvParams().target_min_goal_arc_distance_from_start,
        help="Minimum required vessel-2 start->goal arc distance along the big circle",
    )

    # v2-heading-range view options
    p.add_argument("--heading-samples", type=int, default=36, help="How many possible vessel-2 start points to draw")
    p.add_argument("--heading-line-length", type=float, default=28.0, help="Length (meters) of each heading-boundary line")
    p.add_argument("--auto-close-seconds", type=float, default=0.0, help="Auto-close window after N seconds (0 = wait until closed)")
    p.add_argument("--save-image", type=str, default="", help="Optional image path to save current heading-range view")
    return p.parse_args()


def _make_env_params(args: argparse.Namespace) -> EnvParams:
    return EnvParams(
        episode_seconds=args.episode_seconds,
        seed=args.seed,
        target_min_goal_arc_distance_from_start=args.target_min_goal_arc_distance,
        require_reset_viable_takeover_path=args.require_reset_viable_takeover_path,
        enable_no_takeover_early_done=args.enable_no_takeover_early_done,
    )


def run_episode_view(args: argparse.Namespace, envp: EnvParams) -> None:
    env = SingleTargetFeatureEnv(envp, RewardParams(), render=args.render)
    try:
        for ep in range(1, args.episodes + 1):
            _ = env.reset(seed=args.seed + ep)
            v2_start = env.target_start_pos
            v2_goal = (env.target.goal_x, env.target.goal_y)
            start_angle = float(np.arctan2(v2_start[1] - env.start_y, v2_start[0] - env.start_x))
            goal_angle = float(np.arctan2(v2_goal[1] - env.start_y, v2_goal[0] - env.start_x))
            v2_goal_arc_dist = float(env.envp.target_outer_radius * env._arc_gap(start_angle, goal_angle))
            print(
                f"Episode {ep}: vessel2 start={v2_start}, goal={v2_goal}, "
                f"arc_distance={v2_goal_arc_dist:.2f} m (min={args.target_min_goal_arc_distance:.2f} m)"
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
                f"return={float(total):.3f} target_goal_distance={float(info.get('target_goal_distance', -1.0)):.3f}"
            )
    finally:
        env.close()


def _draw_heading_line(surface, sx: int, sy: int, heading: float, length_px: float, color: tuple[int, int, int]) -> None:
    ex = int(round(sx + length_px * math.cos(heading)))
    ey = int(round(sy + length_px * math.sin(heading)))
    pygame.draw.line(surface, color, (sx, sy), (ex, ey), 2)


def run_heading_range_view(args: argparse.Namespace, envp: EnvParams) -> None:
    if not HAS_PYGAME:
        raise RuntimeError("pygame is required for --view v2-heading-range")

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
    sample_count = max(3, int(args.heading_samples))

    def draw_frame() -> None:
        screen.fill((17, 58, 92))
        pygame.draw.rect(screen, (170, 170, 170), (0, 0, sx(envp.world_w), sy(envp.world_h)), 2)

        if envp.show_grid:
            step = 50
            for x in range(0, int(envp.world_w) + 1, step):
                pygame.draw.line(screen, (40, 80, 110), (sx(x), 0), (sx(x), sy(envp.world_h)))
            for y in range(0, int(envp.world_h) + 1, step):
                pygame.draw.line(screen, (40, 80, 110), (0, sy(y)), (sx(envp.world_w), sy(y)))

        pygame.draw.circle(
            screen,
            (255, 225, 120),
            (sx(cx), sy(cy)),
            int(round(envp.target_outer_radius * envp.pixels_per_meter)),
            1,
        )

        for i in range(sample_count):
            ang = (2.0 * math.pi * i) / sample_count
            x = cx + envp.target_outer_radius * math.cos(ang)
            y = cy + envp.target_outer_radius * math.sin(ang)
            to_center = math.atan2(cy - y, cx - x)
            h_lo = to_center - 0.5 * math.pi
            h_hi = to_center + 0.5 * math.pi
            px, py = sx(x), sy(y)
            pygame.draw.circle(screen, (255, 160, 160), (px, py), 3)
            _draw_heading_line(screen, px, py, h_lo, line_len_px, (100, 255, 120))
            _draw_heading_line(screen, px, py, h_hi, line_len_px, (255, 130, 220))

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


if __name__ == "__main__":
    main()
