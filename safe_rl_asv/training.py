from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass


@dataclass
class HyperParameters:
    """Environment and initialization hyperparameters."""

    world_width: int = 500
    world_height: int = 500
    pixels_per_meter: float = 2.0
    spawn_ring_radius: float = 45.0
    goal_ring_radius: float = 140.0
    vessel_radius: int = 8
    rng_seed: int | None = None


class TrainingEnvironment:
    """Simple training initialization environment.

    The vessel is always initialized at the center of the world.
    The vessel's orientation is randomized.
    A goal point is sampled at a random angle on the outer ring.
    """

    def __init__(self, hp: HyperParameters):
        self.hp = hp
        self.rng = random.Random(hp.rng_seed)
        self.center = (hp.world_width / 2.0, hp.world_height / 2.0)

        self.agent_position = self.center
        self.agent_heading = 0.0
        self.goal_position = self.center

    def reset(self) -> None:
        self.agent_position = self.center
        self.agent_heading = self.rng.uniform(0.0, 2.0 * math.pi)

        theta = self.rng.uniform(0.0, 2.0 * math.pi)
        self.goal_position = (
            self.center[0] + self.hp.goal_ring_radius * math.cos(theta),
            self.center[1] + self.hp.goal_ring_radius * math.sin(theta),
        )

    def state_summary(self) -> str:
        x, y = self.agent_position
        gx, gy = self.goal_position
        heading_deg = math.degrees(self.agent_heading)
        return (
            f"Agent start: ({x:.2f}, {y:.2f}) | "
            f"heading: {heading_deg:.2f} deg | "
            f"goal: ({gx:.2f}, {gy:.2f})"
        )


def _draw_dotted_circle(surface, color, center, radius, width=1, dots=90):
    import pygame

    cx, cy = center
    for i in range(dots):
        if i % 2 == 0:
            a1 = 2 * math.pi * i / dots
            a2 = 2 * math.pi * (i + 1) / dots
            p1 = (cx + radius * math.cos(a1), cy + radius * math.sin(a1))
            p2 = (cx + radius * math.cos(a2), cy + radius * math.sin(a2))
            pygame.draw.line(surface, color, p1, p2, width)


def render_initialization(env: TrainingEnvironment) -> None:
    """Render the initialization layout using pygame."""

    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            "Rendering requested but pygame is not installed. "
            "Install pygame or run without --render."
        ) from exc

    pygame.init()
    screen = pygame.display.set_mode((env.hp.world_width, env.hp.world_height))
    pygame.display.set_caption("safe-RL-ASV: training initialization preview")

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()
                print(env.state_summary())

        screen.fill((18, 22, 30))

        center = (int(env.center[0]), int(env.center[1]))
        _draw_dotted_circle(screen, (130, 180, 255), center, int(env.hp.spawn_ring_radius), width=2)
        pygame.draw.circle(screen, (80, 120, 190), center, int(env.hp.goal_ring_radius), width=2)

        # goal marker
        pygame.draw.circle(
            screen,
            (255, 215, 0),
            (int(env.goal_position[0]), int(env.goal_position[1])),
            6,
        )

        # vessel marker at center
        pygame.draw.circle(screen, (120, 255, 170), center, env.hp.vessel_radius)

        # heading indicator
        hx = center[0] + int(18 * math.cos(env.agent_heading))
        hy = center[1] + int(18 * math.sin(env.agent_heading))
        pygame.draw.line(screen, (0, 0, 0), center, (hx, hy), 2)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def run_training(hp: HyperParameters, render: bool = False) -> None:
    env = TrainingEnvironment(hp)
    env.reset()
    print(env.state_summary())

    if render:
        render_initialization(env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="safe-RL-ASV training bootstrap")
    parser.add_argument("--render", action="store_true", help="Render initialization preview")
    parser.add_argument("--world-width", type=int, default=500)
    parser.add_argument("--world-height", type=int, default=500)
    parser.add_argument("--pixels-per-meter", type=float, default=2.0)
    parser.add_argument(
        "--spawn-ring-radius",
        type=float,
        default=45.0,
        help="Editable inner (dotted) circle radius around the center vessel",
    )
    parser.add_argument(
        "--goal-ring-radius",
        type=float,
        default=140.0,
        help="Outer ring radius where the goal is sampled",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hp = HyperParameters(
        world_width=args.world_width,
        world_height=args.world_height,
        pixels_per_meter=args.pixels_per_meter,
        spawn_ring_radius=args.spawn_ring_radius,
        goal_ring_radius=args.goal_ring_radius,
        rng_seed=args.seed,
    )
    run_training(hp, render=args.render)


if __name__ == "__main__":
    main()
