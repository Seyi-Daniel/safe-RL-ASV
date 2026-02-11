#!/usr/bin/env python3
"""
Boat Sector Visualization (feature-data-first simulation)
--------------------------------------------------------

What this gives you right now:
- A simple "open sea" simulation with N boats.
- Boats are initialized with random (x, y, heading, speed) but **same size, max speed, etc**.
- Boats move in straight lines (no turning) until:
  * a collision occurs (episode ends), or
  * any boat leaves the world bounds (episode ends).
- Runs for a configurable number of EPISODES (set at the top).
- Right-hand panel lists, for *each* boat:
  * Its own features (position, speed, heading)
  * A 12-sector (30° each) breakdown of which other boats sit in each sector
    relative to that boat's heading (0° = straight ahead, 180° = directly astern).
  * For each neighbor: relative bearing, absolute position, distance, TCPA, DCPA, and neighbor's speed/heading.

Controls:
- ESC or close window: quit.
- SPACE: pause/resume.
- N: skip to next episode.
- Mouse wheel / PageUp / PageDown: scroll the info panel if it's long.

Requirements:
- Python 3.9+ recommended
- pygame>=2.3 (pip install pygame)

Run:
    python boat_sector_sim.py

If your environment doesn't have fonts, Pygame will fall back to a default bitmap font.
"""

import math
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import pygame


# =============================
# Simulation parameters (edit)
# =============================
WORLD_W = 1400        # "sea" width (world units == pixels)
WORLD_H = 840         # "sea" height
N_BOATS = 7          # number of boats per episode
EPISODES = 100         # how many episodes to run (auto-resets after each)
FPS = 60              # target framerate
SENSOR_RANGE: Optional[float] = None  # None = unlimited. Or set like 400.0

# All boats share these "fundamental" parameters (size/speed/accel/turn)
BOAT_LENGTH = 28.0      # visual length in pixels (rough "size")
BOAT_WIDTH = 10.0       # visual width in pixels
BOAT_MAX_SPEED = 160.0  # px/s
BOAT_MIN_SPEED = 40.0   # px/s
BOAT_ACCEL = 30.0       # (not used yet since we go straight)
BOAT_DECEL = 30.0       # (not used yet)
BOAT_TURN_RATE = math.radians(60.0)  # (not used yet)

# Collision "radius" ~ circle around boat. Slightly smaller than length for visuals
#COLLISION_RADIUS = BOAT_LENGTH * 0.45
COLLISION_RADIUS = BOAT_LENGTH

# UI layout
INFO_PANEL_W = 520     # width of the right-hand info panel
WINDOW_W = WORLD_W + INFO_PANEL_W
WINDOW_H = WORLD_H

# Colors
BG_WATER = (18, 52, 86)
WHITE = (245, 245, 245)
GREY = (180, 180, 180)
GREEN = (70, 200, 120)
BLUE = (90, 160, 255)
RED = (230, 80, 80)
YELLOW = (250, 200, 50)

# --- Visualization toggles ---
SHOW_SECTOR_RAYS = False   # Set to False to hide the 12 sector rays around each boat
SECTOR_RAY_LEN = None     # None -> uses SENSOR_RANGE if set, else 320 px


# =============================
# Utility math
# =============================
def wrap_angle_rad(a: float) -> float:
    """Wrap any angle (rad) to [-pi, pi)."""
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    return a


def wrap_deg_0_360(d: float) -> float:
    """Wrap degrees to [0, 360)."""
    d = d % 360.0
    if d < 0:
        d += 360.0
    return d


def angle_to_deg(a: float) -> float:
    return a * 180.0 / math.pi


def tcpa_dcpa(a: "Boat", b: "Boat") -> Tuple[float, float]:
    """
    Time to Closest Point of Approach (TCPA) and Distance at CPA (DCPA) for constant-velocity motion.
    If relative speed ~ 0, TCPA = 0 and DCPA = current separation.
    """
    rx, ry = (b.x - a.x), (b.y - a.y)
    rvx = math.cos(b.heading) * b.speed - math.cos(a.heading) * a.speed
    rvy = math.sin(b.heading) * b.speed - math.sin(a.heading) * a.speed
    rv2 = rvx * rvx + rvy * rvy
    if rv2 < 1e-9:
        return 0.0, math.hypot(rx, ry)
    tcpa = - (rx * rvx + ry * rvy) / rv2
    if tcpa < 0.0:
        tcpa = 0.0
    cx = rx + rvx * tcpa
    cy = ry + rvy * tcpa
    dcpa = math.hypot(cx, cy)
    return tcpa, dcpa


# =============================
# Boat class
# =============================
@dataclass
class Boat:
    id: int
    x: float
    y: float
    heading: float  # radians, 0 = +x to the right
    speed: float    # px/s (constant for now)

    # shared physical/visual params
    length: float = BOAT_LENGTH
    width: float = BOAT_WIDTH
    max_speed: float = BOAT_MAX_SPEED
    acceleration: float = BOAT_ACCEL
    deceleration: float = BOAT_DECEL
    turn_rate: float = BOAT_TURN_RATE
    mass: float = 1000.0  # arbitrary

    color: Tuple[int, int, int] = BLUE

    def step(self, dt: float):
        """Straight-line motion; no steering or throttle right now."""
        self.x += math.cos(self.heading) * self.speed * dt
        self.y += math.sin(self.heading) * self.speed * dt

    def as_tuple(self) -> Tuple[float, float]:
        return self.x, self.y


# =============================
# Simulation world
# =============================
class World:
    def __init__(self, width: int, height: int, n_boats: int):
        self.w = width
        self.h = height
        self.n_boats = n_boats
        self.boats: List[Boat] = []
        self.episode_idx = 0
        self.random = random.Random()
        self.random.seed()  # OS entropy

    def reset(self):
        """Create n boats with random positions, headings, speeds (no overlaps)."""
        self.boats = []
        attempts = 0
        MAX_ATTEMPTS = 10_000

        for bid in range(self.n_boats):
            placed = False
            while not placed and attempts < MAX_ATTEMPTS:
                attempts += 1
                # Keep a margin from edges so they don't end instantly
                margin = 40
                x = self.random.uniform(margin, self.w - margin)
                y = self.random.uniform(margin, self.h - margin)

                heading = self.random.uniform(-math.pi, math.pi)
                speed = self.random.uniform(BOAT_MIN_SPEED, BOAT_MAX_SPEED)

                # reject if overlapping any existing boat (simple circle test)
                ok = True
                for b in self.boats:
                    dx = x - b.x
                    dy = y - b.y
                    if dx * dx + dy * dy < (2 * COLLISION_RADIUS) ** 2:
                        ok = False
                        break

                if ok:
                    self.boats.append(Boat(id=bid, x=x, y=y, heading=heading, speed=speed))
                    placed = True

            if not placed:
                print("[WARN] Could not place all boats without overlap; continuing with fewer.")
                break

    def update(self, dt: float) -> Tuple[bool, str]:
        """Advance the world by dt. Return (episode_over, reason)."""
        # 1) move boats
        for b in self.boats:
            b.step(dt)

        # 2) check end conditions
        # (a) collision any pair
        n = len(self.boats)
        for i in range(n):
            for j in range(i + 1, n):
                if collide(self.boats[i], self.boats[j]):
                    return True, f"collision: boat {i} ↔ boat {j}"

        # (b) out of bounds
        for b in self.boats:
            if not (0 <= b.x <= self.w and 0 <= b.y <= self.h):
                return True, f"out_of_bounds: boat {b.id}"

        return False, ""

    def compute_sector_map(self) -> Dict[int, Dict[int, List[Tuple[int, float, float, float, float, float, float, float, float]]]]:
        """
        For each boat i, compute which boats fall into each of 12 angular sectors,
        relative to boat i's heading.

        Returns:
            sector_map: dict[boat_id] -> dict[sector_index 0..11] -> list of tuples
                        Each tuple = (other_id, rel_angle_deg, distance, tcpa, dcpa,
                                      other_x, other_y, other_speed, other_heading_deg)
        """
        sector_map: Dict[int, Dict[int, List[Tuple[int, float, float, float, float, float, float, float, float]]]] = {}
        N_SECTORS = 12
        sector_width = 360.0 / N_SECTORS

        for i, a in enumerate(self.boats):
            per_sector: Dict[int, List[Tuple[int, float, float, float, float, float, float, float, float]]] = {k: [] for k in range(N_SECTORS)}

            for j, b in enumerate(self.boats):
                if i == j:
                    continue

                dx = b.x - a.x
                dy = b.y - a.y
                dist = math.hypot(dx, dy)
                if SENSOR_RANGE is not None and dist > SENSOR_RANGE:
                    continue

                # Absolute angle a->b in degrees
                abs_deg = angle_to_deg(math.atan2(dy, dx))
                # Relative to boat a's heading
                rel_deg = wrap_deg_0_360(abs_deg - angle_to_deg(a.heading))

                # TCPA / DCPA
                tcpa, dcpa = tcpa_dcpa(a, b)

                # Sector bin 0..11
                sector = int(rel_deg // sector_width)
                per_sector[sector].append(
                    (b.id, rel_deg, dist, tcpa, dcpa, b.x, b.y, b.speed, angle_to_deg(b.heading))
                )

            # sort neighbors within each sector by distance
            for k in per_sector:
                per_sector[k].sort(key=lambda t: t[2])

            sector_map[a.id] = per_sector

        return sector_map


def collide(a: Boat, b: Boat) -> bool:
    dx = a.x - b.x
    dy = a.y - b.y
    rr = (COLLISION_RADIUS + COLLISION_RADIUS) ** 2
    return dx * dx + dy * dy <= rr


# =============================
# Rendering
# =============================
class Renderer:
    def __init__(self, world: World):
        pygame.init()
        pygame.display.set_caption("Boats — Open Sea (left) + Info Panel (right)")

        self.world = world
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.surface_sim = pygame.Surface((WORLD_W, WORLD_H))
        self.surface_info = pygame.Surface((INFO_PANEL_W, WORLD_H))

        # Font
        self.font = pygame.font.Font(None, 18)
        self.font_small = pygame.font.Font(None, 16)
        self.font_big = pygame.font.Font(None, 24)

        # Info panel scrolling
        self.scroll_y = 0  # positive scrolls up (content moves down)
        self.total_info_height = 0

    def draw(self, episode_idx: int, step_idx: int, dt: float):
        # Left: sea + boats
        self.surface_sim.fill(BG_WATER)
        self._draw_border(self.surface_sim)

        # grid (optional, light)
        self._draw_grid(self.surface_sim)

        for b in self.world.boats:
            self._draw_boat(self.surface_sim, b)

            if SHOW_SECTOR_RAYS:
                self._draw_sector_rays(self.surface_sim, b)

            # draw ID label
            label = self.font_small.render(str(b.id), True, WHITE)
            self.surface_sim.blit(label, (int(b.x) + 8, int(b.y) - 8))

        # Right: info panel with sector listings
        self._draw_info_panel(episode_idx, step_idx, dt)

        # Blit both panels to main screen
        self.screen.blit(self.surface_sim, (0, 0))
        self.screen.blit(self.surface_info, (WORLD_W, 0))
        pygame.display.flip()

    def _draw_border(self, surf: pygame.Surface):
        pygame.draw.rect(surf, GREY, (0, 0, WORLD_W, WORLD_H), width=2)

    def _draw_grid(self, surf: pygame.Surface, spacing: int = 100):
        for x in range(0, WORLD_W, spacing):
            pygame.draw.line(surf, (40, 70, 100), (x, 0), (x, WORLD_H))
        for y in range(0, WORLD_H, spacing):
            pygame.draw.line(surf, (40, 70, 100), (0, y), (WORLD_W, y))

    def _draw_boat(self, surf: pygame.Surface, b: Boat):
        # Draw as oriented triangle pointing along heading
        L = b.length
        W = b.width
        # local verts: nose at +L/2, base at -L/2, width W
        verts_local = [
            (+L * 0.5, 0.0),           # nose
            (-L * 0.5, -W * 0.5),      # port aft
            (-L * 0.5, +W * 0.5),      # starboard aft
        ]
        ch = math.cos(b.heading)
        sh = math.sin(b.heading)
        verts_world = []
        for vx, vy in verts_local:
            wx = b.x + vx * ch - vy * sh
            wy = b.y + vx * sh + vy * ch
            verts_world.append((wx, wy))

        pygame.draw.polygon(surf, b.color, verts_world)
        # optional collision circle
        pygame.draw.circle(surf, (255, 255, 255), (int(b.x), int(b.y)), int(COLLISION_RADIUS), width=1)

    def _draw_sector_rays(self, surf: pygame.Surface, b: Boat, n: int = 12):
        """Draw N equally spaced rays (sector boundaries) emanating from boat b, aligned to its heading."""
        length = (SECTOR_RAY_LEN if SECTOR_RAY_LEN is not None
                  else (SENSOR_RANGE if SENSOR_RANGE is not None else 320.0))
        for k in range(n):
            ang = b.heading + (2.0 * math.pi / n) * k
            x2 = b.x + length * math.cos(ang)
            y2 = b.y + length * math.sin(ang)
            pygame.draw.line(surf, (220, 220, 220), (int(b.x), int(b.y)), (int(x2), int(y2)), 1)

    def _draw_info_panel(self, episode_idx: int, step_idx: int, dt: float):
        sp = self.surface_info
        sp.fill((24, 24, 28))

        pad = 10
        x0 = pad
        y = pad - self.scroll_y

        # Header
        header = self.font_big.render(f"Episode {episode_idx+1}/{EPISODES}  |  Step {step_idx}", True, WHITE)
        sp.blit(header, (x0, y)); y += 28
        sp.blit(self.font.render(f"Boats: {len(self.world.boats)}  |  dt={dt:.3f}s  |  Sensor={SENSOR_RANGE or '∞'}", True, GREY), (x0, y)); y += 22
        y += 6

        # Sector map
        sector_map = self.world.compute_sector_map()

        # Legend
        sp.blit(self.font.render("12-sector bins (relative to each boat's heading):", True, YELLOW), (x0, y)); y += 22
        sp.blit(self.font_small.render("Sector 0: ahead [0°..30°), 1: [30°..60°), ..., 11: [330°..360°)", True, GREY), (x0, y)); y += 18
        sp.blit(self.font_small.render("Per neighbor: id@rel° pos=(x,y) d=dist T=TCPA(s) D=DCPA spd hdg°", True, GREY), (x0, y)); y += 18
        y += 6

        # For each boat, print features + sector contents
        for b in self.world.boats:
            # Features
            sp.blit(self.font.render(
                f"Boat {b.id} — pos=({b.x:6.1f},{b.y:6.1f})  spd={b.speed:5.1f}  hdg={angle_to_deg(b.heading):6.1f}°",
                True, WHITE), (x0, y)); y += 20

            per_sector = sector_map[b.id]

            # summary line: counts per sector
            counts = [len(per_sector[k]) for k in range(12)]
            counts_str = " ".join(f"{c:2d}" for c in counts)
            sp.blit(self.font_small.render(f"  counts: [{counts_str}]", True, GREY), (x0, y)); y += 18

            # details per sector
            for k in range(12):
                boats_here = per_sector[k]
                if boats_here:
                    parts = []
                    # Limit a bit to avoid overlong lines; increase if you prefer
                    for (oid, rel_deg, dist, tcpa, dcpa, bx, by, spd, hdg_deg) in boats_here[:6]:
                        parts.append(
                            f"{oid}@{rel_deg:5.1f}° pos=({bx:5.1f},{by:5.1f}) d={dist:5.1f} "
                            f"T={tcpa:5.1f}s D={dcpa:5.1f} spd={spd:5.1f} hdg={hdg_deg:5.1f}°"
                        )
                    line = f"  {k:2d}: " + "; ".join(parts)
                else:
                    line = f"  {k:2d}: -"  # empty
                sp.blit(self.font_small.render(line, True, (210, 210, 210)), (x0, y)); y += 16

            y += 8  # spacing between boats

        # Track total rendered height for scrolling bounds
        self.total_info_height = y + self.scroll_y

        # Simple scrollbar indicator (right edge)
        visible_h = sp.get_height()
        total_h = max(visible_h, self.total_info_height)
        if total_h > visible_h:
            bar_h = max(30, int(visible_h * (visible_h / total_h)))
            # scroll fraction
            frac = min(1.0, max(0.0, self.scroll_y / (total_h - visible_h)))
            bar_y = int(frac * (visible_h - bar_h))
            pygame.draw.rect(sp, (90, 90, 100), (sp.get_width() - 8, bar_y, 6, bar_h))

    def scroll(self, delta_px: int):
        # Clamp scroll between 0 and max
        visible_h = self.surface_info.get_height()
        total_h = max(visible_h, self.total_info_height)
        max_scroll = max(0, total_h - visible_h)
        self.scroll_y = min(max(self.scroll_y + delta_px, 0), max_scroll)


# =============================
# Main loop
# =============================
def main():
    world = World(WORLD_W, WORLD_H, N_BOATS)
    renderer = Renderer(world)

    clock = pygame.time.Clock()
    episode = 0
    world.reset()
    step_idx = 0
    paused = False

    while True:
        dt = clock.tick(FPS) / 1000.0  # seconds

        # Events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_n:
                    # skip to next episode
                    episode += 1
                    if episode >= EPISODES:
                        print("All episodes complete. Exiting.")
                        pygame.quit()
                        sys.exit(0)
                    world.reset()
                    step_idx = 0
                elif ev.key == pygame.K_PAGEUP:
                    renderer.scroll(-120)  # scroll up
                elif ev.key == pygame.K_PAGEDOWN:
                    renderer.scroll(+120)  # scroll down

            elif ev.type == pygame.MOUSEWHEEL:
                renderer.scroll(-ev.y * 40)  # wheel up -> negative y

        if not paused:
            done, reason = world.update(dt)
            step_idx += 1
            if done:
                print(f"[Episode {episode+1}] ended — {reason}")
                episode += 1
                if episode >= EPISODES:
                    print("All episodes complete. Exiting.")
                    pygame.quit()
                    sys.exit(0)
                world.reset()
                step_idx = 0

        renderer.draw(episode, step_idx, dt)


if __name__ == "__main__":
    main()
