"""Simplified environment for rendering deterministic crossing scenarios."""
from __future__ import annotations

import math
from collections import deque
from typing import Iterable, List, Optional, Sequence

from .boat import Boat
from .config import BoatParams, EnvConfig, RudderParams
from .utils import angle_deg, tcpa_dcpa

try:  # pragma: no cover - optional dependency
    import pygame

    HAS_PYGAME = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PYGAME = False


class CrossingScenarioEnv:
    """Lightweight pygame environment tailored to the crossing scenario."""

    def __init__(
        self,
        cfg: EnvConfig = EnvConfig(),
        kin: BoatParams = BoatParams(),
        rudder_cfg: RudderParams = RudderParams(),
    ) -> None:
        self.cfg = cfg
        self.kin = kin
        self.rudder_cfg = rudder_cfg

        self.world_w = float(cfg.world_w)
        self.world_h = float(cfg.world_h)
        self.ppm = float(cfg.pixels_per_meter)
        self._width_px = max(320, int(round(self.world_w * self.ppm)))
        self._height_px = max(320, int(round(self.world_h * self.ppm)))

        self.ships: List[Boat] = []
        self._traces: List[deque] = []
        self._meta: dict = {}
        self.time = 0.0
        self.step_index = 0
        self._debug_overlay: Optional[dict] = None

        self._screen = None
        self._font = None
        self._clock = None
        self.hud_on = bool(cfg.show_hud)

        if self.cfg.render and HAS_PYGAME:
            self._setup_render()

    # ------------------------------------------------------------------
    # Setup and configuration helpers
    # ------------------------------------------------------------------
    def _setup_render(self) -> None:
        if not HAS_PYGAME:
            return
        pygame.init()
        self._width_px = max(320, int(round(self.world_w * self.ppm)))
        self._height_px = max(320, int(round(self.world_h * self.ppm)))
        self._screen = pygame.display.set_mode((self._width_px, self._height_px))
        pygame.display.set_caption("Crossing Scenario Renderer")
        self._font = pygame.font.Font(None, 18)
        self._clock = pygame.time.Clock()

    def enable_render(self) -> None:
        if not self._screen and HAS_PYGAME:
            self.cfg.render = True
            self._setup_render()

    def set_debug_overlay(self, overlay: Optional[dict]) -> None:
        self._debug_overlay = overlay

    def close(self) -> None:
        if self._screen and HAS_PYGAME:
            pygame.quit()
        self._screen = None
        self._font = None
        self._clock = None

    def sx(self, x_m: float) -> int:
        return int(round(x_m * self.ppm))

    def sy(self, y_m: float) -> int:
        return self._height_px - int(round(y_m * self.ppm))

    # ------------------------------------------------------------------
    # Scenario management
    # ------------------------------------------------------------------
    def reset_from_states(
        self,
        states: Sequence[dict],
        meta: Optional[dict] = None,
    ) -> None:
        """Replace the fleet with vessels described by ``states``."""

        self.ships.clear()
        self.time = 0.0
        self.step_index = 0
        self._meta = dict(meta or {})

        self._traces = [deque() for _ in states]
        for idx, spec in enumerate(states):
            goal = spec.get("goal")
            if goal is None:
                gx = spec.get("goal_x")
                gy = spec.get("goal_y")
                if gx is not None and gy is not None:
                    goal = (gx, gy)
            boat = Boat(
                boat_id=idx,
                x=float(spec["x"]),
                y=float(spec["y"]),
                heading=float(spec["heading"]),
                speed=float(spec.get("speed", 0.0)),
                kin=self.kin,
                rudder_cfg=self.rudder_cfg,
                goal=goal,
                name=spec.get("name"),
            )
            self.ships.append(boat)

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    def apply_actions(
        self, actions: Optional[Sequence[Optional[tuple[float, int]]]] = None
    ) -> None:
        if not self.ships:
            return

        if actions is None:
            actions = [None] * len(self.ships)

        for boat in self.ships:
            boat.begin_step()

        for boat, action in zip(self.ships, actions):
            if action is not None:
                boat.apply_action(action)

    def advance_applied_actions(
        self, dt: Optional[float] = None, substeps: Optional[int] = None
    ) -> None:
        if not self.ships:
            return

        dt_val = float(self.cfg.dt if dt is None else dt)
        substeps_val = max(1, int(self.cfg.substeps if substeps is None else substeps))
        h = dt_val / substeps_val

        for _ in range(substeps_val):
            for boat in self.ships:
                boat.integrate(h)
            self.time += h
        self.step_index += 1

    def step(self, actions: Optional[Sequence[Optional[tuple[float, int]]]] = None) -> None:
        if not self.ships:
            return

        self.apply_actions(actions)
        self.advance_applied_actions()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _handle_events(self) -> None:
        if not HAS_PYGAME:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    self.hud_on = not self.hud_on
                elif event.key == pygame.K_g:
                    self.cfg.show_grid = not self.cfg.show_grid
                elif event.key == pygame.K_t:
                    self.cfg.show_trails = not self.cfg.show_trails
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise SystemExit

    def _draw_arrow(
        self,
        surf,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        width: int = 2,
        head_len: float = 8.0,
        head_width: float = 5.0,
        direction: str = "forward",
    ) -> None:
        """Draw a simple line with an optional arrowhead.

        ``direction`` selects where to place the arrow tip:
        ``"forward"`` puts it at ``end`` (default),
        ``"backward"`` at ``start``, and ``"none"`` omits it.
        """

        if not HAS_PYGAME:
            return

        pygame.draw.line(surf, color, start, end, width)

        if direction == "none":
            return

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return

        ux, uy = dx / length, dy / length
        if direction == "backward":
            tip = start
        else:
            tip = end

        left = (tip[0] - ux * head_len - uy * head_width, tip[1] - uy * head_len + ux * head_width)
        right = (
            tip[0] - ux * head_len + uy * head_width,
            tip[1] - uy * head_len - ux * head_width,
        )
        pygame.draw.polygon(surf, color, [tip, left, right])

    def _draw_ship(self, surf, boat: Boat, color) -> None:
        if not HAS_PYGAME:
            return
        Lm, Wm = self.kin.length, self.kin.width
        verts_local = [(+0.5 * Lm, 0.0), (-0.5 * Lm, -0.5 * Wm), (-0.5 * Lm, +0.5 * Wm)]
        ch, sh = math.cos(boat.h), math.sin(boat.h)
        pts = []
        for vx_m, vy_m in verts_local:
            wx = boat.x + vx_m * ch - vy_m * sh
            wy = boat.y + vx_m * sh + vy_m * ch
            pts.append((self.sx(wx), self.sy(wy)))
        pygame.draw.polygon(surf, color, pts)
        radius = max(1, int(round(0.5 * math.hypot(Lm, Wm) * self.ppm)))
        pygame.draw.circle(
            surf, (255, 255, 255), (self.sx(boat.x), self.sy(boat.y)), radius * 2.2, 1
        )

        heading_start = pts[0]
        heading_len = 5 * Lm
        heading_end = (
            self.sx(boat.x + (0.5 * Lm + heading_len) * ch),
            self.sy(boat.y + (0.5 * Lm + heading_len) * sh),
        )
        pygame.draw.line(surf, (255, 255, 255), heading_start, heading_end, 2)

        stern_x = boat.x - 0.5 * Lm * ch
        stern_y = boat.y - 0.5 * Lm * sh

        # Draw commanded rudder arrow relative to the boat heading.
        cmd = getattr(boat, "last_rudder_cmd", 0.0)
        if abs(cmd) > 1e-3:
            # perpendicular to heading: port is left (positive cmd), starboard is right (negative cmd)
            perp_x = -math.sin(boat.h)
            perp_y = math.cos(boat.h)
            direction_sign = 1.0 if cmd > 0 else -1.0
            # choose a base arrow length and scale by command magnitude
            base_len = 12.0  # pixels in world space (can adjust)
            arrow_len = 3 * base_len * abs(cmd)
            base_x, base_y = stern_x, stern_y
            end_x = base_x + direction_sign * arrow_len * perp_x
            end_y = base_y + direction_sign * arrow_len * perp_y
            # draw arrow from the stern (screen coordinates)
            self._draw_arrow(
                surf,
                (self.sx(base_x), self.sy(base_y)),
                (20 + self.sx(end_x), 20 + self.sy(end_y)),
                (235, 200, 120),
                width=2,
                direction="forward",
            )

        rudder_len = 5 * Lm
        # Positive boat.rudder corresponds to a port turn (positive yaw).
        # Draw the rudder deflection so that positive values show up on the
        # "port" side of the wake in the viewer.
        rudder_angle = boat.h - boat.rudder
        rudder_ch, rudder_sh = math.cos(rudder_angle), math.sin(rudder_angle)
        rudder_start = (self.sx(stern_x), self.sy(stern_y))
        rudder_end = (
            self.sx(stern_x - rudder_len * rudder_ch),
            self.sy(stern_y - rudder_len * rudder_sh),
        )
        pygame.draw.line(surf, (235, 200, 120), rudder_start, rudder_end, 2)

        dx = rudder_end[0] - rudder_start[0]
        dy = rudder_end[1] - rudder_start[1]
        length = math.hypot(dx, dy)
        if length >= 1e-6:
            ux, uy = dx / length, dy / length

            if boat.last_thr in (1, 2):
                mid_x = 0.5 * (rudder_start[0] + rudder_end[0])
                mid_y = 0.5 * (rudder_start[1] + rudder_end[1])
                arrow_len = 14.0
                p1 = (
                    mid_x - 0.5 * arrow_len * ux,
                    mid_y - 0.5 * arrow_len * uy,
                )
                p2 = (
                    mid_x + 0.5 * arrow_len * ux,
                    mid_y + 0.5 * arrow_len * uy,
                )

                # Draw acceleration arrow along the rudder line (midpoint), pointing
                # in the direction of motion when accelerating and opposite when
                # decelerating. This makes it visually obvious when the throttle
                # is adding or removing speed.
                if boat.last_thr == 1:  # acceleration
                    start, end = p1, p2
                else:  # deceleration
                    start, end = p2, p1
                self._draw_arrow(
                    surf,
                    (int(round(start[0])), int(round(start[1]))),
                    (int(round(end[0])), int(round(end[1]))),
                    (255, 255, 255),
                    width=2,
                    direction="forward",
                )
        if self._font:
            name = boat.name or ("ASV" if boat.id == 0 else "Target Vessel")
            label = self._font.render(name, True, (255, 255, 255))
            surf.blit(label, (self.sx(boat.x) + 8, self.sy(boat.y) - 8))

    def _draw_crossing_marker(self, surf) -> None:
        if not HAS_PYGAME:
            return
        cx = float(self._meta.get("cross_x", self.world_w / 2.0))
        cy = float(self._meta.get("cross_y", self.world_h / 2.0))
        px = self.sx(cx)
        py = self.sy(cy)
        pygame.draw.line(surf, (255, 230, 120), (px - 10, py), (px + 10, py), 2)
        pygame.draw.line(surf, (255, 230, 120), (px, py - 10), (px, py + 10), 2)

    def _draw_grid(self, surf) -> None:
        if not self.cfg.show_grid or not HAS_PYGAME:
            return
        step = 40.0
        x = 0.0
        while x <= self.world_w + 1e-6:
            pygame.draw.line(
                surf,
                (45, 70, 110),
                (self.sx(x), 0),
                (self.sx(x), self._height_px),
            )
            x += step
        y = 0.0
        while y <= self.world_h + 1e-6:
            py = self.sy(y)
            pygame.draw.line(
                surf,
                (45, 70, 110),
                (0, py),
                (self._width_px, py),
            )
            y += step

    def _draw_goals(self, surf) -> None:
        if not HAS_PYGAME:
            return
        goal_colors = [
            (240, 200, 80),
            (120, 220, 160),
            (230, 120, 200),
            (180, 220, 240),
        ]
        outline = (25, 25, 25)
        radius = 6
        for boat, color in zip(self.ships, goal_colors):
            if boat.goal_x is None or boat.goal_y is None:
                continue
            gx = self.sx(boat.goal_x)
            gy = self.sy(boat.goal_y)
            pygame.draw.circle(surf, color, (gx, gy), radius)
            pygame.draw.circle(surf, outline, (gx, gy), radius, 2)

    def _draw_trails(self, surf) -> None:
        if not self.cfg.show_trails or not HAS_PYGAME:
            return
        for trace in self._traces:
            if len(trace) >= 2:
                pygame.draw.lines(surf, (220, 220, 220), False, list(trace), 1)

    def _draw_hud(self, surf) -> None:
        if not HAS_PYGAME or not self.hud_on or not self._font:
            return
        pad = 8
        line_spacing = 20
        fps = self._clock.get_fps() if self._clock else 0.0

        lines = [
            f"FPS {fps:5.1f}   step {self.step_index}   t {self.time:6.2f}s",
        ]

        if self._debug_overlay:
            step_info = self._debug_overlay
            lines.append(
                 " Rudder_cmd: "
                f"{step_info.get('rudder_cmd_for_arrow', 0.0) * 35:+5.2f}  "
                f"step: {step_info.get('step', self.step_index)}"
            )
        scenario_kind = self._meta.get("scenario_kind")
        if scenario_kind:
            lines.append(f"Scenario kind: {scenario_kind}")
        bearing = self._meta.get("bearing")
        if bearing is not None:
            frame = self._meta.get("bearing_frame", "agent")
            if frame == "agent":
                lines.append(f"Requested stand-on bearing: {bearing:6.2f}°")
            else:
                lines.append(
                    f"Requested agent bearing (stand-on frame): {bearing:6.2f}°"
                )
        if self.ships:
            agent = self.ships[0]
            stand_on = self.ships[1] if len(self.ships) > 1 else None

            lines.extend(
                [
                    "Agent Vessel:",
                    f"  Pos ({agent.x:6.1f}, {agent.y:6.1f}) m",
                    f"  Hdg {angle_deg(agent.h):6.1f}°   Spd {agent.u:4.1f} m/s",
                ]
            )

            if stand_on:
                lines.extend(
                    [
                        "Stand-on Vessel:",
                        f"  Pos ({stand_on.x:6.1f}, {stand_on.y:6.1f}) m",
                        f"  Hdg {angle_deg(stand_on.h):6.1f}°   Spd {stand_on.u:4.1f} m/s",
                    ]
                )

                tcpa, dcpa = tcpa_dcpa(agent.snapshot(), stand_on.snapshot())
                tcpa_text = "∞" if math.isinf(tcpa) else f"{tcpa:6.2f} s"
                lines.extend(
                    [
                        "CPA Estimates:",
                        f"  TCPA {tcpa_text}   DCPA {dcpa:5.2f} m",
                    ]
                )

        width = max(self._font.size(text)[0] for text in lines) + 2 * pad
        height = len(lines) * line_spacing + 2 * pad
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        for idx, text in enumerate(lines):
            img = self._font.render(text, True, (235, 235, 235))
            panel.blit(img, (pad, pad + idx * line_spacing))
        surf.blit(panel, (self._width_px - width - 10, 10))

    def render(self) -> None:
        if not self._screen:
            return
        self._handle_events()
        surf = self._screen
        surf.fill((15, 40, 70))
        pygame.draw.rect(
            surf,
            (180, 180, 200),
            (0, 0, self._width_px, self._height_px),
            2,
        )

        self._draw_grid(surf)
        self._draw_goals(surf)

        for trace, boat in zip(self._traces, self.ships):
            trace.append((self.sx(boat.x), self.sy(boat.y)))

        self._draw_trails(surf)

        colors = [(90, 160, 255), (70, 200, 120), (230, 120, 90), (200, 200, 80)]
        for boat, color in zip(self.ships, colors):
            self._draw_ship(surf, boat, color)

        self._draw_hud(surf)
        pygame.display.flip()
        if self._clock:
            self._clock.tick(60)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> List[dict]:
        """Return a copy of the observable state for each vessel."""

        return [boat.snapshot() for boat in self.ships]

    def run_scenario(
        self,
        steps: int,
        actions: Optional[Iterable[Sequence[Optional[int]]]] = None,
    ) -> None:
        """Advance the simulation for ``steps`` frames, rendering each one."""

        if actions is None:
            actions_iter = (None for _ in range(steps))
        else:
            actions_iter = iter(actions)

        for step_actions in actions_iter:
            self.step(step_actions)
            self.render()
