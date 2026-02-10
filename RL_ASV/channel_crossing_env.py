import math
import random
from typing import List, Tuple

import numpy as np
import pygame
import torch
from gymnasium import Env, spaces
from gymnasium.utils import seeding


class BaseBoat:
    def __init__(self, x: float, y: float, speed: float, heading: float):
        self.x = x
        self.y = y
        self.speed = speed
        self.heading = heading  # radians

    def step(self, dt: float):
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt


class ControlledBoat(BaseBoat):
    def __init__(
        self,
        x: float,
        y: float,
        speed: float,
        max_speed: int,
        heading: float,
        turn_rate: float,
        acceleration: float
    ):
        super().__init__(
            x,
            y,
            speed,
            heading
        )
        self.max_speed = max_speed
        self.turn_rate = turn_rate
        self.acceleration = acceleration

    def act(self, action: int, dt: float):
        # Steer: 0 = forward, 1 = turn right, 2 = turn left, 
        # Throttle: 0 = coast, 1 = accelerate, 2 = decelerate
        
        steer, throttle = action

        #steer
        if steer == 1:
            if self.speed > 0:
                self.heading -= self.turn_rate * dt
        elif steer == 2:
            if self.speed > 0:
                self.heading += self.turn_rate * dt
        # normalize
        self.heading = (self.heading + math.pi) % (2*math.pi) - math.pi

        #update speed
        if throttle == 1:
            self.speed = min(self.speed + self.acceleration * dt, self.max_speed)
        elif throttle == 2:
            self.speed = max(self.speed - self.acceleration * dt, 0.0)

        self.step(dt)

class TrafficBoat(BaseBoat):
    def __init__(self, x: float, y: float, speed: float, direction: str):
        heading = 0.0 if direction == "east" else math.pi
        super().__init__(x, y, speed, heading)
        self.direction     = direction         # "east" or "west"
        self.active        = True              # is this boat currently in play?
        self.respawn_timer = 0.0               # if inactive, time until respawn (steps)



class ChannelCrossingEnv(Env):
    """
    Boat tries to cross channel from below without colliding
    with constant-velocity traffic boats going east/west in the channel.
    
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Local‐space triangle verts
    BOAT_VERTS = np.array([
        [ +3.0,  0.0 ],
        [ -3.0, -2.0 ],
        [ -3.0, +2.0 ]
    ], dtype=float)

    BOAT_COLLISION_RADIUS = max(
        math.hypot(vx, vy) for vx, vy in BOAT_VERTS
    )

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        traffic_low: float = 30.0,
        traffic_high: float = 70.0,
        channel_padding: float = 10.0,
        n_traffic: int = 6,
        # — realistic real‑world vessel speeds (m/s) — 1 knot = 0.514 m/s
        traffic_speed: float = 8.0 * 0.514, # Tugboat cruising speed: ~8 knots → ~4.1 m/s
        agent_speed: float   = 40.0 * 0.514, # Speedboat cruising speed: ~40 knots → ~20.6 m/s
        turn_rate = math.pi / 3,  # ≈ 1.047 rad/sec (~60°/sec)
        agent_acceleration: int = 2,
        max_steps: int = 500,
        render_mode: str = None,
        window_name: str = "Channel Crossing"
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.traffic_low = traffic_low
        self.traffic_high = traffic_high
        self.channel_padding = channel_padding
        self.channel_low = self.traffic_low - self.channel_padding
        self.channel_high = self.traffic_high + self.channel_padding
        self.goal_x = None
        self.goal_y = None
        self.n_traffic = n_traffic
        self.agent_speed = agent_speed
        self.turn_rate = turn_rate
        self.agent_acceleration = agent_acceleration
        self.traffic_speed = traffic_speed
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.window_name = window_name

        # boat‐shape half-axes (from BOAT_VERTS)
        verts = self.BOAT_VERTS
        self.boat_half_length = float(np.max(verts[:,0]))        # forward/back half-length
        self.boat_half_width = float(np.max(np.abs(verts[:,1])))# port/star half-width

        # reward-shaping hyperparams
        self.living_reward = 0.01

        # rendering
        self.screen      = None
        self.screen_size = 600
        self.scale       = self.screen_size / self.width
        self.clock       = None

        # spaces
        self.action_space = spaces.Discrete(3*3)
        obs_dim = 3 + 4 * self.n_traffic
        self.observation_space = spaces.Box(
            low=-np.inf*np.ones(obs_dim, dtype=np.float32),
            high= np.inf*np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32
        )

        # placeholders
        self.agent          = None
        self.traffic        = []
        self.prev_distance  = 0.0
        self.prev_y         = 0.0
        self.steps          = 0

        # — fixed time‐step parameters —
        self.fps = self.metadata["render_fps"]
        self.dt  = 1.0 / float(self.fps)

    def decode_action(self, i):
        steer  = i // 3       # 0,1,2
        throttle = i % 3      # 0,1,2
        return steer, throttle

    def _ellipse_params(
        self,
        boat: BaseBoat,
        margin: float = 1.2,
        base_offset_ratio: float = 0.2
    ) -> Tuple[float, float, float, float]:
        """
        Returns:
            a, b   = half-axes in world units
            cx, cy = world-space ellipse center
        """
        # 1) shape-based half-axes + margin
        a = self.boat_half_length * margin
        b = self.boat_half_width * margin

        # 2) shift forward by a * base_offset_ratio
        off = -base_offset_ratio * a
        cx = boat.x + off * math.cos(boat.heading)
        cy = boat.y + off * math.sin(boat.heading)

        return a, b, cx, cy

    def _draw_halo(
        self,
        surf,
        boat: BaseBoat,
        color=(255,0,0,80),
        width: int = 2
    ):
        # use the exact same params
        a, b, cx, cy = self._ellipse_params(boat)

        # size in pixels
        w_px = int(2 * a * self.scale)
        h_px = int(2 * b * self.scale)

        # ellipse on its own transparent surface
        halo = pygame.Surface((w_px, h_px), flags=pygame.SRCALPHA)
        pygame.draw.ellipse(halo, color, halo.get_rect(), width=width)

        # rotate CCW by +heading (degrees)
        rotated = pygame.transform.rotate(halo, math.degrees(boat.heading))

        # convert world‐space center to pixel coords
        px = cx * self.scale
        py = (self.height - cy) * self.scale

        # blit so that rotated is centered on (px,py)
        rect = rotated.get_rect(center=(px, py))
        surf.blit(rotated, rect)

    def _check_halo_overlap(
        self,
        boat1: BaseBoat,
        boat2: BaseBoat,
    ) -> bool:
        """
        Returns True if the two ovals (halos) around boat1 and boat2 overlap.
        Each halo is the same shape (a,b) and uses the same forward/back offset.
        """
        # get (a,b, cx,cy) for both boats
        a1, b1, cx1, cy1 = self._ellipse_params(boat1)
        a2, b2, cx2, cy2 = self._ellipse_params(boat2)

        # Minkowski sum of the two ellipses is another ellipse with half-axes a1+a2, b1+b2
        a_comb = a1 + a2
        b_comb = b1 + b2

        # relative center of boat2 in boat1 frame
        dx = cx2 - cx1
        dy = cy2 - cy1
        ch = math.cos(boat1.heading)
        sh = math.sin(boat1.heading)
        x_prime =  ch*dx + sh*dy
        y_prime = -sh*dx + ch*dy

        # overlap test
        return (x_prime/a_comb)**2 + (y_prime/b_comb)**2 <= 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = seeding.np_random(seed)
        self.steps = 0

        # --- Random goal at channel
        self.goal_x = self.np_random.uniform(0, self.width)
        self.goal_y = self.channel_high + self.channel_padding

        # --- spawn agent boat at random x along channel bottom ---
        agent_x = self.np_random.uniform(0.0, self.width)
        agent_y = self.channel_low - self.channel_padding
        agent_heading = self.np_random.uniform(-math.pi, math.pi)
        self.agent = ControlledBoat(
            x=agent_x,
            y=agent_y,
            speed=self.agent_speed,
            max_speed=self.agent_speed,
            heading=agent_heading,
            turn_rate=self.turn_rate,
            acceleration=self.agent_acceleration
        )

        # --- spawn traffic boats at random x along each lane ---
        traffic_ys  = np.linspace(self.traffic_low, self.traffic_high, self.n_traffic)
        traffic_mid = self.n_traffic // 2

        self.traffic = []
        for n, traffic_y in enumerate(traffic_ys):
            direction = "west" if n < traffic_mid else "east"
            traffic_x = self.np_random.uniform(0.0, self.width)
            traffic_boat = TrafficBoat(traffic_x, traffic_y, self.traffic_speed, direction)
            traffic_boat.active = True
            traffic_boat.respawn_timer = 0
            self.traffic.append(traffic_boat)

        self.prev_distance = math.hypot(self.goal_x - self.agent.x, self.goal_y - self.agent.y)
        self.prev_y = self.agent.y

        return self._get_obs(), {}

    def step(self, action: int):
        self.steps += 1

        steer, throttle = self.decode_action(action)
        self.agent.act((steer, throttle), self.dt)

        # respawn delays in steps (e.g. ~1–3 s at 30 FPS → 30–90 steps)
        min_delay, max_delay = 30, 90
        # how far off-screen before we deactivate
        margin = 3.0  # world‐units roughly = boat length

        # --- update each traffic boat ---
        for tb in self.traffic:
            if tb.active:
                tb.step(self.dt)
                # only deactivate once it's fully off either side
                if tb.x < -margin or tb.x > self.width + margin:
                    tb.active        = False
                    tb.respawn_timer = self.np_random.integers(min_delay, max_delay)
            else:
                # countdown to respawn
                tb.respawn_timer -= 1
                if tb.respawn_timer <= 0:
                    tb.active = True
                    # re-enter at correct boundary
                    tb.x = 0.0 if tb.direction == "east" else self.width
                    # y & heading remain the same

        # Calculate change in distance to the goal and top of channel
        distance = math.hypot(self.goal_x - self.agent.x, self.goal_y - self.agent.y)
        step_reward_distance = -1 if distance >= self.prev_distance else 1

        step_reward_dy = 0.0
        if (self.agent.y < self.channel_high):
            dy = self.agent.y - self.prev_y
            step_reward_dy = -0.1 if dy < 0 else 0.1

        reward = step_reward_distance - self.living_reward
        
        self.prev_distance = distance
        self.prev_y = self.agent.y

        # --- terminal checks (unchanged) ---
        done = False

        for tb in self.traffic:
            if not tb.active:
                continue
            if self._check_halo_overlap(self.agent, tb):
                reward -= 50.0
                done = True
                break

        if not done and distance < 5.0:
            reward += 50.0
            done = True
        if not done and (self.agent.x<0 or self.agent.x>self.width or self.agent.y<0 or self.agent.y>self.height):
            done = True
        if not done and self.steps >= self.max_steps:
            done = True

        return self._get_obs(), float(reward), done, False, {}

    def _get_obs(self) -> np.ndarray:
        obs = [self.agent.x, self.agent.y, self.agent.heading]
        for tb in self.traffic:
            rx = tb.x - self.agent.x
            ry = tb.y - self.agent.y
            rvx = tb.speed*math.cos(tb.heading) - self.agent.speed*math.cos(self.agent.heading)
            rvy = tb.speed*math.sin(tb.heading) - self.agent.speed*math.sin(self.agent.heading)
            obs += [rx, ry, rvx, rvy]
        return np.array(obs, dtype=np.float32)

    def _compute_tcpa_dcpa(self, a: BaseBoat, b: BaseBoat) -> Tuple[float, float]:
        rx, ry = b.x - a.x, b.y - a.y
        rvx = b.speed*math.cos(b.heading) - a.speed*math.cos(a.heading)
        rvy = b.speed*math.sin(b.heading) - a.speed*math.sin(a.heading)
        rv2 = rvx*rvx + rvy*rvy
        if rv2 < 1e-8:
            return 0.0, math.hypot(rx, ry)
        tcpa = max(0.0, - (rx*rvx + ry*rvy)/rv2)
        cx, cy = rx + rvx*tcpa, ry + rvy*tcpa
        return tcpa, math.hypot(cx, cy)

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption(self.window_name)
                self.clock = pygame.time.Clock()
            surface = self.screen
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.close()

        elif self.render_mode == "rgb_array":
            surface = pygame.Surface((self.screen_size, self.screen_size))
        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")
        


        # draw background, start & finish line
        surface.fill((255,255,255))

        line_y1 = int((self.height - self.channel_low)*self.scale)
        line_y2 = int((self.height - self.channel_high)*self.scale)
        pygame.draw.line(surface, (200,0,0), (0,line_y1), (self.screen_size, line_y1), 3)
        pygame.draw.line(surface, (200,0,0), (0,line_y2), (self.screen_size, line_y2), 3)

        # --- Draw goal ---
        goal_px = int(self.goal_x * self.scale)
        goal_py = int((self.height - self.goal_y) * self.scale)
        pygame.draw.circle(surface, (0, 0, 0), (goal_px, goal_py), 7)

        # draw halos
        for boat in [*self.traffic, self.agent]:
            if getattr(boat, "active", True):
                self._draw_halo(surface, boat)

        # draw traffic (blue)
        for tb in self.traffic:
            self._draw_boat(surface, tb.x, tb.y, tb.heading, (0,0,200))
        # draw agent (green)
        self._draw_boat(surface, self.agent.x, self.agent.y, self.agent.heading, (0,200,0))

        # --- capture the image buffer ---
        arr = pygame.surfarray.array3d(surface)      # (W,H,3)
        frame = np.transpose(arr, (1,0,2))           # → (H,W,3)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        
        return frame

    def _draw_boat(self, surf, bx, by, heading, color):
        verts = self.BOAT_VERTS
        pts = []
        for vx, vy in verts:
            wx = bx + vx*math.cos(heading) - vy*math.sin(heading)
            wy = by + vx*math.sin(heading) + vy*math.cos(heading)
            px = int(wx * self.scale)
            py = int((self.height - wy) * self.scale)
            pts.append((px,py))
        pygame.draw.polygon(surf, color, pts)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
