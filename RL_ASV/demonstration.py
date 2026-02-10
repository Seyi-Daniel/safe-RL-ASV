#!/usr/bin/env python3
import argparse
import logging
import math
import os
import pickle
import random
import sys
from collections import deque, namedtuple
from dataclasses import asdict, dataclass
from datetime import datetime

import cv2  # for grayscale & resize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from channel_crossing_env import ChannelCrossingEnv


# --- Hyperparameters ---
@dataclass
class HParams:
    checkpoint_episode: int     = 100         # Save checkpoints and plots every n episodes
    checkpoint_transitions: int = 5000        # Save checkpoints and plots every n transitions in the replay buffer
    episodes: int               = 1000        # Number of episodes for which the simulation is human-controlled
    demo_replay_size: int       = 1_000_000   # Target number of transitions to be saved
    render_mode: str            = "human"     # Set True to visualize training
    replay_buffer_size: int     = 1_000_000   # Replay memory capacity
    seed: int                   = 42

# -----------------------------------------------------------------------------
# Replay buffer
# -----------------------------------------------------------------------------
Transition = namedtuple("Transition",
                        ("state","action","reward","next_state","done"))

class ReplayBuffer:
    """
    A replay buffer with:- deque:- new transitions, with maxlen eviction
    """
    def __init__(
        self,
        size: int,
        resume_path: str = None,
    ):
        self.buffer = deque(maxlen=size)

        if resume_path:
            with open(resume_path, 'rb') as f:
                data = pickle.load(f)
            for t in data:
                self.buffer.append(t)
            print(f"Loaded LIVE buffer with {len(self.buffer)} transitions")

    def push(self, *args):
        """Append a new transition to the live queue only."""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        total = len(self.buffer)
        if batch_size > total:
            raise ValueError(f"Not enough transitions: {total} < {batch_size}")
        picks = random.sample(range(total), batch_size)
        batch = []
        for idx in picks:
            batch.append(self.buffer[idx])
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, path: str):
        """Dump just the live deque to disk (for checkpointing)."""
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)

# -----------------------------------------------------------------------------
# preprocessing & stacking
# -----------------------------------------------------------------------------
# --- Preprocessing function to convert frames to grayscale and resize ---
def preprocess(rgb_frame:np.ndarray) -> np.ndarray:
    grey    = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(grey, (84,84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0  # Shape (84, 84)

def get_initial_motion_frames(de_que:deque, rgb_frame:np.ndarray):
    grey = preprocess(rgb_frame)
    for _ in range(de_que.maxlen):
        de_que.append(grey)

def transform_motion_frames(de_que:deque) -> np.ndarray:
    return np.stack(de_que, axis=0)

# -----------------------------------------------------------------------------
# training loop
# -----------------------------------------------------------------------------
def demonstrate(
        ckpt_episode,
        ckpt_transitions,
        data_file,
        demo_replay_size,
        env, 
        episodes, 
        logger,
        model_dir, 
        plot_dir,
        replay_buffer
):
    pygame.init()
    clock = pygame.time.Clock()

    rewards = []
    avg100  = []

    paused = False

    last_checkpoint_transitions = 0  # ← Track last transition checkpoint

    def checkpoint(tag: str):
        # 1. Save buffer
        buffer_path = os.path.join(result_dir, 'buffers', f"buffer.pkl")
        os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
        save_checkpoint(replay_buffer, buffer_path)
        print(f" → checkpoint @ {tag}")

        # 2. Save plot
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            sharex=True,
            figsize=(12, 8),
            gridspec_kw={'height_ratios': [2, 2]}
        )
        ax1.plot(rewards, label="R",    alpha=0.6)
        ax1.plot(avg100,  label="Avg100", lw=2)
        ax1.set_ylabel("Reward")
        ax1.legend(loc="upper right")
        ax1.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"plot_{tag}.svg"), format="svg")
        plt.close(fig)

        # 3. Save reward data
        pd.DataFrame({
            "episode": np.arange(1, len(rewards)+1),
            "reward":  rewards,
            "avg100":  avg100,
        }).to_csv(data_file, index=False)

    for current_episode in range(1, episodes+1):
        log_start = datetime.now()

        paused = False # reset pause at start of each episode

        _obs, _ = env.reset()

        # get 4 identical start frames
        initial_rgb_frame = env.render()
        motion_frames = deque(maxlen=4)
        get_initial_motion_frames(motion_frames, initial_rgb_frame)
        state  = transform_motion_frames(motion_frames)

        episode_reward = 0.0
        done           = False
        step           = 0

        while not done:

            # 1) Process quit / pause events
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_p:
                    paused = not paused

            # 2) If paused, just render & skip stepping
            if paused:
                env.render()
                clock.tick(env.metadata["render_fps"])
                continue

            # --- human‐controlled for first demo_episodes/length of min replay, then switch to agent ---
            # if current_episode <= demo_episodes:
            if len(replay_buffer) <= demo_replay_size:
                # poll current key state
                keys = pygame.key.get_pressed()
                # default: go straight (steer=0), coast (throttle=0)
                steer    = 0
                throttle = 0
                # steering: D/Right-arrow-key = right, A/Left-arrow-key = left
                if keys[pygame.K_l] and not keys[pygame.K_j]:
                #if keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
                    steer = 1
                elif keys[pygame.K_j] and not keys[pygame.K_l]:
                #elif keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
                    steer = 2
                # throttle: W/Up-arrow-key = accelerate
                if keys[pygame.K_i] and not keys[pygame.K_k]:
                #if keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
                    throttle = 1
                if keys[pygame.K_k] and not keys[pygame.K_i]:
                #elif keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
                    throttle = 2
                # combine into single discrete action
                action = steer * 3 + throttle
            else:
                print("Done")
                return

            # step + get raw image
            _, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated
            step += 1

            next_rgb_frame = env.render()
            motion_frames.append(preprocess(next_rgb_frame))
            next_state = transform_motion_frames(motion_frames)

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            logger.info(
            f"Episode {current_episode} | Step {step} | Action {action} | "
            f"Reward: {reward:.4f} | Cumulative_Reward: {episode_reward:.4f} | Global_Steps: {len(replay_buffer)}" 
            )

        log_end = datetime.now()
        duration = (log_end - log_start).total_seconds()

        rewards.append(episode_reward)
        moving_average = np.mean(rewards[-100:])
        avg100.append(moving_average)
        print(f"\rEp {current_episode}/{episodes} | R {episode_reward:6.1f} | Avg100 {moving_average:6.1f} | global_steps {len(replay_buffer)} | final_speed {env.unwrapped.agent.speed:.3f}")

        logger.info(
            f"--- Episode {current_episode} end: "
            f"Total Reward: {episode_reward:.4f} | Steps: {step} | "
            f"Duration: {duration:.2f}s ---\n"
        )

        # --- Episode-based checkpoint ---
        if current_episode % ckpt_episode == 0 or current_episode == episodes:
            checkpoint(f"{current_episode}episodes")

        # --- Transition-based checkpoint ---
        if (len(replay_buffer) - last_checkpoint_transitions) >= ckpt_transitions:
            checkpoint(f"{len(replay_buffer)}transitions")
            last_checkpoint_transitions = len(replay_buffer)

    env.close()

# ——— Logger setup ———
def setup_logger(log_path, instance_name: str = "training"):
    logger = logging.getLogger(instance_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s — %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def save_checkpoint(replay_buffer, buffer_path):
    replay_buffer.save_buffer(buffer_path)

def load_checkpoint(agent, optimizer, replay_buffer, resume_paths):
    model_path, buffer_path, state_path = resume_paths
    # 1) model + optimizer
    # ckpt = torch.load(model_path)
    # agent.policy_net.load_state_dict(ckpt['model_state'])
    # agent.target_net.load_state_dict(ckpt['target_state'])
    # optimizer.load_state_dict(ckpt['optimizer_state'])
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    agent.policy_net.load_state_dict(ckpt)
    agent.target_net.load_state_dict(ckpt)
    # 2) buffer
    if buffer_path:
        with open(buffer_path, 'rb') as f:
            data = pickle.load(f)
        replay_buffer.buffer = deque(data, maxlen=replay_buffer.buffer.maxlen)
    # 3) ε & global_steps
    if state_path:
        state = torch.load(state_path)
        agent.epsilon = state['epsilon']
        agent.global_steps = state['global_steps']

# -----------------------------------------------------------------------------
#  Main & CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    defaults = HParams()
    parser = argparse.ArgumentParser()
    for field, value in asdict(defaults).items():
        parser.add_argument(f'--{field}', type=type(value), default=value)
    parser.add_argument('--resume',       type=str,   default=None,
                        help="Path to checkpoint .pth to resume model+optimizer")
    parser.add_argument('--resume-state', type=str,   default=None,
                        help="Path to .pt file with training state (ε, steps)")
    parser.add_argument(
        '--resume-buffer',
        type=str,
        default=None,
        help="Path to .pkl to preload into the live deque"
    )
    args = parser.parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # dirs
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base = os.path.abspath(os.path.join(os.getcwd()))
    result_dir = os.path.join(base, "results", timestamp)
    model_dir = os.path.join(result_dir, "models")
    plot_dir = os.path.join(result_dir, "plots")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    data_file = os.path.join(result_dir, f"training_data_{timestamp}.csv")
    hyparam_file = os.path.join(result_dir, f"hyperparameters_{timestamp}.csv")
    log_file = os.path.join(result_dir, f"log_{timestamp}.log")

    # save hyperparameters
    pd.DataFrame({
                "Parameters": asdict(defaults).keys(),
                "Value": asdict(defaults).values()
            }).to_csv(hyparam_file, index=False)
   
    
    buffer = ReplayBuffer(
        size        = args.replay_buffer_size,
        resume_path = args.resume_buffer,
    )

    env = ChannelCrossingEnv(render_mode=args.render_mode)

    logger = setup_logger(log_path=log_file)

    demonstrate(
        ckpt_episode     = args.checkpoint_episode,
        ckpt_transitions = args.checkpoint_transitions,
        data_file        = data_file,
        demo_replay_size = args.demo_replay_size,
        env              = env,
        episodes         = args.episodes,
        logger           = logger,
        model_dir        = model_dir,
        plot_dir         = plot_dir,
        replay_buffer    = buffer
    )
