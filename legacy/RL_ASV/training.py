#!/usr/bin/env python3
import argparse
import logging
import os
import pickle
import random
from collections import deque, namedtuple
from dataclasses import asdict, dataclass
from datetime import datetime

import cv2  # for grayscale & resize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from channel_crossing_env import ChannelCrossingEnv


# --- Hyperparameters ---
@dataclass
class HParams:
    batch_size: int          = 64          # Mini-batch size for training
    replay_buffer_size: int  = 500_000   # Replay memory capacity
    checkpoint_interval: int = 1_000       # Save checkpoints and plots every n episodes
    render_mode: str         = "rgb_array"     # Set True to visualize training
    episodes: int            = 1_000_000     # Number of episodes to train
    epsilon_start: float     = 1.0         # Starting epsilon for exploration
    epsilon_end: float       = 0.1         # Minimum epsilon
    epsilon_decay: float     = 0.998842    # Decay factor per episode
    epsilon_decay_steps: int = 1_000_000   # decay over million steps
    gamma: float             = 0.99        # Discount factor for future rewards
    learning_rate: float     = 1e-4        # Learning rate for optimizer
    min_replay: int          = 1_000
    seed: int                = 42
    target_update_freq: int  = 500         # Update target network every 1000 steps

# -----------------------------------------------------------------------------
# Replay buffer
# -----------------------------------------------------------------------------
Transition = namedtuple("Transition",
                        ("state","action","reward","next_state","done"))

# class ReplayBuffer:
#     def __init__(self, buffer_size:int):
#         self.buffer = deque(maxlen=buffer_size)

#     def push(self, *args):
#         self.buffer.append(Transition(*args))

#     def sample(self, batch_size:int=64):
#         batch = random.sample(self.buffer, batch_size)
#         return Transition(*zip(*batch))
    
#     def __len__(self): return len(self.buffer)

class DualReplayBuffer:
    """
    A replay buffer with:
      - frozen deque: loaded once, never mutated
      - live   deque: new transitions, with maxlen eviction
    Sampling uniformly over both.
    """
    def __init__(
        self,
        live_size: int,
        frozen_path: str = None,
        live_resume_path: str = None,
    ):
        self.live = deque(maxlen=live_size)
        if frozen_path:
            with open(frozen_path, 'rb') as f:
                data = pickle.load(f)
            self.frozen = deque(data, maxlen=len(data))
            print(f"Loaded FROZEN buffer with {len(self.frozen)} transitions")
        else:
            self.frozen = deque()

        if live_resume_path:
            with open(live_resume_path, 'rb') as f:
                data = pickle.load(f)
            for t in data:
                self.live.append(t)
            print(f"Loaded LIVE buffer with {len(self.live)} transitions")

    def push(self, *args):
        """Append a new transition to the live queue only."""
        self.live.append(Transition(*args))

    def sample(self, batch_size: int):
        """Draw batch_size transitions uniformly at random."""
        total = len(self.frozen) + len(self.live)
        if batch_size > total:
            raise ValueError(f"Not enough transitions: {total} < {batch_size}")
        picks = random.sample(range(total), batch_size)
        batch = []
        for idx in picks:
            if idx < len(self.frozen):
                batch.append(self.frozen[idx])
            else:
                batch.append(self.live[idx - len(self.frozen)])
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.frozen) + len(self.live)

    def save_live(self, path: str):
        """Dump just the live deque to disk (for checkpointing)."""
        with open(path, 'wb') as f:
            pickle.dump(list(self.live), f)

# -----------------------------------------------------------------------------
# CNN Q-network for DDQN
# -----------------------------------------------------------------------------
class DQN_CNN(nn.Module):
    def __init__(self, num_actions:int):
        super().__init__()
        # Convolutional backbone (Atari-style)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected head
        self.fc1  = nn.Linear(64 * 7 * 7, 512)
        self.head = nn.Linear(512, num_actions)

        # Weight initialization (Kaiming for ReLU)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns Q-values for each action"""
        x = F.relu(self.conv1(x))       # [B,32,20,20]
        x = F.relu(self.conv2(x))       # [B,64, 9, 9]
        x = F.relu(self.conv3(x))       # [B,64, 7, 7]
        x = x.view(x.size(0), -1)       # [B,64*7*7]
        x = F.relu(self.fc1(x))         # [B,512]
        return self.head(x)             # [B,num_actions]

# -----------------------------------------------------------------------------
# DDQN agent
# -----------------------------------------------------------------------------
class DDQNAgent:
    def __init__(self,
        batch_size: int,
        replay_buffer: DualReplayBuffer,
        device: torch.device,
        epsilon_start: float,
        epsilon_end:   float,
        epsilon_decay: float,
        epsilon_decay_steps: int,
        gamma: float,
        learning_rate: float,
        num_actions: int,
        min_replay: int,
        target_update_freq: int,
    ):
        self.batch_size         = batch_size
        self.device             = device
        self.gamma              = gamma
        self.min_replay         = min_replay
        self.num_actions        = num_actions
        self.target_update_freq = target_update_freq

        # networks
        self.policy_net = DQN_CNN(num_actions).to(device)
        self.target_net = DQN_CNN(num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer + scheduler
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
        #                                            step_size=10000,
        #                                            gamma=0.5)

        # Optimizer (RMSprop as in original DQN)
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=learning_rate,
            alpha=0.95,
            eps=0.01
        )

        # replay
        self.replay_buffer = replay_buffer
        self.global_steps  = 0

        # epsilon
        self.epsilon = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_steps = epsilon_decay_steps

    # @property
    # def epsilon(self) -> float:
    #     # Linear decay per step
    #     frac = min(1.0, self.global_steps / self.epsilon_decay_steps)
    #     return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state:np.ndarray) -> int:
        self.global_steps += 1
        if random.random() < self.epsilon:
            return random.randrange(9)
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)

            # # Split Q-values into two sets of 3
            # action1_qs = q_values[0, :3]   # First 3 values
            # action2_qs = q_values[0, 3:]   # Last 3 values

            # # Take argmax to get discrete actions
            # action1 = torch.argmax(action1_qs).item()
            # action2 = torch.argmax(action2_qs).item()

        #     # Return as list (or tuple), compatible with MultiDiscrete([3, 3])
        # return [action1, action2]

        return q_values.argmax().item()



    def store(self, *args):    self.replay_buffer.push(*args)
    #def __len__(self):         return len(self.replay_buffer)

    def update(self):
        if len(self.replay_buffer) < max(self.min_replay, self.batch_size):
            return
        batch      = self.replay_buffer.sample(self.batch_size)
        states     = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        actions    = torch.tensor(batch.action, device=self.device).long().unsqueeze(1)
        rewards    = torch.tensor(batch.reward, device=self.device).float().unsqueeze(1)
        next_state = torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
        dones      = torch.tensor(batch.done, device=self.device).float().unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)

        # 2) Compute target Q-values
        with torch.no_grad():
            next_action = self.policy_net(next_state).argmax(dim=1, keepdim=True)
            next_q_value = self.target_net(next_state).gather(1, next_action)
            bellman_target = rewards + self.gamma * next_q_value * (1.0 - dones)

        criterion = nn.MSELoss()
        loss = criterion(q_values, bellman_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        #self.scheduler.step()

        if self.global_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return(loss.item())

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
def train(env, agent, episodes, model_dir, plot_dir, data_file,
          ckpt_interval, logger):

    rewards = []
    avg100  = []
    losses = []

    for current_episode in range(1, episodes+1):
        log_start = datetime.now()

        _obs, _ = env.reset()

        # get 4 identical start frames
        initial_rgb_frame = env.render()
        motion_frames = deque(maxlen=4)
        get_initial_motion_frames(motion_frames, initial_rgb_frame)
        state  = transform_motion_frames(motion_frames)

        episode_reward = 0.0
        episode_loss = 0.0
        done           = False
        step           = 0

        while not done:
            action = agent.select_action(state)

            # step + get raw image
            _, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated
            step += 1

            next_rgb_frame = env.render()
            motion_frames.append(preprocess(next_rgb_frame))
            next_state = transform_motion_frames(motion_frames)

            agent.store(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                episode_loss += loss
            state = next_state

            # logger.info(
            # f"Episode {current_episode} | Step {step} | Action {action} | "
            # f"Reward: {reward:.4f} | Loss: {loss} | Cumulative_Reward: {episode_reward:.4f} | Cumulative_Loss: {episode_loss} |  | Global_Steps: {agent.global_steps}" 
            # )

        log_end = datetime.now()
        duration = (log_end - log_start).total_seconds()

        rewards.append(episode_reward)
        moving_average = np.mean(rewards[-100:])
        avg100.append(moving_average)
        losses.append(episode_loss)
        print(f"\rEp {current_episode}/{episodes} | R {episode_reward:6.1f} | L {episode_loss:6.1f} | Avg100 {moving_average:6.1f} | eps {agent.epsilon:.3f} | global_steps {agent.global_steps}")

        # logger.info(
        #     f"--- Episode {current_episode} end: "
        #     f"Total Reward: {episode_reward:.4f} | Total Loss: {episode_loss:.4f} | Steps: {step} | "
        #     f"Duration: {duration:.2f}s ---\n"
        # )

        # decay epsilon
        #agent.epsilon = max(agent.epsilon_end, agent.epsilon - agent.epsilon_decay)
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)

        #Save and plot at checkpoint or final episode
        if current_episode % ckpt_interval == 0 or current_episode == episodes:
            #print()
            # torch.save(agent.policy_net.state_dict(),
            #            os.path.join(model_dir, f"model_{current_episode}.pth"))

            # build paths
            model_path = os.path.join(model_dir, f"ckpt_{current_episode}.pth")
            buffer_path = os.path.join(result_dir, 'buffers', f"buffer.pkl")
            state_path = os.path.join(result_dir, 'states', f"state_{current_episode}.pt")
            os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
            os.makedirs(os.path.dirname(state_path), exist_ok=True)

            save_checkpoint(
                agent,
                agent.optimizer,
                agent.replay_buffer,
                agent.epsilon,
                agent.global_steps,
                model_path,
                buffer_path,
                state_path
            )
            print(f" → full checkpoint @ ep {current_episode}")

            # Create a figure with 2 rows, 1 column, shared x-axis
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                sharex=True,
                figsize=(12, 8),
                gridspec_kw={'height_ratios': [2, 2]}  # make the top plot a bit taller if you like
            )

            # Top plot: rewards and its moving average
            ax1.plot(rewards, label="R",    alpha=0.6)
            ax1.plot(avg100,  label="Avg100", lw=2)
            ax1.set_ylabel("Reward")
            ax1.legend(loc="upper right")
            ax1.grid(True)

            # Bottom plot: losses
            ax2.plot(losses, label="L", alpha=0.7, color='C2')
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Loss")
            ax2.legend(loc="upper right")
            ax2.grid(True)

            # Tidy up and save
            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"plot_{current_episode}.svg"), format="svg")
            plt.close(fig)

            pd.DataFrame({
                "episode": np.arange(1, len(rewards)+1),
                "reward":  rewards,
                "avg100":  avg100,
                "loss": losses,
            }).to_csv(data_file, index=False)
            print(f" → checkpoint @ ep {current_episode}")

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

def save_checkpoint(agent, optimizer, replay_buffer, epsilon, global_steps,
                    model_path, buffer_path, state_path):
    # 1) model + optimizer
    torch.save({
        'model_state': agent.policy_net.state_dict(),
        'target_state': agent.target_net.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, model_path)
    # 2) replay buffer → live only
    replay_buffer.save_live(buffer_path)
    # 3) training state
    torch.save({
        'epsilon': epsilon,
        'global_steps': global_steps,
    }, state_path)

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
        '--frozen-buffer',
        type=str,
        default=None,
        help="Path to .pkl to load into the frozen deque"
    )
    parser.add_argument(
        '--resume-live-buffer',
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
   
    
    buffer = DualReplayBuffer(
        live_size        = args.replay_buffer_size,
        frozen_path      = args.frozen_buffer,
        live_resume_path = args.resume_live_buffer,
    )

    env = ChannelCrossingEnv(render_mode=args.render_mode)
    agent = DDQNAgent(
        num_actions         = env.action_space.n,
        device              = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        replay_buffer       = buffer,
        batch_size          = args.batch_size,
        gamma               = args.gamma,
        learning_rate       = args.learning_rate,
        target_update_freq  = args.target_update_freq,
        min_replay          = args.min_replay,
        epsilon_start       = args.epsilon_start,
        epsilon_end         = args.epsilon_end,
        epsilon_decay       = args.epsilon_decay,
        epsilon_decay_steps = args.epsilon_decay_steps
    )

    optimizer = agent.optimizer

    # optionally resume
    if args.resume or args.resume_live_buffer or args.resume_state:
        load_checkpoint(
            agent,
            optimizer,
            agent.replay_buffer,
            (
               args.resume,
               args.resume_live_buffer,   # ← use the new name here
               args.resume_state
            )
        )
        print(f"Resumed from model={args.resume!r}, live-buffer={args.resume_live_buffer!r}, state={args.resume_state!r}")

    logger = setup_logger(log_path=log_file)
    train(
        env,
        agent,
        episodes         = args.episodes,
        model_dir        = model_dir,
        plot_dir         = plot_dir,
        data_file        = data_file,
        ckpt_interval    = args.checkpoint_interval,
        logger=logger
    )
