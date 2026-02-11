import argparse
import math
import os
import random
from collections import deque

import cv2  # for grayscale & resize
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from channel_crossing_env import ChannelCrossingEnv

# ----- Direct configuration (set these variables) -----
EPISODES      = 100         # number of episodes to run
RENDER_MODE   = "human"     # "human" or "rgb_array"
# ----------------------------------------------------

# Re-define the DQN_CNN architecture (must match training)
class DQN_CNN(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1   = nn.Linear(64 * 7 * 7, 512)
        self.head  = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.head(x)

# Frame preprocessing

def preprocess(frame: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description="Run a trained DDQN model")
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the .pth checkpoint file')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes to run')
    parser.add_argument('--render-mode', type=str, choices=['human', 'rgb_array'],
                        default=None, help='Render mode for the environment')
    parser.add_argument('--window-name', type=str,
                        default=None, help='Name of the Simulation window')
    args = parser.parse_args()

    # Use direct variables if CLI args are not provided
    model_path = args.model_path
    num_episodes = args.episodes if args.episodes is not None else EPISODES
    render_mode = args.render_mode if args.render_mode is not None else RENDER_MODE
    window_name = args.window_name if args.window_name is not None else model_path

    # Create environment
    env = ChannelCrossingEnv(render_mode=render_mode, window_name=window_name)
    env = env.unwrapped  # <-- Unwrap to access .agent, .traffic, etc.
    num_actions = env.action_space.n

    # Initialize network and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN_CNN(num_actions).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(checkpoint['model_state'])
    #policy_net.load_state_dict(checkpoint)
    policy_net.eval()

    # metrics
    success_count = 0
    collision_count = 0
    total_distance_success = 0.0


    for ep in range(1, num_episodes + 1):

        obs, _ = env.reset(ep)

        prev_x, prev_y = env.agent.x, env.agent.y
        distance_travelled = 0.0
        collided = False

        # Initialize deque of 4 frames
        frame = env.render()
        motion_frames = deque(maxlen=4)
        gray = preprocess(frame)
        for _ in range(4):
            motion_frames.append(gray)
        state = np.stack(motion_frames, axis=0)

        done = False
        total_reward = 0.0
        while not done:
            # Select greedy action
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            with torch.no_grad():
                qvals = policy_net(state_tensor)
                action = int(qvals.argmax(dim=1).item())

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)

            # 1) accumulate distance
            curr_x, curr_y = env.agent.x, env.agent.y
            distance_travelled += math.hypot(curr_x - prev_x, curr_y - prev_y)
            prev_x, prev_y = curr_x, curr_y

            # 2) detect collision this step
            if any(env._check_halo_overlap(env.agent, tb) for tb in env.traffic):
                collided = True

            done = terminated or truncated
            total_reward += reward

            # Render next frame and preprocess
            frame = env.render()
            gray = preprocess(frame)
            motion_frames.append(gray)
            state = np.stack(motion_frames, axis=0)

        # determine success (goal OR channel‐line) vs collision
        dist_to_goal = math.hypot(env.goal_x - env.agent.x, env.goal_y - env.agent.y)
        reached_goal  = (dist_to_goal < 5.0)
        crossed_line  = (env.agent.y >= env.channel_high)

        if collided:
            collision_count += 1
            outcome = "COLLISION"
        elif reached_goal or crossed_line:
            success_count += 1
            total_distance_success += distance_travelled
            outcome = "SUCCESS"
        else:
            outcome = "OTHER"

        print(f"Episode {ep}: reward {total_reward:.2f}   {outcome}")

    env.close()

    avg_distance = (total_distance_success / success_count) if success_count else 0.0

    print("\n=== Summary over {} episodes ===".format(num_episodes))
    print(f"  Successes: {success_count}")
    print(f"  Collisions: {collision_count}")
    print(f"  Avg. distance on successes: {avg_distance:.2f}")

    # # 1) bar chart of counts
    # plt.figure()
    # plt.bar(["Successes","Collisions"], [success_count, collision_count])
    # plt.ylabel("Count")
    # plt.title("Episode Outcomes")
    # plt.savefig("episode_outcomes.png")
    # plt.close()

    # # 2) plain‐text “poster” of average distance
    # plt.figure(figsize=(4,2))
    # plt.text(0.5, 0.5, f"Avg Distance:\n{avg_distance:.2f}", 
    #          ha="center", va="center", fontsize=14)
    # plt.axis("off")
    # plt.savefig("average_distance.png")
    # plt.close()


if __name__ == '__main__':
    main()