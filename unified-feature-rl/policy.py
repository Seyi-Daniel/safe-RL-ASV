from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


N_ACTIONS = 9


@dataclass(frozen=True)
class ActionCommand:
    rudder: float
    throttle: float


def decode_action_idx(a: int) -> ActionCommand:
    """Map discrete action index [0,8] to rudder/throttle in {-1,0,1}."""
    steer_idx = a // 3
    throttle_idx = a % 3

    steer_map = {0: 0.0, 1: -1.0, 2: 1.0}      # none, starboard, port
    throttle_map = {0: 0.0, 1: 1.0, 2: -1.0}   # coast, accelerate, decelerate
    return ActionCommand(rudder=steer_map[steer_idx], throttle=throttle_map[throttle_idx])


class DDQNQNet(nn.Module):
    """Feature-DDQN MLP, similar style to feature-RL-ASV (ReLU + Kaiming init)."""

    def __init__(self, in_dim: int = 6, hidden_dim: int = 256, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_actions),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
