from __future__ import annotations

import torch
import torch.nn as nn


N_ACTIONS = 9



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
