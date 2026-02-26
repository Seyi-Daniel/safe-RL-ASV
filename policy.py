from __future__ import annotations

import torch
import torch.nn as nn

ACTION_DIM = 2  # [rudder_cmd, throttle_cmd]
DEFAULT_OBS_DIM = 12  # [agent(6), target(6)] features emitted by SingleTargetFeatureEnv.get_obs()


class ContinuousActor(nn.Module):
    """Continuous actor producing normalized controls in [-1, 1]."""

    def __init__(self, in_dim: int = DEFAULT_OBS_DIM, hidden_dim: int = 256, action_dim: int = ACTION_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ContinuousCritic(nn.Module):
    """State-action critic Q(s,a) for DDPG-style training."""

    def __init__(self, in_dim: int = DEFAULT_OBS_DIM, action_dim: int = ACTION_DIM, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)
