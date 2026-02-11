from __future__ import annotations

import numpy as np


def unpack_theta(theta: np.ndarray, in_dim: int = 10, hidden_dim: int = 32, out_dim: int = 2):
    """Unpack flat parameter vector into MLP weights."""
    idx = 0
    w1_size = in_dim * hidden_dim
    b1_size = hidden_dim
    w2_size = hidden_dim * out_dim
    b2_size = out_dim

    w1 = theta[idx : idx + w1_size].reshape(in_dim, hidden_dim)
    idx += w1_size
    b1 = theta[idx : idx + b1_size]
    idx += b1_size
    w2 = theta[idx : idx + w2_size].reshape(hidden_dim, out_dim)
    idx += w2_size
    b2 = theta[idx : idx + b2_size]
    return w1, b1, w2, b2


def theta_dim(in_dim: int = 10, hidden_dim: int = 32, out_dim: int = 2) -> int:
    return in_dim * hidden_dim + hidden_dim + hidden_dim * out_dim + out_dim


def policy_action(obs: np.ndarray, theta: np.ndarray, hidden_dim: int = 32) -> np.ndarray:
    w1, b1, w2, b2 = unpack_theta(theta, in_dim=10, hidden_dim=hidden_dim, out_dim=2)
    h = np.tanh(obs @ w1 + b1)
    out = np.tanh(h @ w2 + b2)
    return out.astype(np.float32)  # [rudder_cmd, throttle_cmd] in [-1, 1]
