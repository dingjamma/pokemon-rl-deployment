"""
simple_agent.py — CNN actor-critic policy for Pokemon Red.

Architecture (inspired by Nature DQN + PPO actor-critic):
  Input  : (batch, frame_stack, H, W)  — stacked grayscale frames
  Encoder: 3× Conv2d → ReLU → Flatten → Linear(512) → ReLU
  Heads  : Actor  → Linear(512, num_actions)  — logits for Categorical
           Critic → Linear(512, 1)            — state-value estimate

This module is intentionally self-contained: it has no dependency on
PufferLib beyond standard torch.nn.Module so it can be used with any
PPO loop or tested in isolation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _layer_init(layer: nn.Module, std: float = 1.0, bias_const: float = 0.0) -> nn.Module:
    """Orthogonal weight init — standard practice for PPO actor-critics."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class SimpleAgent(nn.Module):
    """
    CNN actor-critic policy for the Pokemon Red environment.

    Args:
        obs_shape:   Observation shape as returned by the env, i.e.
                     (frame_stack, height, width).  E.g. (4, 84, 84).
        num_actions: Size of the discrete action space (default 8).
        hidden_dim:  Width of the fully-connected hidden layer (default 512).
    """

    def __init__(
        self,
        obs_shape: tuple = (4, 84, 84),
        num_actions: int = 8,
        hidden_dim: int = 512,
    ):
        super().__init__()

        frame_stack, height, width = obs_shape

        # Convolutional encoder (Nature DQN layout)
        self.encoder = nn.Sequential(
            # Conv1: (B, C, H,  W) → (B, 32, ?, ?)
            _layer_init(nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            # Conv2
            _layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            # Conv3
            _layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute encoder output size without running a forward pass on real data
        enc_out_dim = self._get_encoder_out_dim(frame_stack, height, width)

        self.fc = nn.Sequential(
            _layer_init(nn.Linear(enc_out_dim, hidden_dim)),
            nn.ReLU(),
        )

        # Separate actor and critic heads
        self.actor  = _layer_init(nn.Linear(hidden_dim, num_actions), std=0.01)
        self.critic = _layer_init(nn.Linear(hidden_dim, 1),           std=1.0)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Shared CNN + FC forward pass.

        Args:
            obs: Float tensor (B, frame_stack, H, W) in range [0, 255].
        Returns:
            features: Float tensor (B, hidden_dim).
        """
        # Normalise to [0, 1]
        x = obs.float() / 255.0
        x = self.encoder(x)
        return self.fc(x)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return state-value estimate V(s). Shape: (B, 1)."""
        return self.critic(self.get_features(obs))

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or evaluate) an action and compute its log-probability,
        entropy, and the state-value.

        Args:
            obs:    (B, frame_stack, H, W) float tensor.
            action: (B,) int tensor — if provided, compute log-prob/entropy
                    for these actions rather than sampling.

        Returns:
            action      : (B,)   sampled or provided actions
            log_prob    : (B,)   log π(a|s)
            entropy     : (B,)   per-sample entropy of the action distribution
            value       : (B, 1) V(s)
        """
        features = self.get_features(obs)
        logits   = self.actor(features)
        dist     = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        value    = self.critic(features)

        return action, log_prob, entropy, value

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Greedy (argmax) action selection — useful for inference/rendering.

        Args:
            obs: (1, frame_stack, H, W) or (B, frame_stack, H, W).
        Returns:
            actions: (B,) integer tensor.
        """
        features = self.get_features(obs)
        logits   = self.actor(features)
        return logits.argmax(dim=-1)

    def _get_encoder_out_dim(self, channels: int, height: int, width: int) -> int:
        """Compute flattened encoder output size with a dummy forward pass."""
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            out   = self.encoder(dummy)
        return int(out.shape[1])

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity-check (run directly: python -m src.agent.simple_agent)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    obs_shape   = (4, 84, 84)
    num_actions = 8
    agent = SimpleAgent(obs_shape=obs_shape, num_actions=num_actions)

    print(f"SimpleAgent — {agent.count_parameters():,} trainable parameters")

    # Simulate a batch of 4 observations
    dummy_obs = torch.randint(0, 256, (4, *obs_shape), dtype=torch.uint8)
    action, log_prob, entropy, value = agent.get_action_and_value(dummy_obs)
    print(f"  action shape   : {action.shape}")
    print(f"  log_prob shape : {log_prob.shape}")
    print(f"  entropy shape  : {entropy.shape}")
    print(f"  value shape    : {value.shape}")
    print("All checks passed.")
