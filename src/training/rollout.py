"""
rollout.py — Fixed-size rollout buffer for PPO.

Stores (obs, action, log_prob, reward, done, value) tuples collected from
parallel environments, then computes GAE advantages and returns.

Design notes:
  - Pre-allocates all tensors on the target device to avoid GC pressure.
  - Uses float32 throughout; uint8 observations are stored raw and cast
    during the advantage/return computation batch.
  - GAE (Schulman et al., 2016) is computed in one vectorised pass.
"""

from __future__ import annotations

import torch


class RolloutBuffer:
    """
    Circular buffer that accumulates one full batch of PPO rollout data.

    Args:
        batch_size:   Total number of (env, step) pairs per rollout.
                      Must equal num_envs * num_steps_per_env.
        obs_shape:    Shape of a single observation (frame_stack, H, W).
        num_actions:  Unused for discrete; kept for forward-compat.
        device:       Torch device to store tensors on.
        gamma:        Discount factor γ.
        gae_lambda:   GAE λ smoothing parameter.
    """

    def __init__(
        self,
        batch_size: int,
        obs_shape: tuple,
        num_actions: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.batch_size  = batch_size
        self.obs_shape   = obs_shape
        self.device      = device
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda

        self._ptr = 0
        self._full = False

        # Pre-allocate storage (observations as uint8 to save memory)
        self.obs      = torch.zeros((batch_size, *obs_shape), dtype=torch.uint8,  device=device)
        self.actions  = torch.zeros((batch_size,),            dtype=torch.long,   device=device)
        self.log_probs= torch.zeros((batch_size,),            dtype=torch.float32,device=device)
        self.rewards  = torch.zeros((batch_size,),            dtype=torch.float32,device=device)
        self.dones    = torch.zeros((batch_size,),            dtype=torch.float32,device=device)
        self.values   = torch.zeros((batch_size,),            dtype=torch.float32,device=device)

        # Filled during advantage computation
        self.advantages = torch.zeros((batch_size,), dtype=torch.float32, device=device)
        self.returns    = torch.zeros((batch_size,), dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Data insertion
    # ------------------------------------------------------------------

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        Store one step of experience for *all* parallel environments.

        Each argument should have a batch dimension matching the number
        of parallel envs.  Call this once per environment step.
        """
        n = obs.shape[0]  # number of envs this step
        end = self._ptr + n
        if end > self.batch_size:
            raise RuntimeError(
                f"RolloutBuffer overflow: tried to write {end} entries "
                f"into buffer of size {self.batch_size}."
            )

        self.obs[self._ptr:end]       = obs.to(self.device)
        self.actions[self._ptr:end]   = action.to(self.device)
        self.log_probs[self._ptr:end] = log_prob.to(self.device)
        self.rewards[self._ptr:end]   = reward.to(self.device)
        self.dones[self._ptr:end]     = done.float().to(self.device)
        self.values[self._ptr:end]    = value.squeeze(-1).to(self.device)

        self._ptr = end
        if self._ptr >= self.batch_size:
            self._full = True

    def is_full(self) -> bool:
        return self._full

    def reset(self):
        self._ptr  = 0
        self._full = False

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    def compute_advantages(self, last_value: torch.Tensor, last_done: torch.Tensor):
        """
        Compute GAE advantages and discounted returns in-place.

        Args:
            last_value: (num_envs,) — critic estimate for the state
                        *after* the final stored step (bootstrap value).
            last_done:  (num_envs,) bool/float — whether the final step
                        ended an episode (used to zero the bootstrap).

        This must be called after the buffer is full and before iterating
        over minibatches.
        """
        if not self._full:
            raise RuntimeError("Buffer is not full — collect more steps first.")

        # We need to know num_envs; infer from last_value
        num_envs = last_value.shape[0]
        num_steps = self.batch_size // num_envs  # steps per env

        # Reshape stored tensors to (num_steps, num_envs)
        rewards_2d = self.rewards.view(num_steps, num_envs)
        dones_2d   = self.dones.view(num_steps, num_envs)
        values_2d  = self.values.view(num_steps, num_envs)

        advantages_2d = torch.zeros_like(rewards_2d)

        last_gae = torch.zeros(num_envs, device=self.device)
        last_val = last_value.squeeze(-1).to(self.device)
        last_done_f = last_done.float().to(self.device)

        # Backward pass through time
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - last_done_f
                next_value        = last_val
            else:
                next_non_terminal = 1.0 - dones_2d[t + 1]
                next_value        = values_2d[t + 1]

            delta    = rewards_2d[t] + self.gamma * next_value * next_non_terminal - values_2d[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages_2d[t] = last_gae

        self.advantages = advantages_2d.view(-1)
        self.returns    = self.advantages + self.values

    # ------------------------------------------------------------------
    # Minibatch iterator
    # ------------------------------------------------------------------

    def get_minibatches(self, minibatch_size: int):
        """
        Yield shuffled minibatches of size `minibatch_size`.

        Normalises advantages per minibatch (zero-mean, unit-variance).

        Yields:
            dict with keys: obs, actions, log_probs, advantages, returns, values
        """
        assert self._full, "Buffer must be full before iterating."
        indices = torch.randperm(self.batch_size, device=self.device)

        for start in range(0, self.batch_size, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]

            mb_advantages = self.advantages[mb_idx]
            # Normalise advantages (reduces variance)
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            yield {
                "obs":       self.obs[mb_idx],
                "actions":   self.actions[mb_idx],
                "log_probs": self.log_probs[mb_idx],
                "advantages": mb_advantages,
                "returns":   self.returns[mb_idx],
                "values":    self.values[mb_idx],
            }
