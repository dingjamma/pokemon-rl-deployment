"""
trainer.py — PPO training loop coordinating PufferLib envs + SimpleAgent.

Workflow per update:
  1. Collect `batch_size` steps from `num_envs` parallel environments.
  2. Compute GAE advantages using the rollout buffer.
  3. Run `update_epochs` passes over the batch, sampling `minibatch_size`
     mini-batches and applying PPO clipped surrogate + value loss.
  4. Log metrics to TensorBoard and optionally save a checkpoint.

Local simplifications (no cloud):
  - Checkpoints written to local disk only (./checkpoints/).
  - Metrics logged to TensorBoard (./runs/).
  - No spot-instance preemption handling.
  - Single-machine parallelism via PufferLib's vectorized envs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

try:
    import pufferlib
    import pufferlib.vector
    HAS_PUFFERLIB = True
except ImportError:
    HAS_PUFFERLIB = False

from src.agent.simple_agent import SimpleAgent
from src.training.rollout import RolloutBuffer


class Trainer:
    """
    PPO trainer for the Pokemon Red environment.

    Args:
        agent:          The actor-critic policy (SimpleAgent instance).
        env_fn:         Zero-argument callable that returns a fresh env.
        config:         Flat dict of hyperparameters (see configs/local.yaml).
        device:         Torch device; auto-detected if None.
        run_dir:        Directory for TensorBoard logs.
        checkpoint_dir: Directory to save .pt checkpoints.
    """

    def __init__(
        self,
        agent: SimpleAgent,
        env_fn: Callable,
        config: Dict,
        device: Optional[torch.device] = None,
        run_dir: str = "./runs",
        checkpoint_dir: str = "./checkpoints",
    ):
        self.agent  = agent
        self.env_fn = env_fn
        self.config = config

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.agent.to(self.device)

        self.run_dir        = Path(run_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training hyperparams (with safe defaults matching local.yaml)
        t = config.get("training", config)  # support both nested and flat dicts
        self.num_envs        = int(t.get("num_envs",        4))
        self.num_workers     = int(t.get("num_workers",     2))
        self.total_timesteps = int(t.get("total_timesteps", 500_000))
        self.batch_size      = int(t.get("batch_size",      2048))
        self.minibatch_size  = int(t.get("minibatch_size",  256))
        self.update_epochs   = int(t.get("update_epochs",   4))
        self.learning_rate   = float(t.get("learning_rate", 2.5e-4))
        self.anneal_lr       = bool(t.get("anneal_lr",      True))
        self.gamma           = float(t.get("gamma",         0.99))
        self.gae_lambda      = float(t.get("gae_lambda",    0.95))
        self.clip_coef       = float(t.get("clip_coef",     0.2))
        self.ent_coef        = float(t.get("ent_coef",      0.01))
        self.vf_coef         = float(t.get("vf_coef",       0.5))
        self.max_grad_norm   = float(t.get("max_grad_norm", 0.5))

        log_cfg             = config.get("logging", {})
        self.log_interval   = int(log_cfg.get("log_interval",  5))
        ckpt_cfg            = config.get("checkpointing", {})
        self.save_interval  = int(ckpt_cfg.get("save_interval", 50))

        self.num_steps_per_env = self.batch_size // self.num_envs

        # Derived
        self.total_updates = self.total_timesteps // self.batch_size

        self.optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate, eps=1e-5)

        # TensorBoard writer (lazily created in train())
        self.writer: Optional[SummaryWriter] = None

        # Global counters
        self.global_step   = 0
        self.update_count  = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self):
        """
        Run the full PPO training loop.
        Creates vectorized environments, collects rollouts, and updates
        the policy until `total_timesteps` is reached.
        """
        run_name = self.config.get("logging", {}).get("run_name", "run")
        self.writer = SummaryWriter(log_dir=str(self.run_dir / run_name))
        print(
            f"[Trainer] Starting PPO on device={self.device}\n"
            f"  total_timesteps={self.total_timesteps:,}  "
            f"num_envs={self.num_envs}  batch_size={self.batch_size}  "
            f"total_updates={self.total_updates}"
        )

        envs = self._make_envs()

        obs_shape = self.agent.encoder[0].weight.shape  # (out_ch, in_ch, kH, kW)
        # Infer obs shape from the environment directly
        sample_obs, _ = envs.reset()
        obs_shape = sample_obs.shape[1:]  # drop batch dim

        buffer = RolloutBuffer(
            batch_size=self.batch_size,
            obs_shape=obs_shape,
            num_actions=self.num_envs,  # not actually used
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Current observations across all envs
        obs_np = sample_obs
        obs_t  = torch.from_numpy(obs_np).to(self.device)
        dones  = torch.zeros(self.num_envs, device=self.device)

        start_time = time.time()

        for update in range(1, self.total_updates + 1):
            # -- Learning rate annealing --
            if self.anneal_lr:
                frac = 1.0 - (update - 1) / self.total_updates
                lr   = frac * self.learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            # -------------------------------------------------------
            # Phase 1: Collect rollout
            # -------------------------------------------------------
            buffer.reset()
            ep_rewards: list[float] = []
            ep_lengths: list[int]   = []
            ep_badges:  list[int]   = []

            for _ in range(self.num_steps_per_env):
                with torch.no_grad():
                    action, log_prob, _, value = self.agent.get_action_and_value(obs_t)

                # Step all envs
                next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(
                    action.cpu().numpy()
                )
                done_np = terminated_np | truncated_np

                # Track episode stats
                for i, info in enumerate(infos):
                    if done_np[i] and isinstance(info, dict):
                        ep_rewards.append(info.get("episode_reward", 0.0))
                        ep_lengths.append(info.get("episode_length", 0))
                        ep_badges.append(bin(info.get("badges", 0)).count("1"))

                reward_t = torch.from_numpy(reward_np).float().to(self.device)
                done_t   = torch.from_numpy(done_np.astype(float)).float().to(self.device)

                buffer.add(obs_t, action, log_prob, reward_t, done_t, value)

                obs_t = torch.from_numpy(next_obs_np).to(self.device)
                dones = done_t
                self.global_step += self.num_envs

            # Bootstrap value for last step
            with torch.no_grad():
                last_value = self.agent.get_value(obs_t)
            buffer.compute_advantages(last_value, dones)

            # -------------------------------------------------------
            # Phase 2: PPO update
            # -------------------------------------------------------
            pg_losses, vf_losses, ent_losses, approx_kls = [], [], [], []

            for _ in range(self.update_epochs):
                for mb in buffer.get_minibatches(self.minibatch_size):
                    mb_obs      = mb["obs"]
                    mb_actions  = mb["actions"]
                    mb_log_probs= mb["log_probs"]
                    mb_adv      = mb["advantages"]
                    mb_returns  = mb["returns"]
                    mb_values   = mb["values"]

                    _, new_log_prob, entropy, new_value = self.agent.get_action_and_value(
                        mb_obs, mb_actions
                    )

                    log_ratio  = new_log_prob - mb_log_probs
                    ratio      = log_ratio.exp()
                    approx_kl  = ((ratio - 1) - log_ratio).mean()
                    approx_kls.append(approx_kl.item())

                    # Policy loss (PPO-Clip)
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss (clipped)
                    new_value  = new_value.squeeze(-1)
                    vf_loss_unc= (new_value - mb_returns).pow(2)
                    vf_clipped = mb_values + torch.clamp(
                        new_value - mb_values, -self.clip_coef, self.clip_coef
                    )
                    vf_loss_clipped = (vf_clipped - mb_returns).pow(2)
                    vf_loss  = 0.5 * torch.max(vf_loss_unc, vf_loss_clipped).mean()

                    ent_loss = entropy.mean()

                    loss = pg_loss - self.ent_coef * ent_loss + self.vf_coef * vf_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    pg_losses.append(pg_loss.item())
                    vf_losses.append(vf_loss.item())
                    ent_losses.append(ent_loss.item())

            self.update_count += 1

            # -------------------------------------------------------
            # Phase 3: Logging
            # -------------------------------------------------------
            if update % self.log_interval == 0:
                elapsed = time.time() - start_time
                sps     = int(self.global_step / elapsed)

                mean_pg_loss  = sum(pg_losses)  / len(pg_losses)
                mean_vf_loss  = sum(vf_losses)  / len(vf_losses)
                mean_ent_loss = sum(ent_losses) / len(ent_losses)
                mean_kl       = sum(approx_kls) / len(approx_kls)

                self.writer.add_scalar("train/policy_loss",   mean_pg_loss,  self.global_step)
                self.writer.add_scalar("train/value_loss",    mean_vf_loss,  self.global_step)
                self.writer.add_scalar("train/entropy_loss",  mean_ent_loss, self.global_step)
                self.writer.add_scalar("train/approx_kl",     mean_kl,       self.global_step)
                self.writer.add_scalar("train/learning_rate",
                    self.optimizer.param_groups[0]["lr"], self.global_step)
                self.writer.add_scalar("perf/steps_per_second", sps,         self.global_step)

                if ep_rewards:
                    self.writer.add_scalar("episode/mean_reward",
                        sum(ep_rewards) / len(ep_rewards), self.global_step)
                    self.writer.add_scalar("episode/mean_length",
                        sum(ep_lengths) / len(ep_lengths), self.global_step)
                    self.writer.add_scalar("episode/mean_badges",
                        sum(ep_badges)  / len(ep_badges),  self.global_step)

                print(
                    f"  Update {update:5d}/{self.total_updates} | "
                    f"step {self.global_step:9,} | "
                    f"SPS {sps:6,} | "
                    f"pg {mean_pg_loss:.4f}  vf {mean_vf_loss:.4f}  "
                    f"ent {mean_ent_loss:.4f}  kl {mean_kl:.4f}"
                    + (f"  ep_rew {sum(ep_rewards)/len(ep_rewards):.2f}" if ep_rewards else "")
                )

            # -------------------------------------------------------
            # Phase 4: Checkpoint
            # -------------------------------------------------------
            if update % self.save_interval == 0 or update == self.total_updates:
                self._save_checkpoint(update)

        envs.close()
        self.writer.close()
        print(f"[Trainer] Training complete — {self.global_step:,} total steps.")

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, update: int):
        path = self.checkpoint_dir / f"checkpoint_update{update:06d}.pt"
        torch.save(
            {
                "update":       update,
                "global_step":  self.global_step,
                "agent_state":  self.agent.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config":       self.config,
            },
            path,
        )
        print(f"  [Checkpoint] Saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(ckpt["agent_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.global_step  = ckpt.get("global_step", 0)
        self.update_count = ckpt.get("update", 0)
        print(f"[Trainer] Loaded checkpoint from {path} (step {self.global_step:,})")

    # ------------------------------------------------------------------
    # Environment factory
    # ------------------------------------------------------------------

    def _make_envs(self):
        """
        Create vectorized environments.  Uses PufferLib's vectorizer if
        available, otherwise falls back to a simple synchronous wrapper.
        """
        if HAS_PUFFERLIB:
            envs = pufferlib.vector.make(
                self.env_fn,
                num_envs=self.num_envs,
                num_workers=min(self.num_workers, self.num_envs),
                backend=pufferlib.vector.Multiprocessing,
            )
            return PufferLibEnvAdapter(envs, self.num_envs)
        else:
            # Synchronous fallback — slower but zero external deps
            print("[Trainer] PufferLib not found — using SyncVecEnv fallback.")
            return SyncVecEnv([self.env_fn for _ in range(self.num_envs)])


# ---------------------------------------------------------------------------
# Minimal env wrappers
# ---------------------------------------------------------------------------

class PufferLibEnvAdapter:
    """
    Thin adapter so PufferLib vectorized envs match our trainer's expected API:
      reset() → (obs_np, [info])
      step(actions_np) → (obs_np, reward_np, terminated_np, truncated_np, [info])
    """

    def __init__(self, envs, num_envs: int):
        self._envs    = envs
        self.num_envs = num_envs

    def reset(self):
        result = self._envs.reset()
        obs    = result[0] if isinstance(result, tuple) else result
        infos  = [{} for _ in range(self.num_envs)]
        return obs, infos

    def step(self, actions):
        obs, rew, term, trunc, infos = self._envs.step(actions)
        if not isinstance(infos, (list, tuple)):
            infos = [{} for _ in range(self.num_envs)]
        return obs, rew, term, trunc, infos

    def close(self):
        self._envs.close()


class SyncVecEnv:
    """
    Simple synchronous vectorised env — runs N envs sequentially in one process.
    No parallelism; useful for debugging when PufferLib is unavailable.
    """

    def __init__(self, env_fns):
        self._envs = [fn() for fn in env_fns]
        self.num_envs = len(self._envs)
        import numpy as np
        self._np = np

    def reset(self):
        import numpy as np
        results = [env.reset() for env in self._envs]
        obs   = np.stack([r[0] for r in results])
        infos = [r[1] for r in results]
        return obs, infos

    def step(self, actions):
        import numpy as np
        results = [env.step(int(a)) for env, a in zip(self._envs, actions)]
        obs       = np.stack([r[0] for r in results])
        rewards   = np.array([r[1] for r in results], dtype=np.float32)
        terminated= np.array([r[2] for r in results], dtype=bool)
        truncated = np.array([r[3] for r in results], dtype=bool)
        infos     = [r[4] for r in results]
        return obs, rewards, terminated, truncated, infos

    def close(self):
        for env in self._envs:
            env.close()
