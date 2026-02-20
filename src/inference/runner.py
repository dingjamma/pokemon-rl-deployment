"""
runner.py — Loads a saved checkpoint and runs the agent in one environment.

Intended for:
  - Watching the trained agent play  (render_mode="human")
  - Collecting evaluation metrics    (render_mode=None / headless)
  - Generating recordings            (render_mode="rgb_array" + save frames)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.agent.simple_agent import SimpleAgent
from src.env.pokemon_env import PokemonRedEnv, make_env


class InferenceRunner:
    """
    Runs a trained Pokemon Red agent for one or more episodes.

    Args:
        checkpoint_path: Path to a .pt checkpoint saved by Trainer.
        rom_path:        Path to PokemonRed.gb ROM.
        device:          Torch device; auto-detected if None.
        render:          If True, display the Game Boy screen via PyBoy's
                         SDL2 window (requires a display).
        frame_delay:     Seconds to sleep between rendered frames (≈1/fps).
        max_steps:       Max steps per episode before forced termination.
    """

    def __init__(
        self,
        checkpoint_path: str,
        rom_path: str = "./roms/PokemonRed.gb",
        device: Optional[torch.device] = None,
        render: bool = False,
        frame_delay: float = 0.016,  # ~60 fps
        max_steps: int = 10_000,
        obs_height: int = 84,
        obs_width: int = 84,
        frame_stack: int = 4,
        frame_skip: int = 24,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.rom_path        = rom_path
        self.render          = render
        self.frame_delay     = frame_delay
        self.max_steps       = max_steps

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Load checkpoint
        ckpt = torch.load(str(self.checkpoint_path), map_location=self.device)

        # Reconstruct the agent from stored config when available
        config      = ckpt.get("config", {})
        env_cfg     = config.get("env", {})
        obs_h       = env_cfg.get("obs_height",  obs_height)
        obs_w       = env_cfg.get("obs_width",   obs_width)
        n_stack     = env_cfg.get("frame_stack", frame_stack)

        obs_shape   = (n_stack, obs_h, obs_w)
        self.agent  = SimpleAgent(obs_shape=obs_shape, num_actions=8)
        self.agent.load_state_dict(ckpt["agent_state"])
        self.agent.to(self.device)
        self.agent.eval()

        self.env_kwargs = dict(
            rom_path    = rom_path,
            obs_height  = obs_h,
            obs_width   = obs_w,
            frame_stack = n_stack,
            frame_skip  = int(env_cfg.get("frame_skip", frame_skip)),
            max_steps   = max_steps,
            headless    = not render,
        )

        print(
            f"[InferenceRunner] Loaded checkpoint from {self.checkpoint_path}\n"
            f"  step={ckpt.get('global_step', '?')}  "
            f"update={ckpt.get('update', '?')}  "
            f"device={self.device}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_episode(self) -> dict:
        """
        Play one episode and return a summary dict.

        Returns:
            {
              "total_reward":  float,
              "steps":         int,
              "badges":        int,     # badges at episode end
              "tiles_seen":    int,     # unique (map, x, y) tiles visited
            }
        """
        env = make_env(**self.env_kwargs)
        obs, _ = env.reset()

        total_reward = 0.0
        steps        = 0

        while True:
            obs_t  = torch.from_numpy(obs[np.newaxis]).to(self.device)  # add batch dim
            action = self.agent.act(obs_t).item()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps        += 1

            if self.render:
                time.sleep(self.frame_delay)

            if terminated or truncated:
                break

        summary = {
            "total_reward": total_reward,
            "steps":        steps,
            "badges":       bin(info.get("badges", 0)).count("1"),
            "tiles_seen":   info.get("tiles_seen", 0),
        }
        env.close()
        return summary

    def run(self, num_episodes: int = 1) -> list[dict]:
        """
        Run `num_episodes` episodes and print a summary table.

        Returns:
            List of per-episode summary dicts.
        """
        summaries = []
        for ep in range(1, num_episodes + 1):
            print(f"  Episode {ep}/{num_episodes} ...", end=" ", flush=True)
            s = self.run_episode()
            summaries.append(s)
            print(
                f"reward={s['total_reward']:.2f}  "
                f"steps={s['steps']}  "
                f"badges={s['badges']}  "
                f"tiles={s['tiles_seen']}"
            )

        if num_episodes > 1:
            mean_reward = sum(s["total_reward"] for s in summaries) / len(summaries)
            mean_badges = sum(s["badges"]       for s in summaries) / len(summaries)
            print(
                f"\n  Mean reward: {mean_reward:.2f}  "
                f"Mean badges: {mean_badges:.2f}"
            )

        return summaries
