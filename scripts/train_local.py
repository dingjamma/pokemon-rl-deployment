#!/usr/bin/env python3
"""
train_local.py — Main entrypoint for local Pokemon Red RL training.

Usage:
    python scripts/train_local.py                        # use configs/local.yaml
    python scripts/train_local.py --config configs/baseline.yaml
    python scripts/train_local.py --resume checkpoints/checkpoint_update000050.pt
    python scripts/train_local.py --num-envs 8 --total-timesteps 1000000

TensorBoard:
    tensorboard --logdir runs/

Checkpoints are saved to ./checkpoints/ every N updates (see config).
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
import torch

# Ensure project root is on the Python path so `src` imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.simple_agent import SimpleAgent
from src.env.pokemon_env import make_env
from src.training.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply command-line flag overrides on top of the loaded YAML config."""
    t = config.setdefault("training", {})
    e = config.setdefault("env",      {})

    if args.num_envs        is not None: t["num_envs"]        = args.num_envs
    if args.num_workers     is not None: t["num_workers"]     = args.num_workers
    if args.total_timesteps is not None: t["total_timesteps"] = args.total_timesteps
    if args.batch_size      is not None: t["batch_size"]      = args.batch_size
    if args.lr              is not None: t["learning_rate"]   = args.lr
    if args.rom             is not None: e["rom_path"]        = args.rom

    if args.run_name:
        config.setdefault("logging", {})["run_name"] = args.run_name

    return config


def build_env_fn(config: dict):
    """Return a zero-argument factory that creates a PokemonRedEnv."""
    e = config.get("env", {})
    reward_cfg = e.get("reward", None)

    def _make():
        return make_env(
            rom_path      = e.get("rom_path",    "./roms/PokemonRed.gb"),
            obs_height    = e.get("obs_height",  84),
            obs_width     = e.get("obs_width",   84),
            grayscale     = e.get("grayscale",   True),
            frame_stack   = e.get("frame_stack", 4),
            frame_skip    = e.get("frame_skip",  24),
            max_steps     = e.get("max_steps",   4096),
            reward_config = reward_cfg,
            headless      = True,
        )
    return _make


def build_agent(config: dict, device: torch.device) -> SimpleAgent:
    e = config.get("env", {})
    obs_shape = (
        e.get("frame_stack", 4),
        e.get("obs_height",  84),
        e.get("obs_width",   84),
    )
    agent = SimpleAgent(obs_shape=obs_shape, num_actions=8)
    print(f"[Agent] SimpleAgent — {agent.count_parameters():,} trainable parameters")
    return agent


def main():
    parser = argparse.ArgumentParser(description="Train Pokemon Red RL agent locally.")
    parser.add_argument(
        "--config", default="configs/local.yaml",
        help="Path to YAML config file (default: configs/local.yaml)",
    )
    parser.add_argument(
        "--resume", default=None, metavar="CHECKPOINT",
        help="Path to checkpoint .pt file to resume training from",
    )
    parser.add_argument("--num-envs",        type=int,   default=None)
    parser.add_argument("--num-workers",     type=int,   default=None)
    parser.add_argument("--total-timesteps", type=int,   default=None)
    parser.add_argument("--batch-size",      type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--rom",             type=str,   default=None,
        help="Override ROM path from config")
    parser.add_argument("--run-name",        type=str,   default=None,
        help="TensorBoard run name (overrides config)")
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    # Load and patch config
    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    print(f"[Config] Loaded from {args.config}")
    print(f"  env.rom_path         = {config['env'].get('rom_path')}")
    print(f"  training.num_envs    = {config['training'].get('num_envs')}")
    print(f"  training.total_steps = {config['training'].get('total_timesteps'):,}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # Build components
    env_fn = build_env_fn(config)
    agent  = build_agent(config, device)

    trainer = Trainer(
        agent          = agent,
        env_fn         = env_fn,
        config         = config,
        device         = device,
        run_dir        = config.get("logging", {}).get("tensorboard_dir", "./runs"),
        checkpoint_dir = config.get("checkpointing", {}).get("save_dir", "./checkpoints"),
    )

    # Resume from checkpoint if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
