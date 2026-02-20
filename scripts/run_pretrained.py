#!/usr/bin/env python3
"""
run_pretrained.py — Load a checkpoint and watch / evaluate the agent.

Usage:
    # Evaluate latest checkpoint silently (no window)
    python scripts/run_pretrained.py --checkpoint checkpoints/checkpoint_update000050.pt

    # Watch the agent play with a visible window (requires display)
    python scripts/run_pretrained.py --checkpoint <path> --render

    # Run 5 evaluation episodes and print stats
    python scripts/run_pretrained.py --checkpoint <path> --episodes 5

    # Auto-find the latest checkpoint in ./checkpoints/
    python scripts/run_pretrained.py --latest
"""

import argparse
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.runner import InferenceRunner


def find_latest_checkpoint(checkpoint_dir: str = "./checkpoints") -> str:
    """Return the path of the most recently modified .pt file in checkpoint_dir."""
    pattern = str(Path(checkpoint_dir) / "*.pt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No .pt checkpoint files found in '{checkpoint_dir}'.\n"
            "Run training first:  python scripts/train_local.py"
        )
    return max(matches, key=lambda p: Path(p).stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate or watch a trained Pokemon Red RL agent."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a specific .pt checkpoint file",
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Automatically use the most recent checkpoint in ./checkpoints/",
    )
    parser.add_argument(
        "--checkpoint-dir", default="./checkpoints",
        help="Directory to search when using --latest (default: ./checkpoints)",
    )
    parser.add_argument(
        "--rom", default="./roms/PokemonRed.gb",
        help="Path to PokemonRed.gb ROM",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes to run (default: 1)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=10_000,
        help="Maximum steps per episode (default: 10000)",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Open a Game Boy window to watch the agent play (requires display)",
    )
    parser.add_argument(
        "--frame-delay", type=float, default=0.016,
        help="Seconds to sleep between frames when rendering (default: 0.016 ≈ 60fps)",
    )
    args = parser.parse_args()

    # Resolve checkpoint path
    if args.latest:
        ckpt_path = find_latest_checkpoint(args.checkpoint_dir)
        print(f"[run_pretrained] Auto-selected checkpoint: {ckpt_path}")
    elif args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # Default: try latest
        try:
            ckpt_path = find_latest_checkpoint(args.checkpoint_dir)
            print(f"[run_pretrained] No --checkpoint given; using: {ckpt_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    runner = InferenceRunner(
        checkpoint_path = ckpt_path,
        rom_path        = args.rom,
        render          = args.render,
        frame_delay     = args.frame_delay,
        max_steps       = args.max_steps,
    )

    print(f"\n[run_pretrained] Running {args.episodes} episode(s) ...\n")
    summaries = runner.run(num_episodes=args.episodes)

    if summaries:
        best = max(summaries, key=lambda s: s["total_reward"])
        print(f"\nBest episode — reward: {best['total_reward']:.2f}  "
              f"badges: {best['badges']}  tiles: {best['tiles_seen']}")


if __name__ == "__main__":
    main()
