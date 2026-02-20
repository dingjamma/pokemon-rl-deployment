# Pokemon Red Reinforcement Learning — Local Training

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A clean, modular RL codebase for training an agent to play Pokemon Red — runs entirely on your local machine.*

[Technical Write-up (Medium)](#) | [Project Report](#) | [arXiv:2502.19920](https://arxiv.org/abs/2502.19920)

---

## Overview

This project takes the recent breakthrough in Pokemon Red RL ([Pleines et al., 2025](https://arxiv.org/abs/2502.19920)) and packages it into a **production-grade, locally-runnable codebase** focused on software engineering best practices:

- Clean Gymnasium environment wrapping PyBoy
- PPO actor-critic with CNN policy trained with PufferLib parallel envs
- TensorBoard logging for every training metric
- Checkpoint saving and resuming
- Simple inference script to watch the trained agent play

**This version runs entirely locally — no cloud accounts or spend required.**  A cloud-deployment layer (AWS EC2/S3) will be added in a future branch.

---

## Project Structure

```
pokemon-rl-deployment/
├── configs/
│   ├── baseline.yaml       # Paper-scale settings (16 envs, 10M steps)
│   └── local.yaml          # Laptop-friendly settings (4 envs, 500K steps)
├── scripts/
│   ├── train_local.py      # Main training entrypoint
│   ├── run_pretrained.py   # Load checkpoint & play / evaluate
│   └── setup_tensorboard.sh
├── src/
│   ├── agent/
│   │   └── simple_agent.py     # CNN actor-critic (Nature DQN + PPO heads)
│   ├── env/
│   │   └── pokemon_env.py      # PyBoy-backed Gymnasium environment
│   ├── training/
│   │   ├── trainer.py          # PPO training loop + checkpointing
│   │   └── rollout.py          # GAE rollout buffer
│   └── inference/
│       └── runner.py           # Episode runner for evaluation / watching
├── notebooks/
│   └── exploration.ipynb   # Env verification, frame viz, checkpoint inspection
├── tests/
│   ├── test_agent.py       # Agent shape, value, gradient tests
│   └── test_env.py         # Environment + RolloutBuffer tests
├── roms/                   # Place PokemonRed.gb here (git-ignored)
├── checkpoints/            # Saved .pt files (git-ignored)
└── requirements.txt
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A **legally-obtained** `PokemonRed.gb` ROM (MD5: `a6924ce1f9ad2228e1c6580779b23878`)
- ~4 GB RAM for 4 parallel envs; ~16 GB for baseline (16 envs)
- No GPU required; CUDA used automatically if available

### 2. Install

```bash
git clone https://github.com/dingjamma/pokemon-rl-deployment
cd pokemon-rl-deployment

python3.11 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Place the ROM

```bash
cp /path/to/PokemonRed.gb ./roms/
```

The `.gitignore` prevents the ROM from being committed.

### 4. Run Training

```bash
# Default: configs/local.yaml (4 envs, 500K steps — fast for testing)
python scripts/train_local.py

# Use baseline config (16 envs, 10M steps — expects multi-core machine)
python scripts/train_local.py --config configs/baseline.yaml

# Override specific settings on the command line
python scripts/train_local.py --num-envs 8 --total-timesteps 2000000

# Resume from a checkpoint
python scripts/train_local.py --resume checkpoints/checkpoint_update000050.pt
```

### 5. Monitor with TensorBoard

In a separate terminal:

```bash
tensorboard --logdir runs/
# Then open http://localhost:6006 in your browser
```

Or use the helper script:

```bash
chmod +x scripts/setup_tensorboard.sh
./scripts/setup_tensorboard.sh
```

**Logged metrics:**

| TensorBoard tag | Description |
|---|---|
| `train/policy_loss` | PPO clipped surrogate loss |
| `train/value_loss` | Critic MSE loss |
| `train/entropy_loss` | Action distribution entropy |
| `train/approx_kl` | Approximate KL divergence |
| `train/learning_rate` | Current (annealed) LR |
| `perf/steps_per_second` | Environment throughput |
| `episode/mean_reward` | Mean cumulative reward per episode |
| `episode/mean_length` | Mean episode length (steps) |
| `episode/mean_badges` | Mean badges collected per episode |

### 6. Run the Agent

```bash
# Auto-pick the latest checkpoint, run 1 episode headlessly
python scripts/run_pretrained.py

# Run 5 evaluation episodes and print summary
python scripts/run_pretrained.py --episodes 5

# Watch the agent play with a Game Boy window (requires display)
python scripts/run_pretrained.py --render

# Point at a specific checkpoint
python scripts/run_pretrained.py --checkpoint checkpoints/checkpoint_update000200.pt
```

### 7. Run Tests

```bash
# All tests (RolloutBuffer + agent; env tests skipped if ROM absent)
pytest tests/ -v

# Only non-ROM tests (always safe in CI)
pytest tests/ -v -m "not requires_rom"

# Include env tests (requires ROM)
ROM_PATH=./roms/PokemonRed.gb pytest tests/ -v
```

---

## Technical Stack

| Component | Library | Notes |
|---|---|---|
| Emulator | [PyBoy](https://github.com/Baekalfen/PyBoy) >=2.0 | Headless Game Boy emulator |
| RL framework | [PufferLib](https://github.com/PufferAI/PufferLib) >=0.7 | Parallel env vectorisation |
| Deep learning | PyTorch >=2.1 | CNN policy + Adam optimiser |
| Gym interface | Gymnasium >=0.29 | Standard env API |
| Logging | TensorBoard >=2.15 | Training metrics |
| Config | PyYAML | YAML-based hyperparameters |

---

## Algorithm Details

### Environment

- **Observation**: 4 stacked 84x84 grayscale frames (standard Atari preprocessing)
- **Actions**: 8 discrete Game Boy buttons — A, B, Up, Down, Left, Right, Start, Select
- **Frame skip**: 24 ticks per step (~0.4 s game-time; gives agent time to act between UI transitions)
- **Episode length**: 4096 steps local / 20480 baseline before forced reset

### Reward Shaping

| Signal | Amount | Trigger |
|---|---|---|
| Badge reward | +4.0 per badge | Each new gym badge earned |
| Exploration | +0.01 | Each new (map, x, y) tile visited |
| Level reward | +0.01 per level | Each party Pokemon level gained |

This is intentionally simple — more sophisticated reward shaping (event flags, Pokedex progress, etc.) will be added in future iterations.

### Policy Network (SimpleAgent)

```
Input: (batch, 4, 84, 84)  uint8 -> normalised to float [0,1]
  Conv2d(4, 32, 8, stride=4) -> ReLU
  Conv2d(32, 64, 4, stride=2) -> ReLU
  Conv2d(64, 64, 3, stride=1) -> ReLU
  Flatten -> Linear(3136->512) -> ReLU
    Actor:  Linear(512->8)   -- action logits
    Critic: Linear(512->1)  -- state value V(s)
```

~1.7M parameters.  Weights initialised with orthogonal init (PPO best practice).

### Training (PPO)

- **Batch collection**: `num_envs` environments each run `batch_size // num_envs` steps
- **GAE**: lambda=0.95, gamma=0.99, computed in a single vectorised backward pass
- **PPO update**: 4 epochs over the batch, shuffled minibatches, clipped ratio eps=0.2
- **Value clipping**: clipped V-loss per OpenAI baseline
- **Entropy bonus**: coefficient 0.01 to encourage exploration
- **LR annealing**: linear decay to 0 over training

---

## Configuration Reference

Both YAML configs share the same schema.  `local.yaml` uses smaller values; `baseline.yaml` uses paper-scale values.

```yaml
env:
  rom_path:    ./roms/PokemonRed.gb
  frame_skip:  24
  obs_height:  84
  obs_width:   84
  grayscale:   true
  frame_stack: 4
  max_steps:   4096
  reward:
    badge_reward:   4.0
    explore_reward: 0.01
    level_reward:   0.01

training:
  num_envs:        4
  num_workers:     2
  total_timesteps: 500_000
  batch_size:      2048
  minibatch_size:  256
  update_epochs:   4
  learning_rate:   2.5e-4
  anneal_lr:       true
  gamma:           0.99
  gae_lambda:      0.95
  clip_coef:       0.2
  ent_coef:        0.01
  vf_coef:         0.5
  max_grad_norm:   0.5

checkpointing:
  save_dir:      ./checkpoints
  save_interval: 50

logging:
  tensorboard_dir: ./runs
  run_name:        local
  log_interval:    5
```

---

## Local vs. Cloud: Known Simplifications

This version intentionally omits cloud infrastructure:

| Feature | Local version | Cloud version (planned) |
|---|---|---|
| Parallelism | Multi-process on one machine | Multi-machine distributed |
| Checkpoints | `./checkpoints/` on disk | AWS S3 |
| Metrics | Local TensorBoard | CloudWatch + TensorBoard |
| Fault tolerance | Manual resume via `--resume` | Spot-instance auto-recovery |
| Scaling | Limited by local RAM/CPU | EC2 c5.4xlarge (16 vCPU) |
| Containerisation | None | Docker |

---

## Roadmap

- [x] Local training loop with PufferLib + PPO
- [x] TensorBoard logging
- [x] Checkpoint save/resume
- [x] Inference / evaluation script
- [ ] Better reward shaping (event flags, Pokedex)
- [ ] Evaluation episodes interleaved during training
- [ ] Hyperparameter sweeps (Optuna)
- [ ] Unit tests for training loop
- [ ] AWS deployment (EC2 + S3 + CloudWatch)
- [ ] Docker containerisation

---

## References

1. Pleines et al., *"Pokemon Red via Reinforcement Learning"* (2025) — [arXiv:2502.19920](https://arxiv.org/abs/2502.19920)
2. [Pokemon RL Website](https://drubinstein.github.io/pokerl/)
3. [PufferLib](https://github.com/PufferAI/PufferLib)
4. [Original PWhiddy Implementation](https://github.com/PWhiddy/PokemonRedExperiments)
5. Schulman et al., *"Proximal Policy Optimization Algorithms"* (2017)
6. Mnih et al., *"Human-level control through deep reinforcement learning"* (2015)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

*Pokemon is (c) Nintendo/Game Freak. This project is for educational purposes.*

## Author

**James Ding** — [GitHub](https://github.com/dingjamma) · [LinkedIn](https://linkedin.com/jam-ding) · [Medium](https://medium.com/@dingjamma)
