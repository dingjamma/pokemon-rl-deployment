# Train RL agents to play Pokemon Red

> **Status (March 2026):** Training live! Running on a local machine with 16 parallel environments. Tracking progress with TensorBoard and W&B.

## Watch the Video on Youtube!

<p float="left">
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/youtube.jpg?raw=true" height="192">
  </a>
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/poke_map.gif?raw=true" height="192">
  </a>
</p>

## Running the Pretrained Model Interactively 🎮
🐍 Python 3.10+ is recommended. Other versions may work but have not been tested.
You also need to install ffmpeg and have it available in the command line.

### Windows Setup
Refer to this [Windows Setup Guide](windows-setup-guide.md)

### Linux / MacOS
1. Copy your legally obtained Pokemon Red ROM into the base directory. You can find this using google, it should be 1MB. Rename it to `PokemonRed.gb` if it is not already. The sha1 sum should be `ea9bcae617fdf159b045185467ae58b2e4a48b9a`, which you can verify by running `shasum PokemonRed.gb`.
2. Move into the `baselines/` directory:
 ```cd baselines```
3. Install dependencies:
```pip install -r requirements.txt```
It may be necessary in some cases to separately install the SDL libraries.
4. Run:
```python run_pretrained_interactive.py```

Interact with the emulator using the arrow keys and the `a` and `s` keys (A and B buttons).
You can pause the AI's input during the game by editing `agent_enabled.txt`

Note: the Pokemon.gb file MUST be in the main directory and your current directory MUST be the `baselines/` directory in order for this to work.

## Training the Model 🏋️

<img src="/assets/grid.png?raw=true" height="156">

Run:
```python run_baseline_parallel_fast.py```

## Tracking Training Progress 📈

### Local Metrics
The current state of each game is rendered to images in the session directory.
You can track the progress in tensorboard by moving into the session directory and running:
```tensorboard --logdir .```
You can then navigate to `localhost:6006` in your browser to view metrics.
W&B live training dashboard: https://wandb.ai/dingjamma/pokemon-train

## Static Visualization 🐜
Map visualization code can be found in `visualization/` directory.

## Supporting Libraries
Check out these awesome projects!
### [PyBoy](https://github.com/Baekalfen/PyBoy)
<a href="https://github.com/Baekalfen/PyBoy">
  <img src="/assets/pyboy.svg" height="64">
</a>

### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
<a href="https://github.com/DLR-RM/stable-baselines3">
  <img src="/assets/sblogo.png" height="64">
</a>
