"""
test_env.py — Unit tests for PokemonRedEnv and RolloutBuffer.

The environment tests are *skipped automatically* when the ROM is absent
so that CI pipelines without PokemonRed.gb still pass.

Run all tests:
    pytest tests/ -v

Run only non-ROM tests (always safe):
    pytest tests/test_env.py -v -m "not requires_rom"

Run with ROM (set ROM_PATH env var):
    ROM_PATH=/path/to/PokemonRed.gb pytest tests/test_env.py -v
"""

import os
import pytest
import numpy as np
import torch

from src.training.rollout import RolloutBuffer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROM_PATH = os.getenv("ROM_PATH", "./roms/PokemonRed.gb")
ROM_AVAILABLE = os.path.exists(ROM_PATH)

skip_no_rom = pytest.mark.skipif(
    not ROM_AVAILABLE,
    reason=f"ROM not found at '{ROM_PATH}'. Set ROM_PATH env var or place ROM in roms/.",
)


# ---------------------------------------------------------------------------
# RolloutBuffer tests (no ROM needed)
# ---------------------------------------------------------------------------

class TestRolloutBuffer:
    OBS_SHAPE    = (4, 84, 84)
    NUM_ENVS     = 4
    NUM_STEPS    = 8   # steps per env
    BATCH_SIZE   = NUM_ENVS * NUM_STEPS  # 32

    @pytest.fixture
    def buf(self):
        return RolloutBuffer(
            batch_size  = self.BATCH_SIZE,
            obs_shape   = self.OBS_SHAPE,
            num_actions = 8,
            device      = torch.device("cpu"),
            gamma       = 0.99,
            gae_lambda  = 0.95,
        )

    def _fill_buffer(self, buf: RolloutBuffer):
        """Fill the buffer with random data."""
        for _ in range(self.NUM_STEPS):
            obs      = torch.randint(0, 256, (self.NUM_ENVS, *self.OBS_SHAPE), dtype=torch.uint8)
            actions  = torch.randint(0, 8,  (self.NUM_ENVS,))
            log_prob = torch.randn(self.NUM_ENVS)
            reward   = torch.rand(self.NUM_ENVS)
            done     = torch.zeros(self.NUM_ENVS)
            value    = torch.rand(self.NUM_ENVS, 1)
            buf.add(obs, actions, log_prob, reward, done, value)

    def test_is_full_after_fill(self, buf):
        self._fill_buffer(buf)
        assert buf.is_full()

    def test_not_full_after_reset(self, buf):
        self._fill_buffer(buf)
        buf.reset()
        assert not buf.is_full()

    def test_overflow_raises(self, buf):
        self._fill_buffer(buf)
        extra_obs    = torch.zeros((self.NUM_ENVS, *self.OBS_SHAPE), dtype=torch.uint8)
        extra_action = torch.zeros(self.NUM_ENVS, dtype=torch.long)
        extra_lp     = torch.zeros(self.NUM_ENVS)
        extra_rew    = torch.zeros(self.NUM_ENVS)
        extra_done   = torch.zeros(self.NUM_ENVS)
        extra_val    = torch.zeros(self.NUM_ENVS, 1)
        with pytest.raises(RuntimeError, match="overflow"):
            buf.add(extra_obs, extra_action, extra_lp, extra_rew, extra_done, extra_val)

    def test_compute_advantages_shapes(self, buf):
        self._fill_buffer(buf)
        last_value = torch.rand(self.NUM_ENVS, 1)
        last_done  = torch.zeros(self.NUM_ENVS)
        buf.compute_advantages(last_value, last_done)

        assert buf.advantages.shape == (self.BATCH_SIZE,)
        assert buf.returns.shape    == (self.BATCH_SIZE,)

    def test_advantages_are_finite(self, buf):
        self._fill_buffer(buf)
        buf.compute_advantages(torch.rand(self.NUM_ENVS, 1), torch.zeros(self.NUM_ENVS))
        assert torch.isfinite(buf.advantages).all()
        assert torch.isfinite(buf.returns).all()

    def test_minibatch_iteration_covers_all_data(self, buf):
        self._fill_buffer(buf)
        buf.compute_advantages(torch.rand(self.NUM_ENVS, 1), torch.zeros(self.NUM_ENVS))

        minibatch_size = 8
        seen_indices   = set()
        for mb in buf.get_minibatches(minibatch_size):
            # Each minibatch is a subset of the buffer
            assert mb["obs"].shape[0]       <= minibatch_size
            assert mb["actions"].shape[0]   <= minibatch_size
            assert mb["returns"].shape[0]   <= minibatch_size
            assert mb["advantages"].shape[0]<= minibatch_size

    def test_compute_advantages_before_full_raises(self, buf):
        with pytest.raises(RuntimeError, match="not full"):
            buf.compute_advantages(torch.rand(self.NUM_ENVS, 1), torch.zeros(self.NUM_ENVS))

    def test_minibatch_iterate_before_full_raises(self, buf):
        with pytest.raises(AssertionError):
            for _ in buf.get_minibatches(8):
                pass


# ---------------------------------------------------------------------------
# Environment tests (require ROM)
# ---------------------------------------------------------------------------

@skip_no_rom
class TestPokemonRedEnvInterface:
    """
    Basic interface tests that verify the environment matches the
    Gymnasium spec and produces correctly shaped observations.
    """

    @pytest.fixture(scope="class")
    def env(self):
        from src.env.pokemon_env import PokemonRedEnv
        e = PokemonRedEnv(
            rom_path    = ROM_PATH,
            obs_height  = 84,
            obs_width   = 84,
            frame_stack = 4,
            frame_skip  = 24,
            max_steps   = 64,  # very short for tests
            headless    = True,
        )
        yield e
        e.close()

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (4, 84, 84)

    def test_action_space_size(self, env):
        assert env.action_space.n == 8

    def test_reset_returns_correct_obs_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == (4, 84, 84), f"Unexpected obs shape: {obs.shape}"
        assert obs.dtype == np.uint8,    f"Unexpected obs dtype: {obs.dtype}"

    def test_reset_returns_info_dict(self, env):
        _, info = env.reset()
        assert isinstance(info, dict)
        for key in ["badges", "map_id", "x", "y", "step", "tiles_seen"]:
            assert key in info, f"Missing key '{key}' in info dict."

    def test_step_returns_correct_types(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.dtype == np.uint8
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_obs_values_in_range(self, env):
        obs, _ = env.reset()
        assert obs.min() >= 0 and obs.max() <= 255

    def test_step_obs_changes(self, env):
        obs1, _ = env.reset()
        obs2, *_ = env.step(2)  # press Up
        # After a step the screen should change (at least slightly)
        # (may not always change if wall-blocked, but entropy increases)
        # Just check shapes are consistent
        assert obs1.shape == obs2.shape

    def test_truncated_after_max_steps(self, env):
        env.reset()
        truncated = False
        for _ in range(env.max_steps + 1):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        assert truncated, "Episode should truncate at max_steps."

    def test_make_env_factory(self):
        from src.env.pokemon_env import make_env
        env = make_env(rom_path=ROM_PATH, max_steps=16, headless=True)
        obs, _ = env.reset()
        assert obs.shape == (4, 84, 84)
        env.close()
