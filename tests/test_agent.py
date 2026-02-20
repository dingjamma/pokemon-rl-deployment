"""
test_agent.py — Unit tests for SimpleAgent.

Run with:  pytest tests/test_agent.py -v
"""

import pytest
import torch
from src.agent.simple_agent import SimpleAgent


OBS_SHAPE   = (4, 84, 84)
NUM_ACTIONS = 8
BATCH_SIZE  = 4


@pytest.fixture(scope="module")
def agent():
    return SimpleAgent(obs_shape=OBS_SHAPE, num_actions=NUM_ACTIONS)


@pytest.fixture(scope="module")
def dummy_obs():
    return torch.randint(0, 256, (BATCH_SIZE, *OBS_SHAPE), dtype=torch.uint8)


class TestSimpleAgentShape:
    """Output shape contracts."""

    def test_get_value_shape(self, agent, dummy_obs):
        value = agent.get_value(dummy_obs)
        assert value.shape == (BATCH_SIZE, 1), f"Expected ({BATCH_SIZE}, 1), got {value.shape}"

    def test_get_action_and_value_shapes(self, agent, dummy_obs):
        action, log_prob, entropy, value = agent.get_action_and_value(dummy_obs)
        assert action.shape   == (BATCH_SIZE,),    f"action shape mismatch: {action.shape}"
        assert log_prob.shape == (BATCH_SIZE,),    f"log_prob shape mismatch: {log_prob.shape}"
        assert entropy.shape  == (BATCH_SIZE,),    f"entropy shape mismatch: {entropy.shape}"
        assert value.shape    == (BATCH_SIZE, 1),  f"value shape mismatch: {value.shape}"

    def test_act_shape(self, agent, dummy_obs):
        actions = agent.act(dummy_obs)
        assert actions.shape == (BATCH_SIZE,)

    def test_act_single_obs(self, agent):
        single = torch.randint(0, 256, (1, *OBS_SHAPE), dtype=torch.uint8)
        action = agent.act(single)
        assert action.shape == (1,)


class TestSimpleAgentValues:
    """Numerical sanity checks."""

    def test_action_in_range(self, agent, dummy_obs):
        action, _, _, _ = agent.get_action_and_value(dummy_obs)
        assert (action >= 0).all() and (action < NUM_ACTIONS).all(), \
            "Sampled actions out of [0, NUM_ACTIONS) range."

    def test_log_prob_is_negative(self, agent, dummy_obs):
        _, log_prob, _, _ = agent.get_action_and_value(dummy_obs)
        assert (log_prob <= 0).all(), "log_prob should be ≤ 0 for a probability."

    def test_entropy_is_non_negative(self, agent, dummy_obs):
        _, _, entropy, _ = agent.get_action_and_value(dummy_obs)
        assert (entropy >= 0).all(), "Entropy should be non-negative."

    def test_entropy_at_most_log_n_actions(self, agent, dummy_obs):
        import math
        _, _, entropy, _ = agent.get_action_and_value(dummy_obs)
        max_entropy = math.log(NUM_ACTIONS)
        assert (entropy <= max_entropy + 1e-5).all(), \
            f"Entropy {entropy.max()} exceeds log(N)={max_entropy:.4f}."

    def test_evaluate_fixed_action(self, agent, dummy_obs):
        """Supplying an explicit action should return its log-prob and entropy."""
        fixed_actions = torch.zeros(BATCH_SIZE, dtype=torch.long)  # always action 0
        action_out, log_prob, entropy, value = agent.get_action_and_value(
            dummy_obs, fixed_actions
        )
        assert (action_out == fixed_actions).all()

    def test_obs_normalisation_does_not_crash(self, agent):
        """White (255) and black (0) frames should not produce NaN."""
        white = torch.full((1, *OBS_SHAPE), 255, dtype=torch.uint8)
        black = torch.zeros((1, *OBS_SHAPE), dtype=torch.uint8)
        for obs in [white, black]:
            a, lp, ent, v = agent.get_action_and_value(obs)
            assert not torch.isnan(v).any(), "NaN in value output."
            assert not torch.isnan(lp).any(), "NaN in log_prob output."


class TestSimpleAgentParameters:
    """Parameter count sanity."""

    def test_has_parameters(self, agent):
        assert agent.count_parameters() > 0

    def test_parameter_count_reasonable(self, agent):
        # CNN + FC for 84×84×4 input with 512 hidden should be well under 10M
        n_params = agent.count_parameters()
        assert n_params < 10_000_000, f"Model unexpectedly large: {n_params:,} params."
        assert n_params > 100_000,    f"Model suspiciously small:  {n_params:,} params."

    def test_parameters_require_grad(self, agent):
        for name, p in agent.named_parameters():
            assert p.requires_grad, f"Parameter {name} does not require grad."


class TestSimpleAgentGradients:
    """Gradient flow through the network."""

    def test_backward_does_not_crash(self, agent):
        obs = torch.randint(0, 256, (2, *OBS_SHAPE), dtype=torch.uint8)
        _, log_prob, entropy, value = agent.get_action_and_value(obs)
        loss = -log_prob.mean() - 0.01 * entropy.mean() + 0.5 * value.pow(2).mean()
        loss.backward()  # should not raise

    def test_gradients_are_finite(self, agent):
        # Fresh agent to avoid leftover .grad from other tests
        fresh = SimpleAgent(obs_shape=OBS_SHAPE, num_actions=NUM_ACTIONS)
        obs  = torch.randint(0, 256, (2, *OBS_SHAPE), dtype=torch.uint8)
        _, log_prob, entropy, value = fresh.get_action_and_value(obs)
        loss = -log_prob.mean() + value.pow(2).mean()
        loss.backward()
        for name, p in fresh.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}."
