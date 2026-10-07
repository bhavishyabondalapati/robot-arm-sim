"""Small sanity tests for the behavior-cloning model (CPU only, fast)."""

import numpy as np
import torch

from robot_arm_sim.policy import BCAgent, BCPolicy, SpatialSoftmax
from robot_arm_sim.train import random_shift


def test_spatial_softmax_finds_the_peak():
    x = torch.full((1, 1, 9, 9), -1e4)
    x[0, 0, 0, 8] = 0.0  # top-right corner
    coords = SpatialSoftmax()(x)
    np.testing.assert_allclose(coords[0].numpy(), [1.0, -1.0], atol=1e-4)  # (x=right, y=top)


def test_policy_output_shape():
    model = BCPolicy(chunk=10)
    out = model(torch.rand(4, 3, 96, 96), torch.rand(4, 3, 96, 96), torch.zeros(4, 6))
    assert out.shape == (4, 10, 6)


def test_random_shift_keeps_shape():
    x = torch.rand(5, 3, 96, 96)
    assert random_shift(x).shape == x.shape


def test_agent_returns_valid_action():
    agent = BCAgent(BCPolicy(chunk=5), torch.device("cpu"))
    agent.reset()
    obs = {
        "front": np.zeros((96, 96, 3), np.uint8),
        "wrist_cam": np.zeros((96, 96, 3), np.uint8),
        "qpos": np.zeros(6, np.float32),
    }
    for _ in range(8):  # more steps than the chunk length exercises the ensembling buffer
        action = agent(obs)
        assert action.shape == (6,)
        assert np.all(np.isfinite(action))
    assert len(agent.history) == 5
