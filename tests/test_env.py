"""Tests for the environment and the scripted expert."""

import numpy as np

from robot_arm_sim.env import CUBE_ANGLE, CUBE_RADIUS, PickPlaceEnv
from robot_arm_sim.scripted import ScriptedPolicy, run_episode


def test_reset_places_cube_in_spawn_region():
    env = PickPlaceEnv(render_images=False)
    for seed in range(20):
        env.reset(seed=seed)
        x, y, z = env.cube_pos
        r, a = np.hypot(x, y), np.arctan2(y, x)
        assert CUBE_RADIUS[0] - 0.005 <= r <= CUBE_RADIUS[1] + 0.005
        assert CUBE_ANGLE[0] - 0.05 <= a <= CUBE_ANGLE[1] + 0.05
        assert abs(z - 0.015) < 0.003  # resting on the floor


def test_same_seed_same_episode():
    env = PickPlaceEnv(render_images=False)
    env.reset(seed=3)
    first = env.cube_pos
    env.reset(seed=4)
    env.reset(seed=3)
    np.testing.assert_allclose(env.cube_pos, first)


def test_observation_shapes():
    env = PickPlaceEnv(image_size=64)
    obs = env.reset(seed=0)
    assert obs["qpos"].shape == (6,)
    assert obs["front"].shape == (64, 64, 3)
    assert obs["wrist_cam"].shape == (64, 64, 3)
    env.close()


def test_untouched_cube_is_not_success():
    env = PickPlaceEnv(render_images=False)
    env.reset(seed=0)
    assert not env.is_success()


def test_scripted_policy_succeeds():
    env = PickPlaceEnv(render_images=False)
    policy = ScriptedPolicy(env)
    results = [run_episode(env, policy, seed=s)[0] for s in range(5)]
    assert all(results)
