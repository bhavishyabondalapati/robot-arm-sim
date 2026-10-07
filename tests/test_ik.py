"""Tests for the inverse kinematics solver."""

import mujoco
import numpy as np
import pytest

from robot_arm_sim.env import CUBE_ANGLE, CUBE_RADIUS, SCENE_XML, TARGET_POS
from robot_arm_sim.ik import IKSolver, orientation_error, topdown_rotation


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE_XML))


@pytest.fixture
def ik(model):
    return IKSolver(model)


def random_configs(ik, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(ik.lower, ik.upper, size=(n, len(ik.lower)))


def test_position_roundtrip(ik):
    """FK of random joints gives a reachable point; IK must find it again."""
    configs = random_configs(ik, 50)
    for i, q in enumerate(configs):
        pos, _ = ik.forward(q)
        res = ik.solve_with_restarts(pos, seed=i)
        assert res.success, f"config {i}: error {res.pos_error:.4f} m"
        # The *solution* may differ from q (several arm poses reach the same
        # point), but its forward kinematics must land on the target.
        reached, _ = ik.forward(res.qpos)
        assert np.linalg.norm(reached - pos) < 1e-3


def test_pose_roundtrip(ik):
    """Same, but matching orientation too. Allow a couple of misses from local minima."""
    configs = random_configs(ik, 50, seed=1)
    solved = 0
    for i, q in enumerate(configs):
        pos, mat = ik.forward(q)
        res = ik.solve_with_restarts(pos, mat, seed=i)
        if res.success:
            solved += 1
            reached_pos, reached_mat = ik.forward(res.qpos)
            assert np.linalg.norm(reached_pos - pos) < 1e-3
            assert np.linalg.norm(orientation_error(mat, reached_mat)) < 1e-2
    assert solved >= 47


def test_solutions_respect_joint_limits(ik):
    for i, q in enumerate(random_configs(ik, 20, seed=2)):
        pos, mat = ik.forward(q)
        res = ik.solve_with_restarts(pos, mat, seed=i)
        assert np.all(res.qpos >= ik.lower - 1e-9)
        assert np.all(res.qpos <= ik.upper + 1e-9)


def test_unreachable_target_reports_failure(ik):
    res = ik.solve_with_restarts(np.array([1.0, 0.0, 0.5]), restarts=2)
    assert not res.success
    assert res.pos_error > 0.3


def test_topdown_grasps_reachable_over_cube_region(ik):
    """Every cube spawn location must allow a top-down grasp (scripted policy relies on it)."""
    for r in np.linspace(*CUBE_RADIUS, 4):
        for a in np.linspace(*CUBE_ANGLE, 5):
            for z in (0.012, 0.07):  # grasp height and lift height
                target = np.array([r * np.cos(a), r * np.sin(a), z])
                res = ik.solve_with_restarts(target, topdown_rotation(a))
                assert res.success, f"r={r:.3f} a={np.rad2deg(a):.0f}deg z={z}"


def test_place_target_reachable(ik):
    for z in (0.03, 0.07):
        res = ik.solve_with_restarts(TARGET_POS + [0, 0, z], topdown_rotation(np.pi / 4))
        assert res.success


def test_topdown_rotation_is_valid():
    for yaw in np.linspace(-np.pi, np.pi, 9):
        R = topdown_rotation(yaw)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
        np.testing.assert_allclose(R[:, 0], [0, 0, -1])  # approach axis points down


def test_orientation_error_matches_known_rotation():
    assert np.allclose(orientation_error(np.eye(3), np.eye(3)), 0)
    angle = 0.3
    Rz = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    np.testing.assert_allclose(orientation_error(Rz, np.eye(3)), [0, 0, angle], atol=1e-9)


def test_ik_does_not_touch_simulation_state(model):
    data = mujoco.MjData(model)
    before = data.qpos.copy()
    IKSolver(model).solve(np.array([0.2, 0.0, 0.05]))
    np.testing.assert_array_equal(data.qpos, before)
