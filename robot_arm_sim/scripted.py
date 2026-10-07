"""Scripted pick-and-place "expert" built on IK.

At the start of an episode it reads the true cube pose from the simulator
(something a real robot would get from a camera + detector), plans a list of
gripper waypoints, converts each to joint angles with IK, and then plays them
back as smooth joint-space motions. It is *open loop*: it does not react if the
cube slips. That makes it a simple, honest baseline.

Run a few episodes and save a video:
    python -m robot_arm_sim.scripted --episodes 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from robot_arm_sim.env import GRIPPER_CLOSED, GRIPPER_OPEN, ROOT, PickPlaceEnv
from robot_arm_sim.ik import IKSolver, topdown_rotation


@dataclass
class Waypoint:
    duration: float  # seconds to move here from the previous waypoint
    arm: np.ndarray  # 5 arm joint angles
    gripper: float  # gripper joint target


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def min_jerk(s: float) -> float:
    """Smooth 0->1 curve with zero velocity and acceleration at both ends."""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class ScriptedPolicy:
    def __init__(
        self,
        env: PickPlaceEnv,
        lift_z: float = 0.07,  # fingertip height while carrying
        grasp_z: float = 0.012,  # fingertip height when grasping (cube centre is 0.015)
        place_z: float = 0.03,  # fingertip height when releasing
    ):
        self.env = env
        self.ik = IKSolver(env.model)
        self.lift_z = lift_z
        self.grasp_z = grasp_z
        self.place_z = place_z
        self.waypoints: list[Waypoint] = []
        self.t = 0.0

    def grasp_yaw(self, cube_pos: np.ndarray, cube_yaw: float) -> float:
        """Pick the jaw angle that lines up with the cube faces and needs the least wrist roll.

        With roll = 0 the jaw closes along the arm's radial direction, so we choose
        the cube face direction (every 90 degrees) closest to that.
        """
        azimuth = np.arctan2(cube_pos[1], cube_pos[0])
        candidates = cube_yaw + np.arange(-4, 5) * np.pi / 2
        return float(candidates[np.argmin([abs(wrap_angle(c - azimuth)) for c in candidates])])

    def _ik(self, pos, yaw, q_prev) -> np.ndarray:
        res = self.ik.solve_with_restarts(np.asarray(pos), topdown_rotation(yaw), q_init=q_prev)
        if not res.success:
            raise RuntimeError(f"IK failed for {np.round(pos, 3)} (err {res.pos_error:.4f} m)")
        return res.qpos

    def reset(self, obs: dict) -> None:
        env = self.env
        cube, target = env.cube_pos, env.target_pos
        yaw = self.grasp_yaw(cube, env.cube_yaw)
        # Keep the same wrist roll while carrying: rotate the jaw yaw with the arm's pan.
        place_yaw = yaw + np.arctan2(target[1], target[0]) - np.arctan2(cube[1], cube[0])

        q = obs["qpos"][:5].astype(float)
        plan = [
            # (duration, xyz, yaw, gripper)
            (1.5, [cube[0], cube[1], self.lift_z], yaw, GRIPPER_OPEN),  # above the cube
            (1.0, [cube[0], cube[1], self.grasp_z], yaw, GRIPPER_OPEN),  # down
            (0.7, [cube[0], cube[1], self.grasp_z], yaw, GRIPPER_CLOSED),  # close jaw
            (1.0, [cube[0], cube[1], self.lift_z], yaw, GRIPPER_CLOSED),  # lift
            (1.5, [target[0], target[1], self.lift_z], place_yaw, GRIPPER_CLOSED),  # carry
            (1.0, [target[0], target[1], self.place_z], place_yaw, GRIPPER_CLOSED),  # lower
            (0.5, [target[0], target[1], self.place_z], place_yaw, GRIPPER_OPEN),  # release
            (0.8, [target[0], target[1], self.lift_z], place_yaw, GRIPPER_OPEN),  # back off
        ]
        self.start = Waypoint(0.0, q, float(obs["qpos"][5]))
        self.waypoints = []
        for duration, pos, wp_yaw, grip in plan:
            q = self._ik(pos, wp_yaw, q)
            self.waypoints.append(Waypoint(duration, q, grip))
        self.t = 0.0

    @property
    def total_duration(self) -> float:
        return sum(w.duration for w in self.waypoints)

    def __call__(self, obs: dict) -> np.ndarray:
        """Return the 6 joint targets for the current time, then advance time."""
        t = self.t
        prev = self.start
        action = None
        for wp in self.waypoints:
            if t < wp.duration:
                s = min_jerk(t / wp.duration)
                arm = prev.arm + s * (wp.arm - prev.arm)
                grip = prev.gripper + s * (wp.gripper - prev.gripper)
                action = np.append(arm, grip)
                break
            t -= wp.duration
            prev = wp
        if action is None:  # finished: hold the last waypoint
            action = np.append(prev.arm, prev.gripper)
        self.t += self.env.control_dt
        return action.astype(np.float32)


def run_episode(env: PickPlaceEnv, policy, seed: int, video_camera: str | None = None):
    """Roll out one episode. Returns (success, frames)."""
    obs = env.reset(seed=seed)
    policy.reset(obs)
    frames = []
    for _ in range(env.max_steps):
        obs = env.step(policy(obs))
        if video_camera:
            frames.append(env.render(video_camera))
    return env.is_success(), frames


def main() -> None:
    import imageio.v2 as imageio

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--video", default=str(ROOT / "videos" / "scripted.mp4"))
    args = parser.parse_args()

    env = PickPlaceEnv(render_images=False)
    policy = ScriptedPolicy(env)
    all_frames, successes = [], []
    for ep in range(args.episodes):
        ok, frames = run_episode(env, policy, seed=args.seed + ep, video_camera="front")
        successes.append(ok)
        all_frames += frames
        print(f"episode {ep}: {'success' if ok else 'FAIL'}  cube ended at {np.round(env.cube_pos, 3)}")
    print(f"success rate: {np.mean(successes):.0%}")
    imageio.mimsave(args.video, all_frames, fps=int(1 / env.control_dt))
    print(f"saved {args.video}")


if __name__ == "__main__":
    main()
