"""Pick-and-place environment around the MuJoCo SO-101 model.

The robot is controlled the same way as the real SO-101: every control step we
send 6 *target joint positions* (5 arm joints + gripper) to position-controlled
servos. MuJoCo's <position> actuators play the role of the STS3215 servos.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = ROOT / "assets" / "so101" / "scene_pick_place.xml"

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CAMERAS = ("front", "wrist_cam")

# A comfortable "ready" pose: gripper above the workspace, pointing down, jaw open.
# (Close to the IK solution for the point (0.16, 0, 0.07); see DECISIONS.md.)
HOME_QPOS = np.array([0.0, -0.75, 0.70, 1.55, 0.0, 1.0])

GRIPPER_OPEN = 1.0  # rad -> ~9 cm between fingertips
GRIPPER_CLOSED = -0.17  # commanded past the cube so the servo squeezes it

CUBE_HALF = 0.015  # cube is 3 cm wide
# The cube spawns in a wedge in front of the robot (polar coordinates around the base).
CUBE_RADIUS = (0.15, 0.22)
CUBE_ANGLE = np.deg2rad((-40.0, 15.0))
# Fixed place location (green disc), to the robot's left.
TARGET_POS = np.array([0.127, 0.127, 0.0])
SUCCESS_RADIUS = 0.03  # cube centre must end within 3 cm of the target centre


class PickPlaceEnv:
    def __init__(
        self,
        image_size: int = 96,
        control_dt: float = 0.05,
        max_steps: int = 200,
        render_images: bool = True,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.n_substeps = int(round(control_dt / self.model.opt.timestep))
        self.control_dt = control_dt
        self.max_steps = max_steps
        self.image_size = image_size
        self.render_images = render_images
        self._renderer = mujoco.Renderer(self.model, image_size, image_size) if render_images else None
        self._video_renderers: dict[tuple[int, int], mujoco.Renderer] = {}

        m = self.model
        self.qpos_adr = np.array([m.jnt_qposadr[m.joint(j).id] for j in JOINT_NAMES])
        self.cube_qpos_adr = m.jnt_qposadr[m.joint("cube_joint").id]
        self.cube_body = m.body("cube").id
        self.ctrl_low = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = m.actuator_ctrlrange[:, 1].copy()
        # Bodies that make up the gripper (used to check the cube was released).
        self.gripper_bodies = {m.body("gripper").id, m.body("moving_jaw_so101_v1").id, m.body("camera_mount").id}

        self.rng = np.random.default_rng()
        self.steps = 0
        self.cube_yaw = 0.0

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        # Robot starts near home, with a little noise so demos are not identical.
        q0 = HOME_QPOS.copy()
        q0[:5] += self.rng.uniform(-0.05, 0.05, size=5)
        self.data.qpos[self.qpos_adr] = q0
        self.data.ctrl[:] = q0

        # Random cube position and yaw. A cube looks the same every 90 degrees,
        # so yaw in [0, 90) covers every case.
        r = self.rng.uniform(*CUBE_RADIUS)
        a = self.rng.uniform(*CUBE_ANGLE)
        self.cube_yaw = self.rng.uniform(0, np.pi / 2)
        adr = self.cube_qpos_adr
        self.data.qpos[adr : adr + 3] = [r * np.cos(a), r * np.sin(a), CUBE_HALF]
        self.data.qpos[adr + 3 : adr + 7] = [np.cos(self.cube_yaw / 2), 0, 0, np.sin(self.cube_yaw / 2)]
        self.data.mocap_pos[0] = TARGET_POS

        mujoco.mj_forward(self.model, self.data)
        # Let everything settle for a moment.
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        self.steps = 0
        return self.get_obs()

    # ------------------------------------------------------------------- step
    def step(self, action: np.ndarray) -> dict:
        """Apply 6 target joint positions for one control step (control_dt seconds)."""
        self.data.ctrl[:] = np.clip(action, self.ctrl_low, self.ctrl_high)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        return self.get_obs()

    # ------------------------------------------------------------ observation
    def get_obs(self) -> dict:
        obs = {"qpos": self.data.qpos[self.qpos_adr].astype(np.float32).copy()}
        if self.render_images:
            for cam in CAMERAS:
                self._renderer.update_scene(self.data, cam)
                obs[cam] = self._renderer.render().copy()
        return obs

    def render(self, camera: str = "front", height: int = 480, width: int = 640) -> np.ndarray:
        """High-resolution frame for videos."""
        key = (height, width)
        if key not in self._video_renderers:
            self._video_renderers[key] = mujoco.Renderer(self.model, height, width)
        r = self._video_renderers[key]
        r.update_scene(self.data, camera)
        return r.render()

    # ---------------------------------------------------------------- helpers
    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body].copy()

    @property
    def target_pos(self) -> np.ndarray:
        return self.data.mocap_pos[0].copy()

    def cube_touching_gripper(self) -> bool:
        for c in self.data.contact[: self.data.ncon]:
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if (b1 == self.cube_body and b2 in self.gripper_bodies) or (
                b2 == self.cube_body and b1 in self.gripper_bodies
            ):
                return True
        return False

    def is_success(self) -> bool:
        """Cube resting on the floor inside the target circle, and let go of."""
        p = self.cube_pos
        near = np.linalg.norm(p[:2] - self.target_pos[:2]) < SUCCESS_RADIUS
        on_floor = p[2] < CUBE_HALF + 0.005
        return bool(near and on_floor and not self.cube_touching_gripper())

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        for r in self._video_renderers.values():
            r.close()
