"""Inverse kinematics for the SO-101 arm.

Inverse kinematics (IK) answers: "which joint angles put the gripper at this
position (and orientation)?". Forward kinematics (FK) is the easy direction:
joint angles -> gripper pose, which MuJoCo computes for us.

We solve IK iteratively with *damped least squares* (DLS):

    1. Compute the pose error e = target - current            (6 numbers)
    2. Ask MuJoCo for the Jacobian J (how the pose changes per joint, 6 x n)
    3. Take a step  dq = J^T (J J^T + lambda^2 I)^-1 e
    4. Clip to joint limits, repeat until the error is tiny.

The damping term lambda keeps steps small near singularities (e.g. when the
arm is fully stretched out), where a plain pseudo-inverse would explode.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# The five joints that position and orient the gripper. The 6th actuator
# ("gripper") only opens/closes the jaw, so IK never touches it.
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
EE_SITE = "gripperframe"  # a site at the fingertips, defined in so101.xml


@dataclass
class IKResult:
    qpos: np.ndarray  # solution for the 5 arm joints (radians)
    success: bool  # True if both errors are under tolerance
    pos_error: float  # metres
    rot_error: float  # radians (0 if orientation was not requested)
    iterations: int


def topdown_rotation(yaw: float) -> np.ndarray:
    """Gripper orientation that points straight down, jaws closing along `yaw`.

    In the `gripperframe` site, +x points out of the fingertips (approach
    direction) and +z is the direction the jaws open/close along.
    """
    x_axis = np.array([0.0, 0.0, -1.0])  # approach: straight down
    z_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])  # jaw closing axis
    y_axis = np.cross(z_axis, x_axis)  # completes a right-handed frame
    return np.column_stack([x_axis, y_axis, z_axis])


def orientation_error(target_mat: np.ndarray, current_mat: np.ndarray) -> np.ndarray:
    """Rotation-vector error (axis * angle, world frame) from current to target."""
    q_target = np.empty(4)
    q_current = np.empty(4)
    mujoco.mju_mat2Quat(q_target, target_mat.reshape(9))
    mujoco.mju_mat2Quat(q_current, current_mat.reshape(9))
    q_conj = np.empty(4)
    mujoco.mju_negQuat(q_conj, q_current)
    q_err = np.empty(4)
    mujoco.mju_mulQuat(q_err, q_target, q_conj)
    err = np.empty(3)
    mujoco.mju_quat2Vel(err, q_err, 1.0)
    return err


class IKSolver:
    """Damped-least-squares IK on a private copy of the MuJoCo state.

    Using a private MjData means solving IK never disturbs the running
    simulation.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site: str = EE_SITE,
        joints: tuple[str, ...] = ARM_JOINTS,
        damping: float = 0.02,
        rot_weight: float = 0.25,
        max_step: float = 0.2,
    ):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site).id
        self.joint_ids = np.array([model.joint(j).id for j in joints])
        self.qpos_adr = model.jnt_qposadr[self.joint_ids]
        self.dof_adr = model.jnt_dofadr[self.joint_ids]
        self.lower = model.jnt_range[self.joint_ids, 0].copy()
        self.upper = model.jnt_range[self.joint_ids, 1].copy()
        self.damping = damping
        # Orientation error is in radians, position error in metres. 1 rad of
        # rotation is "worth" rot_weight metres when balancing the two.
        self.rot_weight = rot_weight
        self.max_step = max_step  # max joint change per iteration (radians)

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: 5 arm joint angles -> (site position, site rotation)."""
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy(), self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def solve(
        self,
        target_pos: np.ndarray,
        target_mat: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        max_iters: int = 300,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-2,
    ) -> IKResult:
        """Find joint angles reaching `target_pos` (and `target_mat` if given)."""
        target_pos = np.asarray(target_pos, dtype=float)
        q = np.zeros(len(self.joint_ids)) if q_init is None else np.array(q_init, dtype=float)
        q = np.clip(q, self.lower, self.upper)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        pos_err = rot_err = np.inf
        for it in range(max_iters):
            pos, mat = self.forward(q)
            e_pos = target_pos - pos
            pos_err = float(np.linalg.norm(e_pos))
            if target_mat is not None:
                e_rot = orientation_error(target_mat, mat)
                rot_err = float(np.linalg.norm(e_rot))
            else:
                rot_err = 0.0
            if pos_err < pos_tol and rot_err < rot_tol:
                return IKResult(q, True, pos_err, rot_err, it)

            # Jacobian needs body/com quantities from mj_comPos.
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            if target_mat is not None:
                J = np.vstack([jacp[:, self.dof_adr], self.rot_weight * jacr[:, self.dof_adr]])
                e = np.concatenate([e_pos, self.rot_weight * e_rot])
            else:
                J = jacp[:, self.dof_adr]
                e = e_pos

            dq = self._dls_step(J, e)
            # If a joint is pinned at a limit and the step pushes it further
            # out, that joint can't help: drop it and let the others do the work.
            # (Without this, the solver often stalls against a limit.)
            blocked = ((q <= self.lower + 1e-6) & (dq < 0)) | ((q >= self.upper - 1e-6) & (dq > 0))
            if blocked.any():
                J = J.copy()
                J[:, blocked] = 0.0
                dq = self._dls_step(J, e)
            # Limit step size for stability, then respect joint limits.
            step = np.max(np.abs(dq))
            if step > self.max_step:
                dq *= self.max_step / step
            q = np.clip(q + dq, self.lower, self.upper)

        return IKResult(q, pos_err < pos_tol and rot_err < rot_tol, pos_err, rot_err, max_iters)

    def _dls_step(self, J: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 e."""
        JJt = J @ J.T + (self.damping**2) * np.eye(J.shape[0])
        return J.T @ np.linalg.solve(JJt, e)

    def solve_with_restarts(
        self,
        target_pos: np.ndarray,
        target_mat: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        restarts: int = 20,
        seed: int = 0,
        **kwargs,
    ) -> IKResult:
        """Try a warm start first, then random initial guesses; keep the best."""
        rng = np.random.default_rng(seed)
        best = self.solve(target_pos, target_mat, q_init, **kwargs)
        for _ in range(restarts):
            if best.success:
                break
            q0 = rng.uniform(self.lower, self.upper)
            res = self.solve(target_pos, target_mat, q0, **kwargs)
            if res.pos_error + 0.1 * res.rot_error < best.pos_error + 0.1 * best.rot_error:
                best = res
        return best
