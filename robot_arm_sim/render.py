"""Step 1: load the SO-101 scene, save a still image and a joint-sweep video.

Run:  python -m robot_arm_sim.render
"""

from __future__ import annotations

import argparse

import imageio.v2 as imageio
import numpy as np

from robot_arm_sim.env import HOME_QPOS, JOINT_NAMES, ROOT, PickPlaceEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "videos"))
    args = parser.parse_args()

    env = PickPlaceEnv(render_images=False)
    env.reset(seed=0)
    print(f"Loaded model: {env.model.nq} qpos, {env.model.nu} actuators")
    print("Joints:", ", ".join(JOINT_NAMES))

    imageio.imwrite(f"{args.out_dir}/scene.png", env.render("front"))

    # Sweep each joint in turn around the home pose, so you can see what it does.
    frames = []
    seconds_per_joint = 2.0
    steps = int(seconds_per_joint / env.control_dt)
    for j in range(len(JOINT_NAMES)):
        for k in range(steps):
            action = HOME_QPOS.copy()
            action[j] += 0.6 * np.sin(2 * np.pi * k / steps)
            env.step(action)
            frames.append(env.render("front"))
    imageio.mimsave(f"{args.out_dir}/joint_sweep.mp4", frames, fps=int(1 / env.control_dt))
    print(f"Saved {args.out_dir}/scene.png and {args.out_dir}/joint_sweep.mp4")
    env.close()


if __name__ == "__main__":
    main()
