"""Record scripted demonstrations for behavior cloning.

Each episode is saved as data/demos/episode_XXXX.npz with:
    qpos     (T, 6)        joint positions the robot measured (what a policy sees)
    front    (T, H, W, 3)  front camera image, uint8
    wrist    (T, H, W, 3)  wrist camera image, uint8
    action   (T, 6)        joint targets the expert commanded (what a policy must output)

Optional "noise injection": the robot *executes* expert action + noise, but we
*record* the clean expert action as the label. That way the demos include
slightly-off states together with the action that steers back on track, which
makes the cloned policy more robust (the idea behind DART, Laskey et al. 2017).

Run:  python -m robot_arm_sim.record --episodes 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from robot_arm_sim.env import ROOT, PickPlaceEnv
from robot_arm_sim.scripted import ScriptedPolicy

DEMO_DIR = ROOT / "data" / "demos"


def record_episode(env: PickPlaceEnv, policy: ScriptedPolicy, seed: int, noise_std: float, rng) -> dict:
    obs = env.reset(seed=seed)
    policy.reset(obs)
    buf = {"qpos": [], "front": [], "wrist": [], "action": []}
    noise = np.zeros(6)
    for _ in range(env.max_steps):
        action = policy(obs)
        buf["qpos"].append(obs["qpos"])
        buf["front"].append(obs["front"])
        buf["wrist"].append(obs["wrist_cam"])
        buf["action"].append(action)
        # Smoothly varying noise on the arm joints (not the gripper).
        noise = 0.9 * noise + np.sqrt(1 - 0.9**2) * rng.normal(0, noise_std, size=6)
        noise[5] = 0.0
        obs = env.step(action + noise)
    episode = {k: np.stack(v) for k, v in buf.items()}
    episode["success"] = np.array(env.is_success())
    episode["seed"] = np.array(seed)
    return episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--noise-std", type=float, default=0.01, help="radians, 0 to disable")
    parser.add_argument("--seed-offset", type=int, default=100_000, help="keeps demo seeds apart from eval seeds")
    parser.add_argument("--out", default=str(DEMO_DIR))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = PickPlaceEnv(image_size=args.image_size)
    policy = ScriptedPolicy(env)
    rng = np.random.default_rng(0)

    saved, failed, seed = 0, 0, args.seed_offset
    start = time.time()
    while saved < args.episodes:
        ep = record_episode(env, policy, seed, args.noise_std, rng)
        seed += 1
        if not ep["success"]:
            failed += 1  # only learn from demos that worked
            continue
        np.savez_compressed(out / f"episode_{saved:04d}.npz", **ep)
        saved += 1
        if saved % 20 == 0:
            print(f"{saved}/{args.episodes} saved ({failed} failed demos skipped), {time.time() - start:.0f}s")
    print(f"Done: {saved} demos in {out}, {failed} failed demos skipped")


if __name__ == "__main__":
    main()
