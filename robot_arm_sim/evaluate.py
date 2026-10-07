"""Compare the learned policy with the scripted baseline on the same random episodes.

    python -m robot_arm_sim.evaluate --episodes 50

Both policies see the same 50 cube poses (seeds 0..49; demos used seeds from
100000 and validation used 50000+, so these are unseen). Writes
results/eval_results.json, results/eval_results.md and videos.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from robot_arm_sim.env import CUBE_HALF, ROOT, PickPlaceEnv
from robot_arm_sim.policy import BCAgent
from robot_arm_sim.scripted import ScriptedPolicy
from robot_arm_sim.train import CKPT_DIR, RESULTS_DIR

VIDEO_DIR = ROOT / "videos"


def frame_with_inset(env: PickPlaceEnv, obs: dict) -> np.ndarray:
    """Front camera at 480x640 with the policy's wrist-camera view in the corner."""
    frame = env.render("front").copy()
    wrist = obs.get("wrist_cam")
    if wrist is None:
        wrist = env.render("wrist_cam", 160, 160)
    else:
        k = max(1, 160 // wrist.shape[0])
        wrist = np.kron(wrist, np.ones((k, k, 1), dtype=np.uint8))
    h, w = wrist.shape[:2]
    frame[8 : 8 + h, -8 - w : -8] = wrist
    return frame


def rollout(env: PickPlaceEnv, policy, seed: int, video: bool) -> dict:
    obs = env.reset(seed=seed)
    policy.reset(obs)
    start_cube = env.cube_pos
    max_lift = 0.0
    frames = [frame_with_inset(env, obs)] if video else []
    for _ in range(env.max_steps):
        obs = env.step(policy(obs))
        max_lift = max(max_lift, env.cube_pos[2] - CUBE_HALF)
        if video:
            frames.append(frame_with_inset(env, obs))
    success = env.is_success()
    final_dist = float(np.linalg.norm(env.cube_pos[:2] - env.target_pos[:2]))
    if success:
        outcome = "success"
    elif max_lift < 0.02:
        outcome = "never lifted cube"
    elif final_dist < 0.06:
        outcome = "placed slightly off target"
    else:
        outcome = "dropped / lost cube"
    return {
        "seed": seed,
        "success": bool(success),
        "outcome": outcome,
        "max_lift_m": round(float(max_lift), 4),
        "final_dist_m": round(final_dist, 4),
        "cube_start": np.round(start_cube, 4).tolist(),
        "frames": frames,
    }


def evaluate(name: str, env: PickPlaceEnv, policy, seeds, n_videos: int) -> dict:
    results, video_frames = [], []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        r = rollout(env, policy, seed, video=i < n_videos)
        video_frames += r.pop("frames")
        results.append(r)
        print(f"[{name}] seed {seed:3d}: {r['outcome']}")
    rate = float(np.mean([r["success"] for r in results]))
    imageio.mimsave(VIDEO_DIR / f"eval_{name}.mp4", video_frames, fps=20)
    # Re-run the first failure with video (both policies are deterministic).
    failures = [r["seed"] for r in results if not r["success"]]
    if failures:
        r = rollout(env, policy, failures[0], video=True)
        imageio.mimsave(VIDEO_DIR / f"eval_{name}_failure_seed{failures[0]}.mp4", r["frames"], fps=20)
    outcomes: dict[str, int] = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    print(f"[{name}] success {rate:.0%} ({time.time() - t0:.0f}s)")
    return {"success_rate": rate, "outcomes": outcomes, "episodes": results}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a success rate (better than +-sqrt(p(1-p)/n) near 0% or 100%)."""
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--checkpoint", default=str(CKPT_DIR / "bc_policy.pt"))
    parser.add_argument("--videos", type=int, default=4, help="episodes per policy to save as video")
    args = parser.parse_args()

    VIDEO_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    seeds = list(range(args.episodes))

    agent = BCAgent.load(args.checkpoint)
    ckpt_info = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = int(ckpt_info.get("image_size", 96))

    env = PickPlaceEnv(image_size=image_size)
    report = {
        "episodes": args.episodes,
        "seeds": f"{seeds[0]}..{seeds[-1]}",
        "checkpoint": Path(args.checkpoint).name,
        "checkpoint_step": int(ckpt_info["step"]),
        "scripted": evaluate("scripted", env, ScriptedPolicy(env), seeds, args.videos),
        "bc_policy": evaluate("bc_policy", env, agent, seeds, args.videos),
    }
    env.close()
    with open(RESULTS_DIR / "eval_results.json", "w") as fh:
        json.dump(report, fh, indent=2)

    lines = [
        f"# Evaluation: {args.episodes} episodes (seeds {report['seeds']})",
        "",
        "| Policy | Success | 95% CI | Outcomes |",
        "|---|---|---|---|",
    ]
    for name, label in (("scripted", "Scripted IK (privileged cube pose)"), ("bc_policy", "Behavior cloning (pixels + joints)")):
        r = report[name]
        k = sum(e["success"] for e in r["episodes"])
        lo, hi = wilson_interval(k, args.episodes)
        outcomes = ", ".join(f"{v} {o}" for o, v in sorted(r["outcomes"].items(), key=lambda x: -x[1]))
        lines.append(f"| {label} | {k}/{args.episodes} ({r['success_rate']:.0%}) | {lo:.0%}–{hi:.0%} | {outcomes} |")
    (RESULTS_DIR / "eval_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
