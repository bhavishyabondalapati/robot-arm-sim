"""Train the behavior-cloning policy on recorded demos.

Behavior cloning = supervised learning on (observation -> expert action) pairs.

    python -m robot_arm_sim.train --steps 30000

Every --eval-every steps the policy is tried in the simulator on a few
*validation* seeds (never the seeds used for the final evaluation), and the
checkpoint with the best validation success rate is kept.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from robot_arm_sim.env import ROOT, PickPlaceEnv
from robot_arm_sim.policy import BCAgent, BCPolicy, get_device
from robot_arm_sim.record import DEMO_DIR
from robot_arm_sim.scripted import run_episode

CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
VAL_SEED_OFFSET = 50_000  # validation rollouts; final eval uses seeds 0..N


def load_demos(demo_dir: Path, chunk: int) -> dict:
    """Load every episode and build, for each timestep, the indices of its next `chunk` actions."""
    files = sorted(Path(demo_dir).glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No demos in {demo_dir}. Run: python -m robot_arm_sim.record")
    keys = ("front", "wrist", "qpos", "action")
    parts = {k: [] for k in keys}
    chunk_idx, episode_id = [], []
    offset = 0
    for i, f in enumerate(files):
        ep = np.load(f)
        T = len(ep["action"])
        for k in keys:
            parts[k].append(ep[k])
        # For step t: actions t, t+1, ..., t+chunk-1 (repeat the last one past the end).
        idx = np.minimum(np.arange(T)[:, None] + np.arange(chunk)[None], T - 1)
        chunk_idx.append(idx + offset)
        episode_id.append(np.full(T, i))
        offset += T
    data = {k: np.concatenate(v) for k, v in parts.items()}
    data["chunk_idx"] = np.concatenate(chunk_idx)
    data["episode_id"] = np.concatenate(episode_id)
    data["n_episodes"] = len(files)
    return data


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Shift each image by up to `pad` pixels (data augmentation, as in DrQ-v2)."""
    n, _, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, size=(n, 1, 1, 2), device=x.device).float() * 2.0 / (h + 2 * pad)
    return F.grid_sample(x, base_grid + shift, padding_mode="zeros", align_corners=False)


def evaluate_rollouts(model: BCPolicy, device, seeds, image_size: int) -> float:
    env = PickPlaceEnv(image_size=image_size)
    agent = BCAgent(model, device)
    successes = [run_episode(env, agent, seed=s)[0] for s in seeds]
    env.close()
    model.train()
    return float(np.mean(successes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo-dir", default=str(DEMO_DIR))
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--val-episodes", type=int, default=10, help="demos held out for validation loss")
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--eval-rollouts", type=int, default=10)
    parser.add_argument("--out", default=str(CKPT_DIR / "bc_policy.pt"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device()
    print(f"device: {device}")

    data = load_demos(Path(args.demo_dir), args.chunk)
    n_ep = data["n_episodes"]
    val_eps = set(rng.choice(n_ep, size=args.val_episodes, replace=False).tolist())
    is_val = np.isin(data["episode_id"], list(val_eps))
    train_idx, val_idx = np.where(~is_val)[0], np.where(is_val)[0]
    image_size = data["front"].shape[1]
    print(f"{n_ep} demos, {len(train_idx)} train / {len(val_idx)} val samples, images {image_size}px")

    # Everything lives on the device; images stay uint8 until a batch is drawn.
    front = torch.as_tensor(data["front"], device=device)
    wrist = torch.as_tensor(data["wrist"], device=device)
    qpos = torch.as_tensor(data["qpos"], device=device)
    action = torch.as_tensor(data["action"], device=device)
    chunk_idx = torch.as_tensor(data["chunk_idx"], device=device)

    config = {"chunk": args.chunk}
    model = BCPolicy(**config).to(device)
    tr = torch.as_tensor(train_idx, device=device)
    model.qpos_mean.copy_(qpos[tr].mean(0))
    model.qpos_std.copy_(qpos[tr].std(0).clamp_min(1e-3))
    model.act_mean.copy_(action[tr].mean(0))
    model.act_std.copy_(action[tr].std(0).clamp_min(1e-3))
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    def batch(indices: torch.Tensor, augment: bool):
        f = front[indices].permute(0, 3, 1, 2).float() / 255.0
        w = wrist[indices].permute(0, 3, 1, 2).float() / 255.0
        if augment:
            f, w = random_shift(f), random_shift(w)
        target = (action[chunk_idx[indices]] - model.act_mean) / model.act_std
        return f, w, qpos[indices], target

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        losses = []
        for i in range(0, len(val_idx), 512):
            f, w, q, target = batch(torch.as_tensor(val_idx[i : i + 512], device=device), augment=False)
            losses.append(F.l1_loss(model(f, w, q), target).item() * len(f))
        model.train()
        return sum(losses) / len(val_idx)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    val_seeds = range(VAL_SEED_OFFSET, VAL_SEED_OFFSET + args.eval_rollouts)
    log = {"step": [], "train_loss": [], "val_step": [], "val_loss": [], "eval_step": [], "val_success": []}
    best = -1.0
    running = None
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        idx = tr[torch.randint(0, len(tr), (args.batch_size,), device=device)]
        f, w, q, target = batch(idx, augment=True)
        loss = F.l1_loss(model(f, w, q), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if step % 250 == 0:
            log["step"].append(step)
            log["train_loss"].append(running)
        if step % 1000 == 0:
            vl = val_loss()
            log["val_step"].append(step)
            log["val_loss"].append(vl)
            print(f"step {step:6d}  train L1 {running:.4f}  val L1 {vl:.4f}  {time.time() - start:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            sr = evaluate_rollouts(model, device, val_seeds, image_size)
            log["eval_step"].append(step)
            log["val_success"].append(sr)
            print(f"step {step:6d}  validation rollout success {sr:.0%}", flush=True)
            ckpt = {
                "model": model.state_dict(),
                "config": config,
                "image_size": image_size,
                "step": step,
                "val_success": sr,
                "args": vars(args),
            }
            torch.save(ckpt, out.with_name(out.stem + "_last.pt"))
            if sr >= best:  # ">=" prefers the later (more trained) model on ties
                best = sr
                torch.save(ckpt, out)
                print(f"  saved best checkpoint -> {out}")

    with open(RESULTS_DIR / "training_log.json", "w") as fh:
        json.dump(log, fh)
    plot_curves(log, RESULTS_DIR / "training_curves.png")
    print(f"done in {(time.time() - start) / 60:.1f} min, best validation success {best:.0%}")


def plot_curves(log: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    a1.plot(log["step"], log["train_loss"], label="train (augmented)")
    a1.plot(log["val_step"], log["val_loss"], label="held-out demos")
    a1.set_xlabel("step")
    a1.set_ylabel("L1 loss (normalized actions)")
    a1.set_yscale("log")
    a1.legend()
    a2.plot(log["eval_step"], [100 * s for s in log["val_success"]], marker="o")
    a2.set_xlabel("step")
    a2.set_ylabel("validation success (%)")
    a2.set_ylim(-5, 105)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
