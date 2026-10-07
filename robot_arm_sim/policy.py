"""A small behavior-cloning policy: (2 camera images + joint angles) -> next actions.

Architecture (about 0.6M parameters):

    front image (96x96) --CNN--> spatial softmax --> 2D keypoints --\
    wrist image (96x96) --CNN--> spatial softmax --> 2D keypoints ---+--> MLP --> next 10 actions
    joint angles (6)    --------------------------------------------/

* Spatial softmax turns each CNN feature map into the (x, y) location where
  that feature is strongest, e.g. "where is the red cube". For manipulation,
  *where* things are matters more than *what* they are, so this works well
  with little data (Levine et al. 2016).
* Predicting a chunk of future actions instead of one action (as in ACT,
  Zhao et al. 2023) makes motions smoother and reduces compounding errors.
  At run time, overlapping chunk predictions are averaged ("temporal ensembling").
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    """Apple Silicon GPU (MPS) if available, else CPU. (No CUDA on a Mac.)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SpatialSoftmax(nn.Module):
    """(B, C, H, W) feature maps -> (B, 2C) expected (x, y) of each channel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        attn = F.softmax(x.reshape(b, c, h * w), dim=-1).reshape(b, c, h, w)
        ys = torch.linspace(-1, 1, h, device=x.device)
        xs = torch.linspace(-1, 1, w, device=x.device)
        ex = (attn.sum(dim=2) * xs).sum(dim=-1)  # sum over rows, then weight columns
        ey = (attn.sum(dim=3) * ys).sum(dim=-1)
        return torch.cat([ex, ey], dim=1)


class ImageEncoder(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()

        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True)
            )

        self.net = nn.Sequential(
            block(3, 32, 2),  # 96 -> 48
            block(32, 64, 2),  # 48 -> 24
            block(64, 64, 1),
            block(64, 64, 1),
            nn.Conv2d(64, channels, 1),
        )
        self.softmax = SpatialSoftmax()
        self.out_dim = 2 * channels

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.net(img))


class BCPolicy(nn.Module):
    def __init__(self, chunk: int = 10, action_dim: int = 6, qpos_dim: int = 6, hidden: int = 512):
        super().__init__()
        self.chunk = chunk
        self.action_dim = action_dim
        self.front_enc = ImageEncoder()
        self.wrist_enc = ImageEncoder()
        feat = self.front_enc.out_dim + self.wrist_enc.out_dim + qpos_dim
        self.head = nn.Sequential(
            nn.Linear(feat, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, chunk * action_dim),
        )
        # Normalization statistics, filled from the dataset and saved with the weights.
        for name, dim in (("qpos_mean", qpos_dim), ("qpos_std", qpos_dim), ("act_mean", action_dim), ("act_std", action_dim)):
            self.register_buffer(name, torch.zeros(dim) if "mean" in name else torch.ones(dim))

    def forward(self, front: torch.Tensor, wrist: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        """front/wrist: float images in [0, 1], (B, 3, H, W). qpos: raw radians (B, 6).

        Returns *normalized* actions (B, chunk, 6).
        """
        q = (qpos - self.qpos_mean) / self.qpos_std
        z = torch.cat([self.front_enc(front), self.wrist_enc(wrist), q], dim=1)
        return self.head(z).reshape(-1, self.chunk, self.action_dim)

    def unnormalize(self, act: torch.Tensor) -> torch.Tensor:
        return act * self.act_std + self.act_mean


def images_to_tensor(imgs: np.ndarray, device) -> torch.Tensor:
    """uint8 (B, H, W, 3) or (H, W, 3) -> float (B, 3, H, W) in [0, 1]."""
    t = torch.as_tensor(imgs, device=device)
    if t.ndim == 3:
        t = t[None]
    return t.permute(0, 3, 1, 2).float() / 255.0


class BCAgent:
    """Wraps a trained BCPolicy for closed-loop control with temporal ensembling."""

    def __init__(self, model: BCPolicy, device, ensemble_decay: float = 0.1):
        self.model = model.eval()
        self.device = device
        self.decay = ensemble_decay
        self.history: list[tuple[int, np.ndarray]] = []
        self.t = 0

    @classmethod
    def load(cls, path, device=None) -> BCAgent:
        device = device or get_device()
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = BCPolicy(**ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model"])
        return cls(model, device)

    def reset(self, obs: dict | None = None) -> None:
        self.history = []
        self.t = 0

    @torch.no_grad()
    def __call__(self, obs: dict) -> np.ndarray:
        front = images_to_tensor(obs["front"], self.device)
        wrist = images_to_tensor(obs["wrist_cam"], self.device)
        qpos = torch.as_tensor(obs["qpos"], device=self.device)[None]
        chunk = self.model.unnormalize(self.model(front, wrist, qpos))[0].cpu().numpy()

        self.history.append((self.t, chunk))
        self.history = [(t0, c) for t0, c in self.history if self.t - t0 < len(c)]
        # Every past chunk that covers "now" votes; older predictions get weight
        # exp(-decay * i) with i=0 for the oldest, as in ACT.
        preds = np.stack([c[self.t - t0] for t0, c in self.history])
        weights = np.exp(-self.decay * np.arange(len(preds)))
        self.t += 1
        return (weights[:, None] * preds).sum(0) / weights.sum()
