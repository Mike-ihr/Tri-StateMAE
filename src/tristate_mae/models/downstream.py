from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .vit3d import ViT3DEncoder


class UNETRSegmentation(nn.Module):
    def __init__(
        self,
        in_chans: int = 4,
        out_chans: int = 3,
        image_size: Sequence[int] = (155, 240, 240),
        patch_size: Sequence[int] = (16, 16, 16),
        embed_dim: int = 768,
        depth: int = 12,
        heads: int = 12,
    ) -> None:
        super().__init__()
        self.image_size = tuple(int(v) for v in image_size)
        self.encoder = ViT3DEncoder(in_chans, patch_size, embed_dim, depth, heads)
        self.proj = nn.Sequential(
            nn.Conv3d(embed_dim, 256, kernel_size=3, padding=1),
            nn.InstanceNorm3d(256),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.InstanceNorm3d(128),
            nn.GELU(),
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.InstanceNorm3d(64),
            nn.GELU(),
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.InstanceNorm3d(32),
            nn.GELU(),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.InstanceNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, out_chans, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        tokens = out["tokens"][:, 1:]
        gd, gh, gw = out["grid_size"]
        feat = tokens.transpose(1, 2).reshape(x.shape[0], -1, gd, gh, gw)
        logits = self.decoder(self.proj(feat))
        return F.interpolate(logits, size=x.shape[-3:], mode="trilinear", align_corners=False)


class ViTClassifier(nn.Module):
    def __init__(
        self,
        in_chans: int = 4,
        num_classes: int = 2,
        patch_size: Sequence[int] = (16, 16, 16),
        embed_dim: int = 768,
        depth: int = 12,
        heads: int = 12,
    ) -> None:
        super().__init__()
        self.encoder = ViT3DEncoder(in_chans, patch_size, embed_dim, depth, heads)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        return self.head(out["tokens"][:, 0])


def load_pretrained_encoder(target: nn.Module, checkpoint: dict) -> None:
    state = checkpoint.get("model", checkpoint)
    encoder_state = {}
    for key, value in state.items():
        if key.startswith("encoder."):
            encoder_state[key.removeprefix("encoder.")] = value
    missing, unexpected = target.encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected encoder keys in checkpoint: {unexpected}")
    if not encoder_state:
        raise RuntimeError("No encoder.* keys found in pre-training checkpoint.")
    print(f"Loaded pre-trained encoder with {len(missing)} missing task-specific keys.")

