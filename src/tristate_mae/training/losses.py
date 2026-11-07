from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        dims = tuple(range(2, logits.ndim))
        inter = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return ce + (1 - dice.mean())


def dice_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred = (torch.sigmoid(logits) >= threshold).float()
    dims = tuple(range(2, logits.ndim))
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + 1e-5) / (denom + 1e-5)
    return dice.mean(dim=0)

