from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .pos_embed import get_3d_sincos_pos_embed


def pad_to_patch_size(x: torch.Tensor, patch_size: Sequence[int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
    _, _, d, h, w = x.shape
    pd, ph, pw = patch_size
    nd, nh, nw = math.ceil(d / pd) * pd, math.ceil(h / ph) * ph, math.ceil(w / pw) * pw
    pads = (0, nw - w, 0, nh - h, 0, nd - d)
    return F.pad(x, pads), (d, h, w)


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        in_chans: int = 4,
        embed_dim: int = 768,
        patch_size: Sequence[int] = (16, 16, 16),
    ) -> None:
        super().__init__()
        self.patch_size = tuple(int(v) for v in patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
        x, original_size = pad_to_patch_size(x, self.patch_size)
        x = self.proj(x)
        grid_size = x.shape[-3:]
        x = x.flatten(2).transpose(1, 2)
        return x, grid_size, original_size


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.norm1(x)
        y, _ = self.attn(q, q, q, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class ViT3DEncoder(nn.Module):
    def __init__(
        self,
        in_chans: int = 4,
        patch_size: Sequence[int] = (16, 16, 16),
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(in_chans, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        visible_tokens: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor | tuple[int, int, int] | list[torch.Tensor]]:
        tokens, grid_size, original_size = self.patch_embed(x)
        pos = get_3d_sincos_pos_embed(self.embed_dim, grid_size, cls_token=False).to(tokens.device)
        tokens = tokens + pos.unsqueeze(0).to(tokens.dtype)

        key_padding_mask = None
        if visible_tokens is not None:
            gathered = []
            masks = []
            for b in range(tokens.shape[0]):
                gathered.append(tokens[b, visible_tokens[b]])
                masks.append(torch.zeros(gathered[-1].shape[0], dtype=torch.bool, device=tokens.device))
            tokens = nn.utils.rnn.pad_sequence(gathered, batch_first=True)
            key_padding_mask = nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=True)

        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        xseq = torch.cat([cls, tokens], dim=1)
        if key_padding_mask is not None:
            cls_mask = torch.zeros(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)
        hidden: list[torch.Tensor] = []
        for block in self.blocks:
            xseq = block(xseq, key_padding_mask=key_padding_mask)
            if return_hidden:
                hidden.append(xseq)
        xseq = self.norm(xseq)
        return {
            "tokens": xseq,
            "grid_size": grid_size,
            "original_size": original_size,
            "hidden": hidden,
            "key_padding_mask": key_padding_mask,
        }


def vit_base_3d(in_chans: int = 4, patch_size: Sequence[int] = (16, 16, 16)) -> ViT3DEncoder:
    return ViT3DEncoder(in_chans=in_chans, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12)
