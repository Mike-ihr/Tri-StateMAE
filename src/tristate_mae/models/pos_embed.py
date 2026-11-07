from __future__ import annotations

import math

import torch


def get_3d_sincos_pos_embed(embed_dim: int, grid_size: tuple[int, int, int], cls_token: bool) -> torch.Tensor:
    d, h, w = grid_size
    grid_d, grid_h, grid_w = torch.meshgrid(
        torch.arange(d, dtype=torch.float32),
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    part = embed_dim // 3
    dims = [part, part, embed_dim - 2 * part]
    dims = [d if d % 2 == 0 else d - 1 for d in dims]
    pos = torch.cat(
        [_sincos_1d(dims[0], grid_d.reshape(-1)), _sincos_1d(dims[1], grid_h.reshape(-1)), _sincos_1d(dims[2], grid_w.reshape(-1))],
        dim=1,
    )
    if pos.shape[1] < embed_dim:
        pos = torch.nn.functional.pad(pos, (0, embed_dim - pos.shape[1]))
    if cls_token:
        pos = torch.cat([torch.zeros(1, embed_dim), pos], dim=0)
    return pos


def _sincos_1d(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("1D sin-cos positional embedding requires even embed_dim.")
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = torch.einsum("n,d->nd", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)
