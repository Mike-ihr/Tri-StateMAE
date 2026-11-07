from __future__ import annotations

from typing import Literal, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .masking import sample_independent_channel_mask, sample_standard_spatial_mask, sample_tristate_mask
from .pos_embed import get_3d_sincos_pos_embed
from .vit3d import TransformerBlock, ViT3DEncoder, pad_to_patch_size


MaskingMode = Literal["tristate", "standard", "independent"]


class TriStateMAE(nn.Module):
    def __init__(
        self,
        in_chans: int = 4,
        patch_size: Sequence[int] = (16, 16, 16),
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        standard_mask_ratio: float = 0.75,
        masking_mode: MaskingMode = "tristate",
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = tuple(int(v) for v in patch_size)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.standard_mask_ratio = standard_mask_ratio
        self.masking_mode = masking_mode

        self.encoder = ViT3DEncoder(in_chans, patch_size, encoder_embed_dim, encoder_depth, encoder_heads)
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.channel_indicator = nn.Linear(in_chans, decoder_embed_dim)
        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(decoder_embed_dim, decoder_heads, mlp_ratio=4.0) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        patch_volume = int(self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * patch_volume)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        imgs_pad, _ = pad_to_patch_size(imgs, self.patch_size)
        patch_grid = (
            imgs_pad.shape[-3] // self.patch_size[0],
            imgs_pad.shape[-2] // self.patch_size[1],
            imgs_pad.shape[-1] // self.patch_size[2],
        )
        num_patches = patch_grid[0] * patch_grid[1] * patch_grid[2]
        masks = self._sample_masks(imgs.shape[0], num_patches, imgs.device)
        masked_imgs = self._apply_channel_patch_mask(imgs_pad, masks["channel_visible"], patch_grid)

        enc = self.encoder(masked_imgs, visible_tokens=masks["patch_visible"])
        encoded = self.decoder_embed(enc["tokens"][:, 1:])
        key_padding_mask = enc["key_padding_mask"]
        enc_pad_mask = key_padding_mask[:, 1:] if key_padding_mask is not None else None

        dec_tokens = self.mask_token.to(dtype=encoded.dtype).expand(imgs.shape[0], num_patches, -1).clone()
        for b in range(imgs.shape[0]):
            keep = masks["patch_visible"][b]
            n = int(keep.sum().item())
            valid = encoded[b, :n] if enc_pad_mask is None else encoded[b, ~enc_pad_mask[b]]
            dec_tokens[b, keep] = valid.to(dtype=dec_tokens.dtype)
        pos = get_3d_sincos_pos_embed(dec_tokens.shape[-1], patch_grid, cls_token=False).to(imgs.device)
        dec_tokens = dec_tokens + pos.unsqueeze(0).to(dec_tokens.dtype)
        dec_tokens = dec_tokens + self.channel_indicator(masks["channel_masked"].float())
        for block in self.decoder_blocks:
            dec_tokens = block(dec_tokens)
        dec_tokens = self.decoder_norm(dec_tokens)
        pred = self.decoder_pred(dec_tokens)

        target = patchify(imgs_pad, self.patch_size)
        loss = masked_patch_mse(pred, target, masks["channel_masked"], self.patch_size)
        return {"loss": loss, "pred": pred, "target": target, **masks}

    def _sample_masks(self, batch_size: int, num_patches: int, device: torch.device) -> dict[str, torch.Tensor]:
        if self.masking_mode == "tristate":
            return sample_tristate_mask(batch_size, num_patches, self.in_chans, self.alpha, self.beta, self.gamma, device)
        if self.masking_mode == "standard":
            return sample_standard_spatial_mask(batch_size, num_patches, self.standard_mask_ratio, self.in_chans, device)
        if self.masking_mode == "independent":
            return sample_independent_channel_mask(batch_size, num_patches, self.standard_mask_ratio, self.in_chans, device)
        raise ValueError(f"Unknown masking_mode: {self.masking_mode}")

    def _apply_channel_patch_mask(
        self,
        imgs_pad: torch.Tensor,
        channel_visible: torch.Tensor,
        patch_grid: tuple[int, int, int],
    ) -> torch.Tensor:
        b, m = imgs_pad.shape[:2]
        gd, gh, gw = patch_grid
        mask = channel_visible.transpose(1, 2).reshape(b, m, gd, gh, gw).float()
        mask = mask.repeat_interleave(self.patch_size[0], 2)
        mask = mask.repeat_interleave(self.patch_size[1], 3)
        mask = mask.repeat_interleave(self.patch_size[2], 4)
        return imgs_pad * mask


def patchify(imgs: torch.Tensor, patch_size: Sequence[int]) -> torch.Tensor:
    p0, p1, p2 = [int(v) for v in patch_size]
    b, c, d, h, w = imgs.shape
    x = imgs.reshape(b, c, d // p0, p0, h // p1, p1, w // p2, p2)
    x = torch.einsum("bcdphqwr->bdhwcpqr", x)
    return x.reshape(b, (d // p0) * (h // p1) * (w // p2), c * p0 * p1 * p2)


def masked_patch_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_masked: torch.Tensor,
    patch_size: Sequence[int],
) -> torch.Tensor:
    patch_volume = int(patch_size[0] * patch_size[1] * patch_size[2])
    b, n, cp = pred.shape
    c = cp // patch_volume
    pred = pred.view(b, n, c, patch_volume)
    target = target.view(b, n, c, patch_volume)
    mask = channel_masked.unsqueeze(-1).float()
    denom = mask.sum() * patch_volume
    if denom.item() == 0:
        return F.mse_loss(pred, target)
    return ((pred - target).pow(2) * mask).sum() / denom.clamp_min(1.0)
