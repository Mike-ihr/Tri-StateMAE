from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

try:
    from _bootstrap import add_project_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_project_src_to_path

add_project_src_to_path()

import torch
from torch import nn
import torch.nn.functional as F

from tristate_mae.models.masking import sample_standard_spatial_mask
from tristate_mae.models.mae import patchify
from tristate_mae.models.pos_embed import get_3d_sincos_pos_embed
from tristate_mae.models.vit3d import TransformerBlock, ViT3DEncoder, pad_to_patch_size
from tristate_mae.utils.config import load_config
from tristate_mae.utils.distributed import device_from_config
from tristate_mae.utils.seed import seed_everything


class LateFusionStandardMAE(nn.Module):
    """Shared single-channel MAE encoder executed once per modality."""

    def __init__(
        self,
        num_modalities: int = 4,
        patch_size: Sequence[int] = (16, 16, 16),
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        self.num_modalities = int(num_modalities)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.mask_ratio = float(mask_ratio)

        self.encoder = ViT3DEncoder(1, self.patch_size, encoder_embed_dim, encoder_depth, encoder_heads)
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(decoder_embed_dim, decoder_heads, mlp_ratio=4.0) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        patch_volume = int(self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_volume)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def sample_visible_tokens(self, batch_size: int, num_patches: int, device: torch.device) -> torch.Tensor:
        masks = sample_standard_spatial_mask(batch_size, num_patches, self.mask_ratio, 1, device)
        return masks["patch_visible"]

    def forward_encoder(
        self,
        imgs: torch.Tensor,
        visible_tokens: torch.Tensor,
    ) -> list[dict[str, torch.Tensor | tuple[int, int, int] | list[torch.Tensor]]]:
        outputs = []
        for modality in range(self.num_modalities):
            outputs.append(self.encoder(imgs[:, modality : modality + 1], visible_tokens=visible_tokens))
        return outputs

    def forward(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        imgs_pad, _ = pad_to_patch_size(imgs, self.patch_size)
        patch_grid = (
            imgs_pad.shape[-3] // self.patch_size[0],
            imgs_pad.shape[-2] // self.patch_size[1],
            imgs_pad.shape[-1] // self.patch_size[2],
        )
        num_patches = patch_grid[0] * patch_grid[1] * patch_grid[2]
        visible_tokens = self.sample_visible_tokens(imgs.shape[0], num_patches, imgs.device)
        encoder_outputs = self.forward_encoder(imgs_pad, visible_tokens)

        preds = []
        for enc in encoder_outputs:
            tokens = self.decoder_embed(enc["tokens"][:, 1:])
            key_padding_mask = enc["key_padding_mask"]
            token_pad_mask = key_padding_mask[:, 1:] if key_padding_mask is not None else None
            dec_tokens = self.mask_token.to(dtype=tokens.dtype).expand(imgs.shape[0], num_patches, -1).clone()
            for b in range(imgs.shape[0]):
                keep = visible_tokens[b]
                valid = tokens[b, : int(keep.sum().item())] if token_pad_mask is None else tokens[b, ~token_pad_mask[b]]
                dec_tokens[b, keep] = valid.to(dtype=dec_tokens.dtype)
            pos = get_3d_sincos_pos_embed(dec_tokens.shape[-1], patch_grid, cls_token=False).to(imgs.device)
            dec_tokens = dec_tokens + pos.unsqueeze(0).to(dec_tokens.dtype)
            for block in self.decoder_blocks:
                dec_tokens = block(dec_tokens)
            dec_tokens = self.decoder_norm(dec_tokens)
            preds.append(self.decoder_pred(dec_tokens))

        pred = torch.stack(preds, dim=1)
        target = patchify(imgs_pad, self.patch_size).view(imgs.shape[0], num_patches, self.num_modalities, -1)
        target = target.transpose(1, 2).contiguous()
        masked = ~visible_tokens[:, None, :, None]
        loss = ((pred - target).pow(2) * masked).sum() / masked.sum().clamp_min(1)
        return {"loss": loss, "pred": pred, "target": target}


def model_from_config(cfg: dict[str, Any]) -> LateFusionStandardMAE:
    model_cfg = cfg["model"]
    return LateFusionStandardMAE(
        num_modalities=model_cfg.get("in_chans", 4),
        patch_size=model_cfg.get("patch_size", [16, 16, 16]),
        encoder_embed_dim=model_cfg.get("encoder_embed_dim", 768),
        encoder_depth=model_cfg.get("encoder_depth", 12),
        encoder_heads=model_cfg.get("encoder_heads", 12),
        decoder_embed_dim=model_cfg.get("decoder_embed_dim", 512),
        decoder_depth=model_cfg.get("decoder_depth", 4),
        decoder_heads=model_cfg.get("decoder_heads", 8),
        mask_ratio=model_cfg.get("standard_mask_ratio", 0.75),
    )


def profile_call(
    name: str,
    fn,
    device: torch.device,
    amp: bool,
    inference_mode: bool,
) -> dict[str, float | str | None]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    before_alloc = 0
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.empty_cache()
        before_alloc = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

    grad_context = torch.inference_mode() if inference_mode else torch.enable_grad()
    with grad_context:
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                output = fn()
    flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
    activation_gb = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        activation_gb = max(0, torch.cuda.max_memory_allocated(device) - before_alloc) / 1024**3
    del output
    return {
        "scope": name,
        "gflops": flops / 1e9,
        "activation_memory_gb": activation_gb,
    }


def write_results(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".json":
        with open(output, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        return
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_tristate.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--input-size", type=int, nargs=3, default=None, metavar=("D", "H", "W"))
    parser.add_argument("--scope", choices=("encoder", "full", "both"), default="both")
    parser.add_argument("--output", default="outputs/late_fusion_standard_mae_profile.csv")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--inference-mode", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    device = device_from_config(args.device or cfg.get("device"))
    model = model_from_config(cfg).to(device).eval()

    eff_cfg = cfg.get("efficiency", {})
    input_size = args.input_size or eff_cfg.get("profile_input_size", cfg["data"].get("image_size", [155, 240, 240]))
    batch_size = args.batch_size or int(eff_cfg.get("profile_batch_size", 1))
    sample = torch.zeros(batch_size, cfg["model"].get("in_chans", 4), *input_size, device=device)
    sample_pad, _ = pad_to_patch_size(sample, model.patch_size)
    patch_grid = (
        sample_pad.shape[-3] // model.patch_size[0],
        sample_pad.shape[-2] // model.patch_size[1],
        sample_pad.shape[-1] // model.patch_size[2],
    )
    num_patches = patch_grid[0] * patch_grid[1] * patch_grid[2]
    visible_tokens = model.sample_visible_tokens(batch_size, num_patches, device)
    amp = bool(cfg.get("training", {}).get("amp", True)) and not args.no_amp

    rows = []
    common = {
        "model": "late_fusion_standard_mae",
        "batch_size": batch_size,
        "input_size": "x".join(str(v) for v in input_size),
        "patch_size": "x".join(str(v) for v in model.patch_size),
        "num_modalities": model.num_modalities,
        "num_patches": num_patches,
        "visible_patches": int(visible_tokens[0].sum().item()),
        "mask_ratio": model.mask_ratio,
        "device": str(device),
        "amp": amp and device.type == "cuda",
        "inference_mode": args.inference_mode,
    }

    if args.scope in ("encoder", "both"):
        row = profile_call(
            "encoder_forward",
            lambda: model.forward_encoder(sample_pad, visible_tokens),
            device,
            amp,
            args.inference_mode,
        )
        rows.append({**common, **row})
    if args.scope in ("full", "both"):
        row = profile_call("full_forward", lambda: model(sample), device, amp, args.inference_mode)
        rows.append({**common, **row})

    for row in rows:
        mem = "n/a" if row["activation_memory_gb"] is None else f"{row['activation_memory_gb']:.4f} GB"
        print(f"{row['scope']}: {row['gflops']:.4f} GFLOPs, activation_memory={mem}")
    write_results(rows, Path(args.output))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
