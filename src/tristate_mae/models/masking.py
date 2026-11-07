from __future__ import annotations

import torch


def sample_tristate_mask(
    batch_size: int,
    num_patches: int,
    num_modalities: int = 4,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    probs = torch.tensor([alpha, beta, gamma], device=device, dtype=torch.float32)
    probs = probs / probs.sum()
    states = torch.multinomial(probs, batch_size * num_patches, replacement=True).view(batch_size, num_patches)
    visible = torch.zeros(batch_size, num_patches, num_modalities, dtype=torch.bool, device=device)

    cmt_patterns = _all_nontrivial_patterns(num_modalities, device=device)
    cmt = states == 1
    if cmt.any():
        chosen = torch.randint(0, cmt_patterns.shape[0], (int(cmt.sum().item()),), device=device)
        visible[cmt] = cmt_patterns[chosen]
    visible[states == 2] = True

    masked = ~visible
    patch_visible = visible.any(dim=-1)
    return {
        "states": states,
        "channel_visible": visible,
        "channel_masked": masked,
        "patch_visible": patch_visible,
    }


def sample_standard_spatial_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float = 0.75,
    num_modalities: int = 4,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    visible = torch.ones(batch_size, num_patches, num_modalities, dtype=torch.bool, device=device)
    num_mask = int(round(num_patches * mask_ratio))
    for b in range(batch_size):
        idx = torch.randperm(num_patches, device=device)[:num_mask]
        visible[b, idx] = False
    return {
        "states": torch.where(visible.any(-1), torch.full_like(visible[..., 0], 2), torch.zeros_like(visible[..., 0])),
        "channel_visible": visible,
        "channel_masked": ~visible,
        "patch_visible": visible.any(dim=-1),
    }


def sample_independent_channel_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float = 0.75,
    num_modalities: int = 4,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    visible = torch.rand(batch_size, num_patches, num_modalities, device=device) >= mask_ratio
    patch_visible = visible.any(dim=-1)
    return {
        "states": torch.where(patch_visible, torch.ones_like(patch_visible, dtype=torch.long), torch.zeros_like(patch_visible, dtype=torch.long)),
        "channel_visible": visible,
        "channel_masked": ~visible,
        "patch_visible": patch_visible,
    }


def effective_mask_ratio(alpha: float, beta: float, gamma: float, num_modalities: int = 4) -> float:
    _ = gamma
    expected_visible_cmt = num_modalities / 2.0
    return alpha + beta * (num_modalities - expected_visible_cmt) / num_modalities


def _all_nontrivial_patterns(num_modalities: int, device: torch.device | str) -> torch.Tensor:
    vals = []
    for mask in range(1, 2**num_modalities - 1):
        vals.append([(mask >> i) & 1 == 1 for i in range(num_modalities)])
    return torch.tensor(vals, dtype=torch.bool, device=device)

