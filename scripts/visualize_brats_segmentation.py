from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _bootstrap import add_project_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_project_src_to_path

add_project_src_to_path()

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from tristate_mae.data.brats import brats_label_to_regions, discover_brats_cases, load_nifti, zscore_nonzero
from tristate_mae.models.downstream import UNETRSegmentation
from tristate_mae.training.common import make_loaders
from tristate_mae.utils.checkpoint import load_checkpoint
from tristate_mae.utils.config import add_config_arg, load_config
from tristate_mae.utils.distributed import device_from_config


REGION_ORDER = ("ET", "TC", "WT")
REGION_COLORS = {
    "WT": torch.tensor([0.0, 1.0, 0.0]),  # green
    "TC": torch.tensor([1.0, 1.0, 0.0]),  # yellow
    "ET": torch.tensor([1.0, 0.0, 0.0]),  # red
}
AXIS_TO_DIM = {"sagittal": 0, "coronal": 1, "axial": 2}


def parse_args() -> argparse.Namespace:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--checkpoint", required=True, help="Path to a fine-tuned segmentation checkpoint.")
    parser.add_argument("--output-dir", default="outputs/brats_seg_visualizations", help="Directory for PNG visualizations.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="Dataset split to visualize.")
    parser.add_argument("--max-cases", type=int, default=8, help="Maximum number of cases to export.")
    parser.add_argument("--case-id", default=None, help="Only export a specific BraTS case id.")
    parser.add_argument("--axis", choices=tuple(AXIS_TO_DIM), default="axial", help="Slice orientation.")
    parser.add_argument("--slice-index", type=int, default=None, help="Slice index. Defaults to the slice with most predicted tumor.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for ET/TC/WT region masks.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity in [0, 1].")
    parser.add_argument(
        "--background-modality",
        choices=("t1", "t1ce", "t2", "flair"),
        default="flair",
        help="MRI modality used as the grayscale background.",
    )
    parser.add_argument(
        "--missing-modality",
        action="append",
        default=[],
        choices=("t1", "t1ce", "t2", "flair"),
        help="Modality to zero out before inference. Can be passed multiple times.",
    )
    return parser.parse_args()


def load_model(cfg: dict, checkpoint_path: str, device: torch.device) -> UNETRSegmentation:
    model = UNETRSegmentation(**cfg["model"]).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def normalize_slice(x: torch.Tensor) -> torch.Tensor:
    mask = x != 0
    if not mask.any():
        return torch.zeros_like(x, dtype=torch.float32)
    values = x[mask].float()
    lo = torch.quantile(values, 0.01)
    hi = torch.quantile(values, 0.99)
    if (hi - lo).abs() < 1e-6:
        return torch.zeros_like(x, dtype=torch.float32)
    out = ((x.float() - lo) / (hi - lo)).clamp(0, 1)
    out[~mask] = 0
    return out


def extract_slice(volume: torch.Tensor, axis: str, index: int) -> torch.Tensor:
    dim = AXIS_TO_DIM[axis]
    if dim == 0:
        return volume[index, :, :]
    if dim == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def orient_slice(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=-1, dims=(-2, -1))


def validate_slice_shape(image_slice: torch.Tensor, axis: str) -> None:
    if axis == "axial" and tuple(image_slice.shape) != (240, 240):
        raise RuntimeError(
            f"Expected axial slice shape (240, 240), got {tuple(image_slice.shape)}. "
            "Check data.image_size and AXIS_TO_DIM."
        )


def resize_to_shape(volume: torch.Tensor, shape: tuple[int, int, int], mode: str) -> torch.Tensor:
    if tuple(volume.shape[-3:]) == tuple(shape):
        return volume
    squeeze_channel = volume.ndim == 3
    x = volume.float()
    if squeeze_channel:
        x = x.unsqueeze(0)
    x = x.unsqueeze(0)
    if mode == "nearest":
        y = F.interpolate(x, size=shape, mode="nearest")
    else:
        y = F.interpolate(x, size=shape, mode="trilinear", align_corners=False)
    y = y.squeeze(0)
    if squeeze_channel:
        y = y.squeeze(0)
    return y


def load_raw_case_tensors(case, modality: str, missing_modalities: list[str]) -> tuple[torch.Tensor, torch.Tensor | None, tuple[int, int, int]]:
    image_np = zscore_nonzero(load_nifti(case.modalities[modality]))
    image = torch.from_numpy(image_np).float()
    if modality in missing_modalities:
        image.zero_()
    label = None
    if case.seg is not None:
        seg = torch.from_numpy(load_nifti(case.seg)).float()
        label = brats_label_to_regions(seg).bool()
    return image, label, tuple(int(v) for v in image.shape)


def pick_slice(mask: torch.Tensor, axis: str) -> int:
    dim = AXIS_TO_DIM[axis]
    spatial_shape = mask.shape[-3:]
    reduce_dims = tuple(i for i in range(3) if i != dim)
    counts = mask.float().sum(dim=reduce_dims)
    if counts.max() <= 0:
        return spatial_shape[dim] // 2
    return int(counts.argmax().item())


def masks_to_rgb(masks: torch.Tensor) -> torch.Tensor:
    colors = torch.zeros((*masks.shape[-2:], 3), dtype=torch.float32, device=masks.device)
    wt = masks[2].bool()
    tc = masks[1].bool()
    et = masks[0].bool()
    for region, mask in (("WT", wt), ("TC", tc), ("ET", et)):
        color = REGION_COLORS[region].to(masks.device)
        colors[mask] = color
    return colors.permute(2, 0, 1)


def overlay_regions(background: torch.Tensor, region_masks: torch.Tensor, alpha: float) -> torch.Tensor:
    gray = normalize_slice(background).unsqueeze(0).repeat(3, 1, 1)
    colored = masks_to_rgb(region_masks)
    foreground = region_masks.any(dim=0, keepdim=True)
    alpha_map = foreground.float() * float(alpha)
    return gray * (1.0 - alpha_map) + colored * alpha_map


def make_panel(image: torch.Tensor, pred: torch.Tensor, label: torch.Tensor | None, axis: str, slice_index: int, alpha: float) -> torch.Tensor:
    image_slice = orient_slice(extract_slice(image, axis, slice_index))
    validate_slice_shape(image_slice, axis)
    pred_slice = torch.stack([orient_slice(extract_slice(pred[c], axis, slice_index)) for c in range(pred.shape[0])], dim=0)

    gray = normalize_slice(image_slice).unsqueeze(0).repeat(3, 1, 1)
    pred_overlay = overlay_regions(image_slice, pred_slice, alpha)
    panels = [gray]

    if label is not None:
        label_slice = torch.stack([orient_slice(extract_slice(label[c], axis, slice_index)) for c in range(label.shape[0])], dim=0)
        panels.append(overlay_regions(image_slice, label_slice, alpha))

    panels.append(pred_overlay)

    separator = torch.ones(3, panels[0].shape[1], 4, dtype=panels[0].dtype, device=panels[0].device)
    out = panels[0]
    for panel in panels[1:]:
        out = torch.cat([out, separator, panel], dim=2)
    return out


def case_ids_from_batch(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["training"]["batch_size"] = 1
    cfg["training"]["num_workers"] = min(2, int(cfg["training"].get("num_workers", 0)))

    device = device_from_config(cfg.get("device"))
    model = load_model(cfg, args.checkpoint, device)
    loaders = make_loaders(cfg, task="seg", missing_modalities=args.missing_modality)
    if args.split not in loaders:
        raise RuntimeError(f"Split '{args.split}' is unavailable. Available splits: {sorted(loaders)}")

    cases_by_id = {
        case.case_id: case
        for case in discover_brats_cases(cfg["data"]["root"], cfg["data"].get("classification_csv"))
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for batch in tqdm(loaders[args.split], desc=f"visualize {args.split}"):
        case_id = case_ids_from_batch(batch["case_id"])[0]
        if args.case_id is not None and case_id != args.case_id:
            continue

        image = batch["image"].to(device, non_blocking=True)
        logits = model(image)
        pred = torch.sigmoid(logits[0]).ge(args.threshold).cpu()

        case = cases_by_id[case_id]
        image_vis, label_tensor, original_shape = load_raw_case_tensors(
            case, args.background_modality, args.missing_modality
        )
        pred = resize_to_shape(pred.float(), original_shape, mode="nearest").ge(0.5)

        axis = args.axis
        spatial_shape = pred.shape[-3:]
        if args.slice_index is None:
            slice_index = pick_slice(pred.any(dim=0), axis)
        else:
            slice_index = args.slice_index
            max_index = spatial_shape[AXIS_TO_DIM[axis]] - 1
            if not 0 <= slice_index <= max_index:
                raise ValueError(f"--slice-index must be in [0, {max_index}] for {axis} slices.")

        panel = make_panel(
            image=image_vis,
            pred=pred,
            label=label_tensor,
            axis=axis,
            slice_index=slice_index,
            alpha=max(0.0, min(1.0, args.alpha)),
        )
        path = output_dir / f"{case_id}_{axis}_{slice_index:03d}_{args.background_modality}.png"
        save_image(panel, path)
        exported += 1

        if args.case_id is not None or exported >= args.max_cases:
            break

    if exported == 0:
        raise RuntimeError("No cases were exported. Check --split, --case-id, and dataset configuration.")
    print(f"Saved {exported} visualization(s) to {output_dir}")
    print("Panel order: grayscale input | ground-truth overlay (if labels are available) | prediction overlay.")
    print("Colors: WT=green, TC=yellow, ET=red.")


if __name__ == "__main__":
    main()
