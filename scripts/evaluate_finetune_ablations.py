from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from _bootstrap import add_project_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_project_src_to_path

add_project_src_to_path()

import torch

from tristate_mae.utils.config import load_config


TASKS = ("seg", "cls")


def find_runs(root: Path, checkpoint_name: str) -> list[tuple[str, Path, Path]]:
    runs: list[tuple[str, Path, Path]] = []
    if not root.exists():
        raise FileNotFoundError(f"Finetune ablation root does not exist: {root}")
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        checkpoint = run_dir / checkpoint_name
        if not checkpoint.exists():
            print(f"skip {run_dir.name}: missing {checkpoint_name}")
            continue
        config = run_dir / "config.yaml"
        if not config.exists():
            config = root / f"{run_dir.name}.yaml"
        if not config.exists():
            print(f"skip {run_dir.name}: missing config.yaml and {run_dir.name}.yaml")
            continue
        runs.append((run_dir.name, checkpoint, config))
    if not runs:
        raise FileNotFoundError(f"No runnable ablations found under {root}")
    return runs


def _loader_case_ids(loader) -> set[str]:
    cases = getattr(loader.dataset, "cases", [])
    return {case.case_id for case in cases}


def test_loader(cfg: dict[str, Any], task: str):
    from tristate_mae.training.common import make_loaders

    loaders = make_loaders(cfg, task=task)
    if "test" not in loaders:
        raise RuntimeError("No test split was created. Refusing to evaluate on val/train as test.")
    test_ids = _loader_case_ids(loaders["test"])
    for split in ("train", "val"):
        if split not in loaders:
            continue
        overlap = test_ids & _loader_case_ids(loaders[split])
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise RuntimeError(f"Test split overlaps with {split}: {examples}")
    return loaders["test"]


@torch.no_grad()
def evaluate_seg_model(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    from tristate_mae.training.losses import dice_score

    model.eval()
    scores = []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        lab = batch["label"].to(device, non_blocking=True)
        scores.append(dice_score(model(img), lab).cpu())
    per_region = torch.stack(scores).mean(0)
    return {
        "et": per_region[0].item(),
        "tc": per_region[1].item(),
        "wt": per_region[2].item(),
        "mean": per_region.mean().item(),
    }


@torch.no_grad()
def evaluate_cls_model(model: torch.nn.Module, loader, device: torch.device) -> tuple[float, float]:
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, roc_auc_score

    model.eval()
    labels, probs = [], []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        lab = batch["label"]
        prob = F.softmax(model(img), dim=1)[:, 1].cpu()
        labels.extend(lab.tolist())
        probs.extend(prob.tolist())
    preds = [int(p >= 0.5) for p in probs]
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return float(acc), float(auc)


def evaluate_seg_run(name: str, checkpoint: Path, config: Path, device: torch.device) -> dict[str, Any]:
    from tristate_mae.models.downstream import UNETRSegmentation
    from tristate_mae.utils.checkpoint import load_checkpoint

    cfg = load_config(config)
    model = UNETRSegmentation(**cfg["model"]).to(device)
    model.load_state_dict(load_checkpoint(checkpoint)["model"], strict=True)
    metrics = evaluate_seg_model(model, test_loader(cfg, "seg"), device)
    return {
        "ablation": name,
        "checkpoint": str(checkpoint),
        "config": str(config),
        "mean_dice": metrics["mean"],
        "dice_et": metrics["et"],
        "dice_tc": metrics["tc"],
        "dice_wt": metrics["wt"],
    }


def evaluate_cls_run(name: str, checkpoint: Path, config: Path, device: torch.device) -> dict[str, Any]:
    from tristate_mae.models.downstream import ViTClassifier
    from tristate_mae.utils.checkpoint import load_checkpoint

    cfg = load_config(config)
    model = ViTClassifier(**cfg["model"]).to(device)
    model.load_state_dict(load_checkpoint(checkpoint)["model"], strict=True)
    acc, auc = evaluate_cls_model(model, test_loader(cfg, "cls"), device)
    return {
        "ablation": name,
        "checkpoint": str(checkpoint),
        "config": str(config),
        "accuracy": acc,
        "auc": auc,
    }


def write_results(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def run_task(
    task: str,
    root: Path,
    checkpoint_name: str,
    output_dir: Path,
    device: torch.device,
    selected: set[str] | None,
    dry_run: bool,
) -> None:
    rows: list[dict[str, Any]] = []
    runs = find_runs(root, checkpoint_name)
    for name, checkpoint, config in runs:
        if selected is not None and name not in selected:
            continue
        print(f"{task} {name}: checkpoint={checkpoint} config={config}")
        if dry_run:
            continue
        if task == "seg":
            row = evaluate_seg_run(name, checkpoint, config, device)
            print(
                f"{name}: mean_dice={row['mean_dice']:.4f} "
                f"ET={row['dice_et']:.4f} TC={row['dice_tc']:.4f} WT={row['dice_wt']:.4f}"
            )
        elif task == "cls":
            row = evaluate_cls_run(name, checkpoint, config, device)
            print(f"{name}: accuracy={row['accuracy']:.4f} auc={row['auc']:.4f}")
        else:
            raise ValueError(f"Unsupported task: {task}")
        rows.append(row)
    if not dry_run:
        write_results(
            rows,
            output_dir / f"{task}_test_results.csv",
            output_dir / f"{task}_test_results.json",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--seg-root", default="outputs/finetune_seg/ablations")
    parser.add_argument("--cls-root", default="outputs/finetune_cls/ablations")
    parser.add_argument("--seg-checkpoint", default="seg_best.pt")
    parser.add_argument("--cls-checkpoint", default="cls_best.pt")
    parser.add_argument("--output-dir", default="outputs/evaluate_finetune_ablations")
    parser.add_argument("--device", default=None)
    parser.add_argument("--only", nargs="*", default=None, help="Optional ablation names to evaluate.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = None
    if not args.dry_run:
        from tristate_mae.utils.distributed import device_from_config

        device = device_from_config(args.device)
    selected = set(args.only) if args.only else None
    tasks = TASKS if args.task == "all" else (args.task,)
    for task in tasks:
        root = Path(args.seg_root if task == "seg" else args.cls_root)
        checkpoint = args.seg_checkpoint if task == "seg" else args.cls_checkpoint
        run_task(task, root, checkpoint, Path(args.output_dir), device, selected, args.dry_run)


if __name__ == "__main__":
    main()
