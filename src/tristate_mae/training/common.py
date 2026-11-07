from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tristate_mae.data.brats import BratsDataset, discover_brats_cases, split_cases


def make_loaders(
    cfg: dict[str, Any],
    task: str,
    missing_modalities: list[str] | None = None,
    distributed: bool = False,
) -> dict[str, DataLoader]:
    data_cfg = cfg["data"]
    cases = discover_brats_cases(data_cfg["root"], data_cfg.get("classification_csv"))
    if not cases:
        raise RuntimeError(f"No complete BraTS cases found under {data_cfg['root']}.")
    splits = split_cases(
        cases,
        train=data_cfg.get("train_cases", 1000),
        val=data_cfg.get("val_cases", 125),
        test=data_cfg.get("test_cases", 126),
        seed=cfg.get("seed", 42),
    )
    loaders = {}
    for split, split_cases_ in splits.items():
        if not split_cases_:
            continue
        ds = BratsDataset(
            split_cases_,
            image_size=data_cfg.get("image_size", [155, 240, 240]),
            task=task,
            missing_modalities=missing_modalities,
        )
        sampler = DistributedSampler(ds, shuffle=True, drop_last=True) if distributed and split == "train" else None
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg["training"].get("batch_size", 1),
            shuffle=(split == "train" and sampler is None),
            sampler=sampler,
            num_workers=cfg["training"].get("num_workers", 4),
            pin_memory=torch.cuda.is_available(),
            drop_last=split == "train" and distributed,
        )
    return loaders


def optimizer_and_scheduler(model: torch.nn.Module, cfg: dict[str, Any], steps_per_epoch: int):
    train_cfg = cfg["training"]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    epochs = int(train_cfg["epochs"])
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = max(1, epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def output_dir(cfg: dict[str, Any]) -> Path:
    path = Path(cfg["training"].get("output_dir", "outputs"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
