from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

MODALITIES = ("t1", "t1ce", "t2", "flair")


@dataclass(frozen=True)
class BratsCase:
    case_id: str
    modalities: dict[str, Path]
    seg: Path | None = None
    cls_label: int | None = None


def _case_id_from_name(path: Path, suffix: str) -> str:
    name = path.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return re.sub(rf"[_-]{suffix}$", "", name, flags=re.IGNORECASE)


def discover_brats_cases(root: str | Path, label_csv: str | Path | None = None) -> list[BratsCase]:
    root = Path(root)
    files = list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz"))
    by_case: dict[str, dict[str, Path]] = {}
    segs: dict[str, Path] = {}

    for path in files:
        low = path.name.lower()
        if any(token in low for token in ("seg", "label", "mask")):
            case_id = _case_id_from_name(path, "seg")
            case_id = _case_id_from_name(Path(case_id), "label")
            segs[case_id] = path
            continue
        for mod in MODALITIES:
            if re.search(rf"(^|[_-]){mod}(\.nii|[_-]|\.)", low):
                case_id = _case_id_from_name(path, mod)
                by_case.setdefault(case_id, {})[mod] = path
                break

    cls_labels = _read_classification_csv(label_csv) if label_csv else {}
    cases = []
    for case_id, mods in sorted(by_case.items()):
        if all(m in mods for m in MODALITIES):
            cases.append(BratsCase(case_id, dict(mods), segs.get(case_id), cls_labels.get(case_id)))
    return cases


def _read_classification_csv(path: str | Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = row.get("case_id") or row.get("id") or row.get("name") or row.get("BraTS21ID")
            label = row.get("label") or row.get("grade") or row.get("diagnosis")
            if case is None or label is None:
                continue
            text = str(label).strip().lower()
            out[str(case)] = 1 if text in {"1", "hgg", "high", "high-grade", "gbm"} else 0
    return out


def split_cases(
    cases: Sequence[BratsCase],
    train: int = 1000,
    val: int = 125,
    test: int = 126,
    seed: int = 42,
) -> dict[str, list[BratsCase]]:
    cases = list(cases)
    rng = random.Random(seed)
    rng.shuffle(cases)
    return {
        "train": cases[:train],
        "val": cases[train : train + val],
        "test": cases[train + val : train + val + test],
    }


def load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def zscore_nonzero(x: np.ndarray) -> np.ndarray:
    mask = x != 0
    if not np.any(mask):
        return x
    mean = x[mask].mean()
    std = x[mask].std()
    if std < 1e-6:
        std = 1.0
    x = (x - mean) / std
    x[~mask] = 0
    return x


def resize_volume(x: torch.Tensor, size: Sequence[int], mode: str) -> torch.Tensor:
    x = x.unsqueeze(0)
    if mode == "nearest":
        y = F.interpolate(x, size=size, mode=mode)
    else:
        y = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
    return y.squeeze(0)


def brats_label_to_regions(seg: torch.Tensor) -> torch.Tensor:
    seg = seg.long()
    et = seg == 4
    tc = (seg == 1) | (seg == 4)
    wt = (seg == 1) | (seg == 2) | (seg == 4)
    return torch.stack([et, tc, wt], dim=0).float()


class BratsDataset(Dataset):
    def __init__(
        self,
        cases: Sequence[BratsCase],
        image_size: Sequence[int] = (155, 240, 240),
        task: str = "pretrain",
        missing_modalities: Sequence[str] | None = None,
    ) -> None:
        self.cases = list(cases)
        self.image_size = tuple(int(v) for v in image_size)
        self.task = task
        self.missing_modalities = set(missing_modalities or [])

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        case = self.cases[idx]
        channels = []
        for mod in MODALITIES:
            arr = zscore_nonzero(load_nifti(case.modalities[mod]))
            t = torch.from_numpy(arr).float().unsqueeze(0)
            t = resize_volume(t, self.image_size, mode="trilinear").squeeze(0)
            if mod in self.missing_modalities:
                t.zero_()
            channels.append(t)
        image = torch.stack(channels, dim=0)
        sample: dict[str, torch.Tensor | str] = {"image": image, "case_id": case.case_id}

        if self.task == "seg":
            if case.seg is None:
                raise RuntimeError(f"Case {case.case_id} has no segmentation label.")
            seg = torch.from_numpy(load_nifti(case.seg)).float().unsqueeze(0)
            seg = resize_volume(seg, self.image_size, mode="nearest").squeeze(0)
            sample["label"] = brats_label_to_regions(seg)
        elif self.task == "cls":
            if case.cls_label is None:
                raise RuntimeError(f"Case {case.case_id} has no classification label.")
            sample["label"] = torch.tensor(case.cls_label, dtype=torch.long)
        return sample

