from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

try:
    from _bootstrap import add_project_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_project_src_to_path

add_project_src_to_path()

from tristate_mae.utils.config import load_config, save_config


def available_pretrains(pretrain_root: Path, checkpoint_name: str) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    if not pretrain_root.exists():
        raise FileNotFoundError(f"Pretrain ablation root does not exist: {pretrain_root}")
    for run_dir in sorted(path for path in pretrain_root.iterdir() if path.is_dir()):
        checkpoint = run_dir / checkpoint_name
        if checkpoint.exists():
            runs.append((run_dir.name, checkpoint))
        else:
            print(f"skip {run_dir.name}: missing {checkpoint_name}")
    if not runs:
        raise FileNotFoundError(f"No {checkpoint_name} files found under {pretrain_root}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_cls.yaml")
    parser.add_argument("--pretrain-root", default="outputs/pretrain_tristate/ablations")
    parser.add_argument("--checkpoint-name", default="pretrain_last.pt")
    parser.add_argument("--output-root", default="outputs/finetune_cls/ablations")
    parser.add_argument("--only", nargs="*", default=None, help="Optional ablation names to run.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    pretrain_root = Path(args.pretrain_root)
    output_root = Path(args.output_root)
    selected = set(args.only) if args.only else None

    for name, checkpoint in available_pretrains(pretrain_root, args.checkpoint_name):
        if selected is not None and name not in selected:
            continue
        cfg = copy.deepcopy(base)
        cfg["training"]["pretrained"] = str(checkpoint)
        cfg["training"]["output_dir"] = str(output_root / name)
        cfg_path = output_root / f"{name}.yaml"
        save_config(cfg, cfg_path)
        cmd = [sys.executable, "scripts/finetune_cls.py", "--config", str(cfg_path)]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
