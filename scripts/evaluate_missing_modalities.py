from __future__ import annotations

import argparse

try:
    from _bootstrap import add_project_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_project_src_to_path

add_project_src_to_path()

import torch

from scripts.finetune_seg import evaluate
from tristate_mae.data.brats import MODALITIES
from tristate_mae.models.downstream import UNETRSegmentation
from tristate_mae.training.common import make_loaders
from tristate_mae.utils.checkpoint import load_checkpoint
from tristate_mae.utils.config import add_config_arg, load_config
from tristate_mae.utils.distributed import device_from_config


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = device_from_config(cfg.get("device"))
    model = UNETRSegmentation(**cfg["model"]).to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint)["model"], strict=True)

    results = {}
    for missing in [None, *MODALITIES]:
        missing_list = [] if missing is None else [missing]
        loaders = make_loaders(cfg, task="seg", missing_modalities=missing_list)
        loader = loaders.get("test", loaders.get("val", loaders["train"]))
        results["full" if missing is None else f"-{missing}"] = evaluate(model, loader, device)
    full = results["full"]["mean"]
    drops = [full - v["mean"] for k, v in results.items() if k != "full"]
    results["drop"] = sum(drops) / len(drops)
    print(results)


if __name__ == "__main__":
    main()
