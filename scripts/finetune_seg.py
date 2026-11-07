from __future__ import annotations

import argparse

try:
    from _bootstrap import add_project_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_project_src_to_path

add_project_src_to_path()

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tristate_mae.models.downstream import UNETRSegmentation, load_pretrained_encoder
from tristate_mae.training.common import make_loaders, optimizer_and_scheduler, output_dir, set_loader_epoch
from tristate_mae.training.losses import DiceCELoss, dice_score
from tristate_mae.utils.checkpoint import load_checkpoint, save_checkpoint
from tristate_mae.utils.config import add_config_arg, load_config, save_config
from tristate_mae.utils.distributed import (
    cleanup_distributed,
    init_distributed,
    maybe_relaunch_with_torchrun,
    reduce_mean,
    unwrap_model,
    wrap_ddp,
)
from tristate_mae.utils.seed import seed_everything


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--pretrained", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    maybe_relaunch_with_torchrun(cfg)
    ctx = init_distributed(cfg)
    seed_everything(cfg.get("seed", 42) + ctx.rank)
    device = ctx.device
    loaders = make_loaders(
        cfg,
        task="seg",
        missing_modalities=cfg["data"].get("missing_modalities"),
        distributed=ctx.enabled,
    )
    model = UNETRSegmentation(**cfg["model"]).to(device)
    ckpt = args.pretrained or cfg["training"].get("pretrained")
    if ckpt:
        load_pretrained_encoder(model, load_checkpoint(ckpt))
    model = wrap_ddp(model, ctx)
    criterion = DiceCELoss()
    opt, sched = optimizer_and_scheduler(model, cfg, len(loaders["train"]))
    out = output_dir(cfg)
    writer = SummaryWriter(out / "tensorboard") if ctx.is_main else None
    if ctx.is_main:
        save_config(cfg, out / "config.yaml")
    best = -1.0
    global_step = 0
    try:
        for epoch in range(cfg["training"]["epochs"]):
            set_loader_epoch(loaders["train"], epoch)
            model.train()
            total = 0.0
            pbar = tqdm(loaders["train"], desc=f"seg {epoch + 1}/{cfg['training']['epochs']}", disable=not ctx.is_main)
            for batch in pbar:
                img = batch["image"].to(device, non_blocking=True)
                lab = batch["label"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(img), lab)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("clip_grad", 1.0))
                opt.step()
                sched.step()
                loss_mean = reduce_mean(loss.detach(), ctx)
                total += loss_mean.item()
                lr = sched.get_last_lr()[0]
                if writer is not None:
                    writer.add_scalar("train/loss", loss_mean.item(), global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                global_step += 1
                pbar.set_postfix(loss=f"{loss_mean.item():.4f}", lr=f"{lr:.2e}")
            metrics = evaluate(model, loaders.get("val", loaders["train"]), device)
            mean_dice = metrics["mean"]
            if writer is not None:
                writer.add_scalar("val/mean_dice", mean_dice, epoch + 1)
                writer.add_scalar("val/dice_et", metrics["et"], epoch + 1)
                writer.add_scalar("val/dice_tc", metrics["tc"], epoch + 1)
                writer.add_scalar("val/dice_wt", metrics["wt"], epoch + 1)
                writer.add_scalar("epoch/loss", total / max(1, len(loaders["train"])), epoch + 1)
            if ctx.is_main:
                print(
                    f"epoch={epoch + 1} "
                    f"val_mean_dice={mean_dice:.4f} "
                    f"ET={metrics['et']:.4f} TC={metrics['tc']:.4f} WT={metrics['wt']:.4f}"
                )
                if mean_dice > best:
                    best = mean_dice
                    save_checkpoint(
                        out / "seg_best.pt",
                        model=unwrap_model(model).state_dict(),
                        epoch=epoch + 1,
                        mean_dice=best,
                        dice_et=metrics["et"],
                        dice_tc=metrics["tc"],
                        dice_wt=metrics["wt"],
                    )
        if ctx.is_main:
            save_checkpoint(out / "seg_last.pt", model=unwrap_model(model).state_dict(), epoch=cfg["training"]["epochs"], mean_dice=best)
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed(ctx)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
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


if __name__ == "__main__":
    main()
