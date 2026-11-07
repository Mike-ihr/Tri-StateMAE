from __future__ import annotations

import argparse

try:
    from _bootstrap import add_project_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_project_src_to_path

add_project_src_to_path()

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tristate_mae.models.downstream import ViTClassifier, load_pretrained_encoder
from tristate_mae.training.common import make_loaders, optimizer_and_scheduler, output_dir, set_loader_epoch
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
    loaders = make_loaders(cfg, task="cls", distributed=ctx.enabled)
    model = ViTClassifier(**cfg["model"]).to(device)
    ckpt = args.pretrained or cfg["training"].get("pretrained")
    if ckpt:
        load_pretrained_encoder(model, load_checkpoint(ckpt))
    model = wrap_ddp(model, ctx)
    opt, sched = optimizer_and_scheduler(model, cfg, len(loaders["train"]))
    out = output_dir(cfg)
    writer = SummaryWriter(out / "tensorboard") if ctx.is_main else None
    if ctx.is_main:
        save_config(cfg, out / "config.yaml")
    best_auc = -1.0
    global_step = 0
    try:
        for epoch in range(cfg["training"]["epochs"]):
            set_loader_epoch(loaders["train"], epoch)
            model.train()
            total = 0.0
            pbar = tqdm(loaders["train"], desc=f"cls {epoch + 1}/{cfg['training']['epochs']}", disable=not ctx.is_main)
            for batch in pbar:
                img = batch["image"].to(device, non_blocking=True)
                lab = batch["label"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(img), lab)
                loss.backward()
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
            acc, auc = evaluate(model, loaders.get("val", loaders["train"]), device)
            if writer is not None:
                writer.add_scalar("val/accuracy", acc, epoch + 1)
                writer.add_scalar("val/auc", auc, epoch + 1)
                writer.add_scalar("epoch/loss", total / max(1, len(loaders["train"])), epoch + 1)
            if ctx.is_main:
                print(f"epoch={epoch + 1} val_acc={acc:.4f} val_auc={auc:.4f}")
                if auc > best_auc:
                    best_auc = auc
                    save_checkpoint(out / "cls_best.pt", model=unwrap_model(model).state_dict(), epoch=epoch + 1, auc=auc, acc=acc)
        if ctx.is_main:
            save_checkpoint(out / "cls_last.pt", model=unwrap_model(model).state_dict(), epoch=cfg["training"]["epochs"], auc=best_auc)
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed(ctx)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    labels, probs = [], []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        lab = batch["label"]
        prob = torch.softmax(model(img), dim=1)[:, 1].cpu()
        labels.extend(lab.tolist())
        probs.extend(prob.tolist())
    preds = [int(p >= 0.5) for p in probs]
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return float(acc), float(auc)


if __name__ == "__main__":
    main()
