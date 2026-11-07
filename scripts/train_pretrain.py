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

from tristate_mae.models.mae import TriStateMAE
from tristate_mae.models.vit3d import pad_to_patch_size
from tristate_mae.training.common import make_loaders, optimizer_and_scheduler, output_dir, set_loader_epoch
from tristate_mae.utils.checkpoint import save_checkpoint
from tristate_mae.utils.config import add_config_arg, load_config, save_config
from tristate_mae.utils.distributed import (
    barrier,
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
    args = parser.parse_args()
    cfg = load_config(args.config)
    maybe_relaunch_with_torchrun(cfg)
    ctx = init_distributed(cfg)
    seed_everything(cfg.get("seed", 42) + ctx.rank)
    device = ctx.device
    loaders = make_loaders(cfg, task="pretrain", distributed=ctx.enabled)
    model = TriStateMAE(**cfg["model"]).to(device)
    model = wrap_ddp(model, ctx)
    opt, sched = optimizer_and_scheduler(model, cfg, len(loaders["train"]))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"].get("amp", True) and device.type == "cuda")
    out = output_dir(cfg)
    writer = SummaryWriter(out / "tensorboard") if ctx.is_main else None
    if ctx.is_main:
        save_config(cfg, out / "config.yaml")
        log_pretrain_efficiency(writer, unwrap_model(model), cfg, device)
    barrier(ctx)

    global_step = 0
    try:
        for epoch in range(cfg["training"]["epochs"]):
            set_loader_epoch(loaders["train"], epoch)
            model.train()
            total = 0.0
            pbar = tqdm(
                loaders["train"],
                desc=f"pretrain {epoch + 1}/{cfg['training']['epochs']}",
                disable=not ctx.is_main,
            )
            for batch in pbar:
                imgs = batch["image"].to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    output = model(imgs)
                    loss = output["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("clip_grad", 1.0))
                scaler.step(opt)
                scaler.update()
                sched.step()
                loss_mean = reduce_mean(loss.detach(), ctx)
                total += loss_mean.item()
                lr = sched.get_last_lr()[0]
                if writer is not None:
                    writer.add_scalar("train/loss", loss_mean.item(), global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    writer.add_scalar("mask/effective_mask_ratio", output["channel_masked"].float().mean().item(), global_step)
                    if (global_step + 1) % cfg["training"].get("tensorboard_flush_steps", 10) == 0:
                        writer.flush()
                global_step += 1
                pbar.set_postfix(loss=f"{total / (pbar.n + 1):.4f}", lr=f"{lr:.2e}")
            epoch_loss = total / max(1, len(loaders["train"]))
            if writer is not None:
                writer.add_scalar("epoch/loss", epoch_loss, epoch + 1)
                writer.flush()
            if ctx.is_main and (epoch + 1) % cfg["training"].get("save_every", 50) == 0:
                save_checkpoint(out / f"pretrain_epoch_{epoch + 1}.pt", model=unwrap_model(model).state_dict(), epoch=epoch + 1)
        if ctx.is_main:
            save_checkpoint(out / "pretrain_last.pt", model=unwrap_model(model).state_dict(), epoch=cfg["training"]["epochs"])
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed(ctx)


def log_pretrain_efficiency(writer: SummaryWriter | None, model: torch.nn.Module, cfg: dict, device: torch.device) -> None:
    if writer is None:
        return
    eff_cfg = cfg.get("efficiency", {})
    if not eff_cfg.get("profile_gflops", True):
        return
    size = eff_cfg.get("profile_input_size", cfg["data"].get("image_size", [155, 240, 240]))
    batch_size = int(eff_cfg.get("profile_batch_size", 1))
    sample = torch.zeros(batch_size, cfg["model"].get("in_chans", 4), *size, device=device)
    sample_pad, _ = pad_to_patch_size(sample, model.patch_size)
    patch_grid = (
        sample_pad.shape[-3] // model.patch_size[0],
        sample_pad.shape[-2] // model.patch_size[1],
        sample_pad.shape[-1] // model.patch_size[2],
    )
    num_patches = patch_grid[0] * patch_grid[1] * patch_grid[2]
    masks = model._sample_masks(batch_size, num_patches, device)
    sample_masked = model._apply_channel_patch_mask(sample_pad, masks["channel_visible"], patch_grid)
    was_training = model.training
    model.eval()
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        before_alloc = 0
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            before_alloc = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        with torch.cuda.amp.autocast(enabled=cfg["training"].get("amp", True) and device.type == "cuda"):
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                output = model.encoder(sample_masked, visible_tokens=masks["patch_visible"])
        flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
        writer.add_scalar("efficiency/encoder_forward_gflops", flops / 1e9, 0)
        if device.type == "cuda":
            activation_gb = max(0, torch.cuda.max_memory_allocated(device) - before_alloc) / 1024**3
            writer.add_scalar("efficiency/encoder_forward_activation_memory_gb", activation_gb, 0)
        del output
    except Exception as exc:
        print(f"Efficiency profiling skipped: {exc}")
    finally:
        if was_training:
            model.train()


if __name__ == "__main__":
    main()
