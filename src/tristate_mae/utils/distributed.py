from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def device_from_config(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def maybe_relaunch_with_torchrun(cfg: dict[str, Any]) -> None:
    dist_cfg = cfg.get("distributed", {})
    visible_cuda = dist_cfg.get("visible_cuda")
    if visible_cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_cuda)
    if not dist_cfg.get("enabled", False):
        return
    if "LOCAL_RANK" in os.environ:
        return
    num_gpus = int(dist_cfg.get("num_gpus", 1))
    if visible_cuda is not None:
        visible = str(visible_cuda)
        num_gpus = len([x for x in visible.split(",") if x.strip()])
    if num_gpus <= 1:
        return
    env = os.environ.copy()
    if visible_cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(visible_cuda)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        *sys.argv,
    ]
    raise SystemExit(subprocess.call(cmd, env=env))


def init_distributed(cfg: dict[str, Any]) -> DistributedContext:
    dist_cfg = cfg.get("distributed", {})
    enabled = bool(dist_cfg.get("enabled", False)) and "LOCAL_RANK" in os.environ
    if not enabled:
        device = device_from_config(cfg.get("device"))
        return DistributedContext(False, 0, 0, 1, device)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available():
        raise RuntimeError("DDP is enabled but CUDA is not available.")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=dist_cfg.get("backend", "nccl"))
    return DistributedContext(True, rank, local_rank, world_size, torch.device("cuda", local_rank))


def wrap_ddp(model: torch.nn.Module, ctx: DistributedContext) -> torch.nn.Module:
    if not ctx.enabled:
        return model
    return DDP(model, device_ids=[ctx.local_rank], output_device=ctx.local_rank, find_unused_parameters=False)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def barrier(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()


def reduce_mean(value: torch.Tensor, ctx: DistributedContext) -> torch.Tensor:
    if not ctx.enabled:
        return value
    value = value.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= ctx.world_size
    return value
