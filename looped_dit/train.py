"""Train Looped-DiT.

Pretraining:
    torchrun --nproc_per_node=8 -m looped_dit.train --config configs/b32_pretrain.yml \
        --output-dir outputs/b32_pretrain

Fine-tuning continues the step counter, weights, EMA and optimizer of a pretrained checkpoint:
    torchrun --nproc_per_node=8 -m looped_dit.train --config configs/b32_finetune.yml \
        --output-dir outputs/b32_finetune --init-from outputs/b32_pretrain/checkpoints/checkpoint_0250000.pt

A run resumes automatically from the newest checkpoint in its output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from .config import TrainConfig
from .data import make_loader
from .diffusion import deep_supervision_weights, euler_sample, training_loss
from .model import LoopedMMDiT
from .text_encoding import TextEncoder
from .utils import (
    amp_dtype,
    atomic_save,
    init_distributed,
    is_main,
    learning_rate,
    rank,
    seed_everything,
    update_ema,
    world_size,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Looped-DiT.")
    parser.add_argument("--config", required=True, help="training config (YAML)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-from", help="checkpoint to start from, e.g. the pretrained model for fine-tuning")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    return parser.parse_args()


def checkpoints(directory: Path) -> list[Path]:
    return sorted(directory.glob("checkpoint_*.pt"))


def save_checkpoint(directory: Path, step: int, model, ema, optimizer, cfg: TrainConfig) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(),
    }
    atomic_save(state, directory / f"checkpoint_{step:07d}.pt")
    old = checkpoints(directory)[: -cfg.keep_last]
    for path in old:
        if cfg.ckpt_keep_every <= 0 or int(path.stem.split("_")[1]) % cfg.ckpt_keep_every:
            path.unlink()


def load_checkpoint(path: Path, cfg: TrainConfig, model, ema, optimizer) -> int:
    """Restore weights, EMA, optimizer state (when present) and the step counter."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    saved, current = ckpt["config"], cfg.model_kwargs()
    mismatch = {k: (saved.get(k), v) for k, v in current.items() if saved.get(k) != (list(v) if isinstance(v, tuple) else v)}
    if mismatch:
        raise ValueError(f"{path} was trained with a different architecture (checkpoint, config): {mismatch}")
    # A checkpoint with only EMA weights initializes both copies from them.
    model.load_state_dict(ckpt.get("model", ckpt["ema"]))
    ema.load_state_dict(ckpt["ema"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        for group in optimizer.param_groups:  # hyperparameters come from the config, not the checkpoint
            group.update(betas=(0.9, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    if is_main():
        state = "" if "optimizer" in ckpt else "; no optimizer state, Adam starts fresh"
        print(f"loaded {path} (step {ckpt['step']}{state})", flush=True)
    return int(ckpt["step"])


def main() -> None:
    args = parse_args()
    overrides = dict(item.split("=", 1) for item in args.set)
    cfg = TrainConfig.from_yaml(args.config, {k: yaml.safe_load(v) for k, v in overrides.items()})
    device = init_distributed()
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    per_step = cfg.micro_batch_size * world_size()
    if cfg.batch_size % per_step:
        raise ValueError(f"batch_size {cfg.batch_size} is not a multiple of micro_batch_size x GPUs = {per_step}")
    accum = cfg.batch_size // per_step
    dtype = amp_dtype(cfg.amp_dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(cfg.seed)  # same initialization on every rank
    model = LoopedMMDiT(**cfg.model_kwargs()).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, betas=(0.9, cfg.adam_beta2), weight_decay=cfg.weight_decay, fused=True
    )
    step = 0
    existing = checkpoints(ckpt_dir)
    resume = existing[-1] if existing else Path(args.init_from) if args.init_from else None
    if resume is not None:
        step = load_checkpoint(resume, cfg, model, ema, optimizer)
    seed_everything(cfg.seed + rank() + step)  # a resumed run does not replay the same noise
    net = DDP(model, device_ids=[device.index], gradient_as_bucket_view=True, bucket_cap_mb=100) if world_size() > 1 else model

    text_encoder = TextEncoder(cfg.text_encoder, cfg.prompt_length, device)
    batches = iter(make_loader(cfg, seed_offset=step))
    exit_weights = deep_supervision_weights(cfg.num_loops, cfg.deep_supervision_weighting) if cfg.deep_supervision else None
    if is_main():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{n_params / 1e6:.1f}M parameters, {world_size()} GPUs x micro-batch {cfg.micro_batch_size} "
              f"x {accum} accumulation steps = batch {cfg.batch_size}; exit weights {exit_weights}", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    wandb = None
    if args.wandb and is_main():
        import wandb

        wandb.init(project="looped-dit", name=out_dir.name, id=out_dir.name, resume="allow", config=cfg.to_dict())

    net.train()
    metric_sums: dict[str, torch.Tensor] = {}  # summed over the micro-batches of a log interval (this rank)
    micro_batches, last_time, last_step = 0, time.time(), step
    while step < cfg.num_steps:
        for micro in range(accum):
            batch = next(batches)
            images = batch["pixel_values"].to(device, non_blocking=True).float().mul_(1 / 127.5).add_(-1.0)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            text_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                text = text_encoder.encode(input_ids, text_mask).to(dtype)
            no_sync = world_size() > 1 and micro < accum - 1
            with net.no_sync() if no_sync else contextlib.nullcontext():
                with torch.autocast("cuda", dtype=dtype):
                    loss, metrics = training_loss(
                        net,
                        images,
                        text,
                        text_mask,
                        exit_weights,
                        noise_scale=cfg.noise_scale,
                        t_logit_mean=cfg.t_logit_mean,
                        t_logit_std=cfg.t_logit_std,
                        label_drop_rate=cfg.label_drop_rate,
                    )
                (loss / accum).backward()
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0) + value
            micro_batches += 1

        lr = learning_rate(step, cfg.learning_rate, cfg.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], cfg.max_grad_norm if cfg.max_grad_norm > 0 else math.inf
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_ema(ema, model, cfg.ema_decay)
        step += 1

        if step % cfg.log_every == 0 and is_main():
            now = time.time()
            log = {"step": step, **{k: v.item() / micro_batches for k, v in metric_sums.items()}, "lr": lr,
                   "grad_norm": grad_norm.item(), "steps_per_sec": (step - last_step) / (now - last_time)}
            print(json.dumps(log), flush=True)
            if wandb:
                wandb.log(log, step=step)
            metric_sums, micro_batches, last_time, last_step = {}, 0, now, step
        if (step % cfg.ckpt_every == 0 or step == cfg.num_steps) and is_main():
            save_checkpoint(ckpt_dir, step, model, ema, optimizer, cfg)
        if cfg.sample_every and step % cfg.sample_every == 0 and is_main():
            images = euler_sample(ema, text[:4], text_mask[:4], cfg.image_size, steps=25, cfg_scale=6.0,
                                  noise_scale=cfg.noise_scale)
            path = out_dir / "samples" / f"{step:07d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_image((images.float() + 1) / 2, path, nrow=2)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
