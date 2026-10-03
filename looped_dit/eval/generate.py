"""Generate the images of one benchmark; launch with torchrun to split prompts over GPUs.

    torchrun --nproc_per_node=8 -m looped_dit.eval.generate --benchmark geneval \
        --data eval_assets/geneval/evaluation_metadata.jsonl --checkpoint CKPT --outdir OUT/geneval_images

Images that already exist are skipped, so an interrupted run can be restarted.
The noise seed of a prompt depends on its index and on the rank that renders it:
use the same number of GPUs to reproduce a run exactly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from ..diffusers import load_pipeline
from ..utils import init_distributed, rank, world_size
from .benchmarks import BENCHMARKS, Prompt, image_path, load_prompts


def output_paths(benchmark: str, outdir: Path, prompt: Prompt, samples: int) -> list[Path]:
    if benchmark == "geneval":
        return [outdir / prompt.item_id / "samples" / f"{i:05d}.png" for i in range(samples)]
    if benchmark == "dpg":
        return [outdir / f"{prompt.item_id}.png"]
    return [image_path(outdir, prompt, i) for i in range(samples)]


def save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    image.save(tmp, format="PNG")
    os.replace(tmp, path)


def grid_2x2(images: list[Image.Image]) -> Image.Image:
    w, h = images[0].size
    canvas = Image.new("RGB", (2 * w, 2 * h))
    for image, xy in zip(images, [(0, 0), (w, 0), (0, h), (w, h)]):
        canvas.paste(image, xy)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark images.")
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--data", required=True, help="benchmark prompt file or directory")
    parser.add_argument("--checkpoint", required=True, help="diffusers folder or legacy .pt checkpoint")
    parser.add_argument("--weights", default="ema", choices=("ema", "model"), help="for .pt checkpoints only")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--loops", type=int, default=None, help="loop depth (default: as trained)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.benchmark == "dpg" and args.samples_per_prompt != 4:
        parser.error("DPG-Bench is scored on 2x2 grids: use --samples-per-prompt 4")

    device = init_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True  # as in training; here it also rounds the fp32 T5 encoder
    torch.backends.cudnn.allow_tf32 = True
    outdir = Path(args.outdir)
    prompts = load_prompts(args.benchmark, args.data)[rank() :: world_size()]
    todo = [p for p in prompts if not all(x.exists() for x in output_paths(args.benchmark, outdir, p, args.samples_per_prompt))]
    print(f"[rank {rank()}] {args.benchmark}: {len(todo)} of {len(prompts)} prompts to generate", flush=True)
    if todo:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        pipe = load_pipeline(args.checkpoint, device, dtype=dtype, weights=args.weights)
        for n, prompt in enumerate(todo, 1):
            images: list[Image.Image] = []
            while len(images) < args.samples_per_prompt:
                batch = min(args.batch_size, args.samples_per_prompt - len(images))
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + prompt.index * 1000 + len(images) + rank() * 1_000_000
                )
                result = pipe(
                    [prompt.text] * batch,
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg_scale,
                    num_loops=args.loops,
                    generator=generator,
                )
                images += result.images
            paths = output_paths(args.benchmark, outdir, prompt, args.samples_per_prompt)
            if args.benchmark == "dpg":
                images = [grid_2x2(images)]
            if args.benchmark == "geneval":
                (outdir / prompt.item_id).mkdir(parents=True, exist_ok=True)
                (outdir / prompt.item_id / "metadata.jsonl").write_text(json.dumps(prompt.metadata) + "\n")
            for image, path in zip(images, paths):
                save(image, path)
            if n % 50 == 0 or n == len(todo):
                print(f"[rank {rank()}] {n}/{len(todo)}", flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
