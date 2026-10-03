"""Generate images with the Looped-DiT diffusers pipeline.

    python -m looped_dit.sample --checkpoint checkpoints/looped-dit-b16.pt --prompt "a red cube on top of a blue sphere" --out cube.png
    python -m looped_dit.sample --checkpoint checkpoints/looped-dit-b16 --prompt "..." --loops 1 2 3 4 --out loops.png

`--checkpoint` is a training `.pt` file or a folder written by `tools/convert_to_diffusers.py`.
"""

from __future__ import annotations

import argparse

import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image

from .checkpoint import load_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="training .pt file or a converted diffusers directory")
    parser.add_argument("--prompt", required=True, action="append", help="repeat for several prompts")
    parser.add_argument("--loops", type=int, nargs="+", default=[None], help="loop depth(s); default: as trained")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-encoder", help="override the text encoder path from the checkpoint")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema", help="which tensors to load from a .pt file")
    parser.add_argument("--out", default="samples.png")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True  # as in training and evaluation
    torch.backends.cudnn.allow_tf32 = True
    pipe = load_pipeline(args.checkpoint, device, text_encoder_name=args.text_encoder, weights=args.weights)
    images = []
    for loops in args.loops:
        assert loops is None or loops >= 1, f"loop depth must be >= 1, got {loops}"
        torch.manual_seed(args.seed)  # same noise for every loop depth
        images += pipe(
            args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg_scale,
            num_loops=loops,
            output_type="pil",
        ).images
    # One row per loop depth, one column per prompt.
    save_image(torch.stack([pil_to_tensor(im) for im in images]).float() / 255, args.out, nrow=len(args.prompt))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
