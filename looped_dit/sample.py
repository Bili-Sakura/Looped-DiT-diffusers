"""Generate images with the Looped-DiT diffusers pipeline.

    python -m looped_dit.sample --model checkpoints/looped-dit-b16.pt \\
        --prompt "a red cube on top of a blue sphere" --out cube.png
    python -m looped_dit.sample --model exports/Looped-DiT-B16 \\
        --prompt "..." --loops 1 2 3 4 --out loops.png
"""

from __future__ import annotations

import argparse

import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image

from looped_dit.diffusers import load_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="diffusers folder or legacy .pt checkpoint")
    parser.add_argument("--prompt", required=True, action="append", help="repeat for several prompts")
    parser.add_argument("--loops", type=int, nargs="+", default=[None], help="loop depth(s); default: as trained")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", default="ema", choices=("ema", "model"), help="for .pt checkpoints only")
    parser.add_argument("--out", default="samples.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    pipe = load_pipeline(args.model, device, dtype=dtype, weights=args.weights)
    images = []
    for loops in args.loops:
        assert loops is None or loops >= 1, f"loop depth must be >= 1, got {loops}"
        generator = torch.Generator(device=device).manual_seed(args.seed)
        result = pipe(
            args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg_scale,
            num_loops=loops,
            generator=generator,
        )
        images += result.images
    save_image(torch.stack([pil_to_tensor(im) for im in images]).float() / 255, args.out, nrow=len(args.prompt))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
