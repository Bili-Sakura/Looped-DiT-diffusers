"""Convert a Looped-DiT training checkpoint into a diffusers folder.

    python tools/convert_to_diffusers.py --checkpoint checkpoints/looped-dit-b16.pt \\
        --output-dir Looped-DiT-diffusers/Looped-DiT-B-16

The folder is what `DiffusionPipeline.from_pretrained` loads (`pipeline.py`, transformer weights,
scheduler, and, by default, FLAN-T5). Pass `--skip-text-encoder` to leave the text encoder as a
Hub id that is downloaded on first use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_dit.checkpoint import convert_to_diffusers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="training .pt file (EMA weights by default)")
    parser.add_argument("--output-dir", required=True, help="diffusers folder to write")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--config", help="YAML config, required when the checkpoint is a raw state dict")
    parser.add_argument("--text-encoder", help="override the text encoder id stored in the checkpoint")
    parser.add_argument(
        "--skip-text-encoder",
        action="store_true",
        help="do not copy FLAN-T5 into the folder; the pipeline downloads it on first use",
    )
    args = parser.parse_args()
    output = convert_to_diffusers(
        args.checkpoint,
        args.output_dir,
        weights=args.weights,
        text_encoder_name=args.text_encoder,
        bundle_text_encoder=not args.skip_text_encoder,
        config_path=args.config,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
