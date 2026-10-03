#!/usr/bin/env python3
"""Export a Looped-DiT training checkpoint to a self-contained diffusers model folder.

Example:
    python tools/convert_checkpoint_to_diffusers.py \\
        --checkpoint checkpoints/looped-dit-b16.pt \\
        --output-dir exports/Looped-DiT-B16 \\
        --weights ema
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoTokenizer, T5EncoderModel

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))

from looped_dit.config import TrainConfig  # noqa: E402
from looped_dit.diffusers.load import train_config_to_transformer_kwargs  # noqa: E402
from looped_dit.diffusers.transformer_looped_dit import LoopedMMDiTModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--weights", default="ema", choices=("ema", "model"))
    parser.add_argument("--variant-name", help="subfolder name (default: derived from checkpoint)")
    parser.add_argument(
        "--text-encoder-source",
        default="google/flan-t5-large",
        help="HF id to copy text_encoder and tokenizer from (default: config text encoder)",
    )
    parser.add_argument("--skip-text-encoder", action="store_true", help="do not bundle text_encoder/tokenizer")
    return parser.parse_args()


def write_scheduler(scheduler_dir: Path) -> None:
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    (scheduler_dir / "scheduler_config.json").write_text(
        json.dumps(
            {
                "_class_name": "FlowMatchEulerDiscreteScheduler",
                "_diffusers_version": "0.32.0",
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "stochastic_sampling": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_model_index(root: Path, variant: str, cfg: TrainConfig) -> None:
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": ["pipeline", "LoopedDiTTextToImagePipeline"],
                "_diffusers_version": "0.32.0",
                "default_num_inference_steps": 100,
                "default_num_loops": cfg.num_loops,
                "noise_scale": cfg.noise_scale,
                "recommended_guidance_scale": 6.0,
                "text_encoder": ["transformers", "T5EncoderModel"],
                "tokenizer": ["transformers", "T5Tokenizer"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "transformer": ["transformer_looped_dit", "LoopedMMDiTModel"],
                "text_encoder_name": cfg.text_encoder,
                "variant": variant,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def copy_pipeline_sources(dest_root: Path) -> None:
    shutil.copy2(REPO_ROOT / "looped_dit/diffusers/pipeline_looped_dit.py", dest_root / "pipeline.py")
    transformer_dir = dest_root / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "looped_dit/diffusers/transformer_looped_dit.py", transformer_dir / "transformer_looped_dit.py")
    shutil.copy2(REPO_ROOT / "looped_dit/model.py", transformer_dir / "looped_mmdit_core.py")


def save_text_encoder(text_encoder_name: str, dest_root: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
    encoder = T5EncoderModel.from_pretrained(text_encoder_name)
    tokenizer.save_pretrained(dest_root / "tokenizer")
    encoder.save_pretrained(dest_root / "text_encoder")


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = TrainConfig.from_dict(ckpt["config"])
    variant = args.variant_name or args.checkpoint.stem.replace("_", "-")
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory must be empty or new: {out}")
    out.mkdir(parents=True, exist_ok=True)

    transformer = LoopedMMDiTModel(**train_config_to_transformer_kwargs(cfg))
    transformer.load_state_dict(ckpt[args.weights])
    transformer_dir = out / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    config = dict(transformer.config)
    config["_class_name"] = "LoopedMMDiTModel"
    config["_diffusers_version"] = "0.32.0"
    (transformer_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    save_file(transformer.state_dict(), transformer_dir / "diffusion_pytorch_model.safetensors")

    copy_pipeline_sources(out)
    write_scheduler(out / "scheduler")
    write_model_index(out, variant, cfg)

    if not args.skip_text_encoder:
        text_source = args.text_encoder_source or cfg.text_encoder
        try:
            save_text_encoder(text_source, out)
        except OSError:
            cache = Path(snapshot_download(text_source))
            shutil.copytree(cache / "tokenizer", out / "tokenizer", dirs_exist_ok=True)
            shutil.copytree(cache / "text_encoder", out / "text_encoder", dirs_exist_ok=True)

    metadata = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "weights": args.weights,
        "train_config": cfg.to_dict(),
    }
    (out / "conversion_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote diffusers model to {out}")


if __name__ == "__main__":
    main()
