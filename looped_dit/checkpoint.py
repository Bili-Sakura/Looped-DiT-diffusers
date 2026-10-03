"""Load training checkpoints into `LoopedDiTPipeline` and export diffusers folders."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from diffusers import __version__ as diffusers_version
from transformers import AutoTokenizer, T5EncoderModel

from .config import TrainConfig
from .pipeline import LoopedDiTPipeline
from .transformer_looped_dit import LoopedDiTTransformer2DModel

PIPELINE_CLASS_NAME = "LoopedDiTPipeline"
TRANSFORMER_FILE = "transformer_looped_dit.py"
TRANSFORMER_CLASS_NAME = "LoopedDiTTransformer2DModel"

_PACKAGE = Path(__file__).resolve().parent


def _is_state_dict(value: object) -> bool:
    return isinstance(value, dict) and bool(value) and all(torch.is_tensor(item) for item in value.values())


def read_training_checkpoint(
    path: str | Path, weights: str = "ema", config_path: str | Path | None = None
) -> tuple[TrainConfig, dict[str, torch.Tensor], int | None]:
    r"""
    Read a Looped-DiT `.pt` checkpoint.

    Training saves `{"config", "ema", "model", "optimizer", "step"}`. A raw state dict is also
    accepted when `config_path` points at the YAML config that built it.

    Args:
        path: Checkpoint file.
        weights: Which tensor dict to load. `"ema"` is the paper sampling weights; `"model"` is the
            raw training weights.
        config_path: YAML config used when the file itself has no `"config"` entry.

    Returns:
        The training config, the selected state dict, and the step counter when the file has one.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "config" in ckpt and weights in ckpt and _is_state_dict(ckpt[weights]):
        return TrainConfig.from_dict(ckpt["config"]), ckpt[weights], int(ckpt["step"]) if "step" in ckpt else None
    if _is_state_dict(ckpt):
        if config_path is None:
            raise ValueError(
                f"{path} is a raw state dict with no training config. Pass the YAML config that built it."
            )
        return TrainConfig.from_yaml(config_path), ckpt, None
    keys = sorted(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__
    raise ValueError(
        f"Unrecognized checkpoint {path}. Expected a training checkpoint with 'config' and {weights!r}, "
        f"or a raw state dict plus a config. Found keys: {keys}."
    )


def build_text_encoder(name: str, prompt_length: int) -> tuple[Any, T5EncoderModel]:
    r"""
    Load the frozen tokenizer and FLAN-T5 encoder in float32.

    Args:
        name: Hub id or local directory of the text encoder.
        prompt_length: Tokenizer `model_max_length`, matching training.

    Returns:
        `(tokenizer, text_encoder)` with the encoder in eval mode and frozen.
    """
    tokenizer = AutoTokenizer.from_pretrained(name, model_max_length=prompt_length)
    text_encoder = T5EncoderModel.from_pretrained(name).eval().requires_grad_(False)
    return tokenizer, text_encoder


def pipeline_from_checkpoint(
    path: str | Path,
    torch_dtype: torch.dtype | None = torch.bfloat16,
    weights: str = "ema",
    text_encoder_name: str | None = None,
    config_path: str | Path | None = None,
    load_text_encoder: bool = True,
) -> LoopedDiTPipeline:
    r"""
    Build a diffusers pipeline from a training `.pt` file.

    Args:
        path: Checkpoint file.
        torch_dtype: Dtype of the denoiser. `None` keeps the checkpoint dtype. The text encoder stays float32.
        weights: `"ema"` or `"model"`.
        text_encoder_name: Override the text-encoder id stored in the checkpoint.
        config_path: YAML config for a raw state dict.
        load_text_encoder: When `False`, tokenizer and text encoder are left empty so tests can pass embeddings.

    Returns:
        A `LoopedDiTPipeline` on CPU.
    """
    cfg, state, _step = read_training_checkpoint(path, weights=weights, config_path=config_path)
    transformer = LoopedDiTTransformer2DModel(**cfg.model_kwargs())
    transformer.load_state_dict(state)
    if torch_dtype is not None:
        transformer.to(dtype=torch_dtype)
    transformer.eval().requires_grad_(False)

    name = text_encoder_name or cfg.text_encoder
    tokenizer = text_encoder = None
    if load_text_encoder:
        tokenizer, text_encoder = build_text_encoder(name, cfg.prompt_length)
    return LoopedDiTPipeline(
        transformer=transformer,
        scheduler=LoopedDiTPipeline._default_scheduler(),
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        text_encoder_name=name,
        prompt_length=cfg.prompt_length,
        noise_scale=cfg.noise_scale,
    )


def load_pipeline(
    checkpoint: str | Path,
    device: torch.device,
    torch_dtype: torch.dtype | None = torch.bfloat16,
    text_encoder_name: str | None = None,
    weights: str = "ema",
) -> LoopedDiTPipeline:
    r"""
    Load either a training `.pt` file or a converted diffusers directory and move it to `device`.

    The denoiser uses `torch_dtype`. The text encoder is kept in float32, which is how the paper's
    sampler ran FLAN-T5.

    Args:
        checkpoint: `.pt` file, local diffusers directory, or a Hub repo id.
        device: Execution device.
        torch_dtype: Denoiser dtype.
        text_encoder_name: Optional replacement for the bundled text encoder.
        weights: Weight key used when `checkpoint` is a `.pt` file.

    Returns:
        An eval-mode pipeline on `device`.
    """
    path = Path(checkpoint)
    if path.is_file():
        pipe = pipeline_from_checkpoint(
            path, torch_dtype=torch_dtype, weights=weights, text_encoder_name=text_encoder_name
        )
    else:
        local = path.is_dir()
        pipe = LoopedDiTPipeline.from_pretrained(
            str(path) if local else str(checkpoint),
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=local,
        )
        name = text_encoder_name or pipe.config.text_encoder_name
        needs_encoder = pipe.text_encoder is None or pipe.tokenizer is None or (
            text_encoder_name is not None and text_encoder_name != pipe.config.text_encoder_name
        )
        if needs_encoder:
            tokenizer, text_encoder = build_text_encoder(name, int(pipe.config.prompt_length))
            pipe.register_modules(tokenizer=tokenizer, text_encoder=text_encoder)
            pipe.register_to_config(text_encoder_name=name)

    if pipe.text_encoder is not None:
        pipe.text_encoder.to(dtype=torch.float32)
        pipe.text_encoder.eval().requires_grad_(False)
    pipe.transformer.eval().requires_grad_(False)
    return pipe.to(device)


def convert_to_diffusers(
    checkpoint: str | Path,
    output_dir: str | Path,
    weights: str = "ema",
    text_encoder_name: str | None = None,
    bundle_text_encoder: bool = True,
    config_path: str | Path | None = None,
) -> Path:
    r"""
    Write a self-contained diffusers folder that `DiffusionPipeline.from_pretrained` can load.

    The folder contains `pipeline.py`, `model_index.json`, `scheduler/scheduler_config.json`, and
    `transformer/` (`config.json`, weights, and `transformer_looped_dit.py`). The text encoder and
    tokenizer are copied in unless `bundle_text_encoder` is false.

    Args:
        checkpoint: Training `.pt` file.
        output_dir: Destination directory. Created if missing.
        weights: `"ema"` or `"model"`.
        text_encoder_name: Override the checkpoint's text encoder id.
        bundle_text_encoder: When true, save FLAN-T5 into the folder. When false, the pipeline
            downloads `text_encoder_name` on first use.
        config_path: YAML config for a raw state dict.

    Returns:
        `output_dir` as a `Path`.
    """
    cfg, state, step = read_training_checkpoint(checkpoint, weights=weights, config_path=config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    transformer = LoopedDiTTransformer2DModel(**cfg.model_kwargs())
    transformer.load_state_dict(state)
    transformer.eval()
    transformer.save_pretrained(output / "transformer", safe_serialization=True)
    shutil.copyfile(_PACKAGE / TRANSFORMER_FILE, output / "transformer" / TRANSFORMER_FILE)

    scheduler = LoopedDiTPipeline._default_scheduler()
    scheduler.save_pretrained(output / "scheduler")

    name = text_encoder_name or cfg.text_encoder
    tokenizer_entry: list[Any] = [None, None]
    text_encoder_entry: list[Any] = [None, None]
    if bundle_text_encoder:
        tokenizer, text_encoder = build_text_encoder(name, cfg.prompt_length)
        tokenizer.save_pretrained(output / "tokenizer")
        text_encoder.save_pretrained(output / "text_encoder")
        tokenizer_entry = ["transformers", tokenizer.__class__.__name__]
        text_encoder_entry = ["transformers", text_encoder.__class__.__name__]

    shutil.copyfile(_PACKAGE / "pipeline.py", output / "pipeline.py")
    model_index = {
        "_class_name": ["pipeline", PIPELINE_CLASS_NAME],
        "_diffusers_version": diffusers_version,
        "default_num_inference_steps": 100,
        "noise_scale": cfg.noise_scale,
        "prompt_length": cfg.prompt_length,
        "text_encoder_name": name,
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": text_encoder_entry,
        "tokenizer": tokenizer_entry,
        "transformer": [Path(TRANSFORMER_FILE).stem, TRANSFORMER_CLASS_NAME],
    }
    (output / "model_index.json").write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "source_checkpoint": str(Path(checkpoint)),
        "weights": weights,
        "step": step,
        "image_size": cfg.image_size,
        "patch_size": cfg.patch_size,
        "num_loops": cfg.num_loops,
        "loop_split": list(cfg.loop_split),
        "noise_scale": cfg.noise_scale,
        "text_encoder": name,
        "bundled_text_encoder": bundle_text_encoder,
    }
    (output / "conversion_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    text_encoder_lines = (
        "pipe.text_encoder.to(dtype=torch.float32)  # paper sampler keeps FLAN-T5 in fp32\n"
        if bundle_text_encoder
        else "# text_encoder was not bundled; the first call downloads it from text_encoder_name\n"
    )
    readme = f"""# Looped-DiT diffusers checkpoint

Converted from `{Path(checkpoint).name}` ({weights} weights). Load with:

```python
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

model_dir = Path({str(output.resolve())!r})
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
{text_encoder_lines}pipe.to("cuda")
image = pipe("a red cube on top of a blue sphere", num_inference_steps=100, guidance_scale=6.0).images[0]
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return output
