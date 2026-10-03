"""Load a Looped-DiT diffusers pipeline from a Hub folder or a training checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import DiffusionPipeline

from looped_dit.config import TrainConfig
from looped_dit.diffusers.pipeline_looped_dit import LoopedDiTTextToImagePipeline
from looped_dit.diffusers.transformer_looped_dit import LoopedMMDiTModel


def train_config_to_transformer_kwargs(cfg: TrainConfig) -> dict:
    return dict(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        mlp_ratio=cfg.mlp_ratio,
        pca_channels=cfg.pca_channels,
        text_dim=cfg.text_dim,
        text_preamble_depth=cfg.text_preamble_depth,
        loop_split=tuple(cfg.loop_split),
        num_loops=cfg.num_loops,
        share_loop_weights=cfg.share_loop_weights,
        use_xsa=cfg.use_xsa,
        use_attn_gate=cfg.use_attn_gate,
        prompt_length=cfg.prompt_length,
        noise_scale=cfg.noise_scale,
        text_encoder_name=cfg.text_encoder,
    )


def pipeline_from_training_checkpoint(
    checkpoint: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    weights: str = "ema",
) -> LoopedDiTTextToImagePipeline:
    """Build a diffusers pipeline from a `.pt` training checkpoint (EMA weights by default)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = TrainConfig.from_dict(ckpt["config"])
    transformer = LoopedMMDiTModel(**train_config_to_transformer_kwargs(cfg))
    transformer.load_state_dict(ckpt[weights])
    pipe = LoopedDiTTextToImagePipeline(
        transformer=transformer.to(device=device, dtype=dtype),
        scheduler=LoopedDiTTextToImagePipeline._default_inference_scheduler(),
        text_encoder_name=cfg.text_encoder,
        default_num_loops=cfg.num_loops,
        noise_scale=cfg.noise_scale,
    )
    return pipe.to(device)


def load_pipeline(
    model_path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    weights: str = "ema",
    local_files_only: bool = False,
) -> LoopedDiTTextToImagePipeline:
    """Load from a diffusers model directory or a legacy `.pt` checkpoint."""
    path = Path(model_path)
    if path.suffix == ".pt":
        return pipeline_from_training_checkpoint(path, device, dtype=dtype, weights=weights)

    custom_pipeline = path / "pipeline.py"
    kwargs: dict = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    if custom_pipeline.is_file():
        kwargs["custom_pipeline"] = str(custom_pipeline)
    pipe = DiffusionPipeline.from_pretrained(str(path), **kwargs)
    if not isinstance(pipe, LoopedDiTTextToImagePipeline):
        raise TypeError(f"Expected LoopedDiTTextToImagePipeline, got {type(pipe)}")
    return pipe.to(device)
