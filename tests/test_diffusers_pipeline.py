"""Diffusers pipeline smoke tests."""

import json
from pathlib import Path

import torch

from looped_dit.diffusers.load import pipeline_from_training_checkpoint, train_config_to_transformer_kwargs
from looped_dit.diffusers.pipeline_looped_dit import LoopedDiTTextToImagePipeline
from looped_dit.diffusers.transformer_looped_dit import LoopedMMDiTModel
from looped_dit.config import TrainConfig


def tiny_transformer() -> LoopedMMDiTModel:
    return LoopedMMDiTModel(
        image_size=32,
        patch_size=8,
        hidden_size=64,
        num_heads=2,
        head_dim=32,
        mlp_ratio=2.0,
        pca_channels=16,
        text_dim=16,
        text_preamble_depth=1,
        loop_split=(1, 2, 1),
        num_loops=3,
        prompt_length=8,
        noise_scale=2.0,
    )


def test_pred_velocity_shape():
    model = tiny_transformer().eval()
    x = torch.randn(2, 3, 32, 32)
    t = torch.full((2,), 0.5)
    text = torch.randn(2, 8, 16)
    mask = torch.ones(2, 8, dtype=torch.long)
    v = model.pred_velocity(x, t, text, mask, num_loops=2)
    assert v.shape == x.shape


def test_pipeline_call_cpu():
    transformer = tiny_transformer().eval()
    pipe = LoopedDiTTextToImagePipeline(
        transformer=transformer,
        scheduler=LoopedDiTTextToImagePipeline._default_inference_scheduler(),
        text_encoder_name="google/flan-t5-large",
        default_num_loops=2,
    )
    torch.manual_seed(0)
    text = torch.randn(1, 8, 16)
    mask = torch.ones(1, 8, dtype=torch.long)

    def fake_encode(prompt, device):
        return text, mask

    pipe._encode_prompt = fake_encode  # type: ignore[method-assign]
    out = pipe(
        "test",
        num_inference_steps=2,
        guidance_scale=1.0,
        num_loops=2,
        output_type="pt",
        progress=False,
    )
    assert out.images.shape == (1, 3, 32, 32)


def test_checkpoint_roundtrip(tmp_path):
    cfg = TrainConfig.from_dict(
        {
            "image_size": 32,
            "patch_size": 8,
            "hidden_size": 64,
            "num_heads": 2,
            "head_dim": 32,
            "mlp_ratio": 2.0,
            "pca_channels": 16,
            "text_dim": 16,
            "text_preamble_depth": 1,
            "loop_split": [1, 2, 1],
            "num_loops": 3,
            "deep_supervision": False,
            "prompt_length": 8,
        }
    )
    model = LoopedMMDiTModel(**train_config_to_transformer_kwargs(cfg))
    ckpt_path = tmp_path / "tiny.pt"
    torch.save({"config": cfg.to_dict(), "ema": model.state_dict()}, ckpt_path)
    pipe = pipeline_from_training_checkpoint(ckpt_path, torch.device("cpu"), dtype=torch.float32)
    assert pipe.transformer.config.image_size == 32


def test_export_layout(tmp_path):
    cfg = TrainConfig.from_dict(
        {
            "image_size": 32,
            "patch_size": 8,
            "hidden_size": 64,
            "num_heads": 2,
            "head_dim": 32,
            "mlp_ratio": 2.0,
            "pca_channels": 16,
            "text_dim": 16,
            "text_preamble_depth": 1,
            "loop_split": [1, 2, 1],
            "num_loops": 3,
            "deep_supervision": False,
            "prompt_length": 8,
        }
    )
    model = LoopedMMDiTModel(**train_config_to_transformer_kwargs(cfg))
    ckpt_path = tmp_path / "tiny.pt"
    torch.save({"config": cfg.to_dict(), "ema": model.state_dict()}, ckpt_path)
    out = tmp_path / "export"
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools/convert_checkpoint_to_diffusers.py"),
            "--checkpoint",
            str(ckpt_path),
            "--output-dir",
            str(out),
            "--skip-text-encoder",
        ],
        check=True,
    )
    index = json.loads((out / "model_index.json").read_text(encoding="utf-8"))
    assert index["transformer"][1] == "LoopedMMDiTModel"
    assert (out / "pipeline.py").is_file()
    assert (out / "transformer/transformer_looped_dit.py").is_file()
