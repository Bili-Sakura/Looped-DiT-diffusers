"""Diffusers pipeline: Euler match, scheduler swap, and checkpoint conversion."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler
from PIL import Image

from looped_dit.checkpoint import HF_VARIANTS, convert_to_diffusers, pipeline_from_checkpoint, prepare_hf_repo
from looped_dit.config import TrainConfig
from looped_dit.diffusion import euler_sample
from looped_dit.pipeline import LoopedDiTPipeline, paper_euler_sigmas
from looped_dit.transformer_looped_dit import LoopedDiTTransformer2DModel


def tiny_config() -> TrainConfig:
    return TrainConfig.from_dict(
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
            "share_loop_weights": True,
            "deep_supervision": False,
            "use_xsa": False,
            "use_attn_gate": False,
            "prompt_length": 6,
            "noise_scale": 2.0,
            "text_encoder": "google/flan-t5-large",
        }
    )


def tiny_bundle():
    cfg = tiny_config()
    torch.manual_seed(0)
    model = LoopedDiTTransformer2DModel(**cfg.model_kwargs()).eval()
    torch.nn.init.normal_(model.final.weight, std=0.02)
    pipe = LoopedDiTPipeline(
        transformer=model,
        scheduler=LoopedDiTPipeline._default_scheduler(),
        prompt_length=cfg.prompt_length,
        noise_scale=cfg.noise_scale,
    )
    torch.manual_seed(1)
    mask = torch.ones(2, 6, dtype=torch.long)
    mask[0, 4:] = 0
    text = torch.randn(2, 6, 16)
    return pipe, model, text, mask


def test_pipeline_matches_training_euler_sampler():
    pipe, model, text, mask = tiny_bundle()
    kwargs = dict(num_inference_steps=2, guidance_scale=2.5, num_loops=2, output_type="latent")
    torch.manual_seed(0)
    got = pipe(prompt_embeds=text, prompt_attention_mask=mask, **kwargs).images
    torch.manual_seed(0)
    ref = euler_sample(model, text, mask, image_size=32, steps=2, cfg_scale=2.5, noise_scale=2.0, num_loops=2)
    assert torch.allclose(got.clamp(-1, 1), ref, atol=1e-5, rtol=1e-5)


def test_generator_is_reproducible_and_output_types_work():
    pipe, _model, text, mask = tiny_bundle()
    first = torch.Generator().manual_seed(4)
    second = torch.Generator().manual_seed(4)
    kwargs = dict(prompt_embeds=text, prompt_attention_mask=mask, num_inference_steps=2, guidance_scale=1.0, num_loops=3)
    a = pipe(generator=first, output_type="latent", **kwargs).images
    b = pipe(generator=second, output_type="latent", **kwargs).images
    assert torch.equal(a, b)

    images = pipe(generator=torch.Generator().manual_seed(4), output_type="pil", **kwargs).images
    assert len(images) == 2 and isinstance(images[0], Image.Image) and images[0].size == (32, 32)
    arrays = pipe(generator=torch.Generator().manual_seed(4), output_type="np", **kwargs).images
    assert arrays.shape == (2, 32, 32, 3) and arrays.dtype.kind == "u"
    pixels = pipe(generator=torch.Generator().manual_seed(4), output_type="pt", **kwargs).images
    assert pixels.shape == (2, 3, 32, 32) and pixels.dtype == torch.float32


def test_scheduler_can_be_swapped():
    pipe, _model, text, mask = tiny_bundle()
    pipe.scheduler = EulerDiscreteScheduler(num_train_timesteps=1000)
    out = pipe(
        prompt_embeds=text,
        prompt_attention_mask=mask,
        num_inference_steps=1,
        guidance_scale=1.0,
        output_type="latent",
    ).images
    assert out.shape == (2, 3, 32, 32)
    assert isinstance(pipe.scheduler, EulerDiscreteScheduler)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    shifted = pipe(
        prompt_embeds=text[:1],
        prompt_attention_mask=mask[:1],
        num_inference_steps=2,
        guidance_scale=1.0,
        output_type="latent",
        generator=torch.Generator().manual_seed(0),
    ).images
    assert shifted.shape == (1, 3, 32, 32)


def test_rejects_bad_inputs_without_loading_a_text_encoder():
    pipe, _model, text, _mask = tiny_bundle()
    with pytest.raises(TypeError):
        pipe(prompt=1)
    with pytest.raises(ValueError):
        pipe(prompt="a cat", height=16, width=16)
    with pytest.raises(ValueError):
        pipe(prompt_embeds=text, num_loops=0)
    with pytest.raises(ValueError):
        pipe(prompt_embeds=text, output_type="gif")


def test_convert_roundtrip_loads_custom_pipeline(tmp_path: Path):
    cfg = tiny_config()
    torch.manual_seed(0)
    model = LoopedDiTTransformer2DModel(**cfg.model_kwargs())
    torch.nn.init.normal_(model.final.weight, std=0.02)
    ckpt = tmp_path / "tiny.pt"
    torch.save({"step": 7, "ema": model.state_dict(), "config": cfg.to_dict()}, ckpt)

    folder = convert_to_diffusers(ckpt, tmp_path / "diffusers", bundle_text_encoder=False)
    index = json.loads((folder / "model_index.json").read_text())
    assert index["_class_name"] == ["pipeline", "LoopedDiTPipeline"]
    assert index["transformer"] == ["transformer_looped_dit", "LoopedDiTTransformer2DModel"]
    assert index["scheduler"] == ["diffusers", "FlowMatchEulerDiscreteScheduler"]
    assert index["text_encoder"] == [None, None]
    assert list((folder / "scheduler").iterdir()) == [folder / "scheduler" / "scheduler_config.json"]
    assert (folder / "transformer" / "transformer_looped_dit.py").is_file()
    assert (folder / "pipeline.py").is_file()

    spec = importlib.util.spec_from_file_location("bundled_looped_dit_pipeline", folder / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.LoopedDiTPipeline is not None

    via_hub_api = DiffusionPipeline.from_pretrained(
        folder,
        trust_remote_code=True,
        local_files_only=True,
        custom_pipeline=str(folder / "pipeline.py"),
    )
    assert via_hub_api.__class__.__name__ == "LoopedDiTPipeline"
    loaded = LoopedDiTPipeline.from_pretrained(folder, trust_remote_code=True, local_files_only=True)
    reference = pipeline_from_checkpoint(ckpt, torch_dtype=None, load_text_encoder=False)
    left = {key: value for key, value in loaded.transformer.state_dict().items()}
    right = reference.transformer.state_dict()
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)

    torch.manual_seed(2)
    mask = torch.ones(1, 6, dtype=torch.long)
    text = torch.randn(1, 6, 16)
    generator = torch.Generator().manual_seed(9)
    a = loaded(prompt_embeds=text, prompt_attention_mask=mask, num_inference_steps=2, guidance_scale=1.0,
               generator=generator, output_type="latent").images
    generator = torch.Generator().manual_seed(9)
    b = reference(prompt_embeds=text, prompt_attention_mask=mask, num_inference_steps=2, guidance_scale=1.0,
                  generator=generator, output_type="latent").images
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_hf_repo_skeleton_has_code_and_configs_without_weights(tmp_path: Path):
    root = prepare_hf_repo(tmp_path / "Looped-DiT-diffusers")
    package = Path(__file__).resolve().parents[1] / "looped_dit"
    assert set(HF_VARIANTS) == {path.name for path in root.iterdir() if path.is_dir()}
    for folder, config_path in HF_VARIANTS.items():
        variant = root / folder
        assert (variant / "pipeline.py").read_bytes() == (package / "pipeline.py").read_bytes()
        assert (variant / "transformer" / "transformer_looped_dit.py").read_bytes() == (
            package / "transformer_looped_dit.py"
        ).read_bytes()
        scheduler_files = list((variant / "scheduler").iterdir())
        assert scheduler_files == [variant / "scheduler" / "scheduler_config.json"]
        index = json.loads((variant / "model_index.json").read_text())
        assert index["_class_name"] == ["pipeline", "LoopedDiTPipeline"]
        assert index["transformer"] == ["transformer_looped_dit", "LoopedDiTTransformer2DModel"]
        assert index["text_encoder"] == [None, None] and index["tokenizer"] == [None, None]
        config = LoopedDiTTransformer2DModel.load_config(variant / "transformer")
        with torch.device("meta"):
            built = LoopedDiTTransformer2DModel.from_config(config)
        trained = TrainConfig.from_yaml(Path(__file__).resolve().parents[1] / config_path)
        for key, value in trained.model_kwargs().items():
            got = built.config[key]
            expect = list(value) if isinstance(value, tuple) else value
            assert got == expect, (folder, key, got, expect)
        assert not list(variant.rglob("*.safetensors"))

    shipped = Path(__file__).resolve().parents[1] / "Looped-DiT-diffusers"
    for folder in HF_VARIANTS:
        assert (shipped / folder / "pipeline.py").read_bytes() == (root / folder / "pipeline.py").read_bytes()
        assert json.loads((shipped / folder / "model_index.json").read_text()) == json.loads(
            (root / folder / "model_index.json").read_text()
        )
        assert json.loads((shipped / folder / "transformer" / "config.json").read_text()) == json.loads(
            (root / folder / "transformer" / "config.json").read_text()
        )


def test_paper_sigma_grid_matches_linspace():
    steps = 4
    sigmas = paper_euler_sigmas(steps)
    flow = torch.linspace(0.0, 1.0, steps + 1)
    assert sigmas == pytest.approx((1.0 - flow)[:-1].tolist())
    assert sigmas[0] == pytest.approx(1.0)
    assert len(sigmas) == steps
