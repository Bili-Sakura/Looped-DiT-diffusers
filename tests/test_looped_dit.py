"""CPU unit tests: python -m pytest tests"""

import io
import itertools
from pathlib import Path

import pytest
import torch
import webdataset as wds
from PIL import Image

from looped_dit.config import TrainConfig
from looped_dit.data import WebDatasetMix
from looped_dit.diffusion import deep_supervision_weights, euler_sample, training_loss
from looped_dit.model import LoopedMMDiT, exclusive_self_attention

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*_*.yml"))


def tiny_model(**overrides) -> LoopedMMDiT:
    kwargs = dict(image_size=32, patch_size=8, hidden_size=64, num_heads=2, head_dim=32, mlp_ratio=2.0,
                  pca_channels=16, text_dim=16, text_preamble_depth=1, loop_split=(1, 2, 1), num_loops=3)
    kwargs.update(overrides)
    torch.manual_seed(0)
    model = LoopedMMDiT(**kwargs)
    torch.nn.init.normal_(model.final.weight, std=0.02)  # the head starts at zero; make outputs informative
    return model


def inputs(batch: int = 2):
    torch.manual_seed(1)
    mask = torch.ones(batch, 6, dtype=torch.long)
    mask[0, 4:] = 0
    return torch.randn(batch, 3, 32, 32), torch.randn(batch, 6, 16), mask


@pytest.mark.parametrize("shared", [True, False])
def test_exits_equal_shallower_loop_depths(shared):
    model = tiny_model(share_loop_weights=shared).eval()
    x, text, mask = inputs()
    with torch.no_grad():
        final, exits = model(x, text, mask, exit_loops=(1, 2))
        assert torch.equal(final, model(x, text, mask))
        for r in (1, 2):
            assert torch.allclose(exits[r], model(x, text, mask, num_loops=r), atol=1e-6)


def test_looping_reuses_weights():
    looped, untied = tiny_model(), tiny_model(share_loop_weights=False)
    assert len(looped.blocks) == 1 + 2 + 1
    assert len(untied.blocks) == 1 + 2 * 3 + 1
    x, text, mask = inputs()
    with torch.no_grad():
        assert not torch.allclose(looped(x, text, mask, num_loops=1), looped(x, text, mask))
        assert looped(x, text, mask, num_loops=6).shape == x.shape  # shared weights: any depth at inference
    with pytest.raises(ValueError):
        untied(x, text, mask, num_loops=4)


@pytest.mark.parametrize("modulation", ["use_xsa", "use_attn_gate"])
def test_modulation_only_in_looped_blocks(modulation):
    model = tiny_model(**{modulation: True})
    assert [getattr(block, modulation) for block in model.blocks] == [False, True, True, False]
    assert [block.update_text for block in model.blocks] == [True, True, True, False]


def test_xsa_output_is_orthogonal_to_own_value():
    out, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 8)
    z = exclusive_self_attention(out, v)
    assert torch.allclose((z * torch.nn.functional.normalize(v, dim=-1)).sum(-1), torch.zeros(2, 3, 5), atol=1e-5)


def test_deep_supervision_weights():
    assert deep_supervision_weights(4, "uniform") == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert deep_supervision_weights(4, "final_plus_mean") == pytest.approx([1 / 3, 1 / 3, 1 / 3, 1])
    assert deep_supervision_weights(4, "exponential") == pytest.approx([w * 16 / 15 for w in (1 / 8, 1 / 4, 1 / 2, 1)])


@pytest.mark.parametrize("deep_supervision", [True, False])
@pytest.mark.parametrize("modulation", [{}, {"use_xsa": True}, {"use_attn_gate": True}])
def test_every_trainable_parameter_is_trained(deep_supervision, modulation):
    model = tiny_model(**modulation)
    x, text, mask = inputs()
    weights = deep_supervision_weights(3, "uniform") if deep_supervision else None
    loss, metrics = training_loss(model, x.clamp(-1, 1), text, mask, weights)
    loss.backward()
    assert torch.isfinite(loss)
    assert ("loss_exit2" in metrics) == deep_supervision
    # Gradient-free exactly where the model is frozen: MiniT2I's unused embedders
    # and the text-stream update of the last block.
    frozen = {n.split(".")[0] if not n.startswith("blocks") else ".".join(n.split(".")[:3])
              for n, p in model.named_parameters() if not p.requires_grad}
    assert frozen == {"t_embed", "pooled_embed", "blocks.3.txt_norm2", "blocks.3.txt_proj", "blocks.3.txt_mlp"}
    assert all((p.grad is not None) == p.requires_grad for p in model.parameters())


def test_sampler_shape():
    model = tiny_model()
    _, text, mask = inputs()
    assert euler_sample(model, text, mask, image_size=32, steps=2, num_loops=2).shape == (2, 3, 32, 32)


def test_webdataset_source_restarts_when_exhausted(tmp_path):
    with wds.TarWriter(str(tmp_path / "shard-0.tar")) as sink:
        for i in range(3):
            buffer = io.BytesIO()
            Image.new("RGB", (40, 30), (i, 0, 0)).save(buffer, format="JPEG")
            sink.write({"__key__": f"s{i}", "jpg": buffer.getvalue(), "txt": "a caption"})
    cfg = TrainConfig.from_dict({"dataset": "webdataset", "image_size": 32, "prompt_length": 8, "shuffle_buffer": 2,
                                 "webdataset_sources": {"a": {"path": str(tmp_path), "weight": 1.0}}})

    def tokenizer(text, max_length, **_):
        return {"input_ids": torch.zeros(1, max_length, dtype=torch.long), "attention_mask": torch.ones(1, max_length, dtype=torch.long)}

    samples = list(itertools.islice(iter(WebDatasetMix(cfg, tokenizer=tokenizer)), 10))  # 3 samples, read 10
    assert len(samples) == 10 and samples[0]["pixel_values"].shape == (3, 32, 32)


def test_captions_from_i1_parquet(tmp_path):
    import importlib.util

    import pyarrow as pa
    import pyarrow.parquet as pq

    spec = importlib.util.spec_from_file_location("make_chunks", Path(__file__).resolve().parents[1] / "tools" / "make_chunks.py")
    make_chunks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_chunks)
    options = {f"id{i}": [f"id{i} caption {j}" for j in range(1, 6)] for i in range(50)}
    options["id0"][1:] = [None] * 4  # only one caption
    columns = {f"caption{j}": [captions[j - 1] for captions in options.values()] for j in range(1, 6)}
    pq.write_table(pa.table({"key": list(options), **columns}), tmp_path / "part.parquet")
    captions = make_chunks.load_captions(str(tmp_path))
    assert captions.keys() == options.keys() and all(captions[key] in options[key] for key in options)
    assert captions["id0"] == "id0 caption 1" and len({caption[-1] for caption in captions.values()}) == 5
    assert make_chunks.load_captions(str(tmp_path)) == captions  # the same choice on every run


def test_config_overrides_are_typed():
    cfg = TrainConfig.from_dict({"learning_rate": "2e-4", "num_steps": "1000", "noise_scale": 2})
    assert (cfg.learning_rate, cfg.num_steps, cfg.noise_scale) == (2e-4, 1000, 2.0)
    assert isinstance(cfg.noise_scale, float) and isinstance(cfg.num_steps, int)


PARAMETERS = {"b32": 260_189_568, "b16": 258_122_880, "l16": 911_766_696}  # as in the paper and MiniT2I


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_released_configs(path):
    cfg = TrainConfig.from_yaml(path)
    assert cfg.deep_supervision and cfg.use_xsa and not cfg.use_attn_gate and cfg.num_loops == 4
    with torch.device("meta"):
        model = LoopedMMDiT(**cfg.model_kwargs())
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS[path.name.split("_")[0]]


def test_tiif_answer_parsing():
    from looped_dit.eval.tiif import parse_answers

    assert parse_answers("Yes, a cat is visible.\n\nno - it is red\nYES", 3) == ["yes", "no", "yes"]
    with pytest.raises(ValueError):
        parse_answers("yes\nmaybe", 2)
