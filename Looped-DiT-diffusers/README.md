# Looped-DiT diffusers

Self-contained Looped-DiT text-to-image folders for Hugging Face diffusers. Each variant ships its own pipeline code, transformer module, and scheduler config. Transformer weights are added by converting a training checkpoint (see below). Looped-DiT denoises in RGB pixel space: there is no VAE.

Text conditioning is frozen `google/flan-t5-large`. The sampler is Euler flow matching for 100 steps at guidance scale 6.0.

## Variants

| Subfolder | Model | Patch | Blocks (pre-loop, looped, post-loop) | Loop depth | Guidance |
| --- | --- | ---: | --- | ---: | ---: |
| `Looped-DiT-B-32/` | Looped-DiT B/32 | 32 | 6, 5, 6 | 4 | 6.0 |
| `Looped-DiT-B-16/` | Looped-DiT B/16 | 16 | 6, 5, 6 | 4 | 6.0 |
| `Looped-DiT-L-16/` | Looped-DiT L/16 | 16 | 8, 7, 8 | 4 | 6.0 |

Load a variant subfolder, not this directory.

## Repo layout

```text
Looped-DiT-diffusers/
├── README.md
├── .gitattributes
├── Looped-DiT-B-32/
│   ├── model_index.json
│   ├── pipeline.py
│   ├── scheduler/
│   │   └── scheduler_config.json
│   └── transformer/
│       ├── config.json
│       └── transformer_looped_dit.py
├── Looped-DiT-B-16/
└── Looped-DiT-L-16/
```

`scheduler/` contains only `scheduler_config.json`. After conversion, a variant also contains `transformer/diffusion_pytorch_model.safetensors` and, unless `--skip-text-encoder` is set, `text_encoder/` and `tokenizer/`.

## Add weights

From the training repository root, convert an EMA checkpoint into the matching variant folder:

```bash
python tools/convert_to_diffusers.py \
    --checkpoint checkpoints/looped-dit-b16.pt \
    --output-dir Looped-DiT-diffusers/Looped-DiT-B-16

python tools/convert_to_diffusers.py \
    --checkpoint checkpoints/looped-dit-b32.pt \
    --output-dir Looped-DiT-diffusers/Looped-DiT-B-32

python tools/convert_to_diffusers.py \
    --checkpoint checkpoints/looped-dit-l16.pt \
    --output-dir Looped-DiT-diffusers/Looped-DiT-L-16
```

The default copy includes FLAN-T5. `--skip-text-encoder` leaves `text_encoder_name` in `model_index.json` so the pipeline downloads `google/flan-t5-large` on first use.

Code and configs are already filled from the training package. Refresh them, without deleting weight files, with `python tools/prepare_hf_repo.py`.

## Load from a local clone

```python
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

model_dir = Path("Looped-DiT-diffusers/Looped-DiT-B-16").resolve()
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
pipe.text_encoder.to(dtype=torch.float32)  # paper sampler keeps FLAN-T5 in fp32
pipe.to("cuda")

generator = torch.Generator(device="cuda").manual_seed(0)
image = pipe(
    "a red cube on top of a blue sphere",
    num_inference_steps=100,
    guidance_scale=6.0,
    num_loops=4,
    generator=generator,
).images[0]
image.save("sample.png")
```

## Load from the Hugging Face Hub

Upload this directory as a repo named like `UserID/Looped-DiT-diffusers`, then load one variant:

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "UserID/Looped-DiT-diffusers/Looped-DiT-B-16",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
pipe.text_encoder.to(dtype=torch.float32)
pipe.to("cuda")
image = pipe("a red cube on top of a blue sphere", num_inference_steps=100, guidance_scale=6.0).images[0]
```

For B/32 and L/16, swap the variant folder. Guidance stays 6.0. `num_loops` defaults to the trained depth (4). Other depths work when loop weights are shared.

## Recommended inference settings

| Variant | Resolution | Steps | CFG scale | `torch_dtype` |
| --- | --- | ---: | ---: | --- |
| `Looped-DiT-B-32` | 512×512 | 100 | 6.0 | `bfloat16` |
| `Looped-DiT-B-16` | 512×512 | 100 | 6.0 | `bfloat16` |
| `Looped-DiT-L-16` | 512×512 | 100 | 6.0 | `bfloat16` |
