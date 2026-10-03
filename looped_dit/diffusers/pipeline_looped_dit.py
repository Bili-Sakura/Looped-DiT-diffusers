# Copyright 2026 Looped-DiT authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import inspect
import os
from typing import Any, List, Optional, Tuple, Union

os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from transformers import logging as transformers_logging

from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor

transformers_logging.set_verbosity_error()

DEFAULT_NUM_INFERENCE_STEPS = 100

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pathlib import Path
        >>> import torch
        >>> from diffusers import DiffusionPipeline

        >>> model_dir = Path("./Looped-DiT-B16").resolve()
        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     str(model_dir),
        ...     local_files_only=True,
        ...     custom_pipeline=str(model_dir / "pipeline.py"),
        ...     trust_remote_code=True,
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> pipe.to("cuda")

        >>> generator = torch.Generator(device="cuda").manual_seed(0)
        >>> image = pipe(
        ...     "a red cube on top of a blue sphere",
        ...     num_inference_steps=100,
        ...     guidance_scale=6.0,
        ...     num_loops=4,
        ...     generator=generator,
        ... ).images[0]
        >>> image.save("sample.png")
        ```
"""


class LoopedDiTTextToImagePipeline(DiffusionPipeline):
    r"""
    Text-to-image pipeline for Looped-DiT pixel-space flow matching.

    Parameters:
        transformer ([`LoopedMMDiTModel`]):
            Looped-DiT MMDiT transformer that predicts clean images in pixel space.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching Euler scheduler. Other [`KarrasDiffusionSchedulers`] can be swapped at inference time.
        tokenizer ([`AutoTokenizer`], *optional*):
            Tokenizer for the text encoder.
        text_encoder ([`T5EncoderModel`], *optional*):
            Text encoder used to embed prompts.
    """

    model_cpu_offload_seq = "text_encoder->transformer"
    _optional_components = ["tokenizer", "text_encoder"]

    def __init__(
        self,
        transformer,
        scheduler: KarrasDiffusionSchedulers,
        tokenizer=None,
        text_encoder=None,
        text_encoder_name: str = "google/flan-t5-large",
        default_num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        default_num_loops: Optional[int] = None,
        noise_scale: float = 2.0,
    ):
        super().__init__()
        if scheduler is None:
            scheduler = self._default_inference_scheduler()
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        trained_loops = getattr(transformer.config, "num_loops", None)
        self.register_to_config(
            text_encoder_name=text_encoder_name,
            default_num_inference_steps=int(default_num_inference_steps),
            default_num_loops=int(default_num_loops if default_num_loops is not None else trained_loops or 4),
            noise_scale=float(noise_scale),
        )

    @staticmethod
    def _default_inference_scheduler() -> FlowMatchEulerDiscreteScheduler:
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            stochastic_sampling=False,
        )

    @staticmethod
    def prepare_extra_step_kwargs(
        scheduler: KarrasDiffusionSchedulers,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        step_params = set(inspect.signature(scheduler.step).parameters.keys())
        if "generator" in step_params:
            kwargs["generator"] = generator
        return kwargs

    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        guidance_scale: float,
        num_inference_steps: int,
        output_type: str,
        num_loops: int,
    ) -> None:
        if not isinstance(prompt, str) and not (isinstance(prompt, list) and all(isinstance(p, str) for p in prompt)):
            raise TypeError(f"`prompt` must be a string or list of strings, got {type(prompt)}.")
        if guidance_scale < 0:
            raise ValueError(f"`guidance_scale` must be non-negative, got {guidance_scale}.")
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}.")
        if num_loops < 1:
            raise ValueError(f"`num_loops` must be >= 1, got {num_loops}.")
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError(f"Unsupported `output_type`: {output_type}")

    def prepare_latents(
        self,
        batch_size: int,
        image_size: int,
        in_channels: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shape = (batch_size, in_channels, image_size, image_size)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = latents * self.config.noise_scale
        else:
            latents = latents.to(device=device, dtype=dtype)
            if tuple(latents.shape) != shape:
                raise ValueError(f"Invalid `latents` shape: {tuple(latents.shape)}. Expected {shape}.")
        return latents

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_encoder_name)
        if self.text_encoder is None:
            self.text_encoder = T5EncoderModel.from_pretrained(self.config.text_encoder_name)
        if next(self.text_encoder.parameters()).device != device:
            self.text_encoder.to(device)
        prompt_length = int(self.transformer.config.prompt_length)
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=prompt_length,
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        text = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return text, attention_mask

    def _cfg_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text: torch.Tensor,
        mask: torch.Tensor,
        cfg_scale: float,
        num_loops: int,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        doubled_x = torch.cat([x, x], dim=0)
        doubled_t = torch.cat([t, t], dim=0)
        doubled_text = torch.cat([text, text], dim=0)
        null_mask = torch.zeros_like(mask)
        doubled_mask = torch.cat([mask, null_mask], dim=0)
        velocity = self.transformer.pred_velocity(doubled_x, doubled_t, doubled_text, doubled_mask, num_loops=num_loops)
        cond, uncond = velocity[:batch_size], velocity[batch_size:]
        cfg_interval = tuple(self.transformer.config.cfg_interval)
        use_cfg = ((t >= cfg_interval[0]) & (t <= cfg_interval[1])).to(velocity.dtype)
        scale = torch.where(
            use_cfg[:, None, None, None] > 0,
            torch.tensor(cfg_scale, device=x.device, dtype=velocity.dtype),
            torch.tensor(1.0, device=x.device, dtype=velocity.dtype),
        )
        return uncond + (cond - uncond) * scale

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        guidance_scale: float = 6.0,
        num_inference_steps: Optional[int] = None,
        num_loops: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        progress: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Generate images from text prompts with Looped-DiT.

        Args:
            prompt (`str` or `list[str]`):
                Text prompt or batch of prompts.
            num_images_per_prompt (`int`, defaults to `1`):
                Number of images to generate per prompt.
            guidance_scale (`float`, defaults to `6.0`):
                Classifier-free guidance scale. CFG is active when `guidance_scale != 1.0`.
            num_inference_steps (`int`, *optional*):
                Number of denoising steps. Defaults to the pipeline config value.
            num_loops (`int`, *optional*):
                Loop depth within each denoising step. Defaults to the trained depth in the transformer config.
            generator (`torch.Generator`, *optional*):
                RNG for reproducibility.
            latents (`torch.Tensor`, *optional*):
                Pre-generated pixel latents with shape `(batch, channels, height, width)`.
            output_type (`str`, defaults to `"pil"`):
                `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Return [`ImagePipelineOutput`] if True.
            progress (`bool`, defaults to `True`):
                Whether to show a progress bar during denoising.
        """
        num_inference_steps = int(num_inference_steps or self.config.default_num_inference_steps)
        num_loops = int(num_loops if num_loops is not None else self.config.default_num_loops)
        self.check_inputs(prompt, guidance_scale, num_inference_steps, output_type, num_loops)

        device = self._execution_device
        self.transformer = self.transformer.to(device)

        if isinstance(prompt, str):
            prompt_batch = [prompt] * num_images_per_prompt
        else:
            prompt_batch = []
            for entry in prompt:
                prompt_batch.extend([entry] * num_images_per_prompt)

        batch_size = len(prompt_batch)
        model_dtype = next(self.transformer.parameters()).dtype

        text, attn = self._encode_prompt(prompt_batch, device)
        text = text.to(dtype=model_dtype)

        if getattr(self.scheduler.config, "stochastic_sampling", False):
            raise ValueError(
                "Looped-DiT expects deterministic FlowMatchEulerDiscreteScheduler stepping "
                "(scheduler.config.stochastic_sampling=False)."
            )

        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        latents = self.prepare_latents(
            batch_size=batch_size,
            image_size=int(self.transformer.config.image_size),
            in_channels=int(self.transformer.config.in_channels),
            device=device,
            dtype=model_dtype,
            generator=generator,
            latents=latents,
        )

        timesteps = self.scheduler.timesteps
        if progress:
            timesteps = self.progress_bar(timesteps)

        using_cfg = guidance_scale != 1.0
        for timestep in timesteps:
            flow_time = 1.0 - float(timestep) / num_train_timesteps
            t = torch.full((batch_size,), flow_time, device=device, dtype=model_dtype)
            if using_cfg:
                velocity = self._cfg_velocity(latents, t, text, attn, guidance_scale, num_loops)
            else:
                velocity = self.transformer.pred_velocity(latents, t, text, attn, num_loops=num_loops)

            latents = self.scheduler.step(-velocity, timestep, latents, **extra_step_kwargs).prev_sample

        if output_type == "latent":
            images = latents
        else:
            images = (latents.clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
            if output_type == "pt":
                images = images.float() / 255.0
            else:
                images = images.permute(0, 2, 3, 1).cpu().numpy()
                if output_type == "pil":
                    images = [Image.fromarray(image) for image in images]

        self.maybe_free_model_hooks()
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
