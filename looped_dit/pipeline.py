# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any, Callable

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, KarrasDiffusionSchedulers
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import deprecate, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Training clamps the flow-matching denominator so the loss stays finite at t -> 1.
VELOCITY_DENOM_MIN = 0.05

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pathlib import Path
        >>> import torch
        >>> from diffusers import DiffusionPipeline

        >>> model_dir = Path("checkpoints/looped-dit-b16").resolve()
        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     str(model_dir),
        ...     local_files_only=True,
        ...     custom_pipeline=str(model_dir / "pipeline.py"),
        ...     trust_remote_code=True,
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> pipe.text_encoder.to(dtype=torch.float32)
        >>> pipe = pipe.to("cuda")

        >>> image = pipe(
        ...     "a red cube on top of a blue sphere",
        ...     num_inference_steps=100,
        ...     guidance_scale=6.0,
        ...     num_loops=4,
        ...     generator=torch.Generator(device="cuda").manual_seed(0),
        ... ).images[0]
        >>> image.save("sample.png")

        >>> # Hugging Face Hub style model id: UserID/RepoID
        >>> # RepoID is usually like "modelname-diffusers"
        >>> # Example: "your-user/Looped-DiT-diffusers"
        ```
"""


def paper_euler_sigmas(num_inference_steps: int) -> list[float]:
    r"""
    Sigma grid of the training Euler sampler.

    Training integrates flow time `t` from 0 (noise) to 1 (data) with
    `torch.linspace(0, 1, steps + 1)`. Flow-match schedulers step in sigma
    `1 - t` and append the terminal 0 themselves, so the returned list omits that 0.

    Args:
        num_inference_steps (`int`):
            Number of Euler steps. Must be positive.

    Returns:
        `list[float]`: `num_inference_steps` sigmas starting at 1 and ending at `1 / steps`.
    """
    if num_inference_steps <= 0:
        raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}.")
    flow_time = torch.linspace(0.0, 1.0, num_inference_steps + 1)
    return (1.0 - flow_time)[:-1].tolist()


class LoopedDiTPipeline(DiffusionPipeline):
    r"""
    Text-to-image pipeline for Looped-DiT.

    Looped-DiT denoises directly in RGB pixel space (no VAE). The transformer predicts the clean
    image `x0`. This pipeline converts that prediction to a flow-matching velocity and integrates it
    with a diffusers scheduler. The default scheduler is [`FlowMatchEulerDiscreteScheduler`] on the
    same uniform grid as the paper (100 steps, shift 1). Any [`KarrasDiffusionSchedulers`] instance
    can be assigned to `pipe.scheduler` without other code changes.

    Classifier-free guidance uses the training null condition: an all-zero text mask, which the
    denoiser replaces with its mask token. There is no separate negative-prompt encoder.

    The pipeline inherits from [`DiffusionPipeline`]. Check the superclass documentation for the
    generic methods (download, save, device placement, CPU offload).

    Args:
        transformer ([`ModelMixin`]):
            Looped-DiT denoiser (`LoopedDiTTransformer2DModel`) that predicts `x0` in pixel space.
        scheduler ([`FlowMatchEulerDiscreteScheduler`] or [`KarrasDiffusionSchedulers`]):
            Scheduler used to step the flow. The paper setting is
            `FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)`.
        tokenizer ([`~transformers.AutoTokenizer`], *optional*):
            Tokenizer for the frozen text encoder. Loaded from `text_encoder_name` when missing.
        text_encoder ([`~transformers.T5EncoderModel`], *optional*):
            Frozen FLAN-T5 encoder. Loaded from `text_encoder_name` when missing.
        text_encoder_name (`str`, defaults to `"google/flan-t5-large"`):
            Hub id or local path used when `tokenizer` / `text_encoder` are not passed.
        prompt_length (`int`, defaults to 256):
            Token length prompts are padded or truncated to. Must match training.
        noise_scale (`float`, defaults to 2.0):
            Standard deviation of the initial Gaussian, matching the training noise scale.
        default_num_inference_steps (`int`, defaults to 100):
            Step count used when `__call__` does not pass `num_inference_steps`.
    """

    model_cpu_offload_seq = "text_encoder->transformer"
    _optional_components = ["tokenizer", "text_encoder"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "prompt_attention_mask"]

    def __init__(
        self,
        transformer: ModelMixin,
        scheduler: KarrasDiffusionSchedulers | SchedulerMixin,
        tokenizer: Any | None = None,
        text_encoder: T5EncoderModel | None = None,
        text_encoder_name: str = "google/flan-t5-large",
        prompt_length: int = 256,
        noise_scale: float = 2.0,
        default_num_inference_steps: int = 100,
    ):
        super().__init__()
        if scheduler is None:
            scheduler = self._default_scheduler()
        if noise_scale <= 0:
            raise ValueError(f"`noise_scale` must be positive, got {noise_scale}.")
        if prompt_length < 1:
            raise ValueError(f"`prompt_length` must be positive, got {prompt_length}.")
        if default_num_inference_steps < 1:
            raise ValueError(f"`default_num_inference_steps` must be positive, got {default_num_inference_steps}.")

        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        self.register_to_config(
            text_encoder_name=text_encoder_name,
            prompt_length=int(prompt_length),
            noise_scale=float(noise_scale),
            default_num_inference_steps=int(default_num_inference_steps),
        )

    @staticmethod
    def _default_scheduler() -> FlowMatchEulerDiscreteScheduler:
        r"""
        Build the paper's Euler scheduler.

        Returns:
            [`FlowMatchEulerDiscreteScheduler`]: 1000 training timesteps, shift 1, deterministic.
        """
        kwargs: dict[str, Any] = {"num_train_timesteps": 1000, "shift": 1.0}
        if "stochastic_sampling" in inspect.signature(FlowMatchEulerDiscreteScheduler.__init__).parameters:
            kwargs["stochastic_sampling"] = False
        return FlowMatchEulerDiscreteScheduler(**kwargs)

    def _encode_prompt(
        self,
        prompt: str | list[str] | None,
        device: torch.device,
        num_images_per_prompt: int,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Deprecated alias of [`~LoopedDiTPipeline.encode_prompt`].

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Prompt or prompts to tokenize and encode.
            device (`torch.device`):
                Device of the returned tensors.
            num_images_per_prompt (`int`):
                How many times to repeat each prompt embedding.
            prompt_embeds (`torch.Tensor`, *optional*):
                Already encoded prompts of shape `(batch, sequence, text_dim)`.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Mask of shape `(batch, sequence)` with 1 on real tokens. Required with `prompt_embeds`
                only when padding should be replaced by the mask token; otherwise a mask of ones is used.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: Prompt embeddings and the attention mask.
        """
        deprecation_message = (
            "`_encode_prompt()` is deprecated and will be removed in a future version. Use `encode_prompt()` instead."
        )
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
        return self.encode_prompt(prompt, device, num_images_per_prompt, prompt_embeds, prompt_attention_mask)

    def encode_prompt(
        self,
        prompt: str | list[str] | None,
        device: torch.device,
        num_images_per_prompt: int,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Encode prompts with the frozen FLAN-T5 encoder.

        Prompts are padded or truncated to `config.prompt_length`. The unconditional branch of
        classifier-free guidance is not encoded here: the denoiser builds it by zeroing this mask.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Prompt or prompts to tokenize. Ignored when `prompt_embeds` is passed.
            device (`torch.device`):
                Device of the returned tensors.
            num_images_per_prompt (`int`):
                Number of times to repeat each encoded prompt along the batch dimension.
            prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed embeddings of shape `(batch, sequence, text_dim)`. When set, `prompt` is ignored.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Mask of shape `(batch, sequence)`, 1 for tokens that should condition the model. When
                `prompt_embeds` is set and this is omitted, every position is treated as a real token.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`:
                Embeddings `(batch * num_images_per_prompt, sequence, text_dim)` and a mask of the same batch.
        """
        if num_images_per_prompt < 1:
            raise ValueError(f"`num_images_per_prompt` must be >= 1, got {num_images_per_prompt}.")

        if prompt_embeds is None:
            if isinstance(prompt, str):
                prompt = [prompt]
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.text_encoder_name, model_max_length=int(self.config.prompt_length)
                )
            if self.text_encoder is None:
                self.text_encoder = T5EncoderModel.from_pretrained(self.config.text_encoder_name)
                self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
            encoder_device = next(self.text_encoder.parameters()).device
            if encoder_device != device:
                self.text_encoder.to(device)
            tokens = self.tokenizer(
                prompt,
                max_length=int(self.config.prompt_length),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(device)
            prompt_attention_mask = tokens.attention_mask.to(device)
            prompt_embeds = self.text_encoder(input_ids=input_ids, attention_mask=prompt_attention_mask).last_hidden_state
        else:
            prompt_embeds = prompt_embeds.to(device)
            if prompt_attention_mask is None:
                prompt_attention_mask = torch.ones(
                    prompt_embeds.shape[:2], device=device, dtype=torch.long
                )
            else:
                prompt_attention_mask = prompt_attention_mask.to(device)
            if prompt_embeds.shape[0] != prompt_attention_mask.shape[0]:
                raise ValueError(
                    "`prompt_embeds` and `prompt_attention_mask` must have the same batch size, got "
                    f"{prompt_embeds.shape[0]} and {prompt_attention_mask.shape[0]}."
                )

        if num_images_per_prompt != 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            prompt_attention_mask = prompt_attention_mask.repeat_interleave(num_images_per_prompt, dim=0)
        return prompt_embeds, prompt_attention_mask

    def prepare_extra_step_kwargs(
        self, generator: torch.Generator | list[torch.Generator] | None, eta: float
    ) -> dict[str, Any]:
        r"""
        Extra arguments forwarded to `scheduler.step`, depending on what that method accepts.

        Args:
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Generator passed through when the scheduler step samples noise.
            eta (`float`):
                DDIM eta in `[0, 1]`. Ignored by schedulers whose `step` has no `eta` argument.

        Returns:
            `dict`: Keyword arguments for `scheduler.step`.
        """
        extra_step_kwargs: dict[str, Any] = {}
        step_params = set(inspect.signature(self.scheduler.step).parameters.keys())
        if "eta" in step_params:
            extra_step_kwargs["eta"] = eta
        if "generator" in step_params:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        height: int,
        width: int,
        callback_steps: int | None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 6.0,
        num_loops: int | None = None,
        output_type: str = "pil",
    ) -> None:
        r"""
        Validate generation arguments and raise `ValueError` or `TypeError` on misuse.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Prompt text. Mutually exclusive with `prompt_embeds`.
            height (`int`):
                Output height in pixels. Must equal the transformer's trained `image_size`.
            width (`int`):
                Output width in pixels. Must equal the transformer's trained `image_size`.
            callback_steps (`int`, *optional*):
                Deprecated callback period. When set, it must be a positive integer.
            prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed text embeddings. Required when `prompt` is omitted.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Mask paired with `prompt_embeds`.
            callback_on_step_end_tensor_inputs (`list[str]`, *optional*):
                Tensor names the step callback may read. Each name must be listed on
                `_callback_tensor_inputs`.
            num_inference_steps (`int`):
                Denoising steps. Must be positive.
            guidance_scale (`float`):
                Classifier-free guidance scale. Must be finite. `1` disables guidance.
            num_loops (`int`, *optional*):
                Loop depth. `None` uses the depth stored on the transformer. Otherwise `>= 1`, and
                untied models cannot exceed the trained depth.
            output_type (`str`):
                One of `"pil"`, `"np"`, `"pt"`, or `"latent"`.
        """
        image_size = int(self.transformer.config.image_size)
        patch_size = int(self.transformer.config.patch_size)
        if height != image_size or width != image_size:
            raise ValueError(
                f"Looped-DiT uses a fixed positional grid of {image_size}x{image_size} "
                f"(patch size {patch_size}). Got height={height}, width={width}."
            )
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(f"height and width must be divisible by patch_size={patch_size}, got {(height, width)}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            key in self._callback_tensor_inputs for key in callback_on_step_end_tensor_inputs
        ):
            unexpected = [key for key in callback_on_step_end_tensor_inputs if key not in self._callback_tensor_inputs]
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {unexpected}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`. Pass only one of them.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and not isinstance(prompt, str) and not (
            isinstance(prompt, list) and all(isinstance(item, str) for item in prompt)
        ):
            raise TypeError(f"`prompt` has to be a string or a list of strings, got {type(prompt)}.")
        if prompt_embeds is not None and prompt_embeds.ndim != 3:
            raise ValueError(f"`prompt_embeds` must have shape (batch, sequence, dim), got {tuple(prompt_embeds.shape)}.")
        if prompt_attention_mask is not None and prompt_embeds is None:
            raise ValueError("`prompt_attention_mask` was passed without `prompt_embeds`.")

        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}.")
        if not torch.isfinite(torch.tensor(guidance_scale)):
            raise ValueError(f"`guidance_scale` must be finite, got {guidance_scale}.")
        if num_loops is not None:
            if int(num_loops) < 1:
                raise ValueError(f"`num_loops` must be >= 1, got {num_loops}.")
            trained = int(self.transformer.config.num_loops)
            if not bool(self.transformer.config.share_loop_weights) and int(num_loops) > trained:
                raise ValueError(
                    f"This checkpoint does not share loop weights, so `num_loops` cannot exceed the trained "
                    f"depth {trained}. Got {num_loops}."
                )
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError(f"Unsupported `output_type` {output_type!r}. Choose from 'pil', 'np', 'pt', 'latent'.")

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Sample the initial pixel-space noise, or validate a tensor the caller already sampled.

        Looped-DiT has no VAE, so "latents" here are RGB images. Fresh noise is scaled by
        `config.noise_scale` (2.0 in the paper). A provided `latents` tensor is not rescaled.

        Args:
            batch_size (`int`):
                Number of images, including `num_images_per_prompt`.
            num_channels (`int`):
                Channel count. 3 for RGB.
            height (`int`):
                Image height in pixels.
            width (`int`):
                Image width in pixels.
            dtype (`torch.dtype`):
                Dtype of freshly sampled noise. The integration itself is accumulated in float32.
            device (`torch.device`):
                Device of the returned tensor.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Per-call RNG. A list must have length `batch_size`.
            latents (`torch.Tensor`, *optional*):
                Starting noise of shape `(batch_size, num_channels, height, width)`.

        Returns:
            `torch.Tensor`: Starting noise of shape `(batch_size, num_channels, height, width)`.
        """
        shape = (batch_size, num_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You passed a list of {len(generator)} generators for a batch of {batch_size}. "
                "The two lengths must match."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = latents * float(self.config.noise_scale)
        else:
            latents = latents.to(device=device)
            if tuple(latents.shape) != shape:
                raise ValueError(f"`latents` shape {tuple(latents.shape)} does not match the expected {shape}.")
        return latents

    @property
    def guidance_scale(self) -> float:
        r"""
        Classifier-free guidance scale of the call that is currently running.

        Returns:
            `float`: The scale set by the active `__call__`.
        """
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        r"""
        Whether the active call runs a conditional and an unconditional forward.

        Returns:
            `bool`: True when `guidance_scale != 1`.
        """
        return self._guidance_scale != 1.0

    @property
    def num_timesteps(self) -> int:
        r"""
        Number of scheduler timesteps in the active call.

        Returns:
            `int`: Length of the timestep schedule.
        """
        return self._num_timesteps

    @property
    def interrupt(self) -> bool:
        r"""
        Whether the active denoising loop should skip remaining steps.

        Returns:
            `bool`: True after the caller sets `pipeline._interrupt = True`.
        """
        return self._interrupt

    def _images_from_latents(self, latents: torch.Tensor, output_type: str) -> torch.Tensor | list[Image.Image] | Any:
        r"""
        Convert pixel-space samples in `[-1, 1]` to the requested output type.

        Quantization matches the original sampler: `uint8(clamp(x, -1, 1) * 127.5 + 128)`.

        Args:
            latents (`torch.Tensor`):
                Samples of shape `(batch, channels, height, width)` in model range `[-1, 1]`.
            output_type (`str`):
                `"latent"` returns `latents` unchanged. `"pt"` is float RGB in `[0, 1]`. `"np"` is
                `uint8` HWC arrays. `"pil"` is a list of `PIL.Image.Image`.

        Returns:
            Images in the requested type.
        """
        if output_type == "latent":
            return latents
        images = (latents.float().clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
        if output_type == "pt":
            return images.float() / 255.0
        arrays = images.permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "np":
            return arrays
        return [Image.fromarray(image) for image in arrays]

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        guidance_scale: float = 6.0,
        num_images_per_prompt: int = 1,
        num_loops: int | None = None,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], dict] | PipelineCallback | MultiPipelineCallbacks | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        **kwargs,
    ) -> ImagePipelineOutput | tuple:
        r"""
        Generate images from text prompts.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Prompt or prompts to guide image generation. Required unless `prompt_embeds` is passed.
            height (`int`, *optional*):
                Image height in pixels. Defaults to the transformer's trained resolution (512).
                Other resolutions are rejected: the positional embedding is a fixed grid.
            width (`int`, *optional*):
                Image width in pixels. Defaults to the trained resolution and must match `height`.
            num_inference_steps (`int`, *optional*):
                Denoising steps. Defaults to `config.default_num_inference_steps` (100).
            timesteps (`list[int]`, *optional*):
                Custom scheduler timesteps, descending. Mutually exclusive with `sigmas`. Ignored by the
                paper Euler grid, which is selected only when both `timesteps` and `sigmas` are omitted
                and the scheduler is [`FlowMatchEulerDiscreteScheduler`].
            sigmas (`list[float]`, *optional*):
                Custom sigmas passed to `scheduler.set_timesteps`. Mutually exclusive with `timesteps`.
            guidance_scale (`float`, defaults to 6.0):
                Classifier-free guidance scale from the paper. Guidance is on when this is not `1`.
                The unconditional branch is an empty text mask, not a negative prompt.
            num_images_per_prompt (`int`, defaults to 1):
                How many images to sample for each prompt.
            num_loops (`int`, *optional*):
                How many times to run the shared middle blocks. `None` uses the trained depth. Other
                depths work without retraining when loop weights are shared.
            eta (`float`, defaults to 0.0):
                DDIM eta. Ignored by the flow-match Euler scheduler.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                RNG for the initial noise. `None` uses PyTorch's global generator, which is what
                `torch.manual_seed` seeds.
            latents (`torch.Tensor`, *optional*):
                Initial noise `(batch, 3, height, width)`. Not multiplied by `noise_scale`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed FLAN-T5 states `(batch, sequence, text_dim)` in place of `prompt`.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Mask `(batch, sequence)` paired with `prompt_embeds`. 1 marks real tokens.
            output_type (`str`, defaults to `"pil"`):
                `"pil"`, `"np"`, `"pt"` (float RGB in `[0, 1]`), or `"latent"` (pixels in model range).
            return_dict (`bool`, defaults to `True`):
                Return [`ImagePipelineOutput`] when `True`, otherwise a one-tuple of images.
            callback_on_step_end (`Callable` or `PipelineCallback`, *optional*):
                Called as `callback_on_step_end(pipeline, step, timestep, callback_kwargs)` after each
                scheduler step. Return a dict to replace tensors listed in
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["latents"]`):
                Tensor names passed to the step callback. Must be a subset of `_callback_tensor_inputs`.

        Examples:

        Returns:
            [`ImagePipelineOutput`] or `tuple`:
                When `return_dict` is `True`, [`ImagePipelineOutput`] with the images. Otherwise a tuple
                whose first element is the images.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        if kwargs:
            raise TypeError(f"Unexpected arguments: {sorted(kwargs)}.")

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        image_size = int(self.transformer.config.image_size)
        height = image_size if height is None else int(height)
        width = image_size if width is None else int(width)
        if num_inference_steps is None:
            num_inference_steps = int(self.config.default_num_inference_steps)

        # 1. Check inputs.
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            prompt_embeds,
            prompt_attention_mask,
            callback_on_step_end_tensor_inputs,
            num_inference_steps,
            guidance_scale,
            num_loops,
            output_type,
        )

        self._guidance_scale = float(guidance_scale)
        self._interrupt = False

        # 2. Define call parameters.
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        # 3. Encode input prompt.
        prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=self.transformer.dtype)
        prompt_attention_mask = prompt_attention_mask.to(device=device)

        # 4. Prepare timesteps.
        # The paper's Euler grid is the training linspace. Other schedulers keep their own spacing.
        # A caller-supplied `timesteps` or `sigmas` always wins.
        if timesteps is None and sigmas is None and isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            sigmas = paper_euler_sigmas(num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, sigmas)
        if getattr(self.scheduler.config, "stochastic_sampling", False):
            raise ValueError(
                "Looped-DiT's training sampler is deterministic. Set `stochastic_sampling=False` on "
                "FlowMatchEulerDiscreteScheduler, or assign a different scheduler."
            )

        # 5. Prepare latent variables (pixel-space noise; there is no VAE).
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            int(self.transformer.config.in_channels),
            height,
            width,
            self.transformer.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # 7. Denoising loop.
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                model_latents = latents
                if hasattr(self.scheduler, "scale_model_input"):
                    model_latents = self.scheduler.scale_model_input(model_latents, t)
                text = prompt_embeds
                mask = prompt_attention_mask
                if self.do_classifier_free_guidance:
                    model_latents = torch.cat([model_latents, model_latents], dim=0)
                    text = torch.cat([text, text], dim=0)
                    mask = torch.cat([mask, torch.zeros_like(mask)], dim=0)

                # fp32 latents with bf16 weights match the old sampler, which autocasts the forward.
                amp_dtype = self.transformer.dtype
                use_amp = model_latents.is_cuda and amp_dtype in (torch.float16, torch.bfloat16)
                if not use_amp:
                    model_latents = model_latents.to(dtype=amp_dtype)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    x0 = self.transformer(model_latents, text, mask, num_loops=num_loops)
                x0 = x0.float()
                if self.do_classifier_free_guidance:
                    x0_cond, x0_uncond = x0.chunk(2)
                    x0 = x0_uncond + self.guidance_scale * (x0_cond - x0_uncond)

                # sigma = 1 - t_flow. Passing -velocity makes `x + (sigma_next - sigma) * model_output`
                # equal the training update `x + velocity * (t_next - t)`.
                flow_time = (1.0 - t.to(device=latents.device, dtype=torch.float32) / num_train_timesteps)
                velocity = (x0 - latents.float()) / (1.0 - flow_time).clamp_min(VELOCITY_DENOM_MIN)
                latents = self.scheduler.step(-velocity, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {key: locals()[key] for key in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    prompt_attention_mask = callback_outputs.pop("prompt_attention_mask", prompt_attention_mask)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        images = self._images_from_latents(latents, output_type)

        # Offload all models.
        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
