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

from typing import Optional

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

try:
    from looped_mmdit_core import LoopedMMDiT
except ImportError:
    from looped_dit.model import LoopedMMDiT


class LoopedMMDiTModel(ModelMixin, ConfigMixin):
    r"""Looped-DiT pixel-space MMDiT denoiser for flow matching."""

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 32,
        in_channels: int = 3,
        hidden_size: int = 768,
        num_heads: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 2.6666666666666665,
        pca_channels: int = 128,
        text_dim: int = 1024,
        text_preamble_depth: int = 2,
        loop_split: tuple[int, int, int] | list[int] = (6, 5, 6),
        num_loops: int = 4,
        share_loop_weights: bool = True,
        use_xsa: bool = True,
        use_attn_gate: bool = False,
        prompt_length: int = 256,
        noise_scale: float = 2.0,
        cfg_interval: tuple[float, float] | list[float] = (0.0, 1.0),
        text_encoder_name: str = "google/flan-t5-large",
    ):
        super().__init__()
        if isinstance(loop_split, list):
            loop_split = tuple(loop_split)
        if isinstance(cfg_interval, list):
            cfg_interval = tuple(cfg_interval)
        self.transformer = LoopedMMDiT(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            pca_channels=pca_channels,
            text_dim=text_dim,
            text_preamble_depth=text_preamble_depth,
            loop_split=loop_split,
            num_loops=num_loops,
            share_loop_weights=share_loop_weights,
            use_xsa=use_xsa,
            use_attn_gate=use_attn_gate,
        )

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        num_loops: Optional[int] = None,
    ) -> torch.Tensor:
        return self.transformer(x, text, text_mask, num_loops=num_loops)

    def pred_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text: torch.Tensor,
        mask: torch.Tensor,
        num_loops: Optional[int] = None,
    ) -> torch.Tensor:
        x0 = self.transformer(x, text, mask, num_loops=num_loops)
        return (x0 - x) / torch.clamp(1.0 - t[:, None, None, None], min=0.05)
