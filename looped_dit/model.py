"""Historical import path for the Looped-DiT denoiser.

The module that is saved into a diffusers checkpoint is `transformer_looped_dit`.
Training and the unit tests keep importing `LoopedMMDiT` from here.
"""

from .transformer_looped_dit import LoopedDiTTransformer2DModel as LoopedMMDiT
from .transformer_looped_dit import exclusive_self_attention

__all__ = ["LoopedMMDiT", "exclusive_self_attention"]
