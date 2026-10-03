"""Frozen FLAN-T5 encoder used by training.

Inference encodes prompts inside `LoopedDiTPipeline` instead.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, T5EncoderModel


class TextEncoder:
    """Frozen FLAN-T5 encoder; prompts are padded to prompt_length tokens."""

    def __init__(self, name: str, prompt_length: int, device: torch.device):
        self.prompt_length = prompt_length
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name, model_max_length=prompt_length)
        self.model = T5EncoderModel.from_pretrained(name).to(device).eval().requires_grad_(False)

    def tokenize(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            prompts, max_length=self.prompt_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return tokens["input_ids"].to(self.device), tokens["attention_mask"].to(self.device)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
