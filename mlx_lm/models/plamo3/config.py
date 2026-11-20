# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "plamo3"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    scale_embedding: bool = False
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 2048
    window_size: int = 2048
    sliding_window: Optional[int] = None
    sliding_window_pattern: int = 8
    rope_theta: float = 1_000_000
    rope_local_theta: float = 10_000
    rope_scaling_factor: Optional[float] = None
    initial_context_length: Optional[int] = None
    intermediate_size: int = 13312
    vocab_size: int = 32000
    image_token_id: Optional[int] = None
    image_feature_size: Optional[int] = None
    image_proj_type: str = "linear"
    linear_type: str = "normal"

    def __post_init__(self):
        if self.sliding_window is not None:
            self.window_size = self.sliding_window
        if (
            self.rope_scaling_factor not in (None, 1)
            and self.initial_context_length is None
        ):
            raise ValueError("Scaled RoPE requires initial_context_length")

    @property
    def attention_window_size(self):
        # Older configs omit RoPE scaling and count only past tokens in the window.
        return self.window_size + (self.rope_scaling_factor is None)
