# Copyright © 2026 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "talkie"
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 40
    head_dim: int = 128
    intermediate_size: int = 13696
    vocab_size: int = 65540
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 2048
    # Talkie applies F.rms_norm without an explicit eps: the reduction is done
    # in fp32 with the fp32 default eps.
    rms_norm_eps: float = 1.1920928955078125e-07
