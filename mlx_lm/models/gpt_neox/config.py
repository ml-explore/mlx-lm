# Copyright © 2023 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


# Based on the transformers implementation at:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py
@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    max_position_embeddings: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    layer_norm_eps: float
    vocab_size: int
    rotary_emb_base: int
    rotary_pct: float
    use_parallel_residual: bool = True
    num_key_value_heads: int = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
