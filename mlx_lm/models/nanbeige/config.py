# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False
    num_loops: int = 1
    loop_loss_weights: Optional[List[float]] = None
    skip_loop_final_norm: bool = False
    # Features of the reference implementation without dedicated weights which
    # would otherwise load silently and produce incorrect results:
    enable_double_loop_split: bool = False
    loop_share_kv: bool = False
    enable_depth_attention: bool = False

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if (
            self.enable_double_loop_split
            or self.loop_share_kv
            or self.enable_depth_attention
        ):
            raise NotImplementedError(
                "enable_double_loop_split, loop_share_kv, and "
                "enable_depth_attention are not yet supported."
            )

    @property
    def effective_num_loops(self) -> int:
        # Checkpoints trained with per-loop losses store one weight per extra loop
        if self.loop_loss_weights:
            return len(self.loop_loss_weights) + 1
        return self.num_loops
