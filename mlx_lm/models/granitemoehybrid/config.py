# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import List, Optional, Tuple

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    # Required fields (no defaults)
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_bias: bool
    embedding_multiplier: float
    attention_multiplier: float
    logits_scaling: float
    residual_multiplier: float
    layer_types: List[str]
    rms_norm_eps: float

    # Optional fields (with defaults)
    # MoE parameters (optional for dense mode)
    num_local_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    shared_intermediate_size: Optional[int] = None

    # Mamba parameters (optional for non-hybrid mode)
    mamba_n_heads: Optional[int] = None
    mamba_d_head: Optional[int] = None
    mamba_proj_bias: Optional[bool] = None
    mamba_d_state: Optional[int] = None
    mamba_d_conv: Optional[int] = None
    mamba_n_groups: Optional[int] = None
    mamba_conv_bias: Optional[bool] = None

    # Dense MLP parameters (for non-MoE mode)
    mlp_bias: bool = False

    # Other optional parameters
    position_embedding_type: str = "rope"
    tie_word_embeddings: bool = True
    time_step_limit: Tuple[float, float] = (0.001, 100.0)

    # rope_theta may arrive as a flat field (older configs) or nested under
    # rope_parameters (newer transformers releases, e.g. Granite 4.0 Hybrid
    # checkpoints). Accept both and normalize to a flat rope_theta.
    rope_theta: float = 10000.0
    rope_parameters: Optional[dict] = None

    def __post_init__(self):
        if self.rope_parameters is not None and "rope_theta" in self.rope_parameters:
            self.rope_theta = self.rope_parameters["rope_theta"]

    # Mode flags - inferred from num_local_experts
    @property
    def use_moe(self) -> bool:
        return bool(self.num_local_experts)
