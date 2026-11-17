# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
    "TextArgs",
]


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "kimi_linear"
    vocab_size: int = 163840
    hidden_size: int = 7168
    num_hidden_layers: int = 93
    num_attention_heads: int = 96
    num_key_value_heads: int = 96
    intermediate_size: int = 33792
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 1048576
    linear_attn_config: Optional[Dict[str, Any]] = None
    hidden_act: str = "situ"
    activation_situ_beta: Optional[float] = None
    activation_situ_linear_beta: Optional[float] = None
    attn_res_block_size: Optional[int] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    mla_use_nope: bool = True
    mla_use_output_gate: bool = False
    num_experts: Optional[int] = None
    num_experts_per_token: int = 16
    num_shared_experts: int = 0
    moe_intermediate_size: Optional[int] = None
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1
    routed_expert_hidden_size: Optional[int] = None
    latent_moe_use_norm: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.hidden_act != "situ":
            raise ValueError(f"Unsupported activation '{self.hidden_act}'")
        if self.moe_router_activation_func != "sigmoid":
            raise ValueError(
                f"Unsupported MoE router activation '{self.moe_router_activation_func}'"
            )


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: dict
    model_type: str = "kimi_k3"

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config)
