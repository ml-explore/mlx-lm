# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional

from mlx_lm.models.deepseek_v3.config import ModelArgs as DeepseekV3ModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(DeepseekV3ModelArgs):
    """Sarvam MLA configuration mapped to the DeepSeek-V3 implementation."""

    model_type: str = "sarvam_mla"
    vocab_size: int = 262144
    hidden_size: int = 4096
    intermediate_size: int = 16384
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 32
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: Optional[int] = 1
    n_routed_experts: Optional[int] = 128
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: Optional[int] = None
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 163840
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_parameters: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.rope_parameters is not None:
            self.rope_scaling = self.rope_parameters
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ModelArgs":
        params = dict(params)

        aliases = {
            "num_experts": "n_routed_experts",
            "num_shared_experts": "n_shared_experts",
            "score_function": "scoring_func",
        }
        for source, target in aliases.items():
            if target not in params and source in params:
                params[target] = params[source]

        # The released checkpoints omit these fields. The latest vLLM executor
        # supplies one group, which is equivalent to ungrouped top-k routing.
        params.setdefault("n_group", 1)
        params.setdefault("topk_group", 1)
        params.setdefault("q_lora_rank", None)

        rope = params.get("rope_scaling") or params.get("rope_parameters")
        if rope is not None:
            rope = dict(rope)
            params["rope_parameters"] = rope
            params["rope_scaling"] = rope
            params["rope_theta"] = rope.get(
                "rope_theta", params.get("rope_theta", 10000.0)
            )

        return super().from_dict(params)
