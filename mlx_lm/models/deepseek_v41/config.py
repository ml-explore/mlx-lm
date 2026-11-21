# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional, Union

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v41"
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    head_dim: int = 512
    q_lora_rank: int = 1280
    qk_rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    moe_intermediate_size: int = 2304
    n_routed_experts: int = 384
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    rms_norm_eps: float = 1e-20

    sliding_window: int = 128
    compress_ratios: tuple[int, ...] = ()
    kv_source_layer_ids: tuple[int, ...] = ()
    index_source_layer_ids: tuple[int, ...] = ()
    compress_rope_theta: float = 160000.0
    candidate_source_layer_id: int = -1
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512

    rope_theta: float = 10000.0
    rope_scaling: Optional[dict[str, Union[float, str]]] = None
    max_position_embeddings: int = 1048576
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    engram_layer_ids: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ModelArgs":
        if "text_config" in params:
            params = {**params["text_config"], "model_type": params["model_type"]}
        return super().from_dict(params)

    def __post_init__(self) -> None:
        if self.engram_layer_ids:
            raise ValueError(
                f"Layers {list(self.engram_layer_ids)} use engram memory, which "
                "this implementation does not have: the n-gram hash tables, the "
                "hashed embedding lookup and the engram attention are all missing."
            )
