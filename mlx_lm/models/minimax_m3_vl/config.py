# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import List, Optional, Union

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
    "TextArgs",
]


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = ""
    hidden_size: int = 6144
    intermediate_size: int = 3072
    dense_intermediate_size: int = 12288
    shared_intermediate_size: int = 3072
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    num_hidden_layers: int = 60
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    rope_parameters: Optional[dict] = None
    head_dim: int = 128
    rotary_dim: Optional[int] = None
    partial_rotary_factor: float = 0.5
    vocab_size: int = 200064
    tie_word_embeddings: bool = False
    routed_scaling_factor: float = 2.0
    use_qk_norm: bool = True
    use_gemma_norm: bool = True
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    mlp_layer_types: Optional[List[str]] = None
    moe_layer_freq: Optional[List[int]] = None
    layer_types: Optional[List[str]] = None
    sparse_attention_config: Optional[dict] = None
    sparse_attention_freq: Optional[List[int]] = None
    index_n_heads: Optional[int] = None
    index_head_dim: Optional[int] = None
    index_block_size: Optional[int] = None
    index_topk_blocks: Optional[int] = None
    index_local_blocks: int = 1

    def __post_init__(self):
        # The published config nests these; a 5.x re-save spells them flat.
        sac = self.sparse_attention_config or {}
        self.index_n_heads = sac.get("sparse_num_index_heads", self.index_n_heads)
        self.index_head_dim = sac.get("sparse_index_dim", self.index_head_dim)
        self.index_block_size = sac.get("sparse_block_size", self.index_block_size)
        self.index_topk_blocks = sac.get("sparse_topk_blocks", self.index_topk_blocks)
        self.index_local_blocks = sac.get("sparse_local_block", self.index_local_blocks)
        self.sparse_attention_freq = sac.get(
            "sparse_attention_freq", self.sparse_attention_freq
        )
        rope = self.rope_parameters or {}
        self.rope_theta = rope.get("rope_theta", self.rope_theta)
        self.partial_rotary_factor = rope.get(
            "partial_rotary_factor", self.partial_rotary_factor
        )
        if self.rotary_dim is None:
            self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)

    def is_moe(self, layer_idx: int) -> bool:
        # transformers 5.x renamed moe_layer_freq to mlp_layer_types. The published
        # config carries only the old key, a re-save only the new one.
        if self.mlp_layer_types is not None:
            return self.mlp_layer_types[layer_idx] == "sparse"
        if self.moe_layer_freq is not None:
            return bool(self.moe_layer_freq[layer_idx])
        return True

    def is_sparse_attn(self, layer_idx: int) -> bool:
        if self.index_block_size is None:
            return False
        if self.layer_types is not None:
            return self.layer_types[layer_idx] != "full_attention"
        if self.sparse_attention_freq is not None:
            return bool(self.sparse_attention_freq[layer_idx])
        return True


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: Union[TextArgs, dict]
    model_type: str = "minimax_m3_vl"

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextArgs.from_dict(self.text_config)
