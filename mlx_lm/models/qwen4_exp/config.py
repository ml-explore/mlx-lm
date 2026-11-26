# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
    "TextArgs",
]


@dataclass
class TextArgs(BaseModelArgs):
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_parameters: dict[str, Any]
    eos_token_id: int | list[int]
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    hc_count: int
    hc_lowrank: int
    indexer_n_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    ple_embed_dim: int
    ple_layer_ids: list[int]
    # Optional, with the reference's defaults. config.json has no `seed` (n-gram hash).
    ple_conv_kernel_size: int = 4
    seed: int = 1234
    layer_types: list[str] = field(default_factory=list)
    full_attention_interval: int = 4
    output_gate_type: str | None = None
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    # QSA reads only the picked keys (False: masked dense attention over all keys).
    sparse_attention: bool = True
    # Without the Metal kernel, prefill reads the picked keys from this cache length on.
    sparse_prefill_min_context: int = 32768
    # Sparse prefill attention in one Metal kernel.
    sparse_prefill_fused: bool = True
    rope_theta: float = field(init=False)
    partial_rotary_factor: float = field(init=False)

    def __post_init__(self) -> None:
        self.rope_theta = float(self.rope_parameters["rope_theta"])
        self.partial_rotary_factor = float(
            self.rope_parameters.get("partial_rotary_factor", 1.0)
        )
        self.output_gate_type = self.output_gate_type or self.hidden_act
        if not self.layer_types:
            k = self.full_attention_interval
            self.layer_types = [
                "full_attention" if (i + 1) % k == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict[str, Any]
    text: TextArgs = field(init=False)

    def __post_init__(self) -> None:
        self.text = TextArgs.from_dict(self.text_config)
