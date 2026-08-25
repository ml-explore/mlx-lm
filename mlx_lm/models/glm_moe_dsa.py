# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import BaseModelArgs
from .deepseek_v32 import Model as DSV32Model


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_parameters: Dict
    attention_bias: bool
    rope_scaling: Dict = None
    rope_theta: Optional[float] = None
    # GLM-5.2's HF config omits these; deepseek_v32 defaults them, so default here too
    # (loading a real GLM-5.2 config.json fails otherwise).
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    moe_layer_freq: int = 1
    # DSA IndexShare: "full" layers run the indexer, "shared" layers reuse the
    # previous full layer's top-k. indexer_types is provided by GLM-5.2's config;
    # if absent it is derived from index_topk_freq. index_skip_topk_offset is only
    # used for MTP iteration in the reference and is accepted-but-ignored here.
    index_topk_freq: int = 1
    indexer_types: Optional[Any] = None
    index_skip_topk_offset: int = 0
    indexer_rope_interleave: bool = False
    indexer_norm_eps: float = 1e-6
    # GLM-5.2 ships one nextn (MTP) layer; kept as an MtpModule when the
    # checkpoint retains layer 78 (see deepseek_v32.Model.sanitize).
    num_nextn_predict_layers: int = 0

    def __post_init__(self):
        self.rope_scaling = self.rope_parameters
        self.rope_theta = self.rope_parameters["rope_theta"]
        # GLM-5.2 configs ship `indexer_rope_interleave: true`, but HF's modeling
        # ignores the flag and hardcodes half-split (non-interleaved) RoPE in the
        # indexer (verified vs HF); force the empirically-correct behavior.
        self.indexer_rope_interleave = False


class Model(DSV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
