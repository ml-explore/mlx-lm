# Copyright © 2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .deepseek_v3 import DeepseekV3Model


@dataclass
class TextArgs(BaseModelArgs):
    vocab_size: int = 102400
    hidden_size: int = 4096
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1407
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: Optional[int] = None
    n_routed_experts: Optional[int] = None
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Dict = None
    attention_bias: bool = False


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: Union[TextArgs, dict]
    model_type: str

    def __post_init__(self):
        self.text_config = TextArgs.from_dict(self.text_config)


class LanguageModel(nn.Module):
    def __init__(self, config: TextArgs):
        super().__init__()
        self.args = config
        self.model = DeepseekV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, mask)
        return self.lm_head(out)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
    ):
        return self.language_model(inputs, cache, mask)

    def sanitize(self, weights):
        def keep(key):
            return (
                "vision_tower" not in key
                and "rotary_emb" not in key
                and "multi_modal_projector" not in key
            )

        weights = {k: v for k, v in weights.items() if keep(k)}
        # Stack experts
        for l in range(self.args.text_config.num_hidden_layers):
            prefix = f"language_model.model.layers.{l}"
            for m in [("gate_proj"), ("down_proj"), ("up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.text_config.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        return weights

    @property
    def layers(self):
        return self.language_model.model.layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate
