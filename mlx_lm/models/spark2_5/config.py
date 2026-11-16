# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    sliding_window: int
    rope_parameters: Dict[str, Dict[str, Any]]
    layer_types: Optional[List[str]] = None
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 8192
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "gelu"
    gate_attn_act_mode: str = "sigmoid"
    headwise_attn_output_gate: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.layer_types is None:
            # Official checkpoints list every layer type; the reference
            # pattern is three sliding layers followed by one full layer.
            pattern = ["sliding_attention"] * 3 + ["full_attention"]
            self.layer_types = [
                pattern[i % len(pattern)] for i in range(self.num_hidden_layers)
            ]

        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                "layer_types length must match num_hidden_layers: "
                f"{len(self.layer_types)} != {self.num_hidden_layers}"
            )

        unsupported = set(self.layer_types) - {
            "sliding_attention",
            "full_attention",
        }
        if unsupported:
            raise ValueError(f"Unsupported layer types: {sorted(unsupported)}")

        missing_rope = set(self.layer_types) - set(self.rope_parameters)
        if missing_rope:
            raise ValueError(
                f"Missing RoPE parameters for layer types: {sorted(missing_rope)}"
            )

        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        if self.num_attention_heads * self.head_dim <= 0:
            raise ValueError("num_attention_heads * head_dim must be positive")

        for layer_type in set(self.layer_types):
            factor = float(
                self.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
            )
            dims = int(self.head_dim * factor)
            if dims <= 0 or dims > self.head_dim or dims % 2:
                raise ValueError(
                    f"Invalid rotary dimension {dims} for {layer_type} "
                    f"with head_dim={self.head_dim}"
                )

        if self.hidden_act != "gelu":
            raise ValueError(f"Spark2_5 requires GELU, got {self.hidden_act!r}")

        if not self.headwise_attn_output_gate:
            raise ValueError("Spark2_5 requires head-wise attention output gates")

        if self.gate_attn_act_mode != "sigmoid":
            raise ValueError(
                "Spark2_5 requires sigmoid attention gates, got "
                f"{self.gate_attn_act_mode!r}"
            )
