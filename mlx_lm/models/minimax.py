# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    num_experts_per_tok: int
    num_local_experts: int
    shared_intermediate_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    vocab_size: int
    tie_word_embeddings: bool = False
    postnorm: bool = True
    shared_moe_mode: str = "sigmoid"
    full_attn_alpha_factor: float = 3.5565588200778455
    full_attn_beta_factor: float = 1.0
    linear_attn_alpha_factor: float = 3.5565588200778455
    linear_attn_beta_factor: float = 1.0
    mlp_alpha_factor: float = 3.5565588200778455
    mlp_beta_factor: float = 1.0
    layer_types: List[str] = None
    head_dim: Optional[int] = None





{
  "layer_types": [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention"
  ],
  "head_dim": 128,
  "hidden_size": 6144,
  "intermediate_size": 9216,
  "full_attn_alpha_factor": 3.5565588200778455,
  "full_attn_beta_factor": 1.0,
  "linear_attn_alpha_factor": 3.5565588200778455,
  "linear_attn_beta_factor": 1.0,
  "mlp_alpha_factor": 3.5565588200778455,
  "mlp_beta_factor": 1.0,
  "max_position_embeddings": 10240000,
  "num_attention_heads": 64,
  "num_experts_per_tok": 2,
  "num_hidden_layers": 80,
  "num_key_value_heads": 8,
  "num_local_experts": 32,
  "postnorm": true,
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000000,
  "rotary_dim": 64,
  "shared_intermediate_size": 0,
  "shared_moe_mode": "sigmoid",
  "sliding_window": null,
  "tie_word_embeddings": false,
  "vocab_size": 200064
}




class MiniMaxAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads if args.head_dim is None else args.head_dim

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.rope = nn.RoPE(
            args.rotary_dim,
            traditional=False,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MiniMaxSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok

        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.intermediate_size, args.num_local_experts)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=self.num_experts_per_tok - 1, axis=-1)[..., :self.num_experts_per_tok])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y


class MiniMaxDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = args.layer_types[layer_idx]
        self.mlp_alpha_factor = args.mlp_alpha_factor
        self.mlp_beta_factor = args.mlp_beta_factor
    
        if self.layer_type == "linear_attention":
            self.self_attn = MiniMaxLightningAttention(args)
            self.attn_alpha_factor = args.linear_attn_alpha_factor
            self.attn_beta_factor = args.linear_attn_beta_factor
        else:
            self.self_attn = MiniMaxAttention(args)
            self.attn_alpha_factor = args.full_attn_alpha_factor
            self.attn_beta_factor = args.full_attn_beta_factor
        
        self.block_sparse_moe = MiniMaxSparseMoeBlock(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x * self.attn_alpha_factor + r * self.attn_beta_factor
        r = self.block_sparse_moe(self.post_attention_layernorm(h))
        out = h * self.mlp_alpha_factor + r * self.mlp_beta_factor
        return out













class MiniMaxText01Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = [
            MiniMaxText01DecoderLayer(args=args, attention_type=args.attn_type_list[i])
            for i in range(args.num_hidden_layers)
        ]

        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.slopes = self._build_slope_tensor(args.num_attention_heads)

    def _build_slope_tensor(self, n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        return mx.array(get_slopes(n_attention_heads)).reshape(n_attention_heads, 1, 1)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        slope_rates = [self.slopes for _ in range(len(self.layers))]

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            sr = slope_rates[i] * (1 - i / (len(self.layers) - 1) + 1e-5)
            h = layer(h, mask=mask, cache=c, slope_rate=sr)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniMaxText01Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs=inputs, mask=mask, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if "model.slopes" not in weights:
            slopes = self.model._build_slope_tensor(self.args.num_attention_heads)
            weights["model.slopes"] = slopes

        if "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in weights:
            return weights
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
            for orig_name, new_name in mapping.items():
                if f"{prefix}.block_sparse_moe.experts.0.{orig_name}.weight" in weights:
                    to_join = [
                        weights.pop(
                            f"{prefix}.block_sparse_moe.experts.{e}.{orig_name}.weight"
                        )
                        for e in range(self.args.num_local_experts)
                    ]
                    weights[
                        f"{prefix}.block_sparse_moe.switch_mlp.{new_name}.weight"
                    ] = mx.stack(to_join)
        return weights

    @property
    def layers(self):
        return self.model.layers


# Goekdeniz-Guelmez/MiniMax01Text-Dev