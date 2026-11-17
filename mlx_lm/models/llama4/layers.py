# Copyright © 2023 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.common import swiglu
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.llama4.config import ModelArgs, TextArgs

__all__ = [
    "Attention",
    "MLP",
    "MoE",
    "TransformerBlock",
]


class Attention(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.use_rope = int((layer_idx + 1) % 4 != 0)  # rope unused for dense layers
        self.attn_temperature_tuning = args.attn_temperature_tuning
        self.floor_scale = args.floor_scale
        self.attn_scale = args.attn_scale

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.use_qk_norm = args.use_qk_norm and self.use_rope

        if self.use_rope:
            self.rope = initialize_rope(
                head_dim,
                args.rope_theta,
                traditional=True,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
        else:
            offset = 0

        if self.use_rope:
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if self.use_qk_norm:
            queries = mx.fast.rms_norm(queries, weight=None, eps=1e-6)
            keys = mx.fast.rms_norm(keys, weight=None, eps=1e-6)

        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                mx.log(
                    mx.floor(mx.arange(offset + 1, offset + L + 1) / self.floor_scale)
                    + 1.0
                )
                * self.attn_scale
                + 1.0
            )
            attn_scales = attn_scales[:, None]
            queries = (queries * attn_scales).astype(queries.dtype)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = intermediate_size or args.intermediate_size

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        assert self.top_k == 1, "Only 1 expert per token supported"
        self.num_experts = args.num_local_experts
        self.experts = SwitchGLU(
            args.hidden_size, args.intermediate_size, self.num_experts
        )
        self.router = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.shared_expert = MLP(args)

    def __call__(self, x) -> mx.array:
        logits = self.router(x)
        k = self.top_k
        indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        indices = mx.stop_gradient(indices)
        scores = mx.take_along_axis(logits, indices, axis=-1)
        scores = mx.sigmoid(scores.astype(mx.float32)).astype(x.dtype)

        out = self.experts(x * scores, indices).squeeze(2)
        return out + self.shared_expert(x)


class TransformerBlock(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args, layer_idx)
        self.is_moe_layer = (layer_idx % args.interleave_moe_layer_step) == (
            args.interleave_moe_layer_step - 1
        )
        if self.is_moe_layer:
            self.feed_forward = MoE(args)
        else:
            self.feed_forward = MLP(args, args.intermediate_size_mlp)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.post_attention_layernorm(h))
        out = h + r
        return out
