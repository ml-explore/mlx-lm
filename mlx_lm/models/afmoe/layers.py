# Copyright © 2024 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.activations import swiglu
from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.layers.switch import SwitchGLU
from mlx_lm.models.afmoe.config import ModelArgs

__all__ = [
    "AfmoeMoE",
    "Attention",
    "DecoderLayer",
    "MLP",
    "MoERouter",
]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, is_local_attention: bool = False):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.is_local_attention = is_local_attention

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.gate_proj = nn.Linear(
            self.hidden_size, self.n_heads * self.head_dim, bias=False
        )

        if is_local_attention:
            self.rope = initialize_rope(
                self.head_dim,
                args.rope_theta,
                False,  # traditional
                args.rope_scaling,
                args.max_position_embeddings,
            )
        else:
            self.rope = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        if self.is_local_attention and self.rope is not None:
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        gate = mx.sigmoid(self.gate_proj(x))
        output = output * gate

        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = (
            intermediate_size
            if intermediate_size is not None
            else args.intermediate_size
        )

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoERouter(nn.Module):
    """Router module that wraps the gate for proper weight naming."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.gate(x)


class AfmoeMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_experts = args.num_experts
        self.num_experts_per_tok = args.num_experts_per_tok
        self.route_norm = args.route_norm
        self.route_scale = args.route_scale
        self.score_func = args.score_func
        self.n_group = args.n_group
        self.topk_group = args.topk_group

        self.router = MoERouter(args)

        self.expert_bias = mx.zeros((args.num_experts,))

        self.experts = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
        )

        if args.num_shared_experts > 0:
            shared_intermediate_size = (
                args.moe_intermediate_size * args.num_shared_experts
            )
            self.shared_experts = MLP(args, intermediate_size=shared_intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.router(x)

        if self.score_func == "sigmoid":
            scores = mx.sigmoid(gates.astype(mx.float32))
        else:
            scores = mx.softmax(gates.astype(mx.float32), axis=-1)

        # Add expert bias for selection
        selection_scores = scores + self.expert_bias

        # Group-based expert selection if n_group > 1
        if self.n_group > 1:
            selection_scores = mx.unflatten(
                selection_scores, axis=-1, shape=(self.n_group, -1)
            )
            group_scores = mx.topk(selection_scores, 2, axis=-1).sum(
                axis=-1, keepdims=True
            )
            k = self.n_group - self.topk_group
            group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
            selection_scores = mx.put_along_axis(
                selection_scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
            )
            selection_scores = mx.flatten(selection_scores, -2, -1)

        # Select top-k experts
        k = self.num_experts_per_tok
        inds = mx.argpartition(-selection_scores, kth=k - 1, axis=-1)[..., :k]
        inds = mx.stop_gradient(inds)

        selected_scores = mx.take_along_axis(scores, inds, axis=-1)

        if self.route_norm and self.num_experts_per_tok > 1:
            denominator = selected_scores.sum(axis=-1, keepdims=True)
            selected_scores = selected_scores / denominator

        selected_scores = selected_scores * self.route_scale

        y = self.experts(x, inds)
        y = (y * selected_scores[..., None]).sum(axis=-2).astype(y.dtype)

        if self.args.num_shared_experts > 0:
            y = y + self.shared_experts(x)

        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int, use_sliding: bool = False):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.use_sliding = use_sliding
        self.layer_idx = layer_idx

        self.self_attn = Attention(args, is_local_attention=use_sliding)

        if layer_idx < args.num_dense_layers:
            self.mlp = MLP(args)
        else:
            self.mlp = AfmoeMoE(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.pre_mlp_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_mlp_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        r = self.post_attention_layernorm(r)
        h = x + r

        r = self.mlp(self.pre_mlp_layernorm(h))
        r = self.post_mlp_layernorm(r)
        return h + r
