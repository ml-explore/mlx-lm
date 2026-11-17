# Copyright © 2023 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.common import swiglu
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.ernie4_5_moe.config import ModelArgs

__all__ = [
    "Attention",
    "Ernie4_5_DecoderLayer",
    "Ernie4_5_MLP",
    "Ernie4_5_MoeMLP",
]


class Ernie4_5_MLP(nn.Module):
    def __init__(self, dim, hidden_dim, use_bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=use_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=use_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or dim // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.use_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.use_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.use_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.use_bias)

        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=True,
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


class Ernie4_5_MoeMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.k = args.moe_k
        self.moe_intermediate_size = (
            args.moe_intermediate_size
            if args.moe_intermediate_size
            else args.intermediate_size
        )

        self.gate = nn.Linear(args.hidden_size, args.moe_num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.moe_num_experts,))

        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            self.moe_intermediate_size,
            args.moe_num_experts,
            bias=args.use_bias,
        )

        if getattr(args, "moe_num_shared_experts", 0) > 0:
            shared_intermediate_size = (
                args.moe_intermediate_size * args.moe_num_shared_experts
                if getattr(args, "moe_intermediate_size", None)
                else args.intermediate_size * args.moe_num_shared_experts
            )
            self.shared_experts = Ernie4_5_MLP(
                args.hidden_size, shared_intermediate_size, args.use_bias
            )
        else:
            self.shared_experts = None

        if args.moe_gate_act == "softmax":
            self.gate_act = nn.Softmax()
        elif args.moe_gate_act == "sigmoid":
            self.gate_act = nn.Sigmoid()
        else:
            raise ValueError(f"{args.moe_gate_act} is not supported.")

    def __call__(self, x: mx.array) -> mx.array:
        # HF multiplies in fp32, so near-ties pick the same experts.
        gates = self.gate_act(self.gate(x.astype(mx.float32)))

        k = self.k
        bias = self.e_score_correction_bias
        # The bias only ranks the experts. The weights use the unbiased gates.
        inds = mx.stop_gradient(
            mx.argpartition(-(gates + bias), kth=k - 1, axis=-1)[..., :k]
        )
        scores = mx.take_along_axis(gates, inds, axis=-1)

        scores = scores / mx.maximum(scores.sum(axis=-1, keepdims=True), 1e-12)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

        if self.shared_experts is not None:
            y = y + self.shared_experts(x)

        return y


class Ernie4_5_DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args)

        moe_layer_start_index = (
            min(args.moe_layer_start_index)
            if isinstance(args.moe_layer_start_index, (tuple, list))
            else args.moe_layer_start_index
        )

        if args.moe_layer_end_index is None:
            moe_layer_end_index = args.num_hidden_layers - 1
        else:
            moe_layer_end_index = (
                max(args.moe_layer_end_index)
                if isinstance(args.moe_layer_end_index, (tuple, list))
                else args.moe_layer_end_index
            )

        if (
            ((layer_idx + 1) % args.moe_layer_interval == 0)
            and layer_idx >= moe_layer_start_index
            and layer_idx <= moe_layer_end_index
        ):
            self.mlp = Ernie4_5_MoeMLP(args)
        else:
            self.mlp = Ernie4_5_MLP(
                args.hidden_size, args.intermediate_size, args.use_bias
            )

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
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r
