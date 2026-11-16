# Copyright © 2023 Apple Inc.

from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.activations import swiglu
from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.switch import SwitchGLU
from mlx_lm.models.hunyuan.config import ModelArgs

__all__ = [
    "Attention",
    "DecoderLayer",
    "DynamicNTKAlphaRoPE",
    "Gate",
    "MLP",
    "MoeBlock",
]


def _int_or_list(arg, idx):
    if isinstance(arg, list):
        return arg[idx]
    return arg


class DynamicNTKAlphaRoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float = 10000,
        scaling_alpha: float = 1.0,
    ):
        super().__init__()
        self.dims = dims
        base = base * scaling_alpha ** (dims / (dims - 2))
        self._freqs = base ** (mx.arange(0, self.dims, 2) / self.dims)

    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class Attention(nn.Module):
    def __init__(self, kv_proj: bool, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        if kv_proj:
            self.k_proj = nn.Linear(
                dim, n_kv_heads * head_dim, bias=args.attention_bias
            )
            self.v_proj = nn.Linear(
                dim, n_kv_heads * head_dim, bias=args.attention_bias
            )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)
        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(head_dim, args.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(head_dim, args.rms_norm_eps)

        self.rope = DynamicNTKAlphaRoPE(
            head_dim,
            base=args.rope_theta,
            scaling_alpha=args.rope_scaling["alpha"],
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        kv_states=None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x)
        if kv_states is None:
            keys, values = self.k_proj(x), self.v_proj(x)
            kv_states = keys, values
        else:
            keys, values = kv_states

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = cache.offset if cache else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), kv_states


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Gate(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.wg = nn.Linear(dim, num_experts, bias=False)

    def __call__(self, x) -> mx.array:
        return self.wg(x)


class MoeBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.intermediate_size
        self.use_shared_mlp = args.use_mixed_mlp_moe

        if args.use_mixed_mlp_moe:
            num_shared = _int_or_list(args.num_shared_expert, layer_idx)
            self.shared_mlp = MLP(dim, int(intermediate_size * num_shared))

        self.num_experts = num_experts = args.num_experts
        self.top_k = _int_or_list(args.moe_topk, layer_idx)

        self.gate = Gate(dim, num_experts)

        # Use moe_intermediate_size if available, otherwise use intermediate_size
        expert_intermediate_size = intermediate_size
        if args.moe_intermediate_size is not None:
            expert_intermediate_size = _int_or_list(
                args.moe_intermediate_size, layer_idx
            )

        self.switch_mlp = SwitchGLU(dim, expert_intermediate_size, num_experts)

    def __call__(
        self,
        x: mx.array,
    ):
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None].astype(mx.float32)).sum(axis=-2).astype(y.dtype)

        if self.use_shared_mlp:
            shared_expert_output = self.shared_mlp(x)
            y = y + shared_expert_output

        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, kv_proj: bool, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(kv_proj, args)
        if args.num_experts == 1:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = MoeBlock(args, layer_idx)

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
        shared_kv_states: Optional[Tuple[mx.array, mx.array]] = None,
    ):
        r, shared_kv_states = self.self_attn(
            self.input_layernorm(x), mask, cache, shared_kv_states
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, shared_kv_states
