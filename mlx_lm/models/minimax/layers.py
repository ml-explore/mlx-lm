# Copyright © 2025 Apple Inc.

from functools import lru_cache
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.models.minimax.config import ModelArgs

__all__ = [
    "MiniMaxAttention",
    "MiniMaxDecoderLayer",
    "MiniMaxSparseMoeBlock",
    "ShardedRMSNorm",
    "sharded_rms_norm",
]


@lru_cache
def sharded_rms_norm(group):
    @mx.compile
    def _cast_square_sum(x):
        return x.astype(mx.float32).square().sum(-1, keepdims=True)

    @mx.compile
    def _normalize(x, norm2, w, eps):
        norm2 = mx.distributed.all_sum(norm2, group=group)
        norm = mx.rsqrt(norm2 / (x.shape[-1] * group.size()) + eps)
        return (x.astype(mx.float32) * norm * w).astype(x.dtype)

    # Split the compile so that x upcasting doesn't break the compile and we
    # have 2 kernels generated 1 for f(x) = square(upcast(x)) and another
    # g(x) = downcast(upcast(x) * norm * w)
    def _inner_sharded_rms_norm(x, w, eps):
        return _normalize(x, _cast_square_sum(x), w, eps)

    return _inner_sharded_rms_norm


class ShardedRMSNorm(nn.Module):
    def __init__(
        self, dims: int, eps: float = 1e-5, group: Optional[mx.distributed.Group] = None
    ):
        super().__init__()
        group = group or mx.distributed.init()
        self.weight = mx.ones((dims // group.size(),))
        self.group = group
        self.eps = eps

    def _extra_repr(self):
        return f"{self.weight.shape[0] * self.group.size()}, eps={self.eps}"

    def __call__(self, x):
        return sharded_rms_norm(self.group)(x, self["weight"], self.eps)

    @classmethod
    def from_rms_norm(
        cls, norm_module, *, group: Optional[mx.distributed.Group] = None
    ):
        sn = cls(norm_module.weight.shape[0], norm_module.eps, group=group)
        sn.weight = mx.contiguous(
            mx.split(norm_module.weight, group.size(), axis=-1)[group.rank()]
        )

        return sn


class MiniMaxAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_dim = hidden_size = args.hidden_size

        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = head_dim = (
            args.head_dim or hidden_size // args.num_attention_heads
        )
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * head_dim, args.hidden_size, bias=False
        )

        self.use_qk_norm = args.use_qk_norm if hasattr(args, "use_qk_norm") else False
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(
                head_dim * self.num_attention_heads, eps=args.rms_norm_eps
            )
            self.k_norm = nn.RMSNorm(
                head_dim * self.num_key_value_heads, eps=args.rms_norm_eps
            )

        self.rope = nn.RoPE(args.rotary_dim, traditional=False, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(
            0, 2, 1, 3
        )
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
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.intermediate_size, args.num_local_experts
        )
        self.e_score_correction_bias = mx.zeros((args.num_local_experts,))
        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        gates = self.gate(x.astype(mx.float32))

        scores = mx.sigmoid(gates)
        orig_scores = scores
        scores = scores + self.e_score_correction_bias

        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        inds = mx.stop_gradient(inds)
        scores = mx.take_along_axis(orig_scores, inds, axis=-1)

        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        scores = scores.astype(x.dtype)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


class MiniMaxDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.self_attn = MiniMaxAttention(args)

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
        r = x + self.self_attn(self.input_layernorm(x), mask, cache)
        r = r + self.block_sparse_moe(self.post_attention_layernorm(r))
        return r
