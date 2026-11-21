# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.models.talkie.config import ModelArgs

__all__ = [
    "ActGain",
    "Attention",
    "HeadGain",
    "MLP",
    "TransformerBlock",
    "WeightGain",
    "rms_norm",
]


def rms_norm(x: mx.array, eps: float) -> mx.array:
    # All of Talkie's norms are weightless.
    return mx.fast.rms_norm(x, None, eps)


class WeightGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_g = mx.ones((1,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.w_g.astype(x.dtype)


class HeadGain(nn.Module):
    def __init__(self, n_head: int):
        super().__init__()
        self.head_g = mx.ones((n_head,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, L, D)
        return x * self.head_g.astype(x.dtype).reshape(1, -1, 1, 1)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.eps = args.rms_norm_eps

        proj_dim = self.n_heads * self.head_dim
        self.attn_query = nn.Linear(dim, proj_dim, bias=False)
        self.attn_key = nn.Linear(dim, proj_dim, bias=False)
        self.attn_value = nn.Linear(dim, proj_dim, bias=False)
        self.attn_resid = nn.Linear(proj_dim, dim, bias=False)
        self.head_gain = HeadGain(self.n_heads)

        # Talkie rotates with the opposite sign to NeoX RoPE
        # (y1 = x1*cos + x2*sin, y2 = -x1*sin + x2*cos), which is
        # equivalent to standard RoPE with negated frequencies.
        self._rope_freqs = -(
            args.rope_theta
            ** (mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim)
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.attn_query(x).reshape(B, L, self.n_heads, self.head_dim)
        keys = self.attn_key(x).reshape(B, L, self.n_heads, self.head_dim)
        values = self.attn_value(x).reshape(B, L, self.n_heads, self.head_dim)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = self._rope(queries, offset)
        keys = self._rope(keys, offset)

        # QK norm is applied after RoPE, with a learned per-head gain on the
        # queries only.
        queries = rms_norm(queries, self.eps)
        keys = rms_norm(keys, self.eps)
        queries = self.head_gain(queries)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.attn_resid(output)

    def _rope(self, x, offset):
        return mx.fast.rope(
            x,
            dims=self.head_dim,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._rope_freqs,
        )


class ActGain(nn.Module):
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.a_g = mx.full((1,), init_value)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.a_g.astype(x.dtype)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        self.mlp_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.mlp_linear = nn.Linear(dim, hidden_dim, bias=False)
        self.mlp_resid = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp_resid(nn.silu(self.mlp_gate(x)) * self.mlp_linear(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        gain_init = (2 * args.num_hidden_layers) ** -0.5
        self.eps = args.rms_norm_eps
        self.attn = Attention(args)
        self.attn_gain = ActGain(gain_init)
        self.mlp = MLP(args)
        self.mlp_gain = ActGain(gain_init)
        self.embed_skip = ActGain(0.0)

    def __call__(
        self,
        e_x: mx.array,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        x = x + self.attn_gain(self.attn(rms_norm(x, self.eps), mask, cache))
        x = x + self.mlp_gain(self.mlp(rms_norm(x, self.eps)))
        # Every block adds a skip connection from the normalized token
        # embeddings, with a learned scalar gain.
        x = x + self.embed_skip(e_x)
        return x
