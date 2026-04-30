# Copyright © 2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int = 65540
    hidden_size: int = 5120
    intermediate_size: int = 13696
    num_hidden_layers: int = 40
    num_attention_heads: int = 40
    head_dim: int = 128
    max_position_embeddings: int = 2048
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False


def _weightless_rms_norm(x: mx.array) -> mx.array:
    xf = x.astype(mx.float32)
    return (xf * mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + mx.finfo(mx.float32).eps)).astype(x.dtype)


class HeadGain(nn.Module):
    def __init__(self, n_head: int):
        super().__init__()
        self.head_g = mx.ones(n_head)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.head_g.astype(x.dtype).reshape(1, -1, 1, 1)


class ActGain(nn.Module):
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.a_g = mx.full((1,), init_value)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.a_g.astype(x.dtype)


class TalkieAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5

        proj_dim = self.n_heads * self.head_dim
        self.attn_query = nn.Linear(dim, proj_dim, bias=False)
        self.attn_key = nn.Linear(dim, proj_dim, bias=False)
        self.attn_value = nn.Linear(dim, proj_dim, bias=False)
        self.attn_resid = nn.Linear(proj_dim, dim, bias=False)
        self.head_gain = HeadGain(self.n_heads)

    def _apply_rotary_emb(
        self, x: mx.array, cos: mx.array, sin: mx.array
    ) -> mx.array:
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = -x1 * sin + x2 * cos
        return mx.concatenate([y1, y2], axis=-1).astype(x.dtype)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.attn_query(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.attn_key(x).reshape(B, L, self.n_heads, self.head_dim)
        v = self.attn_value(x).reshape(B, L, self.n_heads, self.head_dim)

        q = self._apply_rotary_emb(q, cos, sin).transpose(0, 2, 1, 3)
        k = self._apply_rotary_emb(k, cos, sin).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = _weightless_rms_norm(q)
        k = _weightless_rms_norm(k)
        q = self.head_gain(q)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.attn_resid(out)


class TalkieMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden = args.intermediate_size
        self.mlp_gate = nn.Linear(dim, hidden, bias=False)
        self.mlp_linear = nn.Linear(dim, hidden, bias=False)
        self.mlp_resid = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp_resid(nn.silu(self.mlp_gate(x)) * self.mlp_linear(x))


class TalkieDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        gain_init = (2 * args.num_hidden_layers) ** -0.5
        self.attn = TalkieAttention(args)
        self.attn_gain = ActGain(gain_init)
        self.mlp = TalkieMLP(args)
        self.mlp_gain = ActGain(gain_init)
        self.embed_skip = ActGain(0.0)

    def __call__(
        self,
        e_x: mx.array,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        x = x + self.attn_gain(self.attn(_weightless_rms_norm(x), cos, sin, mask, cache))
        x = x + self.mlp_gain(self.mlp(_weightless_rms_norm(x)))
        x = x + self.embed_skip(e_x)
        return x


class TalkieModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_size, args.hidden_size)
        self.blocks = [TalkieDecoderLayer(args) for _ in range(args.num_hidden_layers)]

    def __call__(
        self,
        inputs: mx.array,
        cos: mx.array,
        sin: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed(inputs)
        h = _weightless_rms_norm(h)
        e_h = h

        if cache is None:
            cache = [None] * len(self.blocks)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.blocks, cache):
            h = layer(e_h, h, cos, sin, mask, c)

        return _weightless_rms_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = TalkieModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        self._cos, self._sin = self._precompute_rope(args.max_position_embeddings)

    def _precompute_rope(self, seq_len: int) -> tuple[mx.array, mx.array]:
        channel_range = mx.arange(0, self.args.head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (self.args.rope_theta ** (channel_range / self.args.head_dim))
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = t[:, None] * inv_freq[None, :]
        cos = mx.cos(freqs)[None, :, None, :]
        sin = mx.sin(freqs)[None, :, None, :]
        return cos, sin

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if cache is None:
            cache = [None] * len(self.model.blocks)
            offset = 0
        else:
            offset = int(cache[0].offset)

        L = inputs.shape[1] if input_embeddings is None else input_embeddings.shape[1]
        if offset + L > self._cos.shape[1]:
            self._cos, self._sin = self._precompute_rope(offset + L)
        cos = self._cos[:, offset : offset + L]
        sin = self._sin[:, offset : offset + L]

        h = self.model(inputs, cos, sin, cache, input_embeddings)
        return self.lm_head(h)

    def sanitize(self, weights):
        if "lm_head" in weights and "lm_head.weight" not in weights:
            gain = weights.pop("lm_head_gain.w_g").reshape(())
            weights["lm_head.weight"] = weights.pop("lm_head") * gain
        else:
            weights.pop("lm_head_gain.w_g", None)
        return weights

    @property
    def layers(self):
        return self.model.blocks

    def make_cache(self):
        return [KVCache() for _ in self.layers]
