# Copyright © 2026 MLX Contributors

from dataclasses import dataclass
import math
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "talkie"
    vocab_size: int = 65540
    n_layer: int = 40
    n_head: int = 40
    n_embd: int = 5120
    head_dim: int = 128
    rope_base: float = 1_000_000.0
    max_seq_len: int = 2048
    dtype: str = "bfloat16"
    style: str = "it"


class HeadGain(nn.Module):
    def __init__(self, n_head: int):
        super().__init__()
        self.head_g = mx.ones((n_head,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.head_g.astype(x.dtype).reshape(1, -1, 1, 1)


class WeightGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_g = mx.ones((1,))

    def __call__(self, w: mx.array) -> mx.array:
        return w * self.w_g.astype(w.dtype)


class ActGain(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.a_g = mx.ones((1,)) * init_value

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.a_g.astype(x.dtype)


def rms_norm(x: mx.array) -> mx.array:
    # Match torch.nn.functional.rms_norm(..., eps=None) for bf16 checkpoints:
    # PyTorch computes the reduction in fp32 and uses fp32 epsilon.
    xf = x.astype(mx.float32)
    out = xf * mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + mx.finfo(mx.float32).eps)
    return out.astype(x.dtype)


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_head = args.n_head
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        n_state = args.n_embd

        self.attn_query = nn.Linear(n_state, n_state, bias=False)
        self.attn_key = nn.Linear(n_state, n_state, bias=False)
        self.attn_value = nn.Linear(n_state, n_state, bias=False)
        self.attn_resid = nn.Linear(n_state, n_state, bias=False)
        self.head_gain = HeadGain(args.n_head)

    def _apply_rotary_emb(
        self, x: mx.array, cos: mx.array, sin: mx.array
    ) -> mx.array:
        # Talkie rotates first and second halves of each head, matching the
        # original PyTorch implementation.
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return mx.concatenate([y1, y2], axis=-1).astype(x.dtype)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape
        q = self.attn_query(x).reshape(B, L, self.n_head, self.head_dim)
        k = self.attn_key(x).reshape(B, L, self.n_head, self.head_dim)
        v = self.attn_value(x).reshape(B, L, self.n_head, self.head_dim)

        q = self._apply_rotary_emb(q, cos, sin).transpose(0, 2, 1, 3)
        k = self._apply_rotary_emb(k, cos, sin).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = rms_norm(q)
        k = rms_norm(k)
        q = self.head_gain(q)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        y = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        y = y.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.attn_resid(y)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        n_state = args.n_embd
        n_mlp = int(round(((8 / 3) * n_state) / 128) * 128)
        self.mlp_gate = nn.Linear(n_state, n_mlp, bias=False)
        self.mlp_linear = nn.Linear(n_state, n_mlp, bias=False)
        self.mlp_resid = nn.Linear(n_mlp, n_state, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp_resid(silu(self.mlp_gate(x)) * self.mlp_linear(x))


class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = CausalSelfAttention(args)
        self.attn_gain = ActGain((2 * args.n_layer) ** -0.5)
        self.mlp = MLP(args)
        self.mlp_gain = ActGain((2 * args.n_layer) ** -0.5)
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
        x = x + self.attn_gain(self.attn(rms_norm(x), cos, sin, mask, cache))
        x = x + self.mlp_gain(self.mlp(rms_norm(x)))
        x = x + self.embed_skip(e_x)
        return x


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.n_layer

        self.embed = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = [Block(args) for _ in range(args.n_layer)]
        self.lm_head = mx.zeros((args.vocab_size, args.n_embd))
        self.lm_head_gain = WeightGain()

        self._cos, self._sin = self._precompute_rope(args.max_seq_len)

    def _precompute_rope(self, seq_len: int) -> tuple[mx.array, mx.array]:
        dtype = mx.bfloat16 if self.args.dtype == "bfloat16" else mx.float16
        channel_range = mx.arange(0, self.args.head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (self.args.rope_base ** (channel_range / self.args.head_dim))
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = t[:, None] * inv_freq[None, :]
        cos = mx.cos(freqs).astype(dtype)[None, :, None, :]
        sin = mx.sin(freqs).astype(dtype)[None, :, None, :]
        return cos, sin

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            x = input_embeddings
        else:
            x = self.embed(inputs)

        if cache is None:
            cache = [None] * len(self.blocks)
            offset = 0
        else:
            offset = cache[0].offset

        L = x.shape[1]
        if offset + L > self._cos.shape[1]:
            raise ValueError(
                f"sequence length {offset + L} exceeds max_seq_len {self._cos.shape[1]}"
            )
        cos = self._cos[:, offset : offset + L]
        sin = self._sin[:, offset : offset + L]
        mask = create_attention_mask(x, cache[0])

        x = rms_norm(x)
        e_x = x
        for block, layer_cache in zip(self.blocks, cache):
            x = block(e_x, x, cos, sin, mask=mask, cache=layer_cache)
        x = rms_norm(x)

        lm_head = self.lm_head_gain(self.lm_head)
        return x @ lm_head.T

    @property
    def layers(self):
        return self.blocks

    def make_cache(self):
        return [KVCache() for _ in self.blocks]

    def sanitize(self, weights):
        return {k: v for k, v in weights.items() if k not in {"cos", "sin"}}
