# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "nanochat"
    hidden_size: int = 1280
    num_hidden_layers: int = 20
    num_attention_heads: int = 10
    num_key_value_heads: int = 10
    vocab_size: int = 65536
    max_position_embeddings: int = 2048
    intermediate_size: int = 5120  # 4 * hidden_size
    rope_theta: float = 10000.0


def rms_norm(x):
    """Functional RMSNorm with no learnable parameters."""
    return mx.fast.rms_norm(x, None, 1e-5)


def apply_rotary_emb(x, offset, base=10000.0, freqs=None):
    """Apply RoPE with blocked layout.


    Args:
        x: Input tensor in (B, H, T, D) format
        offset: Position offset for KV caching
        base: RoPE base frequency (default 10000.0)
        freqs: Precomputed negated frequencies (optional)

    Returns:
        Tensor with RoPE applied, same shape as input
    """
    head_dim = x.shape[-1]

    if freqs is None:
        # Compute negated frequencies
        half_D = head_dim // 2
        freqs = -mx.exp(
            mx.arange(0.0, half_D, dtype=mx.float32) * (math.log(base) / half_D)
        )

    # Use traditional=False + negated freqs
    return mx.fast.rope(
        x,
        dims=head_dim,
        traditional=False,
        base=None,
        freqs=freqs,
        scale=1.0,
        offset=offset,
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.rope_theta = args.rope_theta

        self.c_q = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.c_k = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.c_v = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Precompute negated RoPE frequencies for awni's approach
        half_D = self.head_dim // 2
        self._rope_freqs = -mx.exp(
            mx.arange(0.0, half_D, dtype=mx.float32)
            * (math.log(self.rope_theta) / half_D)
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.c_q(x)
        keys = self.c_k(x)
        values = self.c_v(x)

        # Reshape to (B, L, H, D) then transpose to (B, H, L, D)
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # Apply RoPE using precomputed frequencies (expects B, H, T, D format)
        offset = cache.offset if cache is not None else 0
        queries = apply_rotary_emb(
            queries, offset=offset, base=self.rope_theta, freqs=self._rope_freqs
        )
        keys = apply_rotary_emb(
            keys, offset=offset, base=self.rope_theta, freqs=self._rope_freqs
        )

        # QK norm (critical feature of nanochat!)
        queries = rms_norm(queries)
        keys = rms_norm(keys)

        # Handle KV cache after transpose
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return self.c_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.c_fc = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.c_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # Critical: nanochat uses ReLU^2, not GELU!
        x = self.c_fc(x)
        x = nn.relu2(x)
        return self.c_proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = Attention(args)
        self.mlp = MLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # Pre-norm architecture with functional RMSNorm
        h = x + self.attn(rms_norm(x), mask=mask, cache=cache)
        out = h + self.mlp(rms_norm(h))
        return out


class NanoChatModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.h = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ) -> mx.array:
        h = self.wte(inputs)
        # Critical: norm after token embedding
        h = rms_norm(h)

        if cache is None:
            cache = [None] * len(self.h)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.h, cache):
            h = layer(h, mask=mask, cache=c)

        # Critical: final norm before lm_head
        h = rms_norm(h)

        return h


@partial(mx.compile, shapeless=True)
def softcap(logits, cap=15.0):
    return cap * mx.tanh(logits / cap)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = NanoChatModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ) -> mx.array:
        out = self.transformer(inputs, cache=cache)
        logits = self.lm_head(out)

        # Critical: logits softcap (nanochat uses softcap=15)
        logits = softcap(logits)

        return logits

    @property
    def layers(self):
        return self.transformer.h
