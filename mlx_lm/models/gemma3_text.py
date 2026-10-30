# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .prefill import prefill_layers
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 1152
    num_hidden_layers: int = 26
    intermediate_size: int = 6912
    num_attention_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1.0e-6
    vocab_size: int = 262144
    num_key_value_heads: int = 1
    rope_theta: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    query_pre_attn_scalar: float = 256
    sliding_window: int = 512
    sliding_window_pattern: int = 6
    max_position_embeddings: int = 32768
    rope_scaling: Dict = None


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.repeats = n_heads // n_kv_heads
        self.head_dim = head_dim = args.head_dim
        self.layer_idx = layer_idx

        self.scale = args.query_pre_attn_scalar**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = RMSNorm(dims=head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(dims=head_dim, eps=args.rms_norm_eps)
        self.is_sliding = (layer_idx + 1) % args.sliding_window_pattern != 0

        if self.is_sliding:
            self.rope = initialize_rope(
                dims=head_dim,
                base=args.rope_local_base_freq,
                traditional=False,
            )
        else:
            self.rope = initialize_rope(
                dims=head_dim,
                base=args.rope_theta,
                traditional=False,
                max_position_embeddings=args.max_position_embeddings,
                scaling_config=args.rope_scaling,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        query_length: Optional[int] = None,
        position_offset: int = 0,
    ) -> mx.array:
        B, L, _ = x.shape
        query_length = query_length or L
        queries = self.q_proj(x[:, -query_length:])
        keys, values = self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, query_length, self.n_heads, -1).transpose(
            0, 2, 1, 3
        )

        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        if cache is not None:
            offset = cache.offset + position_offset
            queries = self.rope(queries, offset=offset + L - query_length)
            keys = self.rope(keys, offset=offset)
            keys, values = cache.update_and_fetch(keys, values)
            if position_offset:
                cache.offset += position_offset
        else:
            queries = self.rope(queries, offset=L - query_length)
            keys = self.rope(keys)

        if isinstance(mask, mx.array):
            mask = mask[..., -query_length:, :]

        # Sliding window
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, query_length, -1)
        return self.o_proj(output)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


@partial(mx.compile, shapeless=True)
def clip_residual(x, y):
    if x.dtype != mx.float16:
        return x + y
    bound = mx.finfo(mx.float16).max
    return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -bound, bound).astype(
        mx.float16
    )


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        query_length: Optional[int] = None,
        position_offset: int = 0,
    ) -> mx.array:
        r = self.self_attn(
            self.input_layernorm(x), mask, cache, query_length, position_offset
        )
        if query_length is not None:
            x = x[:, -query_length:]
        h = clip_residual(x, self.post_attention_layernorm(r))
        r = self.mlp(self.pre_feedforward_layernorm(h))
        out = clip_residual(h, self.post_feedforward_layernorm(r))
        return out


class Gemma3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.window_size = args.sliding_window
        self.sliding_window_pattern = args.sliding_window_pattern
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_idx=layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        prefill: bool = False,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        h *= mx.array(self.args.hidden_size**0.5, mx.bfloat16).astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        if prefill:
            layer_types = [
                "sliding_attention" if layer.self_attn.is_sliding else "full_attention"
                for layer in self.layers
            ]
            out = prefill_layers(h, self.layers, layer_types, self.window_size, cache)
            if out is not None:
                return self.norm(out)

        global_mask = create_attention_mask(h, cache[self.sliding_window_pattern - 1])

        if self.sliding_window_pattern > 1:
            sliding_window_mask = create_attention_mask(
                h,
                cache[0],
                window_size=self.window_size,
            )
        else:
            sliding_window_mask = None
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            is_global = (
                i % self.sliding_window_pattern == self.sliding_window_pattern - 1
            )
            mask = global_mask if is_global else sliding_window_mask
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Gemma3Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.tie_word_embeddings = False

    def prefill(self, inputs, cache=None, input_embeddings=None):
        """Update prompt caches without returning logits."""
        self.model(inputs, cache, input_embeddings, prefill=True)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if "lm_head.weight" not in weights:
            self.tie_word_embeddings = True
            self.pop("lm_head")
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for i in range(self.args.num_hidden_layers):
            if (
                i % self.args.sliding_window_pattern
                == self.args.sliding_window_pattern - 1
            ):
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches
