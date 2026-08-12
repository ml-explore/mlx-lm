# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelArgs:
    vocab_size: int
    hidden_size: int
    head_dim: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rope_theta: float = 10_000
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    layer_norm: str = "pre"  # "pre" or "post"
    init_std: float = 0.02
    rope_scaling_factor: float = 1.0
    original_max_position_embeddings: int = 8192


class YarnRoPE(nn.Module):
    def __init__(
        self,
        dims,
        traditional=False,
        base=500_000,
        scaling_factor=8.0,
        original_max_position_embeddings=8192,
        beta_fast=32,
        beta_slow=1,
    ):
        super().__init__()

        def find_correction_dim(num_rotations):
            return (
                dims
                * math.log(
                    original_max_position_embeddings / (num_rotations * 2 * math.pi)
                )
            ) / (2 * math.log(base))

        def find_correction_range():
            low = math.floor(find_correction_dim(beta_fast))
            high = math.ceil(find_correction_dim(beta_slow))
            return max(low, 0), min(high, dims - 1)

        def linear_ramp_mask(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001  # Prevent singularity
            ramp = (mx.arange(dim, dtype=mx.float32) - min_val) / (max_val - min_val)
            return mx.clip(ramp, 0, 1)

        self.mscale = 0.1 * math.log(scaling_factor) + 1.0
        freq_extra = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        freq_inter = scaling_factor * freq_extra
        low, high = find_correction_range()
        freq_mask = 1.0 - linear_ramp_mask(low, high, dims // 2)
        self._freqs = (freq_inter * freq_extra) / (
            freq_inter * freq_mask + freq_extra * (1 - freq_mask)
        )
        self.dims = dims
        self.traditional = traditional

    def __call__(self, x, offset=0):
        x = self.mscale * x
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        head_dim = args.head_dim
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        if args.rope_scaling_factor > 1:
            self.rope = YarnRoPE(
                head_dim,
                base=args.rope_theta,
                scaling_factor=args.rope_scaling_factor,
                original_max_position_embeddings=args.original_max_position_embeddings,
            )
        else:
            self.rope = nn.RoPE(head_dim, base=args.rope_theta, traditional=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1))
        keys = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1))
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.post_norm = args.layer_norm == "post"
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.post_norm:
            h = x + self.attention_layernorm(self.self_attn(x, mask, cache))
            return h + self.mlp_layernorm(self.mlp(h))
        h = x + self.self_attn(self.attention_layernorm(x), mask, cache)
        return h + self.mlp(self.mlp_layernorm(h))


class LanguageModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
    ):
        h = self.embed_tokens(inputs)
        if mask is None and h.shape[1] > 1:
            mask = "causal"
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model = LanguageModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, mask=mask, cache=cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def init_weights(self):
        std = self.args.init_std
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight = mx.random.normal(shape=module.weight.shape) * std
