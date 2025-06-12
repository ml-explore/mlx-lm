# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from functools import partial
from itertools import accumulate
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import ConcatenateKVCache, KVCache
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_kv_reuse_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_dim_scale_factor: float = 3.25
    rope_theta: float = 50000
    rms_norm_eps: float = 1e-5


class FusedLoRALinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: list[int],
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        super().__init__()

        self.linear = FusedLinear(input_dims, output_dims)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale

        scale = 1 / math.sqrt(input_dims)
        self.lora_a = [
            mx.random.uniform(low=-scale, high=scale, shape=(input_dims, r))
            for _ in output_dims
        ]
        self.lora_b = [mx.zeros((r, od)) for od in output_dims]

    def fuse(self, de_quantize: bool = False):
        linear = self.linear
        weight = linear.weight
        is_quantized = isinstance(linear, FusedQuantizedLinear)

        # Use the same type as the linear weight if not quantized
        dtype = weight.dtype

        if is_quantized:
            dtype = linear.scales.dtype
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                linear.group_size,
                linear.bits,
            )

        input_dims = weight.shape[-1]
        output_dims = linear.output_dims
        fused_linear = FusedLinear(input_dims, output_dims)
        fused_linear.weight = weight
        deltas = [
            ((self.scale * b.T) @ a.T).astype(dtype)
            for a, b in zip(self.lora_a, self.lora_b)
        ]
        delta = mx.concatenate(deltas, axis=0)
        fused_linear.weight = weight + delta

        if is_quantized and not de_quantize:
            fused_linear = fused_linear.to_quantized(linear.group_size, linear.bits)

        return fused_linear

    def __call__(self, x):
        dt = x.dtype
        y = self.linear(x)
        x = self.dropout(x)
        z = [(x @ a) @ b for a, b in zip(self.lora_a, self.lora_b)]
        return tuple(yi + (self.scale * zi).astype(dt) for yi, zi in zip(y, z))


class FusedQuantizedLinear(nn.QuantizedLinear):
    def __init__(self, input_dims, output_dims, group_size: int = 64, bits: int = 4):
        *indices, output_dims = accumulate(output_dims)
        self.indices = indices
        super().__init__(
            input_dims, output_dims, bias=False, group_size=group_size, bits=bits
        )

    @property
    def input_dims(self):
        return self.scales.shape[-1] * self.group_size

    @property
    def output_dims(self):
        indices = [0] + self.indices + [self.weight.shape[0]]
        return [indices[i] - indices[i - 1] for i in range(1, len(indices))]

    def __call__(self, x):
        x = super().__call__(x)
        return x.split(self.indices, axis=-1)

    def to_lora(self, r: int = 8, dropout: float = 0.0, scale: float = 20.0):
        lora_lin = FusedLoRALinear(self.input_dims, self.output_dims, r, dropout, scale)
        lora_lin.linear = self
        return lora_lin


class FusedLinear(nn.Linear):
    def __init__(self, input_dims, output_dims):
        *indices, output_dims = accumulate(output_dims)
        self.indices = indices
        super().__init__(input_dims, output_dims, bias=False)

    @property
    def input_dims(self):
        return self.weight.shape[-1]

    @property
    def output_dims(self):
        indices = [0] + self.indices + [self.weight.shape[0]]
        return [indices[i] - indices[i - 1] for i in range(1, len(indices))]

    def __call__(self, x):
        x = super().__call__(x)
        return x.split(self.indices, axis=-1)

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        input_dims = self.input_dims
        output_dims = self.output_dims
        ql = FusedQuantizedLinear(input_dims, output_dims, group_size, bits)
        ql.weight, ql.scales, ql.biases = mx.quantize(self.weight, group_size, bits)

        return ql

    def to_lora(self, r: int = 8, dropout: float = 0.0, scale: float = 20.0):
        lora_lin = FusedLoRALinear(self.input_dims, self.output_dims, r, dropout, scale)
        lora_lin.linear = self
        return lora_lin


@partial(mx.compile, shapeless=True)
def fake_8bit_quant(x, scale):
    dt = x.dtype
    x = x.astype(mx.float32)
    x = (x / scale).round()
    x = mx.clip(x, -128, 127)
    return (x * scale).astype(dt)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_dim
        self.n_heads = n_heads = args.num_heads
        self.n_kv_heads = n_kv_heads = args.num_kv_heads
        self.head_dim = head_dim = args.hidden_dim // n_heads
        self.scale = head_dim**-0.5

        qkv_dim = (n_heads + 2 * n_kv_heads) * head_dim
        self.qkv_proj = FusedLinear(
            dim, [n_heads * head_dim] + 2 * [n_kv_heads * head_dim]
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            True,
        )
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.quant_key_scale = mx.array(1.0)
        self.quant_value_scale = mx.array(1.0)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        # Get the queries, keys and values
        queries, keys, values = self.qkv_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.q_norm(self.rope(queries, offset=cache.offset))
            keys = self.k_norm(self.rope(keys, offset=cache.offset))
            keys = fake_8bit_quant(keys, self.quant_key_scale)
            values = fake_8bit_quant(values, self.quant_value_scale)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.q_norm(self.rope(queries))
            keys = self.k_norm(self.rope(keys))
            keys = fake_8bit_quant(keys, self.quant_key_scale)
            values = fake_8bit_quant(values, self.quant_value_scale)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class KVReuseAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_dim
        self.n_heads = n_heads = args.num_heads
        self.head_dim = head_dim = args.hidden_dim // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            True,
        )
        self.q_norm = nn.RMSNorm(head_dim)

    def __call__(
        self,
        x: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, D = x.shape
        _, _, S, _ = keys.shape

        queries = self.q_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        queries = self.q_norm(self.rope(queries, offset=S - L))

        output = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


@partial(mx.compile, shapeless=True)
def _swiglu(g, x):
    return nn.silu(g) * x


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_dim
        hidden_dim = int(dim * args.hidden_dim_scale_factor)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        g = self.gate_proj(x)
        x = self.up_proj(x)
        return self.down_proj(_swiglu(g, x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_dim, eps=args.rms_norm_eps
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
        out = h + r
        return out


class KVReuseTransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = KVReuseAttention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_dim, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), keys, values, mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class AFMModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size

        self.embedding = nn.Embedding(args.vocab_size, args.hidden_dim)
        self.layers = [
            TransformerBlock(args)
            for _ in range(args.num_layers - args.num_kv_reuse_layers)
        ]
        self.kv_reuse_layers = [
            KVReuseTransformerBlock(args) for _ in range(args.num_kv_reuse_layers)
        ]
        self.output_norm = nn.RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.embedding(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)
            cache[-1] = ConcatenateKVCache()

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        keys, values = cache[-1].state
        for layer in self.kv_reuse_layers:
            h = layer(h, keys, values, mask)

        return self.output_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = AFMModel(args)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)
        out = self.model.embedding.as_linear(out)
        return out

    def make_cache(self):
        return [KVCache() for _ in range(len(self.model.layers))]

    @property
    def layers(self):
        return self.model.layers + self.model.kv_reuse_layers
