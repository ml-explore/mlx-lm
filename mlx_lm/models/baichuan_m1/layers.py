# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.activations import swiglu
from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.models.baichuan_m1.config import ModelArgs

__all__ = [
    "Attention",
    "DecoderLayer",
    "MLP",
]


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            raise ValueError("Layer index must be provided to Attention module.")

        self.is_swa = layer_idx in config.sliding_window_layers
        self.num_heads = (
            config.num_swa_attention_heads
            if self.is_swa and config.num_swa_attention_heads
            else config.num_attention_heads
        )
        self.num_kv_heads = (
            config.num_swa_key_value_heads
            if self.is_swa and config.num_swa_key_value_heads
            else config.num_key_value_heads
        )

        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        assert self.head_dim * self.num_heads == self.hidden_size

        self.scale = self.head_dim**-0.5

        self.W_pack = nn.Linear(
            config.hidden_size,
            self.hidden_size + 2 * self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

        self.conv_window = config.conv_window
        assert self.conv_window == 2
        self.conv_k = mx.zeros((1, 1, self.num_kv_heads, 1, self.conv_window))
        self.conv_v = mx.zeros((1, 1, self.num_kv_heads, 1, self.conv_window))

    def _custom_convolution(self, u, weights, state=None):
        B, H, L, D = u.shape
        weights = weights.reshape((1, H, self.conv_window, 1, 1))
        w0 = weights[:, :, 0]
        w1 = weights[:, :, 1]
        if state is None:
            state = mx.zeros((B, H, 1, D), u.dtype)
        if L > 1:
            u_prev = mx.concatenate([state, u[:, :, :-1]], axis=2)
        else:
            u_prev = state
        return u_prev * w0 + u * w1

    def __call__(
        self, x: mx.array, mask: mx.array = None, cache: Any = None
    ) -> mx.array:
        B, L, D = x.shape

        proj = self.W_pack(x)
        q, k, v = mx.split(proj, (D, D + self.num_kv_heads * self.head_dim), axis=-1)

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is None:
            cache = (None, None)

        if cache[0] is not None:
            offset = cache[1].offset
            last_k, last_v = cache[0][0], cache[0][1]
        else:
            offset = 0
            last_k, last_v = None, None

        k_init = k
        v_init = v
        k = self._custom_convolution(k, self.conv_k, state=last_k)
        v = self._custom_convolution(v, self.conv_v, state=last_v)
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache[0] is not None:
            k, v = cache[1].update_and_fetch(k, v)
            if L > 0:
                cache[0][0] = k_init[:, :, -1:, :]
                cache[0][1] = v_init[:, :, -1:, :]

        out = scaled_dot_product_attention(
            q, k, v, cache=cache[1], scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self, x: mx.array, mask: mx.array = None, cache: Any = None
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        x = x + r
        r = self.mlp(self.post_attention_layernorm(x))
        return x + r
