# Copyright © 2023-2024 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rope_theta: float
    sliding_window: int
    sliding_window_layers: List[int]
    conv_window: int
    rms_norm_eps: float
    model_type: str = "baichuan_m1"
    num_swa_attention_heads: Optional[int] = None
    num_swa_key_value_heads: Optional[int] = None
    tie_word_embeddings: bool = False


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
        self.last_k = None
        self.last_v = None

    def _custom_convolution(self, u, weights):
        B, H, L, D = u.shape
        W = self.conv_window
        weights = weights.reshape((1, H, W, 1, 1))
        u_padded = mx.pad(u, [(0, 0), (0, 0), (W - 1, 0), (0, 0)])
        Lp = L + (W - 1)
        u_unfolded = mx.as_strided(
            u_padded,
            (B, H, W, L, D),
            (H * Lp * D, Lp * D, D, D, 1),
        )
        return mx.sum(u_unfolded * weights, axis=2)

    def __call__(
        self, x: mx.array, mask: mx.array = None, cache: Any = None
    ) -> mx.array:
        B, L, D = x.shape

        proj = self.W_pack(x)
        q, k, v = mx.split(proj, (D, D + self.num_kv_heads * self.head_dim), axis=-1)

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k_rope = self.rope(k, offset=offset)
        k_unrotated = k
        v_unrotated = v

        if cache is not None and cache.offset == 0:
            if L > 0:
                self.last_k = k_unrotated[:, :, -1:, :]
                self.last_v = v_unrotated[:, :, -1:, :]
            else:
                self.last_k = mx.zeros(
                    (B, self.num_kv_heads, 1, self.head_dim), dtype=k_unrotated.dtype
                )
                self.last_v = mx.zeros(
                    (B, self.num_kv_heads, 1, self.head_dim), dtype=v_unrotated.dtype
                )

            k_conv = self._custom_convolution(k_unrotated, self.conv_k)
            v_conv = self._custom_convolution(v_unrotated, self.conv_v)
            k_rotated_conv = self.rope(k_conv, offset=offset)
            k_final, v_final = cache.update_and_fetch(k_rotated_conv, v_conv)

        elif cache is not None and cache.offset > 0:
            if self.last_k is None or self.last_v is None:
                raise ValueError("last_k or last_v is None. Ensure prefill occurred.")

            # Need to handle L > 1 case (continuation prefill) iteratively
            k_conv_list = []
            v_conv_list = []
            # Use temporary variables to hold the 'last' state during the loop
            temp_last_k = self.last_k
            temp_last_v = self.last_v

            # Iterate through the input sequence (L tokens)
            for i in range(L):
                # Get current token's unrotated k/v
                current_k_unrotated = k_unrotated[:, :, i : i + 1, :]
                current_v_unrotated = v_unrotated[:, :, i : i + 1, :]

                # Convolve using previous step's k/v (temp_last_k/v)
                # Reshape weights for broadcasting: (1, H, 1, 1)
                w0_k = self.conv_k[..., 0].reshape(1, self.num_kv_heads, 1, 1)
                w1_k = self.conv_k[..., 1].reshape(1, self.num_kv_heads, 1, 1)
                w0_v = self.conv_v[..., 0].reshape(1, self.num_kv_heads, 1, 1)
                w1_v = self.conv_v[..., 1].reshape(1, self.num_kv_heads, 1, 1)

                k_conv_token = w0_k * temp_last_k + w1_k * current_k_unrotated
                v_conv_token = w0_v * temp_last_v + w1_v * current_v_unrotated

                k_conv_list.append(k_conv_token)
                v_conv_list.append(v_conv_token)

                # Update the 'last' state for the *next* token in this loop
                temp_last_k = current_k_unrotated
                temp_last_v = current_v_unrotated

            # Concatenate results for the whole sequence
            k_conv = mx.concatenate(k_conv_list, axis=2)
            v_conv = mx.concatenate(v_conv_list, axis=2)

            # Apply RoPE *after* convolution for the value going into cache
            k_rotated_conv = self.rope(k_conv, offset=offset)

            # Update cache with the *convolved* and *rotated* current K/V sequence
            k_final, v_final = cache.update_and_fetch(k_rotated_conv, v_conv)

            # Update the instance's last K/V state for the *next* call to __call__
            # Use the *last* unrotated values from the *input* sequence
            self.last_k = k_unrotated[:, :, -1:, :]
            self.last_v = v_unrotated[:, :, -1:, :]

        else:
            k_conv = self._custom_convolution(k_unrotated, self.conv_k)
            v_conv = self._custom_convolution(v_unrotated, self.conv_v)
            k_final = self.rope(k_conv, offset=offset)
            v_final = v_conv

        out = scaled_dot_product_attention(
            q, k_final, v_final, cache=cache, scale=self.scale, mask=mask
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
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


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


class BaichuanModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self, inputs: mx.array, mask: mx.array = None, cache: Any = None
    ) -> mx.array:
        x = self.embed_tokens(inputs)
        if mask is None:
            mask = create_attention_mask(x, cache)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            x = layer(x, mask, c)
        return self.norm(x)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = BaichuanModel(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> List[Any]:
        caches = []
        for i, layer in enumerate(self.model.layers):
            is_swa = i in self.config.sliding_window_layers
            if is_swa:
                caches.append(RotatingKVCache(max_size=self.config.sliding_window))
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: dict) -> dict:
        if self.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        else:
            # Pre-normalize the lm_head
            w = weights["lm_head.weight"]
            w = w / (mx.linalg.norm(w, axis=-1, keepdims=True) + 1e-7)
            weights["lm_head.weight"] = w
        return weights

    def __call__(
        self, inputs: mx.array, mask: mx.array = None, cache: Any = None
    ) -> mx.array:
        outputs = self.model(inputs, mask, cache)
        return self.lm_head(outputs)

    @property
    def layers(self) -> List[nn.Module]:
        return self.model.layers
