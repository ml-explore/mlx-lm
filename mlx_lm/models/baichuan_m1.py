# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import CacheList, KVCache, MambaCache, RotatingKVCache


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

        if cache is not None:
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

        if cache is not None:
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
            if cache is not None:
                c = [cache[0][1]]
            mask = create_attention_mask(x, c)
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
            conv_cache = MambaCache()
            if is_swa:
                kv_cache = RotatingKVCache(max_size=self.config.sliding_window)
            else:
                kv_cache = KVCache()
            caches.append(CacheList(conv_cache, kv_cache))
        return caches

    def sanitize(self, weights: dict) -> dict:
        is_quantized = "lm_head.scales" in weights
        if not is_quantized and "lm_head.weight" in weights:
            w = weights["lm_head.weight"]
            dtype = w.dtype
            w = w.astype(mx.float32)
            norm = mx.linalg.norm(w, axis=-1, keepdims=True)
            w = (w / (norm + 1e-7)).astype(dtype)
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
