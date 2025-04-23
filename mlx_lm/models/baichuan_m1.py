# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import List, Any, Optional
import math

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
    attention_dropout: float
    rms_norm_eps: float
    pad_token_id: int
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
        self.num_heads = config.num_swa_attention_heads if self.is_swa and config.num_swa_attention_heads else config.num_attention_heads
        self.num_kv_heads = config.num_swa_key_value_heads if self.is_swa and config.num_swa_key_value_heads else config.num_key_value_heads

        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        assert self.head_dim * self.num_heads == self.hidden_size

        self.scale = self.head_dim ** -0.5

        self.W_pack = nn.Linear(config.hidden_size, self.hidden_size + 2 * self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

        self.conv_window = config.conv_window
        assert self.conv_window == 2
        self.conv_k = mx.zeros((1, 1, self.num_kv_heads, 1, self.conv_window))
        self.conv_v = mx.zeros((1, 1, self.num_kv_heads, 1, self.conv_window))
        self.last_k = None
        self.last_v = None

    def _custom_convolution(self, u, k_conv_weights):
        if k_conv_weights is None:
            raise ValueError("Convolution weights (conv_k or conv_v) not loaded.")
        B, H, L, D = u.shape
        W = self.conv_window

        u_padded = mx.pad(u, [(0, 0), (0, 0), (W - 1, 0), (0, 0)])

        outputs = []
        for w in range(W):
           u_slice = u_padded[:, :, w:w+L, :]
           raw_weight_slice = k_conv_weights[..., w]
           weight_slice = raw_weight_slice.squeeze(0).squeeze(-1).reshape(1, self.num_kv_heads, 1, 1)
           outputs.append(u_slice * weight_slice)

        v = mx.sum(mx.stack(outputs, axis=0), axis=0)
        return v

    def __call__(self, x: mx.array, mask: mx.array = None, cache: Any = None) -> mx.array:
        B, L, D = x.shape

        proj = self.W_pack(x)
        q_proj_end = self.hidden_size
        k_proj_end = q_proj_end + self.num_kv_heads * self.head_dim
        q = proj[:, :, :q_proj_end]
        k = proj[:, :, q_proj_end:k_proj_end]
        v = proj[:, :, k_proj_end:]

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
                 self.last_k = mx.zeros((B, self.num_kv_heads, 1, self.head_dim), dtype=k_unrotated.dtype)
                 self.last_v = mx.zeros((B, self.num_kv_heads, 1, self.head_dim), dtype=v_unrotated.dtype)

            k_conv = self._custom_convolution(k_unrotated, self.conv_k)
            v_conv = self._custom_convolution(v_unrotated, self.conv_v)
            k_rotated_conv = self.rope(k_conv, offset=offset)
            k_final, v_final = cache.update_and_fetch(k_rotated_conv, v_conv)

        elif cache is not None and cache.offset > 0:
            if self.last_k is None or self.last_v is None:
                raise ValueError("last_k or last_v is None during decode step. Ensure prefill occurred.")
            if L != 1:
                raise ValueError(f"Decode step expected sequence length 1, but got {L}")

            current_k_conv = self.conv_k[..., 0:1].squeeze(0).squeeze(-1)[:,:,None,:] * self.last_k + \
                             self.conv_k[..., 1:2].squeeze(0).squeeze(-1)[:,:,None,:] * k_unrotated
            current_v_conv = self.conv_v[..., 0:1].squeeze(0).squeeze(-1)[:,:,None,:] * self.last_v + \
                             self.conv_v[..., 1:2].squeeze(0).squeeze(-1)[:,:,None,:] * v_unrotated

            current_k_rotated_conv = self.rope(current_k_conv, offset=offset)
            k_final, v_final = cache.update_and_fetch(current_k_rotated_conv, current_v_conv)
            self.last_k = k_unrotated
            self.last_v = v_unrotated

        else:
             k_conv = self._custom_convolution(k_unrotated, self.conv_k)
             v_conv = self._custom_convolution(v_unrotated, self.conv_v)
             k_final = self.rope(k_conv, offset=offset)
             v_final = v_conv

        out = scaled_dot_product_attention(q, k_final, v_final, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)

class MLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, mask: mx.array = None, cache: Any = None) -> mx.array:
        h = self.input_layernorm(x)
        r = self.self_attn(h, mask, cache)
        x = x + r
        h2 = self.post_attention_layernorm(x)
        m = self.mlp(h2)
        return x + m

class BaichuanModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, inputs: mx.array, mask: mx.array = None, cache: Any = None) -> mx.array:
        x = self.embed_tokens(inputs)
        if mask is None:
            mask = create_attention_mask(x, cache)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            x = layer(x, mask, c)
        return self.norm(x)

class NormHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.weight = mx.random.normal((vocab_size, hidden_size))
        scale = math.sqrt(6.0 / (vocab_size + hidden_size))
        self.weight = self.weight * scale

    def __call__(self, hidden_states: mx.array) -> mx.array:
        norm_factor = mx.linalg.norm(self.weight, axis=-1, keepdims=True)
        norm_weight = self.weight / (norm_factor + 1e-7)
        return hidden_states @ norm_weight.T

class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = BaichuanModel(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = NormHead(config.hidden_size, config.vocab_size)

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
        return weights

    def __call__(self, inputs: mx.array, mask: mx.array = None, cache: Any = None) -> mx.array:
        outputs = self.model(inputs, mask, cache)
        return self.lm_head(outputs)

    @property
    def layers(self) -> List[nn.Module]:
        return self.model.layers 