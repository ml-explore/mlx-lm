# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import ArraysCache, KVCache, RotatingKVCache
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    layer_types: List[str]
    max_position_embeddings: Optional[int] = None
    hidden_activation: str = "gelu_pytorch_tanh"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    query_pre_attn_scalar: Optional[float] = None
    sliding_window: int = 512
    conv_L_cache: int = 3
    rope_theta: float = 1000000.0
    rope_local_base_freq: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    final_logit_softcapping: Optional[float] = None
    attn_logit_softcapping: Optional[float] = None
    use_bidirectional_attention: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.query_pre_attn_scalar is None:
            self.query_pre_attn_scalar = self.head_dim
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"Expected {self.num_hidden_layers} layer types, "
                f"got {len(self.layer_types)}."
            )
        if self.use_bidirectional_attention:
            raise ValueError("Gear bidirectional attention is not supported.")


class GearRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        weight = (1.0 + self.weight).astype(x.dtype)
        return mx.fast.rms_norm(x, weight, self.eps)


def _activation(name: str):
    if name in {"gelu_pytorch_tanh", "gelu_new", "gelu_fast"}:
        return nn.gelu_approx
    if name == "gelu":
        return nn.gelu
    if name in {"silu", "swish"}:
        return nn.silu
    raise ValueError(f"Unsupported Gear activation: {name}")


def _repeat_kv(x: mx.array, repeats: int) -> mx.array:
    if repeats == 1:
        return x
    B, n_kv_heads, L, D = x.shape
    x = mx.expand_dims(x, axis=2)
    x = mx.broadcast_to(x, (B, n_kv_heads, repeats, L, D))
    return x.reshape(B, n_kv_heads * repeats, L, D)


def _softcap_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    softcap: float,
    mask: Optional[Union[mx.array, str]],
) -> mx.array:
    repeats = queries.shape[1] // keys.shape[1]
    keys = _repeat_kv(keys, repeats)
    values = _repeat_kv(values, repeats)

    scores = (queries * scale) @ keys.swapaxes(-1, -2)
    scores = mx.tanh(scores / softcap) * softcap

    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask

    scores = mx.softmax(scores, axis=-1, precise=True)
    return scores @ values


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.hidden_activation = args.hidden_activation

    def __call__(self, x: mx.array) -> mx.array:
        act_fn = _activation(self.hidden_activation)
        return self.down_proj(act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = args.query_pre_attn_scalar**-0.5
        self.attn_logit_softcapping = args.attn_logit_softcapping

        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias
        )

        self.q_norm = GearRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = GearRMSNorm(self.head_dim, eps=args.rms_norm_eps)

        rope_theta = (
            args.rope_local_base_freq
            if args.layer_types[layer_idx] == "sliding_attention"
            else args.rope_theta
        )
        rope_scaling = (
            None
            if args.layer_types[layer_idx] == "sliding_attention"
            else args.rope_scaling
        )
        self.rope = initialize_rope(
            self.head_dim,
            rope_theta,
            traditional=False,
            scaling_config=rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[Union[mx.array, str]] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if self.attn_logit_softcapping:
            output = _softcap_attention(
                queries,
                keys,
                values,
                scale=self.scale,
                softcap=self.attn_logit_softcapping,
                mask=mask,
            )
        else:
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class ConvKVGatedMixer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.head_dim = args.head_dim
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.L_cache = args.conv_L_cache

        self.key_conv = nn.Conv1d(
            in_channels=self.kv_dim,
            out_channels=self.kv_dim,
            kernel_size=self.L_cache,
            groups=self.kv_dim,
            bias=args.attention_bias,
            padding=0,
        )
        self.value_conv = nn.Conv1d(
            in_channels=self.kv_dim,
            out_channels=self.kv_dim,
            kernel_size=self.L_cache,
            groups=self.kv_dim,
            bias=args.attention_bias,
            padding=0,
        )

        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(args.hidden_size, self.kv_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, self.kv_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=args.attention_bias
        )

        self.q_norm = GearRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = GearRMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def _apply_mask(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        if mask is None:
            return x
        return mx.where(mask[..., None], x, 0)

    def _conv(
        self,
        x: mx.array,
        conv: nn.Conv1d,
        cache: Optional[ArraysCache],
        cache_idx: int,
    ) -> mx.array:
        n_keep = self.L_cache - 1
        if cache is not None:
            state = cache[cache_idx]
            if state is None:
                state = mx.zeros((x.shape[0], n_keep, x.shape[-1]), dtype=x.dtype)
            conv_input = mx.concatenate([state, x], axis=1)
            if n_keep > 0:
                if cache.lengths is not None:
                    ends = mx.clip(cache.lengths, 0, x.shape[1])
                    positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                    cache[cache_idx] = mx.take_along_axis(conv_input, positions, axis=1)
                else:
                    cache[cache_idx] = mx.contiguous(conv_input[:, -n_keep:, :])
            else:
                cache[cache_idx] = mx.zeros((x.shape[0], 0, x.shape[-1]), dtype=x.dtype)
        else:
            conv_input = mx.pad(x, [(0, 0), (n_keep, 0), (0, 0)])
        return conv(conv_input)

    def _expand_kv(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape
        x = x.reshape(B, L, self.n_kv_heads, self.head_dim)
        x = mx.expand_dims(x, axis=3)
        x = mx.broadcast_to(
            x,
            (
                B,
                L,
                self.n_kv_heads,
                self.num_key_value_groups,
                self.head_dim,
            ),
        )
        return x.reshape(B, L, self.n_heads * self.head_dim)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[ArraysCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        x = self._apply_mask(x, mask)

        query = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        key = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        value = self.v_proj(x)

        query = self.q_norm(query).reshape(B, L, -1)
        key = self.k_norm(key).reshape(B, L, -1)

        key = self._conv(key, self.key_conv, cache, cache_idx=0)
        value = self._conv(value, self.value_conv, cache, cache_idx=1)
        if cache is not None:
            cache.advance(L)

        key = self._expand_kv(key)
        value = self._expand_kv(value)

        return self.o_proj(mx.sigmoid(query * key) * value)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attention_type = args.layer_types[layer_idx]
        self.is_attention_layer = self.attention_type != "conv_mixer"
        self.is_sliding = self.attention_type == "sliding_attention"

        if self.is_attention_layer:
            self.self_attn = Attention(args, layer_idx)
        else:
            self.local_mixer = ConvKVGatedMixer(args)

        self.mlp = MLP(args)
        self.input_layernorm = GearRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = GearRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.pre_feedforward_layernorm = GearRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_feedforward_layernorm = GearRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[Union[mx.array, str]] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        h = self.input_layernorm(x)

        if self.is_attention_layer:
            h = self.self_attn(h, mask=mask, cache=cache)
        else:
            h = self.local_mixer(h, mask=mask, cache=cache)

        h = residual + self.post_attention_layernorm(h)
        residual = h
        h = self.mlp(self.pre_feedforward_layernorm(h))
        return residual + self.post_feedforward_layernorm(h)


class GearModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window
        self.embed_scale = math.sqrt(args.hidden_size)

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = GearRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.fa_idx = self._first_layer_idx("full_attention")
        self.swa_idx = self._first_layer_idx("sliding_attention")
        self.conv_idx = self._first_layer_idx("conv_mixer")

    def _first_layer_idx(self, layer_type: str) -> Optional[int]:
        try:
            return self.layer_types.index(layer_type)
        except ValueError:
            return None

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs) * self.embed_scale

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = (
            create_attention_mask(h, cache[self.fa_idx])
            if self.fa_idx is not None
            else None
        )
        sliding_mask = (
            create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )
            if self.swa_idx is not None
            else None
        )
        conv_mask = (
            cache[self.conv_idx].make_mask(h.shape[1])
            if self.conv_idx is not None and cache[self.conv_idx] is not None
            else None
        )

        for layer, c in zip(self.layers, cache):
            if layer.attention_type == "full_attention":
                mask = full_mask
            elif layer.attention_type == "sliding_attention":
                mask = sliding_mask
            else:
                mask = conv_mask
            h = layer(h, mask=mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GearModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        if self.args.final_logit_softcapping:
            cap = self.args.final_logit_softcapping
            out = mx.tanh(out / cap) * cap
        return out

    def sanitize(self, weights):
        sanitized_weights = {}
        for name, param in weights.items():
            if "rotary_emb.inv_freq" in name or name.endswith("embed_scale"):
                continue
            if self.args.tie_word_embeddings and name == "lm_head.weight":
                continue
            if name.endswith(("key_conv.weight", "value_conv.weight")):
                if len(param.shape) == 3 and param.shape[1] == 1:
                    param = param.transpose(0, 2, 1)
            sanitized_weights[name] = param
        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.attention_type == "full_attention":
                caches.append(KVCache())
            elif layer.attention_type == "sliding_attention":
                caches.append(RotatingKVCache(max_size=self.model.sliding_window))
            else:
                caches.append(ArraysCache(size=2))
        return caches
