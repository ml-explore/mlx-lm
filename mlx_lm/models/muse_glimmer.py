# Copyright © 2026 Apple Inc.
# Muse Glimmer (Meta, 2026-08-10) — text-tower support for mlx-lm.
#
# Glimmer is multimodal; this implements the language model only, which is what
# text/agentic inference needs. Architecture verified against Meta's reference
# transformers implementation (transformers 5.15 modeling_muse_glimmer.py).
#
# Glimmer-specific details (vs a standard Gemma/Llama block):
#   * gated attention output: attn * sigmoid(gate_proj(x)) before o_proj
#   * scaleless RMSNorm on Q and K, then Q scaled by qk_scale_factor
#   * [local, local, local, global] layer pattern; RoPE on local layers only,
#     global layers run NoPE (no positional embedding)
#   * scaleless RMSNorm applied to the token embeddings
#   * "centered" RMSNorm (output = norm(x) * (1 + weight)) on the four
#     per-block norms; standard RMSNorm on the final norm
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "muse_glimmer"
    hidden_size: int = 6656
    num_hidden_layers: int = 52
    intermediate_size: int = 19968
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    vocab_size: int = 202048
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    sliding_window: int = 2048
    qk_scale_factor: float = 3.87
    hidden_activation: str = "silu"
    max_position_embeddings: int = 131072
    rope_theta: float = 500000.0
    rope_parameters: Optional[dict] = None
    layer_types: Optional[List[str]] = None
    layer_rope_theta: Optional[List[float]] = None
    tie_word_embeddings: bool = False
    final_logit_softcapping: float = 20.0
    output_multiplier: float = 0.19611613513818404

    @classmethod
    def from_dict(cls, params):
        # Meta's original checkpoint nests the language-model fields under
        # "text_config"; the mlx-community conversions flatten them. Accept both.
        if "text_config" in params:
            merged = dict(params)
            merged.update(params["text_config"])
            params = merged
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in params.items() if k in fields})

    def __post_init__(self):
        # rope_theta may live under rope_parameters (as in Meta's config)
        if self.rope_parameters and "rope_theta" in self.rope_parameters:
            self.rope_theta = self.rope_parameters["rope_theta"]
        if self.layer_types is None:
            # default [local, local, local, global] repeating
            pat = ["sliding_attention"] * 3 + ["full_attention"]
            self.layer_types = (pat * (self.num_hidden_layers // 4 + 1))[
                : self.num_hidden_layers
            ]
        if self.layer_rope_theta is None:
            self.layer_rope_theta = [
                0 if t == "full_attention" else self.rope_theta
                for t in self.layer_types
            ]


class CenteredRMSNorm(nn.Module):
    """RMSNorm with a zero-centered scale: out = norm(x) * (1 + weight)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.qk_scale_factor = args.qk_scale_factor

        self.layer_type = args.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.use_rope = bool(args.layer_rope_theta[layer_idx])

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)

        self.eps = args.rms_norm_eps
        if self.use_rope:
            self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        # Scaleless QK-norm over head_dim; Q additionally scaled.
        q = mx.fast.rms_norm(q, None, self.eps) * self.qk_scale_factor
        k = mx.fast.rms_norm(k, None, self.eps)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if self.use_rope:
            offset = cache.offset if cache is not None else 0
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Glimmer's output gate.
        out = out * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = CenteredRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = CenteredRMSNorm(
            args.hidden_size, args.post_norm_eps
        )
        self.pre_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.rms_norm_eps
        )
        self.post_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.post_norm_eps
        )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache)
        h = self.post_attention_layernorm(h)
        h = residual + h

        residual = h
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        return residual + h


class MuseGlimmerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_eps = args.rms_norm_eps
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.window = args.sliding_window

    def _masks(self, h, cache):
        made, out = {}, []
        for layer, c in zip(self.layers, cache):
            t = layer.self_attn.layer_type
            if t not in made:
                if t == "sliding_attention":
                    made[t] = create_attention_mask(h, c, window_size=self.window)
                else:
                    made[t] = create_attention_mask(h, c)
            out.append(made[t])
        return out

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)
        # Scaleless RMSNorm on the embeddings (not Gemma's sqrt scaling).
        h = mx.fast.rms_norm(h, None, self.embed_eps)

        if cache is None:
            cache = [None] * len(self.layers)

        masks = self._masks(h, cache)
        for layer, c, mask in zip(self.layers, cache, masks):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MuseGlimmerModel(args)
        self.tie_word_embeddings = args.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache=cache)
        if self.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        out = out * self.args.output_multiplier
        cap = self.args.final_logit_softcapping
        return mx.tanh(out / cap) * cap

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            # Drop the vision tower — this is a text-only port.
            if k.startswith(("vision_tower", "vision_adapter", "vision_projection")):
                continue
            # Meta/MLX nest the text tower under language_model.*
            if k.startswith("language_model.model."):
                k = "model." + k[len("language_model.model.") :]
            elif k.startswith("language_model.lm_head."):
                k = "lm_head." + k[len("language_model.lm_head.") :]
            elif k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model.") :]
            out[k] = v
        return out

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def make_cache(self):
        caches = []
        for lt in self.args.layer_types:
            if lt == "sliding_attention":
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches
