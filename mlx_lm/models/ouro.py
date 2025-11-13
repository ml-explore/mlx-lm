# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "ouro"
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    intermediate_size: int = 5632
    vocab_size: int = 49152
    head_dim: int = 128
    max_position_embeddings: int = 65536
    rope_theta: float = 1000000.0
    rope_scaling: Optional[dict] = None
    rms_norm_eps: float = 1e-6
    total_ut_steps: int = 4
    early_exit_step: Optional[int] = None
    early_exit_threshold: Optional[float] = None
    tie_word_embeddings: bool = False


@partial(mx.compile, shapeless=True)
def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    return nn.silu(gate) * up


class OuroAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        keys = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)

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

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class OuroMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class OuroDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = OuroAttention(args, layer_idx)
        self.mlp = OuroMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.input_layernorm_2 = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_attention_layernorm_2 = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache)
        h = self.input_layernorm_2(h)
        h = residual + h

        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        h = self.post_attention_layernorm_2(h)
        return residual + h


class OuroModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.total_ut_steps = args.total_ut_steps
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            OuroDecoderLayer(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.early_exit_gate = nn.Linear(args.hidden_size, 1, bias=True)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[List[KVCache]] = None,
        use_weighted_exit: bool = False,
        exit_at_step: Optional[int] = None,
        exit_threshold: Optional[float] = None,
    ) -> mx.array:
        hs, gates = self._forward_ut_steps(x, cache)

        exit_at_step = (
            exit_at_step if exit_at_step is not None else self.args.early_exit_step
        )
        if exit_at_step is not None and 0 <= exit_at_step < len(hs):
            return hs[exit_at_step]

        exit_threshold = (
            exit_threshold
            if exit_threshold is not None
            else self.args.early_exit_threshold
        )

        if exit_threshold is not None or use_weighted_exit:
            exit_pdf = self._compute_exit_distribution(gates)

            if exit_threshold is not None:
                return self._threshold_exit(hs, exit_pdf, exit_threshold)

            if use_weighted_exit:
                return self._weighted_exit(hs, exit_pdf)

        return hs[-1]

    def _forward_ut_steps(self, x: mx.array, cache: Optional[List[KVCache]] = None):
        h = self.embed_tokens(x)

        num_layers = len(self.layers)
        if cache is None:
            cache = [None] * (self.total_ut_steps * num_layers)

        mask = create_attention_mask(h, cache[0])
        hs = []
        gates = []

        for current_ut in range(self.total_ut_steps):
            for layer_idx, layer in enumerate(self.layers):
                cache_idx = current_ut * num_layers + layer_idx
                h = layer(h, mask, cache[cache_idx])

            h = self.norm(h)
            hs.append(h)
            gates.append(self.early_exit_gate(h))

        return hs, gates

    def _compute_exit_distribution(self, gates: List[mx.array]) -> mx.array:
        pdf = []
        remaining = mx.ones_like(gates[0].squeeze(-1))

        for gate in gates[:-1]:
            lambda_i = mx.sigmoid(gate.squeeze(-1))
            p_i = lambda_i * remaining
            remaining = remaining * (1.0 - lambda_i)
            pdf.append(p_i)

        pdf.append(remaining)
        return mx.stack(pdf, axis=-1)

    def _weighted_exit(
        self,
        hs: List[mx.array],
        exit_pdf: mx.array,
    ) -> mx.array:
        hs = mx.stack(hs, axis=2)
        weights = mx.expand_dims(exit_pdf, axis=-1)
        return mx.sum(hs * weights, axis=2)

    def _threshold_exit(
        self,
        hs: List[mx.array],
        exit_pdf: mx.array,
        exit_threshold: float,
    ) -> mx.array:
        cumulative_probs = mx.cumsum(exit_pdf, axis=-1)
        threshold_mask = cumulative_probs >= exit_threshold

        exit_steps = mx.argmax(threshold_mask, axis=-1)
        exit_steps = mx.where(mx.any(threshold_mask, axis=-1), exit_steps, len(hs) - 1)

        hs = mx.stack(hs, axis=2)
        batch_indices = mx.arange(exit_steps.shape[0])[:, None]
        seq_indices = mx.arange(exit_steps.shape[1])[None, :]
        return hs[batch_indices, seq_indices, exit_steps, :]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = OuroModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[KVCache]] = None,
        use_weighted_exit: bool = False,
        exit_at_step: Optional[int] = None,
        exit_threshold: Optional[float] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, use_weighted_exit, exit_at_step, exit_threshold)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if "early_exit_gate" in path:
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def make_cache(self):
        total_caches = self.args.total_ut_steps * len(self.layers)
        return [KVCache() for _ in range(total_caches)]

    def sanitize(self, weights):
        return weights
