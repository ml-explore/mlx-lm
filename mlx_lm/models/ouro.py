"""
Ouro model implementation for MLX.

Ouro is a Universal Transformer that processes inputs through
layers multiple times (UT steps) with an early exit mechanism.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple

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
    attention_dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    total_ut_steps: int = 4
    early_exit_threshold: Optional[float] = 1.0
    layer_types: Optional[List[str]] = None
    sliding_window: Optional[int] = None
    max_window_layers: Optional[int] = None
    use_sliding_window: bool = False
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02


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

        layer_types = args.layer_types or ["full_attention"] * args.num_hidden_layers
        self.sliding_window = (
            args.sliding_window
            if layer_types[layer_idx] == "sliding_attention"
            else None
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
        h = residual + h

        return h


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
    ) -> Tuple[mx.array, Optional[List[mx.array]], Optional[List[mx.array]]]:
        h = self.embed_tokens(x)

        # For UT, we need total_ut_steps * num_layers cache slots
        if cache is None:
            cache = [None] * (self.total_ut_steps * len(self.layers))

        mask = create_attention_mask(h, cache[0])

        hidden_states_list = []
        gate_list = []

        for current_ut in range(self.total_ut_steps):
            for layer_idx, layer in enumerate(self.layers):
                # Cache index: current_ut * num_layers + layer_idx
                cache_idx = current_ut * len(self.layers) + layer_idx
                layer_cache = cache[cache_idx] if cache else None
                h = layer(h, mask, layer_cache)

            h = self.norm(h)
            hidden_states_list.append(h)

            gate = self.early_exit_gate(h)
            gate_list.append(gate)

        return h, hidden_states_list, gate_list


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
    ):
        hidden_states, _, _ = self.model(inputs, cache)

        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden_states)
        else:
            return self.lm_head(hidden_states)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        total_caches = self.args.total_ut_steps * len(self.layers)
        return [KVCache() for _ in range(total_caches)]
