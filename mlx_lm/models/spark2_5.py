# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    sliding_window: int
    rope_parameters: Dict[str, Dict[str, Any]]
    layer_types: Optional[List[str]] = None
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 8192
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "gelu"
    gate_attn_act_mode: str = "sigmoid"
    headwise_attn_output_gate: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.layer_types is None:
            # Official checkpoints list every layer type; the reference
            # pattern is three sliding layers followed by one full layer.
            pattern = ["sliding_attention"] * 3 + ["full_attention"]
            self.layer_types = [
                pattern[i % len(pattern)] for i in range(self.num_hidden_layers)
            ]

        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                "layer_types length must match num_hidden_layers: "
                f"{len(self.layer_types)} != {self.num_hidden_layers}"
            )

        unsupported = set(self.layer_types) - {
            "sliding_attention",
            "full_attention",
        }
        if unsupported:
            raise ValueError(f"Unsupported layer types: {sorted(unsupported)}")

        missing_rope = set(self.layer_types) - set(self.rope_parameters)
        if missing_rope:
            raise ValueError(
                f"Missing RoPE parameters for layer types: {sorted(missing_rope)}"
            )

        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        if self.num_attention_heads * self.head_dim <= 0:
            raise ValueError("num_attention_heads * head_dim must be positive")

        for layer_type in set(self.layer_types):
            factor = float(
                self.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
            )
            dims = int(self.head_dim * factor)
            if dims <= 0 or dims > self.head_dim or dims % 2:
                raise ValueError(
                    f"Invalid rotary dimension {dims} for {layer_type} "
                    f"with head_dim={self.head_dim}"
                )

        if self.hidden_act != "gelu":
            raise ValueError(f"Spark2_5 requires GELU, got {self.hidden_act!r}")

        if not self.headwise_attn_output_gate:
            raise ValueError("Spark2_5 requires head-wise attention output gates")

        if self.gate_attn_act_mode != "sigmoid":
            raise ValueError(
                "Spark2_5 requires sigmoid attention gates, got "
                f"{self.gate_attn_act_mode!r}"
            )


def _make_rope(args: ModelArgs, layer_type: str):
    params = args.rope_parameters[layer_type]
    base = float(params.get("rope_theta", 10000.0))
    factor = float(params.get("partial_rotary_factor", 1.0))
    dims = int(args.head_dim * factor)
    return nn.RoPE(dims, traditional=False, base=base)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        layer_type = args.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"

        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        self.q_size = q_size
        self.kv_size = kv_size

        self.q_k_v_proj = nn.Linear(
            args.hidden_size,
            q_size + 2 * kv_size,
            bias=args.attention_bias,
        )
        self.g_proj = nn.Linear(
            args.hidden_size,
            self.n_heads,
            bias=args.attention_bias,
        )
        self.out_proj = nn.Linear(
            q_size,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.rope = _make_rope(args, layer_type)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = x.shape

        qkv = self.q_k_v_proj(x)
        queries, keys, values = mx.split(
            qkv,
            [self.q_size, self.q_size + self.kv_size],
            axis=-1,
        )

        queries = queries.reshape(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # The Q/K/V views alias the fused projection. Materialize them before
        # RoPE so the rope kernel reads contiguous inputs on every backend.
        queries = mx.contiguous(queries)
        keys = mx.contiguous(keys)
        values = mx.contiguous(values)

        # The cache offset is the absolute token position, so RoPE stays
        # correct even after a rotating cache wraps its bounded storage.
        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = mx.contiguous(output.transpose(0, 2, 1, 3))

        gate = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
        output = output * gate[..., None]

        return self.out_proj(output.reshape(batch_size, sequence_length, -1))


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.hidden_size,
            args.intermediate_size,
            bias=args.mlp_bias,
        )
        self.up_proj = nn.Linear(
            args.hidden_size,
            args.intermediate_size,
            bias=args.mlp_bias,
        )
        self.down_proj = nn.Linear(
            args.intermediate_size,
            args.hidden_size,
            bias=args.mlp_bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    @property
    def is_sliding(self) -> bool:
        return self.self_attn.is_sliding

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        attention_output = self.self_attn(self.input_layernorm(x), mask, cache)
        hidden_states = x + attention_output
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states + mlp_output


class Spark2_5Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self._first_full = next(
            (
                idx
                for idx, layer_type in enumerate(args.layer_types)
                if layer_type == "full_attention"
            ),
            None,
        )
        self._first_sliding = next(
            (
                idx
                for idx, layer_type in enumerate(args.layer_types)
                if layer_type == "sliding_attention"
            ),
            None,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = (
            input_embeddings if input_embeddings is not None else self.embedding(inputs)
        )

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = None
        if self._first_full is not None:
            full_mask = create_attention_mask(hidden_states, cache[self._first_full])
        sliding_mask = None
        if self._first_sliding is not None:
            sliding_mask = create_attention_mask(
                hidden_states,
                cache[self._first_sliding],
                window_size=self.args.sliding_window,
            )

        for layer, layer_cache in zip(self.layers, cache):
            mask = sliding_mask if layer.is_sliding else full_mask
            hidden_states = layer(hidden_states, mask, layer_cache)

        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Spark2_5Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embedding.as_linear(hidden_states)
        return self.lm_head(hidden_states)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                if layer.is_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    @property
    def quant_predicate(self):
        # The head-gate is a tiny per-head gating projection; keeping it at
        # source precision preserves the sigmoid gate values.
        def predicate(path, _):
            return not path.endswith("self_attn.g_proj")

        return predicate

    def sanitize(self, weights):
        # Some MLX conversions rename the embedding to embed_tokens while the
        # Hugging Face checkpoint ships it as model.embedding.weight.
        if (
            "model.embed_tokens.weight" in weights
            and "model.embedding.weight" not in weights
        ):
            weights["model.embedding.weight"] = weights.pop("model.embed_tokens.weight")
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        # Drop the serialized RoPE inverse-frequency buffer some conversions ship.
        return {
            name: value
            for name, value in weights.items()
            if "rotary_emb.inv_freq" not in name
        }
