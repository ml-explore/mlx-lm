# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache, _BaseCache
from .rope_utils import initialize_rope


class _OffsetCache(_BaseCache):
    """Lightweight cache for KV-shared layers that only tracks offset."""

    def __init__(self):
        self.offset = 0

    @property
    def nbytes(self):
        return 0

    def empty(self):
        return True


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4_text"
    hidden_size: int = 1536
    num_hidden_layers: int = 35
    intermediate_size: int = 6144
    num_attention_heads: int = 8
    head_dim: int = 256
    global_head_dim: int = 512
    global_partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    vocab_size_per_layer_input: int = 262144
    num_key_value_heads: int = 1
    num_global_key_value_heads: Optional[int] = None
    num_kv_shared_layers: int = 20
    pad_token_id: int = 0
    hidden_size_per_layer_input: int = 256
    rope_traditional: bool = False
    partial_rotary_factor: float = 1.0
    rope_parameters: Optional[Dict] = None
    sliding_window: int = 512
    sliding_window_pattern: int = 5
    max_position_embeddings: int = 131072
    attention_k_eq_v: bool = False
    final_logit_softcapping: float = 30.0
    use_double_wide_mlp: bool = True
    enable_moe_block: bool = False
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    layer_types: Optional[List[str]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {
                    "partial_rotary_factor": 1.0,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {
                    "partial_rotary_factor": 1.0,
                    "rope_theta": 10000.0,
                    "rope_type": "default",
                },
            }
        if self.layer_types is None:
            pattern = ["sliding_attention"] * (self.sliding_window_pattern - 1) + [
                "full_attention"
            ]
            self.layer_types = (pattern * (self.num_hidden_layers // len(pattern) + 1))[
                : self.num_hidden_layers
            ]


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class RMSNormNoScale(nn.Module):
    """RMSNorm without learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class RMSNormZeroShift(nn.Module):
    """RMSNorm with scale_shift=0 (used for per-layer projection norm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


@partial(mx.compile, shapeless=True)
def logit_softcap(softcap, x):
    return mx.tanh(x / softcap) * softcap


class MLP(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int = 0):
        super().__init__()
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide = (
            getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer
        )
        intermediate_size = config.intermediate_size * (2 if use_double_wide else 1)

        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    """Expert router: norm -> scale -> project -> top-k -> renormalize."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.norm = RMSNormNoScale(config.hidden_size, eps=config.rms_norm_eps)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = mx.ones((config.hidden_size,))
        self.per_expert_scale = mx.ones((config.num_experts,))
        self._root_size = config.hidden_size**-0.5

    def __call__(self, x: mx.array):
        x = self.norm(x)
        x = x * self._root_size
        x = x * self.scale

        expert_scores = self.proj(x)
        router_probs = mx.softmax(expert_scores, axis=-1)

        top_k_indices = mx.argpartition(
            -expert_scores, kth=self.config.top_k_experts - 1, axis=-1
        )[..., : self.config.top_k_experts]

        top_k_weights = mx.take_along_axis(router_probs, top_k_indices, axis=-1)
        top_k_weights = top_k_weights / mx.sum(top_k_weights, axis=-1, keepdims=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices]
        return top_k_indices, top_k_weights


class GeGLU(nn.Module):
    """GELU-gated linear unit activation for SwitchGLU."""

    def __call__(self, x, gate):
        return nn.gelu_approx(gate) * x


class Experts(nn.Module):
    """Sparse MoE using SwitchGLU with gather_mm."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        from .switch_layers import SwitchGLU

        self.switch_glu = SwitchGLU(
            input_dims=config.hidden_size,
            hidden_dims=config.moe_intermediate_size,
            num_experts=config.num_experts,
            activation=GeGLU(),
            bias=False,
        )

    def __call__(
        self, x: mx.array, top_k_indices: mx.array, top_k_weights: mx.array
    ) -> mx.array:
        B, S, H = x.shape
        K = top_k_indices.shape[-1]

        x_flat = x.reshape(B * S, H)
        indices_flat = top_k_indices.reshape(B * S, K)

        expert_out = self.switch_glu(x_flat, indices_flat)

        weights = top_k_weights.reshape(B * S, K)[..., None]
        return (expert_out * weights).sum(axis=-2).reshape(B, S, H)


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"

        self.head_dim = (
            config.global_head_dim
            if self.layer_type == "full_attention"
            and hasattr(config, "global_head_dim")
            and config.global_head_dim
            else config.head_dim
        )

        dim = config.hidden_size
        self.n_heads = config.num_attention_heads

        # K-eq-V for full attention layers (26B/31B models)
        self.use_k_eq_v = (
            getattr(config, "attention_k_eq_v", False) and not self.is_sliding
        )
        if self.use_k_eq_v and config.num_global_key_value_heads is not None:
            self.n_kv_heads = config.num_global_key_value_heads
        else:
            self.n_kv_heads = config.num_key_value_heads

        self.scale = 1.0

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        if not self.use_k_eq_v:
            self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNormNoScale(self.head_dim, eps=config.rms_norm_eps)

        # RoPE (with partial rotation support)
        layer_key = "sliding_attention" if self.is_sliding else "full_attention"
        rope_params = config.rope_parameters.get(layer_key, {})
        rope_theta = rope_params.get("rope_theta", 10000.0)
        self.rope = initialize_rope(
            dims=self.head_dim,
            traditional=config.rope_traditional,
            base=rope_theta,
            scaling_config=rope_params,
            max_position_embeddings=config.max_position_embeddings,
        )

        # KV sharing (2B/4B models)
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        if self.is_kv_shared_layer:
            prev_layers = config.layer_types[:first_kv_shared_layer_idx]
            self.kv_shared_layer_index = (
                len(prev_layers)
                - 1
                - prev_layers[::-1].index(config.layer_types[layer_idx])
            )
        else:
            self.kv_shared_layer_index = None

        if not self.is_kv_shared_layer:
            prev_layers = config.layer_types[:first_kv_shared_layer_idx]
            self.store_full_length_kv = layer_idx == len(prev_layers) - 1 - prev_layers[
                ::-1
            ].index(config.layer_types[layer_idx])
        else:
            self.store_full_length_kv = False

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        shared_kv: Optional[tuple] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        if self.is_kv_shared_layer and shared_kv is not None:
            keys, values = shared_kv
            offset = cache.offset if cache is not None else 0
        else:
            offset = cache.offset if cache is not None else 0

            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            # k_eq_v: values from raw k_proj (before k_norm)
            if self.use_k_eq_v:
                values = keys
            else:
                values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            keys = self.k_norm(keys)
            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)

            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        if self.store_full_length_kv:
            self._last_kv = (keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        if mask is not None and isinstance(mask, mx.array):
            if mask.shape[-1] != keys.shape[-2]:
                mask = mask[..., -keys.shape[-2] :]

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # MoE (26B model)
        self.enable_moe = getattr(config, "enable_moe_block", False)
        if self.enable_moe:
            self.router = Router(config)
            self.experts = Experts(config)
            self.post_feedforward_layernorm_1 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        # Per-layer input gating (2B/4B models)
        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        )
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(
                config.hidden_size, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

        # Layer scalar
        self.layer_scalar = mx.ones((1,))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        per_layer_input: Optional[mx.array] = None,
        shared_kv: Optional[tuple] = None,
    ) -> mx.array:
        residual = x

        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache, shared_kv=shared_kv)
        h = self.post_attention_layernorm(h)
        h = residual + h

        residual = h

        if self.enable_moe:
            h1 = self.pre_feedforward_layernorm(h)
            h1 = self.mlp(h1)
            h1 = self.post_feedforward_layernorm_1(h1)

            top_k_indices, top_k_weights = self.router(h)
            h2 = self.pre_feedforward_layernorm_2(h)
            h2 = self.experts(h2, top_k_indices, top_k_weights)
            h2 = self.post_feedforward_layernorm_2(h2)

            h = h1 + h2
        else:
            h = self.pre_feedforward_layernorm(h)
            h = self.mlp(h)

        h = self.post_feedforward_layernorm(h)
        h = residual + h

        # Per-layer input gating
        if (
            self.per_layer_input_gate is not None
            and self.per_layer_projection is not None
            and self.post_per_layer_input_norm is not None
            and per_layer_input is not None
        ):
            residual = h
            gate = self.per_layer_input_gate(h)
            gate = nn.gelu_approx(gate)
            gate = mx.multiply(gate, per_layer_input)
            gate = self.per_layer_projection(gate)
            gate = self.post_per_layer_input_norm(gate)
            h = residual + gate

        if self.layer_scalar is not None:
            h = h * self.layer_scalar

        return h


class ScaledLinear(nn.Module):
    """Linear layer with output scaling."""

    def __init__(self, in_features: int, out_features: int, scalar: float):
        super().__init__()
        self.weight = mx.zeros((out_features, in_features))
        self.scalar = scalar

    def __call__(self, x: mx.array) -> mx.array:
        return (x @ self.weight.T) * self.scalar


class Gemma4TextModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.window_size = config.sliding_window
        self.sliding_window_pattern = config.sliding_window_pattern
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size**0.5
        self.layers = [
            DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Per-layer input embeddings (2B/4B models)
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = nn.Embedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
            )
            self.embed_tokens_per_layer_scale = config.hidden_size_per_layer_input**0.5
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = ScaledLinear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                scalar=config.hidden_size**-0.5,
            )
            self.per_layer_projection_norm = RMSNormZeroShift(
                config.hidden_size_per_layer_input, eps=config.rms_norm_eps
            )
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_input_scale = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None

    def get_per_layer_inputs(
        self,
        input_ids: Optional[mx.array],
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_ids is None:
            if input_embeddings is None:
                raise RuntimeError(
                    "input_embeddings must be provided when input_ids are omitted."
                )

            flat_inputs = input_embeddings.reshape(-1, input_embeddings.shape[-1])
            resolved = np.full((flat_inputs.shape[0],), -1, dtype=np.int32)
            match_counts = np.zeros((flat_inputs.shape[0],), dtype=np.int32)

            # Reverse the exact embedding lookup in chunks to avoid allocating
            # a full [batch, seq, vocab, hidden] comparison tensor.
            chunk_size = 1024
            for start in range(0, self.config.vocab_size, chunk_size):
                end = min(start + chunk_size, self.config.vocab_size)
                vocab_ids = mx.arange(start, end, dtype=mx.int32)
                vocab_embeddings = self.embed_tokens(vocab_ids)
                matches = np.array(
                    mx.all(
                        flat_inputs[:, None, :] == vocab_embeddings[None, :, :],
                        axis=-1,
                    )
                )
                counts = matches.sum(axis=1)
                matched_rows = counts > 0
                if matched_rows.any():
                    resolved[matched_rows] = (
                        start + matches.argmax(axis=1)[matched_rows]
                    )
                match_counts += counts

            if not np.all(match_counts == 1):
                raise RuntimeError(
                    "Could not uniquely recover input_ids from input_embeddings."
                )

            input_ids = mx.array(
                resolved.reshape(input_embeddings.shape[:2]), dtype=mx.int32
            )

        result = self.embed_tokens_per_layer(input_ids)
        result = result * self.embed_tokens_per_layer_scale
        return result.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: mx.array,
        per_layer_inputs: Optional[mx.array] = None,
    ) -> mx.array:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def __call__(
        self,
        inputs: mx.array = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        per_layer_inputs: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            raw_input_embeddings = input_embeddings
        else:
            raw_input_embeddings = self.embed_tokens(inputs)

        h = raw_input_embeddings
        h = h * self.embed_scale
        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(
                    inputs, raw_input_embeddings
                )
            per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        global_cache_idx = None
        sliding_cache_idx = None
        for i, layer in enumerate(self.layers):
            if layer.layer_type == "full_attention" and global_cache_idx is None:
                global_cache_idx = i
            elif layer.layer_type == "sliding_attention" and sliding_cache_idx is None:
                sliding_cache_idx = i

        global_mask = create_attention_mask(
            h, cache[global_cache_idx] if global_cache_idx is not None else None
        )
        sliding_window_mask = create_attention_mask(
            h,
            cache[sliding_cache_idx] if sliding_cache_idx is not None else None,
            window_size=self.window_size,
        )

        shared_kv_store = {}

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            is_global = layer.layer_type == "full_attention"

            layer_shared_kv = None
            if (
                layer.self_attn.is_kv_shared_layer
                and layer.self_attn.kv_shared_layer_index is not None
            ):
                ref_idx = layer.self_attn.kv_shared_layer_index
                if ref_idx in shared_kv_store:
                    layer_shared_kv, ref_offset = shared_kv_store[ref_idx]
                    if c is not None:
                        c.offset = ref_offset

            mask = global_mask if is_global else sliding_window_mask

            per_layer_input = None
            if per_layer_inputs is not None:
                per_layer_input = per_layer_inputs[:, :, i, :]

            pre_offset = c.offset if c is not None else 0

            h = layer(
                h,
                mask,
                c,
                per_layer_input=per_layer_input,
                shared_kv=layer_shared_kv,
            )

            if layer.self_attn.store_full_length_kv:
                shared_kv_store[i] = (layer.self_attn._last_kv, pre_offset)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Gemma4TextModel(args)
        self.final_logit_softcapping = getattr(args, "final_logit_softcapping", None)
        self.tie_word_embeddings = getattr(args, "tie_word_embeddings", True)
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        per_layer_inputs: Optional[mx.array] = None,
    ):
        out = self.model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            per_layer_inputs=per_layer_inputs,
        )
        if self.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        if self.final_logit_softcapping is not None:
            out = logit_softcap(self.final_logit_softcapping, out)
        return out

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "self_attn.rotary_emb" in k:
                continue
            if any(
                s in k for s in ["input_max", "input_min", "output_max", "output_min"]
            ):
                continue
            if k.endswith(".experts.gate_up_proj") or k.endswith(
                ".experts.gate_up_proj.weight"
            ):
                base = k.rsplit(".experts.gate_up_proj", 1)[0]
                if base == k:
                    base = k.rsplit(".experts.gate_up_proj.weight", 1)[0]
                split = v.shape[-2] // 2
                sanitized[f"{base}.experts.switch_glu.gate_proj.weight"] = v[
                    ..., :split, :
                ]
                sanitized[f"{base}.experts.switch_glu.up_proj.weight"] = v[
                    ..., split:, :
                ]
                continue
            if k.endswith(".experts.down_proj") or k.endswith(".experts.down_proj.weight"):
                base = k.rsplit(".experts.down_proj", 1)[0]
                if base == k:
                    base = k.rsplit(".experts.down_proj.weight", 1)[0]
                sanitized[f"{base}.experts.switch_glu.down_proj.weight"] = v
                continue
            sanitized[k] = v
        if "lm_head.weight" not in sanitized:
            self.tie_word_embeddings = True
            if hasattr(self, "lm_head"):
                self.pop("lm_head")
        return sanitized

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("router.proj"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

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
        first_kv_shared = self.args.num_hidden_layers - getattr(
            self.args, "num_kv_shared_layers", 0
        )
        caches = []
        for i in range(self.args.num_hidden_layers):
            is_shared = i >= first_kv_shared > 0
            if is_shared:
                caches.append(_OffsetCache())
            elif self.args.layer_types[i] == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(
                        max_size=self.args.sliding_window,
                        keep=0,
                    )
                )
        return caches
