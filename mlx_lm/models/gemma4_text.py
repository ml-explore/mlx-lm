# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache, _BaseCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


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


class RMSNormNoScale(nn.Module):
    """RMSNorm without learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


@partial(mx.compile, shapeless=True)
def logit_softcap(softcap, x):
    return mx.tanh(x / softcap) * softcap


@partial(mx.compile, shapeless=True)
def _complete_square(x2, y2, xy):
    return x2 + mx.expand_dims(y2, -1) - 2 * xy


@partial(mx.compile, shapeless=True)
def geglu(gate, x):
    return nn.gelu_approx(gate) * x


class MLP(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int = 0):
        super().__init__()
        first_kv_shared_layer_idx = (
            config.num_hidden_layers - config.num_kv_shared_layers
        )
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide = config.use_double_wide_mlp and is_kv_shared_layer
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
        self.eps = config.rms_norm_eps
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = mx.ones((config.hidden_size,))
        self.per_expert_scale = mx.ones((config.num_experts,))
        self._root_size = config.hidden_size**-0.5

    def __call__(self, x: mx.array):
        x = mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)

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
        return geglu(gate, x)


class Experts(nn.Module):
    """Sparse MoE using SwitchGLU with gather_mm."""

    def __init__(self, config: ModelArgs):
        super().__init__()

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
        self.use_k_eq_v = config.attention_k_eq_v and not self.is_sliding
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

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
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

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        shared_kv: Optional[tuple] = None,
        offset: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        if shared_kv is not None:
            keys, values = shared_kv
        else:
            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            values = keys
            if not self.use_k_eq_v:
                values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            offset = cache.offset if cache is not None else 0

            keys = self.k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)

            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        if mask is not None and isinstance(mask, mx.array):
            if mask.shape[-1] != keys.shape[-2]:
                mask = mask[..., -keys.shape[-2] :]

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output), (keys, values), offset


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # MoE (26B model)
        self.enable_moe = config.enable_moe_block
        if self.enable_moe:
            self.router = Router(config)
            self.experts = Experts(config)
            self.post_feedforward_layernorm_1 = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        # Per-layer input gating (2B/4B models)
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(
                config.hidden_size, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = nn.RMSNorm(
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
        offset: Optional[Any] = None,
    ) -> mx.array:
        residual = x

        h = self.input_layernorm(x)
        h, shared_kv, offset = self.self_attn(
            h, mask, cache, shared_kv=shared_kv, offset=offset
        )
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

        return h, shared_kv, offset


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
        self.args = config
        self.config = config  # backward compat
        self.vocab_size = config.vocab_size
        self.window_size = config.sliding_window
        self.sliding_window_pattern = config.sliding_window_pattern
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size**0.5
        self.layers = [
            DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            self.per_layer_projection_norm = nn.RMSNorm(
                config.hidden_size_per_layer_input, eps=config.rms_norm_eps
            )
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_input_scale = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None

        # Arrange for shared KVs
        self.previous_kvs = list(range(len(self.layers)))
        if config.num_kv_shared_layers > 0:
            N = len(self.layers)
            M = N - config.num_kv_shared_layers
            kvs_by_type = {}
            for i in range(M):
                kvs_by_type[self.layers[i].layer_type] = i
            for j in range(M, N):
                self.previous_kvs[j] = kvs_by_type[self.layers[j].layer_type]

    def _get_per_layer_inputs(
        self,
        input_ids: Optional[mx.array],
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_ids is None:
            if input_embeddings is None:
                raise RuntimeError(
                    "input_embeddings must be provided when input_ids are omitted."
                )

            # Split the sequence dimension if this still holds too much
            # memory. 260k vocab means the distance tensor would be ~1GB
            # per 2k tokens in bf16.
            #
            # If the embedding is quantized we have to dequantize it anyway to
            # perform the match test.
            norms_embedding = self.embed_tokens.weight.square().sum(-1)
            norms_input = input_embeddings.square().sum(-1)
            distance = _complete_square(
                norms_embedding,
                norms_input,
                self.embed_tokens.as_linear(input_embeddings),
            )

            # Checks can be added if needed but they necessarily break the GPU
            # pipelining and force an eval.
            #
            #   match_counts = (distance < eps).sum(-1)
            #
            input_ids = mx.argmin(distance, -1)

        result = self.embed_tokens_per_layer(input_ids)
        result = result * self.embed_tokens_per_layer_scale
        return mx.unflatten(
            result,
            -1,
            (self.args.num_hidden_layers, self.hidden_size_per_layer_input),
        )

    def _project_per_layer_inputs(
        self,
        input_embeddings: mx.array,
        per_layer_inputs: Optional[mx.array] = None,
    ) -> mx.array:
        per_layer_projection = self.per_layer_model_projection(input_embeddings)
        per_layer_projection = mx.unflatten(
            per_layer_projection,
            -1,
            (self.args.num_hidden_layers, self.hidden_size_per_layer_input),
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def _make_masks(self, h, cache):
        mask = {}
        masks = []
        for l, c in zip(self.layers, cache):
            if l.layer_type not in mask:
                if l.layer_type == "full_attention":
                    mask["full_attention"] = create_attention_mask(h, c)
                elif l.layer_type == "sliding_attention":
                    mask["sliding_attention"] = create_attention_mask(
                        h, c, window_size=self.window_size
                    )
            masks.append(mask[l.layer_type])
        return masks

    def __call__(
        self,
        inputs: mx.array = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        per_layer_inputs: Optional[mx.array] = None,
    ):
        # Make the initial hidden state
        if input_embeddings is None:
            input_embeddings = self.embed_tokens(inputs)
        h = input_embeddings
        h = h * self.embed_scale

        # Get the extra inputs per layer if we have per layer embeddings
        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self._get_per_layer_inputs(inputs, input_embeddings)
            per_layer_inputs = self._project_per_layer_inputs(h, per_layer_inputs)
        if per_layer_inputs is not None:
            per_layer_inputs = [
                per_layer_inputs[:, :, i, :] for i, _ in enumerate(self.layers)
            ]
        else:
            per_layer_inputs = [None] * len(self.layers)

        # Make the kv cache list, be sure to append None for all the shared kv
        # layers
        if cache is None:
            cache = [None] * len(self.layers)
        else:
            cache = cache + [None] * (len(self.layers) - len(cache))

        # Apply each layer. We save all intermediate kvs and offset and grab
        # the previous one for the shared kv layers.
        masks = self._make_masks(h, cache)
        intermediates = [(None, None)] * len(self.layers)
        for idx, (layer, c, mask, prev_idx, per_layer_input) in enumerate(
            zip(
                self.layers,
                cache,
                masks,
                self.previous_kvs,
                per_layer_inputs,
            )
        ):
            kvs, offset = intermediates[prev_idx]

            h, kvs, offset = layer(
                h,
                mask,
                c,
                per_layer_input=per_layer_input,
                shared_kv=kvs,
                offset=offset,
            )

            intermediates[idx] = (kvs, offset)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Gemma4TextModel(args)
        self.final_logit_softcapping = args.final_logit_softcapping
        self.tie_word_embeddings = args.tie_word_embeddings
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
            if any(
                s in k
                for s in (
                    "self_attn.rotary_emb",
                    "input_max",
                    "input_min",
                    "output_max",
                    "output_min",
                )
            ):
                continue

            if k.endswith(".experts.gate_up_proj"):
                base = k.removesuffix(".gate_up_proj")
                gate, up = map(mx.contiguous, mx.split(v, 2, axis=-2))
                sanitized[f"{base}.switch_glu.gate_proj.weight"] = gate
                sanitized[f"{base}.switch_glu.up_proj.weight"] = up
                continue

            if k.endswith(".experts.down_proj"):
                base = k.removesuffix(".down_proj")
                sanitized[f"{base}.switch_glu.down_proj.weight"] = v
                continue

            sanitized[k] = v

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
        first_kv_shared = self.args.num_hidden_layers - self.args.num_kv_shared_layers
        caches = []
        for i in range(first_kv_shared):
            if self.args.layer_types[i] == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(
                        max_size=self.args.sliding_window,
                        keep=0,
                    )
                )
        return caches
