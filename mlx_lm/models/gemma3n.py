# Copyright © 2025 Apple Inc.

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class TextConfig(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    num_kv_shared_layers: int
    query_pre_attn_scalar: float
    vocab_size_per_layer_input: int
    sliding_window: int
    max_position_embeddings: int
    rope_local_base_freq: float
    rope_theta: float
    final_logit_softcapping: float
    layer_types: List[str]
    activation_sparsity_pattern: List[float]
    hidden_size_per_layer_input: int
    altup_num_inputs: int
    altup_coef_clip: float
    altup_correct_scale: bool
    altup_active_idx: int
    laurel_rank: int
    rope_scaling: Optional[Dict] = None


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict


class RMSNoScale(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, None, self.eps)


class Gemma3nLaurelBlock(nn.Module):
    """Learned Augmented Residual Layer"""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config

        self.linear_left = nn.Linear(
            self.config.hidden_size, self.config.laurel_rank, bias=False
        )
        self.linear_right = nn.Linear(
            self.config.laurel_rank, self.config.hidden_size, bias=False
        )
        self.post_laurel_norm = nn.RMSNorm(
            dims=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def __call__(self, x: mx.array) -> mx.array:
        laurel_x = self.linear_left(x)
        laurel_x = self.linear_right(laurel_x)
        normed_laurel_x = self.post_laurel_norm(laurel_x)
        return x + normed_laurel_x


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    unsqueeze_dim: int = 1,
):

    cos = mx.expand_dims(cos, unsqueeze_dim)
    sin = mx.expand_dims(sin, unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def _compute_default_rope_parameters(
    config: Optional[TextConfig] = None,
    **rope_kwargs,
) -> tuple[mx.array, float]:

    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = (
            config.partial_rotary_factor
            if hasattr(config, "partial_rotary_factor")
            else 1.0
        )
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        base ** (mx.arange(0, dim, 2, dtype=mx.int64).astype(mx.float32) / dim)
    )
    return inv_freq, attention_factor


class Gemma3nRotaryEmbedding(nn.Module):
    def __init__(self, config: TextConfig, device=None):
        super().__init__()

        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_init_fn = _compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config)
        self._inv_freq = mx.array(inv_freq, dtype=mx.float32)
        self._original_inv_freq = mx.array(inv_freq, dtype=mx.float32)

    def __call__(self, x, position_ids):
        inv_freq_expanded = self._inv_freq[None, :, None].astype(mx.float32)
        position_ids_expanded = position_ids[:, None, :].astype(mx.float32)

        freqs = (
            inv_freq_expanded.astype(mx.float32)
            @ position_ids_expanded.astype(mx.float32)
        ).transpose(0, 2, 1)
        emb = mx.concatenate((freqs, freqs), axis=-1)
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling

        return cos.astype(x.dtype), sin.astype(x.dtype)


class Gemma3nAttention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.repeats = n_heads // n_kv_heads
        self.head_dim = head_dim = config.head_dim
        self.layer_idx = layer_idx

        self.scale = 1.0

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(dims=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(dims=config.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNoScale(eps=config.rms_norm_eps)

        first_kv_shared_layer_idx = (
            config.num_hidden_layers - config.num_kv_shared_layers
        )

        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx

        # Compute the layer index from which shared KV cache values will be retrieved.
        if not self.is_kv_shared_layer:
            self.kv_shared_layer_index = None
        elif self.is_sliding:
            # The last layer that computes local sliding attention is always 2 before sharing starts
            self.kv_shared_layer_index = first_kv_shared_layer_idx - 2
        else:
            # The last layer before sharing starts is always the last that computes global attention layer
            self.kv_shared_layer_index = first_kv_shared_layer_idx - 1

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        caches: Optional[List[Any]] = None,
        position_embeddings: Optional[mx.array] = None,
        cache_position: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        cos, sin = position_embeddings

        queries = self.q_proj(x)
        queries = queries.reshape(B, L, -1, self.head_dim)
        queries = self.q_norm(queries)
        queries = apply_rotary_pos_emb(queries, cos, sin, unsqueeze_dim=2)
        queries = queries.transpose(0, 2, 1, 3)

        if (
            self.is_kv_shared_layer
            and self.kv_shared_layer_index is not None
            and cache is not None
        ):
            # For shared layers, retrieve KV from the designated cache layer
            shared_cache = caches[self.kv_shared_layer_index]
            keys, values = shared_cache.state

        else:
            keys = self.k_proj(x).reshape(B, L, -1, self.head_dim)
            keys = self.k_norm(keys)
            keys = apply_rotary_pos_emb(keys, cos, sin, unsqueeze_dim=2)
            keys = keys.transpose(0, 2, 1, 3)

            values = self.v_proj(x).reshape(B, L, -1, self.head_dim)
            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        if isinstance(mask, mx.array) and mask.shape[-1] != keys.shape[-2]:
            mask = mask[:, : keys.shape[-2]]

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.activation_sparsity_pattern is not None:
            self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]
        else:
            self.activation_sparsity = 0.0
        if self.activation_sparsity > 0:
            self._std_multiplier = math.sqrt(2.0) * mx.erfinv(
                2 * self.activation_sparsity - 1
            )

    def __call__(self, x: mx.array):
        gate_proj = self.gate_proj(x)
        if self.activation_sparsity > 0.0:
            gate_proj = self._gaussian_topk(gate_proj)
        activations = nn.gelu_approx(gate_proj)
        up_proj = self.up_proj(x)
        down_proj = self.down_proj(activations * up_proj)
        return down_proj

    def _gaussian_topk(self, inputs: mx.array) -> mx.array:
        # TODO does that need higher precision ?
        inputs_mean = mx.mean(inputs, axis=-1, keepdims=True)
        inputs_std = mx.std(inputs, axis=-1, keepdims=True)
        cutoff_x = inputs_mean + inputs_std * self._std_multiplier.astype(
            inputs_std.dtype
        )
        return mx.maximum(0, inputs - cutoff_x)


class Gemma3nAltUp(nn.Module):
    """Alternating Updates (AltUp)"""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config

        self.correct_output_scale = mx.zeros((self.config.hidden_size,))
        self.correction_coefs = nn.Linear(
            self.config.altup_num_inputs, self.config.altup_num_inputs, bias=False
        )
        self.prediction_coefs = nn.Linear(
            self.config.altup_num_inputs, self.config.altup_num_inputs**2, bias=False
        )
        self.modality_router = nn.Linear(
            self.config.hidden_size, self.config.altup_num_inputs, bias=False
        )
        self.router_norm = nn.RMSNorm(
            dims=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self._router_input_scale = mx.array(self.config.hidden_size**-1.0)

    def compute_router_modalities(self, x: mx.array) -> mx.array:
        router_inputs = self.router_norm(x) * self._router_input_scale.astype(
            self.router_norm.weight.dtype
        )
        routed = self.modality_router(router_inputs).astype(mx.float32)
        return mx.tanh(routed)

    def predict(self, x: mx.array) -> mx.array:
        modalities = self.compute_router_modalities(x[self.config.altup_active_idx])

        # Force float32 computation like PyTorch
        self.prediction_coefs.weight = self.prediction_coefs.weight.astype(mx.float32)

        if self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight = mx.clip(
                self.prediction_coefs.weight,
                -self.config.altup_coef_clip,
                self.config.altup_coef_clip,
            )

        # Fix: Use permute pattern that matches PyTorch exactly
        all_coefs = (
            self.prediction_coefs(modalities)
            .reshape(
                *modalities.shape[:-1],
                self.config.altup_num_inputs,
                self.config.altup_num_inputs,
            )
            .transpose(0, 1, 3, 2)  # This should match PyTorch's permute(0, 1, 3, 2)
        )

        # Fix: Match PyTorch's tensor manipulation exactly
        # PyTorch: hidden_states.float().permute(1, 2, 3, 0)
        x_permuted = x.astype(mx.float32).transpose(1, 2, 3, 0)
        predictions = mx.matmul(x_permuted, all_coefs)
        predictions = predictions.transpose(
            3, 0, 1, 2
        )  # Match PyTorch's permute(3, 0, 1, 2)
        predictions += x
        return predictions.astype(x.dtype)

    def correct(self, predictions: mx.array, activated: mx.array):
        modalities = self.compute_router_modalities(activated)

        # Force float32 computation like PyTorch
        self.correction_coefs.weight = self.correction_coefs.weight.astype(mx.float32)

        if self.config.altup_coef_clip is not None:
            self.correction_coefs.weight = mx.clip(
                self.correction_coefs.weight,
                -self.config.altup_coef_clip,
                self.config.altup_coef_clip,
            )

        # Fix: Match PyTorch's broadcasting approach instead of loop
        all_coefs = self.correction_coefs(modalities) + 1.0

        active_x = predictions[self.config.altup_active_idx]
        innovation = activated - active_x

        # Replicate innovation for all inputs like PyTorch
        innovation_expanded = mx.broadcast_to(
            mx.expand_dims(innovation, axis=0),
            (self.config.altup_num_inputs,) + innovation.shape,
        )

        # Fix: Match PyTorch's tensor manipulation
        # PyTorch: all_coefs.permute(2, 1, 0).unsqueeze(1)
        all_coefs_reshaped = all_coefs.transpose(2, 1, 0)
        all_coefs_reshaped = mx.expand_dims(all_coefs_reshaped, axis=1)

        # Broadcast multiply like PyTorch
        corrected = innovation_expanded * all_coefs_reshaped
        corrected += predictions

        return corrected.astype(activated.dtype)

    def scale_corrected_output(self, corrected: mx.array):
        scale = self.correct_output_scale if self.config.altup_correct_scale else 1.0
        return corrected * scale

    def __call__(self, x: mx.array, activated: mx.array):
        predictions = self.predict(x)
        corrected = self.correct(predictions=predictions, activated=activated)
        output = corrected[self.config.altup_active_idx]
        if self.config.altup_correct_scale:
            output = self.scale_corrected_output(output)
        return corrected, output


class Gemma3nDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma3nAttention(config, layer_idx)
        self.mlp = MLP(config, layer_idx=layer_idx)
        self.input_layernorm = nn.RMSNorm(
            self.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.post_attention_layernorm = nn.RMSNorm(
            self.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = nn.RMSNorm(
            self.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            self.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.is_sliding = self.self_attn.is_sliding
        self.sliding_window = config.sliding_window

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input

        self.altup = Gemma3nAltUp(config)
        self.laurel = Gemma3nLaurelBlock(config)
        self.per_layer_input_gate = nn.Linear(
            self.hidden_size, self.hidden_size_per_layer_input, bias=False
        )
        self.per_layer_projection = nn.Linear(
            self.hidden_size_per_layer_input, self.hidden_size, bias=False
        )
        self.post_per_layer_input_norm = nn.RMSNorm(
            self.hidden_size,
            eps=config.rms_norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        per_layer_input: Optional[mx.array] = None,
        caches: Optional[List[Any]] = None,
        cache_position: Optional[mx.array] = None,
        position_embeddings_global: Optional[mx.array] = None,
        position_embeddings_local: Optional[mx.array] = None,
    ):
        if isinstance(x, list):
            x = mx.stack(x, axis=0)

        predictions = self.altup.predict(x)
        active_prediction = predictions[self.config.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        # apply global RoPE to non-sliding layer only
        if self.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        attn = self.self_attn(
            active_prediction_normed,
            mask,
            cache,
            caches,
            position_embeddings,
            cache_position,
        )

        attn = self.post_attention_layernorm(attn)

        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) / mx.sqrt(
            mx.array(2.0, dtype=active_prediction.dtype)
        )

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm

        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        first_prediction = corrected_predictions[self.config.altup_active_idx]
        if self.config.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        first_prediction = self.per_layer_input_gate(first_prediction)
        first_prediction = nn.gelu_approx(first_prediction)

        first_prediction = mx.multiply(first_prediction, per_layer_input)

        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)

        for i in range(1, len(corrected_predictions)):
            corrected_predictions[i] = corrected_predictions[i] + first_prediction

        return corrected_predictions


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.vocab_size = config.vocab_size
        self.vocab_size_per_layer_input = config.vocab_size_per_layer_input
        self.num_hidden_layers = config.num_hidden_layers
        self.final_logit_softcapping = config.final_logit_softcapping

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Gemma3nDecoderLayer(config=config, layer_idx=layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.embed_tokens_per_layer = nn.Embedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
        )

        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )

        self.per_layer_projection_norm = nn.RMSNorm(
            dims=config.hidden_size_per_layer_input,
            eps=config.rms_norm_eps,
        )

        self.altup_projections = [
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(1, self.config.altup_num_inputs)
        ]

        self.altup_unembed_projections = [
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(1, self.config.altup_num_inputs)
        ]

        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._per_layer_projection_scale = mx.array(self.hidden_size**-0.5)
        self._per_layer_input_scale = mx.rsqrt(mx.array(2.0))

        self.rope_embedding = Gemma3nRotaryEmbedding(config)

        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rope_embedding_local = Gemma3nRotaryEmbedding(config)
        self.first_sliding_idx = self.config.layer_types.index("sliding_attention")
        self.first_full_idx = self.config.layer_types.index("full_attention")

    def __call__(
        self,
        inputs: mx.array = None,
        mask: mx.array = None,
        cache=None,
        input_embeddings: mx.array = None,
    ):
        per_layer_inputs = self.get_per_layer_inputs(inputs)
        if input_embeddings is None:
            h = self.embed_tokens(inputs) * (self.hidden_size**0.5)
        else:
            h = input_embeddings

        if per_layer_inputs is None and inputs is not None:
            per_layer_inputs = self.get_per_layer_inputs(inputs)

        per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        cache_position = None
        if cache_position is None:
            past_seen_tokens = 0
            if cache is not None and cache[0] is not None:
                past_seen_tokens = cache[0].offset

            cache_position = mx.arange(
                past_seen_tokens,
                past_seen_tokens + h.shape[1],
            )

        if mask is None:
            # TOOD why return_array?
            full_mask = create_attention_mask(
                h, cache[self.first_full_idx :], return_array=True
            )
            sliding_window_mask = create_attention_mask(
                h, cache[self.first_sliding_idx :], return_array=True
            )
        h0 = h

        # Initialize RoPE embeddings
        position_ids = cache_position[None, :]
        position_embeddings_global = self.rope_embedding(h0, position_ids)
        position_embeddings_local = self.rope_embedding_local(h0, position_ids)

        # Expand hidden_states to support per-layer inputs
        target_magnitude = mx.mean(h0**2, axis=-1, keepdims=True) ** 0.5
        epsilon_tensor = mx.array(mx.finfo(h0.dtype).min, dtype=h0.dtype)

        h_list = [h0] * self.config.altup_num_inputs

        for i in range(1, self.config.altup_num_inputs):
            altup_proj = self.altup_projections[i - 1](h_list[i])
            h_list[i] = altup_proj.astype(h0.dtype)
            new_magnitude = mx.mean(h_list[i] ** 2, axis=-1, keepdims=True) ** 0.5
            h_list[i] *= target_magnitude / mx.maximum(new_magnitude, epsilon_tensor)

        h = mx.stack(h_list, axis=0)

        for i, (layer, c) in enumerate(
            zip(self.layers[: self.config.num_hidden_layers], cache)
        ):
            per_layer_input = per_layer_inputs[:, :, i, :]

            is_global = self.config.layer_types[i] == "full_attention"

            local_mask = mask
            if mask is None and is_global:
                local_mask = full_mask
            elif mask is None:
                local_mask = sliding_window_mask

            h = layer(
                h,
                local_mask,
                c,
                per_layer_input,
                cache,
                cache_position,
                position_embeddings_global,
                position_embeddings_local,
            )

        # Per-layer inputs to single output
        target_magnitude = mx.mean(h[0] ** 2, axis=-1, keepdims=True) ** 0.5

        for i in range(1, self.config.altup_num_inputs):
            altup_unemb_proj = self.altup_unembed_projections[i - 1](h[i])
            h[i] = altup_unemb_proj.astype(h0.dtype)
            new_magnitude = mx.mean(h[i] ** 2, axis=-1, keepdims=True) ** 0.5
            h[i] *= target_magnitude / mx.maximum(new_magnitude, epsilon_tensor)

        h = mx.mean(h, axis=0)

        out = self.norm(h)
        out = self.embed_tokens.as_linear(out)
        # TODO compile that
        if self.final_logit_softcapping is not None:
            out = mx.tanh(out / self.final_logit_softcapping)
            out = out * self.final_logit_softcapping
        return out

    def get_per_layer_inputs(self, input_ids: mx.array) -> mx.array:
        per_layer_inputs_mask = mx.logical_and(
            input_ids >= 0, input_ids < self.vocab_size_per_layer_input
        )
        tokens = mx.where(per_layer_inputs_mask, input_ids, mx.zeros_like(input_ids))
        result = self.embed_tokens_per_layer(tokens) * (
            self.hidden_size_per_layer_input**0.5
        )
        return result.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self, inputs_embeds: mx.array, per_layer_inputs: Optional[mx.array] = None
    ) -> mx.array:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection *= self._per_layer_projection_scale.astype(
            inputs_embeds.dtype
        )

        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        if per_layer_projection.shape != per_layer_inputs.shape:
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]

        return (
            per_layer_projection + per_layer_inputs
        ) * self._per_layer_input_scale.astype(inputs_embeds.dtype)


class Gemma3n(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = LanguageModel(TextConfig.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, mask=mask, input_embeddings=input_embeddings
        )


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = Gemma3n(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.model(
            inputs, cache=cache, mask=mask, input_embeddings=input_embeddings
        )

    def sanitize(self, weights):
        weights = tree_unflatten(list(weights.items()))
        for k in ["vision_tower", "audio_tower", "embed_audio", "embed_vision"]:
            weights["model"].pop(k, None)
        return dict(tree_flatten(weights))

    @property
    def layers(self):
        return self.model.language_model.layers
