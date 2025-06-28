# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache


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


class Gemma3nAttention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int, is_kv_shared_layer: bool):
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

        self.is_kv_shared_layer = is_kv_shared_layer

        self.rope = nn.RoPE(
            head_dim,
            traditional=False,
            base=(
                config.rope_local_base_freq if self.is_sliding else config.rope_theta
            ),
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        queries = queries.reshape(B, L, -1, self.head_dim)
        queries = self.q_norm(queries)

        offset = 0
        if self.is_kv_shared_layer and cache is not None:
            # For shared layers, retrieve KV from the designated cache layer
            keys, values = cache.state
            offset = cache.offset

        else:
            if cache is not None:
                offset = cache.offset
            keys = self.k_proj(x).reshape(B, L, -1, self.head_dim)
            keys = self.k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)

            values = self.v_proj(x).reshape(B, L, -1, self.head_dim)
            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

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

    def compute_router_modalities(self, x: mx.array) -> mx.array:
        router_inputs = self.router_norm(x) * (self.config.hidden_size**-1.0)
        routed = self.modality_router(router_inputs).astype(mx.float32)
        return mx.tanh(routed)

    def predict(self, x: mx.array) -> mx.array:
        modalities = self.compute_router_modalities(x[self.config.altup_active_idx])

        self.prediction_coefs.weight = self.prediction_coefs.weight.astype(mx.float32)

        if self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight = mx.clip(
                self.prediction_coefs.weight,
                -self.config.altup_coef_clip,
                self.config.altup_coef_clip,
            )

        all_coefs = (
            self.prediction_coefs(modalities)
            .reshape(
                *modalities.shape[:-1],
                self.config.altup_num_inputs,
                self.config.altup_num_inputs,
            )
            .transpose(0, 1, 3, 2)
        )

        x_permuted = x.astype(mx.float32).transpose(1, 2, 3, 0)
        predictions = mx.matmul(x_permuted, all_coefs)
        predictions = predictions.transpose(3, 0, 1, 2)
        predictions += x
        return predictions.astype(x.dtype)

    def correct(self, predictions: mx.array, activated: mx.array):
        modalities = self.compute_router_modalities(activated)

        self.correction_coefs.weight = self.correction_coefs.weight.astype(mx.float32)

        if self.config.altup_coef_clip is not None:
            self.correction_coefs.weight = mx.clip(
                self.correction_coefs.weight,
                -self.config.altup_coef_clip,
                self.config.altup_coef_clip,
            )

        all_coefs = self.correction_coefs(modalities) + 1.0

        active_x = predictions[self.config.altup_active_idx]
        innovation = activated - active_x

        innovation_expanded = mx.broadcast_to(
            mx.expand_dims(innovation, axis=0),
            (self.config.altup_num_inputs,) + innovation.shape,
        )

        all_coefs_reshaped = all_coefs.transpose(2, 1, 0)
        all_coefs_reshaped = mx.expand_dims(all_coefs_reshaped, axis=1)

        corrected = innovation_expanded * all_coefs_reshaped
        corrected += predictions

        return corrected.astype(activated.dtype)

    def scale_corrected_output(self, corrected: mx.array):
        scale = self.correct_output_scale if self.config.altup_correct_scale else 1.0
        return corrected * scale


class Gemma3nDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int, is_kv_shared_layer: bool):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma3nAttention(config, layer_idx, is_kv_shared_layer)
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
    ):
        if isinstance(x, list):
            x = mx.stack(x, axis=0)

        predictions = self.altup.predict(x)
        active_prediction = predictions[self.config.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        attn = self.self_attn(
            active_prediction_normed,
            mask,
            cache,
        )

        attn = self.post_attention_layernorm(attn)

        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) * (2.0**-0.5)

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
        self.first_kv_shared_layer_idx = (
            config.num_hidden_layers - config.num_kv_shared_layers
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Gemma3nDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                is_kv_shared_layer=layer_idx >= self.first_kv_shared_layer_idx,
            )
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

        self.first_sliding_idx = self.config.layer_types.index("sliding_attention")
        self.first_full_idx = self.config.layer_types.index("full_attention")

        concrete_layers = self.config.layer_types[: self.first_kv_shared_layer_idx]
        shared_full_idx = (
            len(concrete_layers) - 1 - concrete_layers[::-1].index("full_attention")
        )
        shared_sliding_idx = (
            len(concrete_layers) - 1 - concrete_layers[::-1].index("sliding_attention")
        )

        self.layer_idx_to_cache_idx = []
        for i, layer_type in enumerate(self.config.layer_types):
            if i < self.first_kv_shared_layer_idx:
                self.layer_idx_to_cache_idx.append(i)
            else:
                if layer_type == "full_attention":
                    self.layer_idx_to_cache_idx.append(shared_full_idx)
                elif layer_type == "sliding_attention":
                    self.layer_idx_to_cache_idx.append(shared_sliding_idx)
                else:
                    raise NotImplementedError(f"Unknown layer type: {layer_type}")

    def __call__(
        self,
        inputs: mx.array = None,
        mask: mx.array = None,
        cache=None,
        input_embeddings: mx.array = None,
    ):
        if input_embeddings is None:
            h = self.embed_tokens(inputs) * (self.hidden_size**0.5)
        else:
            h = input_embeddings

        per_layer_inputs = self.get_per_layer_inputs(inputs)
        per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            full_mask = create_attention_mask(
                h,
                cache[self.first_full_idx :],
            )
            sliding_window_mask = create_attention_mask(
                h,
                cache[self.first_sliding_idx :],
            )
        h0 = h

        # Expand hidden_states to support per-layer inputs
        target_magnitude = mx.mean(h0**2, axis=-1, keepdims=True) ** 0.5
        epsilon_tensor = mx.array(mx.finfo(h0.dtype).min, dtype=h0.dtype)

        h_list = [h0]

        for i in range(1, self.config.altup_num_inputs):
            h_list.append(self.altup_projections[i - 1](h0))
            new_magnitude = mx.mean(h_list[i] ** 2, axis=-1, keepdims=True) ** 0.5
            h_list[i] *= target_magnitude / mx.maximum(new_magnitude, epsilon_tensor)

        h = mx.stack(h_list, axis=0)

        for i, layer in enumerate(self.layers):
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
                cache[self.layer_idx_to_cache_idx[i]],
                per_layer_input,
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
        per_layer_inputs_mask = input_ids < self.vocab_size_per_layer_input
        tokens = mx.where(per_layer_inputs_mask, input_ids, mx.zeros_like(input_ids))
        result = self.embed_tokens_per_layer(tokens) * (
            self.hidden_size_per_layer_input**0.5
        )
        return result.reshape(
            *input_ids.shape,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: mx.array,
        per_layer_inputs: mx.array,
    ) -> mx.array:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * (
            self.hidden_size**-0.5
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        return (per_layer_projection + per_layer_inputs) * (2.0**-0.5)

    def make_cache(self):
        caches = []
        for layer_type in self.config.layer_types[: self.first_kv_shared_layer_idx]:
            if layer_type == "full_attention":
                caches.append(KVCache())
            elif layer_type == "sliding_attention":
                caches.append(
                    RotatingKVCache(max_size=self.config.sliding_window, keep=0)
                )
            else:
                raise NotImplementedError(f"Unknown layer type: {layer_type}")
        return caches


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

    def make_cache(self):
        return self.language_model.make_cache()


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
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

    def make_cache(self):
        return self.model.make_cache()
