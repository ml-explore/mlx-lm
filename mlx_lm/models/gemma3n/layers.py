# Copyright © 2025 Apple Inc.

import math
from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.models.gemma3n.config import TextConfig

__all__ = [
    "Gemma3nAltUp",
    "Gemma3nAttention",
    "Gemma3nDecoderLayer",
    "Gemma3nLaurelBlock",
    "MLP",
    "RMSNoScale",
    "gelu_topk",
]


class RMSNoScale(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, None, self.eps)


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
        offset: int = 0,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        queries = queries.reshape(B, L, -1, self.head_dim)
        queries = self.q_norm(queries)

        if self.is_kv_shared_layer and cache is not None:
            # For shared layers, retrieve KV from the designated cache layer
            keys, values = cache.keys_and_values()

        else:
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

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output)


@partial(mx.compile, shapeless=True)
def gelu_topk(inputs, std_multiplier):
    inputs_mean = mx.mean(inputs, axis=-1, keepdims=True)
    inputs_std = mx.std(inputs, axis=-1, keepdims=True)
    cutoff_x = inputs_mean + inputs_std * std_multiplier.astype(inputs_std.dtype)
    return nn.gelu_approx(mx.maximum(0, inputs - cutoff_x))


class MLP(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.intermediate_size[layer_idx]
            if isinstance(config.intermediate_size, list)
            else config.intermediate_size
        )
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
            activations = gelu_topk(gate_proj, self._std_multiplier)
        else:
            activations = nn.gelu_approx(gate_proj)
        up_proj = self.up_proj(x)
        down_proj = self.down_proj(activations * up_proj)
        return down_proj


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

        x_up = x.astype(mx.float32)
        x_permuted = x_up.transpose(1, 2, 3, 0)
        predictions = mx.matmul(x_permuted, all_coefs)
        predictions = predictions.transpose(3, 0, 1, 2)
        predictions += x_up
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

        all_coefs = all_coefs.moveaxis(2, 0)
        corrected = innovation[None] * all_coefs[..., None]
        corrected += predictions

        return corrected.astype(activated.dtype)


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
        offset: int = 0,
    ):
        predictions = self.altup.predict(x)
        active_prediction = predictions[self.config.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        attn = self.self_attn(
            active_prediction_normed,
            mask,
            cache,
            offset=offset,
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
            first_prediction = first_prediction * self.altup.correct_output_scale

        first_prediction = self.per_layer_input_gate(first_prediction)
        first_prediction = nn.gelu_approx(first_prediction)

        first_prediction = mx.multiply(first_prediction, per_layer_input)

        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)

        corrected_predictions[1:] = corrected_predictions[1:] + first_prediction

        return corrected_predictions
