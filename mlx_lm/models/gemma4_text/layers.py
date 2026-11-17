# Copyright © 2025 Apple Inc.

from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.gemma4_text.config import ModelArgs

__all__ = [
    "Attention",
    "DecoderLayer",
    "Experts",
    "GeGLU",
    "MLP",
    "RMSNormNoScale",
    "Router",
    "geglu",
]


class RMSNormNoScale(nn.Module):
    """RMSNorm without learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


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
        return self.down_proj(geglu(self.gate_proj(x), self.up_proj(x)))


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

        top_k_indices = mx.argpartition(
            expert_scores, kth=-self.config.top_k_experts, axis=-1
        )
        top_k_indices = mx.stop_gradient(
            top_k_indices[..., -self.config.top_k_experts :]
        )

        top_k_weights = mx.take_along_axis(expert_scores, top_k_indices, axis=-1)
        top_k_weights = mx.softmax(top_k_weights, axis=-1)
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
        w = mx.expand_dims(top_k_weights, -1)
        y = self.switch_glu(x, top_k_indices)

        return (w * y).sum(-2)


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.has_kv = layer_idx < config.num_hidden_layers - config.num_kv_shared_layers

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
        if self.has_kv:
            self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
            if not self.use_k_eq_v:
                self.v_proj = nn.Linear(
                    dim, self.n_kv_heads * self.head_dim, bias=False
                )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if self.has_kv:
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
        elif not self.has_kv:
            raise ValueError(
                f"Layer {self.layer_idx} is a KV-shared layer but received no shared_kv"
            )
        else:
            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            values = keys
            if not self.use_k_eq_v:
                values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            offset = mx.array(cache.offset) if cache is not None else 0

            keys = self.k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)

            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

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
