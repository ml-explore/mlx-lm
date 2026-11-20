# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.common import swiglu
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.g9v3.config import ModelArgs

__all__ = [
    "G9v3Attention",
    "G9v3MLP",
    "G9v3TopkRouter",
    "G9v3MoE",
    "G9v3DecodeLayer",
    "group_expert_select",
]


class G9v3Attention(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5

        self.use_gated_attention = getattr(config, "use_gated_attention", False)

        q_proj_dim = (
            self.num_attention_heads * self.head_dim * 2
            if self.use_gated_attention
            else self.num_attention_heads * self.head_dim
        )

        self.q_proj = nn.Linear(config.hidden_size, q_proj_dim, bias=False)
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.rope = initialize_rope(
            dims=config.head_dim,
            base=config.rope_theta,
            traditional=False,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        batch_size, query_length, dimension = hidden_states.shape

        if self.use_gated_attention:
            queries_and_gates = self.q_proj(hidden_states).reshape(
                batch_size, query_length, self.num_attention_heads, self.head_dim * 2
            )  # include gate
            queries = queries_and_gates[..., : self.head_dim].transpose(0, 2, 1, 3)
            gates = queries_and_gates[..., self.head_dim :].reshape(
                batch_size, query_length, -1
            )
        else:
            queries = (
                self.q_proj(hidden_states)
                .reshape(batch_size, query_length, self.num_attention_heads, -1)
                .transpose(0, 2, 1, 3)
            )

        keys = (
            self.k_proj(hidden_states)
            .reshape(batch_size, query_length, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(hidden_states)
            .reshape(batch_size, query_length, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        outputs = scaled_dot_product_attention(
            queries, keys, values, cache=cache, mask=mask, scale=self.scaling
        )

        outputs = outputs.transpose(0, 2, 1, 3).reshape(batch_size, query_length, -1)

        if self.use_gated_attention:
            outputs = outputs * nn.sigmoid(gates)

        return self.o_proj(outputs)


class G9v3MLP(nn.Module):

    def __init__(self, config: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()

        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(
            config.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(
            swiglu(self.gate_proj(hidden_states), self.up_proj(hidden_states))
        )


class G9v3TopkRouter(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, hidden_status: mx.array) -> mx.array:
        return group_expert_select(
            hidden_status @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class G9v3MoE(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.experts = SwitchGLU(
            input_dims=config.hidden_size,
            hidden_dims=config.moe_intermediate_size,
            num_experts=config.n_routed_experts,
        )
        self.gate = G9v3TopkRouter(config=config)
        self.shared_experts = G9v3MLP(
            config=config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:

        inds, scores = self.gate(hidden_states)
        expert_outputs = self.experts(hidden_states, indices=inds)
        expert_outputs = (
            (expert_outputs * scores[..., None])
            .sum(axis=-2)
            .astype(expert_outputs.dtype)
        )
        expert_outputs = expert_outputs + self.shared_experts(hidden_states)

        return expert_outputs


class G9v3DecodeLayer(nn.Module):

    def __init__(self, layer_idx: int, config: ModelArgs):
        super().__init__()

        self.self_attn = G9v3Attention(config=config)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = G9v3MoE(config=config)
        else:
            self.mlp = G9v3MLP(config=config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states, cache=cache, mask=mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):

    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    inds = mx.stop_gradient(inds)
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / (denominator + 1e-20)
    scores = scores * routed_scaling_factor

    return inds, scores
