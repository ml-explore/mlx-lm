from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
from mlx import nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


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
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / (denominator + 1e-20)
    scores = scores * routed_scaling_factor

    return inds, scores


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "g9v3"
    vocab_size: int = 130560
    hidden_size: int = 2048
    num_attention_heads: int = 32
    head_dim: int = 128
    num_key_value_heads: int = 2
    use_gated_attention: Optional[bool] = False
    rope_theta: int = 5000000
    rope_scaling: Optional[Dict] = None
    first_k_dense_replace: int = 1
    intermediate_size: int = 8192
    n_routed_experts: int = 320
    num_experts_per_tok: int = 32
    routed_scaling_factor: float = 3.66
    n_group: Optional[int] = 1
    topk_group: Optional[int] = 1
    norm_topk_prob: Optional[bool] = True
    moe_intermediate_size: int = 512
    n_shared_experts: int = 1
    rms_norm_eps: float = 1e-06
    num_hidden_layers: int = 38
    max_position_embeddings: int = 131072


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


class G9v3Model(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(self.vocab_size, dims=config.hidden_size)
        self.layers = [
            G9v3DecodeLayer(layer_idx=idx, config=config)
            for idx in range(config.num_hidden_layers)
        ]

        self.norm = nn.RMSNorm(dims=config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:

        hidden_states = self.embed_tokens(hidden_states)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(hidden_states, cache[0])

        for c, block in zip(cache, self.layers):
            hidden_states = block(hidden_states=hidden_states, cache=c, mask=mask)

        return self.norm(hidden_states)


class Model(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = G9v3Model(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:

        hidden_states = self.model(hidden_states=hidden_states, cache=cache)
        return self.lm_head(hidden_states)

    def sanitize(self, weights):
        for layer_idx in range(self.args.num_hidden_layers):

            if layer_idx < self.args.first_k_dense_replace:
                continue

            prefix = f"model.layers.{layer_idx}.mlp.experts"

            for name in ["gate_proj", "up_proj", "down_proj"]:
                key = f"{prefix}.0.{name}.weight"
                if key not in weights:
                    continue

                expert_weights = [
                    weights.pop(f"{prefix}.{expert_idx}.{name}.weight")
                    for expert_idx in range(self.args.n_routed_experts)
                ]
                weights[f"{prefix}.{name}.weight"] = mx.stack(expert_weights)

        return weights

    @property
    def layers(self):
        return self.model.layers
