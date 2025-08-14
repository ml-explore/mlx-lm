# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    moe_intermediate_size: int
    num_experts: int
    num_shared_experts: int
    norm_topk_prob: bool
    num_attention_heads: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    first_k_dense_replace: int
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    use_bias: bool = False
    use_qkv_bias: bool = False
    norm_head: bool = False
    norm_softmax: bool = False


class BailingMoeMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else args.intermediate_size
        )

        self.gate_proj = nn.Linear(
            args.hidden_size, self.intermediate_size, bias=args.use_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, args.hidden_size, bias=args.use_bias
        )
        self.up_proj = nn.Linear(
            args.hidden_size, self.intermediate_size, bias=args.use_bias
        )

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class BailingMoeAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // self.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            args.hidden_size,
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=args.use_qkv_bias,
        )
        self.dense = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.use_bias,
        )

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.query_key_value(x)

        q_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        q, k, v = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)

        queries = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = k.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = v.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)


class BailingMoeGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size

        self.gate_proj = nn.Linear(self.gating_dim, self.num_experts, bias=False)

    def __call__(self, hidden_states):
        B, L, D = hidden_states.shape
        x = hidden_states.reshape(-1, D)

        logits = self.gate_proj(x)
        scores = mx.softmax(logits, axis=-1, precise=True)

        topk_idx = mx.argpartition(scores, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        topk_scores = mx.take_along_axis(scores, topk_idx, axis=-1)

        if self.top_k > 1 and self.norm_topk_prob:
            denom = mx.sum(topk_scores, axis=-1, keepdims=True)
            topk_scores = topk_scores / mx.maximum(denom, 1e-9)

        return topk_idx, topk_scores


class BailingMoeSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_experts_per_tok = args.num_experts_per_tok

        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.use_bias,
        )

        self.gate = BailingMoeGate(config=args)

        if args.num_shared_experts > 0:
            self.shared_experts = BailingMoeMLP(
                args=args,
                intermediate_size=args.moe_intermediate_size * args.num_shared_experts,
            )
        else:
            self.shared_experts = None

    def __call__(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        if self.shared_experts is not None:
            identity = hidden_states

        x = hidden_states.reshape(-1, hidden_dim)

        expert_indices, expert_weights = self.gate(hidden_states)
        expert_outputs = self.switch_mlp(x, expert_indices)

        weighted_output = mx.sum(expert_outputs * expert_weights[..., None], axis=-2)
        output = weighted_output.reshape(batch_size, seq_len, hidden_dim)

        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)

        return output


class BailingMoeDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attention = BailingMoeAttention(args)

        self.mlp = (
            BailingMoeSparseMoeBlock(args)
            if (
                args.num_experts is not None and layer_idx >= args.first_k_dense_replace
            )
            else BailingMoeMLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.attention(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class BailingMoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            BailingMoeDecoderLayer(args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        h = self.word_embeddings(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.norm_head = args.norm_head
        self.model_type = args.model_type
        self.model = BailingMoeModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.model(inputs, mask, cache)
        return self.lm_head(h)

    def sanitize(self, weights):
        if self.norm_head:
            w = weights["lm_head.weight"]
            dtype = w.dtype
            weight_norm = (
                mx.linalg.norm(w.astype(mx.float32), axis=0, keepdims=True) + 1e-7
            )
            weights["lm_head.weight"] = (w / weight_norm).astype(dtype)

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"

            if l >= self.args.first_k_dense_replace:
                for m in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                            to_join = [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                                for e in range(self.args.num_experts)
                            ]
                            weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(
                                to_join
                            )

                if f"{prefix}.mlp.gate.weight" in weights:
                    gate_weight = weights.pop(f"{prefix}.mlp.gate.weight")
                    weights[f"{prefix}.mlp.gate.gate_proj.weight"] = gate_weight

                if f"{prefix}.mlp.gate.bias" in weights:
                    gate_bias = weights.pop(f"{prefix}.mlp.gate.bias")
                    weights[f"{prefix}.mlp.gate.gate_proj.bias"] = gate_bias

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers
