# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "sarvam_moe"
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    use_qkv_bias: bool = False
    use_bias: bool = False
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    num_experts: int = 128
    num_shared_experts: int = 1
    num_experts_per_tok: int = 6
    n_group: int = 1
    topk_group: int = 1
    moe_intermediate_size: int = 1024
    first_k_dense_replace: int = 1
    head_dim: int = 256
    use_qk_norm: bool = True
    moe_router_enable_expert_bias: bool = True
    routed_scaling_factor: float = 2.5
    moe_shared_expert_intermediate_size: Optional[int] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_scaling:
            if not isinstance(self.rope_scaling, dict):
                self.rope_scaling = None


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or (dim // self.n_heads)
        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            dim,
            (self.n_heads + 2 * self.n_kv_heads) * self.head_dim,
            bias=args.use_qkv_bias,
        )

        if args.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        else:
            self.query_layernorm = None
            self.key_layernorm = None

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=args.rope_theta,
        )

        self.dense = nn.Linear(self.n_heads * self.head_dim, dim, bias=args.use_bias)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.query_key_value(x)
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim

        queries, keys, values = mx.split(
            qkv, [q_size, q_size + kv_size], axis=-1
        )

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if self.query_layernorm is not None:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

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


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_experts
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor

        scale = args.hidden_size**-0.5
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(args.num_experts, args.hidden_size),
        )

        if args.moe_router_enable_expert_bias:
            self.expert_bias = mx.zeros((args.num_experts,))
        else:
            self.expert_bias = None

    def _topk(self, x: mx.array, k: int):
        inds = mx.argpartition(x, kth=-k, axis=-1)[..., -k:]
        vals = mx.take_along_axis(x, inds, axis=-1)
        order = mx.argsort(vals, axis=-1)[..., ::-1]
        inds = mx.take_along_axis(inds, order, axis=-1)
        vals = mx.take_along_axis(vals, order, axis=-1)
        return inds, vals

    def group_limited_topk(self, scores: mx.array):
        if self.n_group == 1:
            return self._topk(scores, self.top_k)

        num_tokens, num_experts = scores.shape
        group_scores = scores.reshape(num_tokens, self.n_group, -1)
        top2_vals = mx.topk(group_scores, k=2, axis=-1)
        group_score_sums = top2_vals.sum(axis=-1)

        group_idx, _ = self._topk(group_score_sums, k=self.topk_group)

        group_mask = mx.zeros((num_tokens, self.n_group), dtype=scores.dtype)
        batch_col = mx.arange(num_tokens)[:, None]
        group_mask[batch_col, group_idx] = 1

        experts_per_group = num_experts // self.n_group
        score_mask = mx.repeat(group_mask[:, :, None], experts_per_group, axis=2)
        score_mask = score_mask.reshape(num_tokens, num_experts)

        masked_scores = mx.where(score_mask > 0, scores, -1e9)
        return self._topk(masked_scores, self.top_k)

    def __call__(self, x: mx.array):
        logits = x @ self.weight.T
        scores = mx.sigmoid(logits)

        scores_for_routing = scores
        if self.expert_bias is not None:
            scores_for_routing = scores_for_routing + self.expert_bias

        B, L, E = scores_for_routing.shape
        scores_flat = scores_for_routing.reshape(-1, E)

        inds_flat, _ = self.group_limited_topk(scores_flat)
        inds = inds_flat.reshape(B, L, self.top_k)

        gathered_scores = mx.take_along_axis(scores, inds, axis=-1)

        if self.top_k > 1:
            denom = gathered_scores.sum(axis=-1, keepdims=True) + 1e-20
            topk_weight = gathered_scores / denom
        else:
            topk_weight = gathered_scores

        topk_weight = topk_weight * self.routed_scaling_factor
        return inds, topk_weight


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = Gate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=False,
        )

        if args.num_shared_experts > 0:
            shared_inter_size = (
                args.moe_shared_expert_intermediate_size
                or args.moe_intermediate_size * args.num_shared_experts
            )
            self.shared_experts = MLP(args, shared_inter_size)
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        topk_inds, topk_weights = self.gate(x)

        y = self.switch_mlp(x, topk_inds)
        y = (y * topk_weights[..., None]).sum(axis=-2)

        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.attention = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        is_moe = (args.num_experts > 0) and (layer_idx >= args.first_k_dense_replace)

        if is_moe:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.attention(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class SarvamMoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = SarvamMoeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        keys_to_remove = [
            k for k in weights if "input_scale" in k or "weight_scale" in k
        ]
        for k in keys_to_remove:
            weights.pop(k, None)

        weights.pop("model.rotary_emb.inv_freq", None)

        if "model.word_embeddings.weight" in weights:
            weights["model.embed_tokens.weight"] = weights.pop(
                "model.word_embeddings.weight"
            )

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp"

            # Stack individual expert weights into SwitchGLU format
            if f"{prefix}.experts.0.gate_proj.weight" in weights:
                for n in ["gate_proj", "up_proj", "down_proj"]:
                    to_join = [
                        weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

            # Handle already-converted checkpoints with experts.switch_mlp path
            for n in ["gate_proj", "up_proj", "down_proj"]:
                for suffix in ["weight", "scales", "biases"]:
                    old = f"{prefix}.experts.switch_mlp.{n}.{suffix}"
                    new = f"{prefix}.switch_mlp.{n}.{suffix}"
                    if old in weights:
                        weights[new] = weights.pop(old)

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if "lm_head" in path:
                return False
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim or (
            self.args.hidden_size // self.args.num_attention_heads
        )

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
