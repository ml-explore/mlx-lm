# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 1024
    head_dim: int = 128
    num_hidden_layers: int = 36
    intermediate_size: int = 1024
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    rope_theta: float = 50000.0
    vocab_size: int = 256000
    layer_norm_eps: float = 1e-05
    logit_scale: float = 0.0625
    attention_bias: bool = False
    layer_norm_bias: bool = False
    sliding_window: int = 4096
    sliding_window_pattern: int = 4
    num_experts: int = 128
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    num_shared_experts: Optional[int] = None
    moe_num_shared_experts: int = 4
    moe_gate_act: str = "sigmoid"
    expert_selection_fn: Optional[str] = None
    shared_expert_combination_strategy: str = "average"
    rms_norm_eps: Optional[float] = None
    first_k_dense_replace: int = 0
    prefix_dense_intermediate_size: Optional[int] = None
    prefix_dense_sliding_window_pattern: int = 1
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.num_shared_experts is not None:
            self.moe_num_shared_experts = self.num_shared_experts
        if self.expert_selection_fn is not None:
            self.moe_gate_act = self.expert_selection_fn
        if self.prefix_dense_intermediate_size is None:
            self.prefix_dense_intermediate_size = self.intermediate_size


def is_prefix_dense_layer(args: ModelArgs, layer_idx: int):
    return layer_idx < args.first_k_dense_replace


def is_sliding_layer(args: ModelArgs, layer_idx: int):
    if is_prefix_dense_layer(args, layer_idx):
        return False
    if args.layer_types is not None:
        return args.layer_types[layer_idx] == "sliding_attention"
    return (layer_idx + 1) % args.sliding_window_pattern != 0


def norm_layer(args: ModelArgs):
    if args.rms_norm_eps is not None:
        return nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    return nn.LayerNorm(
        args.hidden_size, eps=args.layer_norm_eps, bias=args.layer_norm_bias
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        attetion_bias = args.attention_bias

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attetion_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attetion_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attetion_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attetion_bias)

        self.rope = nn.RoPE(head_dim, traditional=True, base=args.rope_theta)

        self.use_sliding_window = is_sliding_layer(args, layer_idx)
        self.force_rope = (
            is_prefix_dense_layer(args, layer_idx)
            and args.prefix_dense_sliding_window_pattern == 1
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Cohere2Moe applies RoPE to sliding layers and optionally to prefix
        # dense full-attention layers.
        if self.use_sliding_window or self.force_rope:
            if cache is None:
                queries = self.rope(queries)
                keys = self.rope(keys)
            else:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        sdpa_type = mx.float32 if queries.dtype == mx.float16 else queries.dtype
        output = scaled_dot_product_attention(
            queries.astype(sdpa_type),
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        ).astype(queries.dtype)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class CohereMoeSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.intermediate_size

        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)

        if getattr(args, "moe_num_shared_experts", 0) > 0:
            shared_intermediate_size = (
                args.intermediate_size * args.moe_num_shared_experts
            )
            self.shared_experts = MLP(
                args.hidden_size, shared_intermediate_size,
            )
            self.shared_expert_combination_strategy = \
                args.shared_expert_combination_strategy
            assert self.shared_expert_combination_strategy in [
                "average", "sum"
            ], "shared_expert_combination_strategy "
            "must be one of ['average', 'sum']"
        else:
            self.shared_experts = None
            self.shared_expert_combination_strategy = None

        if args.moe_gate_act == "softmax":
            self.gate_act = nn.Softmax()
        elif args.moe_gate_act == "sigmoid":
            self.gate_act = nn.Sigmoid()
        else:
            raise ValueError(f"{args.moe_gate_act} is not supported.")

    def __call__(
        self,
        x: mx.array,
    ):
        gates = self.gate(x)
        gates = self.gate_act(gates.astype(mx.float32))

        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.maximum(scores.sum(axis=-1, keepdims=True), 1e-12)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

        if self.shared_experts is not None:
            if self.shared_expert_combination_strategy == "average":
                y = (y + self.shared_experts(x)) / 2
            else:
                y = y + self.shared_experts(x)

        return y


class CohereMoEDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads

        self.self_attn = Attention(args, layer_idx)
        self.mlp = (
            MLP(args.hidden_size, args.prefix_dense_intermediate_size)
            if is_prefix_dense_layer(args, layer_idx)
            else CohereMoeSparseMoeBlock(args)
        )
        self.input_layernorm = norm_layer(args)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:

        h = self.input_layernorm(x)
        attn_h = self.self_attn(h, mask, cache)
        ff_h = self.mlp(h)

        return attn_h + ff_h + x


class CohereModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.window_size = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            CohereMoEDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = norm_layer(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            mask = create_attention_mask(
                h,
                c,
                window_size=(
                    self.window_size if layer.self_attn.use_sliding_window else None
                ),
            )

            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = CohereModel(args)
        self.args = args

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        out = self.model.embed_tokens.as_linear(out)
        out = out * self.model.args.logit_scale
        return out

    def make_cache(self):
        caches = []
        for i in range(self.args.num_hidden_layers):
            if is_sliding_layer(self.args, i):
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights):
        for l in range(self.args.num_hidden_layers):
            if is_prefix_dense_layer(self.args, l):
                continue
            prefix = f"model.layers.{l}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{n}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{n}.{k}")
                            for e in range(self.args.num_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{n}.{k}"] = mx.stack(to_join)

        for key in list(weights.keys()):
            if "rotary_emb.inv_freq" in key:
                weights.pop(key)
            elif key.endswith(".bias"):
                if ".mlp." in key:
                    weights.pop(key)
                elif ".self_attn." in key and not self.args.attention_bias:
                    weights.pop(key)
                elif "layernorm" in key.lower() and not self.args.layer_norm_bias:
                    weights.pop(key)

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, module):
            if ".self_attn." in path:
                return False
            if ".mlp.gate" in path and "gate_proj" not in path:
                return False
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers
