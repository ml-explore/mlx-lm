# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_local_experts: int
    num_experts_per_tok: int
    shared_intermediate_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    embedding_multiplier: float
    attention_multiplier: float
    residual_multiplier: float
    logits_scaling: float
    sliding_window: int
    layer_types: List[str]
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    rope_parameters: Optional[dict] = None

    def __post_init__(self):
        # transformers 5.x nests rope config under `rope_parameters`
        if self.rope_parameters:
            self.rope_theta = float(
                self.rope_parameters.get("rope_theta", self.rope_theta)
            )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = dim // self.n_heads
        self.scale = args.attention_multiplier
        bias = args.attention_bias

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=bias)

        # gpt-oss-style attention sinks (one learned logit per head)
        self.sinks = mx.zeros((self.n_heads,))

        self.rope = initialize_rope(
            head_dim, args.rope_theta, False, None, args.max_position_embeddings
        )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask, sinks=self.sinks
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.router = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.experts = SwitchGLU(
            args.hidden_size, args.intermediate_size, args.num_local_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        logits = self.router(x).astype(
            mx.float32
        )  # route in fp32 (HF/granitemoe parity)
        idx = mx.argpartition(logits, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        gates = mx.softmax(
            mx.take_along_axis(logits, idx, axis=-1), precise=True, axis=-1
        )
        y = self.experts(x, idx)
        return (y * gates[..., None]).sum(axis=-2).astype(y.dtype)


class SharedMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.input_linear = nn.Linear(
            args.hidden_size, args.shared_intermediate_size * 2, bias=False
        )
        self.output_linear = nn.Linear(
            args.shared_intermediate_size, args.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.input_linear(x), 2, axis=-1)
        return self.output_linear(swiglu(gate, up))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.residual_multiplier = args.residual_multiplier
        self.self_attn = Attention(args)
        self.block_sparse_moe = MoE(args)
        self.shared_mlp = SharedMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache) * (
            self.residual_multiplier
        )
        normed = self.post_attention_layernorm(h)
        mlp_out = self.block_sparse_moe(normed) + self.shared_mlp(normed)
        return h + mlp_out * self.residual_multiplier


class GraniteMoeSwaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, lt) for lt in args.layer_types]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.embedding_multiplier = args.embedding_multiplier
        self.layer_types = args.layer_types
        self.window_size = args.sliding_window
        self.ga_idx = args.layer_types.index("full_attention")
        self.swa_idx = (
            args.layer_types.index("sliding_attention")
            if "sliding_attention" in args.layer_types
            else self.ga_idx
        )

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        h = self.embed_tokens(inputs) * self.embedding_multiplier
        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(h, cache[self.ga_idx])
        swa_mask = create_attention_mask(
            h, cache[self.swa_idx], window_size=self.window_size
        )

        for layer, c in zip(self.layers, cache):
            mask = full_mask if layer.layer_type == "full_attention" else swa_mask
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    # Sliding-window layers use RotatingKVCache, which records an exact rollback
    # within the verify window — enables --draft-model speculative decoding.
    supports_speculative_rollback = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GraniteMoeSwaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.logits_scaling = args.logits_scaling

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out / self.logits_scaling

    def sanitize(self, weights):
        if any("experts.gate_proj" in k for k in weights):
            return weights  # already sanitized
        new = {}
        for k, v in weights.items():
            if k.endswith("block_sparse_moe.experts.gate_up_proj"):
                out = v.shape[1]
                new[k.replace("gate_up_proj", "gate_proj") + ".weight"] = v[
                    :, : out // 2, :
                ]
                new[k.replace("gate_up_proj", "up_proj") + ".weight"] = v[
                    :, out // 2 :, :
                ]
            elif k.endswith("block_sparse_moe.experts.down_proj"):
                new[k + ".weight"] = v
            elif k == "lm_head.weight" and self.args.tie_word_embeddings:
                continue
            else:
                new[k] = v
        return new

    def make_cache(self, max_kv_size=None):
        caches = []
        for lt in self.model.layer_types:
            if lt == "full_attention":
                caches.append(
                    RotatingKVCache(max_size=max_kv_size) if max_kv_size else KVCache()
                )
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("block_sparse_moe.router"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
