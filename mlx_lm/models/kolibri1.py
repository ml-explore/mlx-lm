# Copyright © 2026 Apple Inc.

# Ported from Aleph Alpha's Apache-2.0 vLLM plugin:
# https://github.com/Aleph-Alpha/aleph-alpha-inference

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from .cache import KVCache, RotatingKVCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    sliding_window: int
    layer_types: List[str]
    rope_theta: float = 10000.0
    norm_topk_prob: bool = False
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 262144


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, is_full_attention: bool):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        # vLLM's get_rope defaults to the neox (non-interleaved) layout.
        self.rope = (
            None
            if is_full_attention
            else nn.RoPE(head_dim, traditional=False, base=args.rope_theta)
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = mx.zeros((args.num_experts,), dtype=mx.float32)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_experts = MLP(
            args.hidden_size, args.shared_expert_intermediate_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        logits = self.gate(x).astype(mx.float32)
        k = self.top_k
        biased = logits + self.expert_bias.astype(mx.float32)
        inds = mx.stop_gradient(mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k])
        scores = mx.sigmoid(mx.take_along_axis(logits, inds, axis=-1))
        if self.norm_topk_prob:
            scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        return y + self.shared_experts(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_full_attention = args.layer_types[layer_idx] == "full_attention"
        self.self_attn = Attention(args, self.is_full_attention)
        self.mlp = SparseMoeBlock(args)

        eps = args.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attn_norm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_ffn_norm = nn.RMSNorm(args.hidden_size, eps=eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.post_attn_norm(self.self_attn(self.input_layernorm(x), mask, cache))
        h = x + r
        r = self.post_ffn_norm(self.mlp(self.post_attention_layernorm(h)))
        return h + r


class Kolibri1Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.window_size = args.sliding_window
        self.swa_idx = args.layer_types.index("sliding_attention")
        self.fa_idx = args.layer_types.index("full_attention")

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

        full_mask = create_attention_mask(h, cache[self.fa_idx])
        swa_mask = create_attention_mask(
            h, cache[self.swa_idx], window_size=self.window_size
        )

        for layer, c in zip(self.layers, cache):
            mask = full_mask if layer.is_full_attention else swa_mask
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Kolibri1Model(args)
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
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        def dequant(weight, scale_inv, bs=128):
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            m, n = weight.shape
            pad_bottom, pad_side = (-m) % bs, (-n) % bs
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
            weight = weight.reshape(
                (m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs
            )
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_bottom, n + pad_side
            )
            return weight[:m, :n].astype(mx.bfloat16)

        new_weights = {}
        for k, v in weights.items():
            if k.endswith(".weight_scale_inv"):
                continue
            scale_key = k + "_scale_inv"
            if scale_key in weights:
                v = dequant(v, weights[scale_key])
            k = k.replace(".moe.router.expert_bias", ".mlp.expert_bias")
            new_weights[k] = v
        weights = new_weights

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.mlp"
            for n in ["gate_proj", "down_proj", "up_proj"]:
                if f"{prefix}.experts.0.{n}.weight" in weights:
                    weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(
                        [
                            weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                            for e in range(self.args.num_experts)
                        ]
                    )
        return weights

    @property
    def cast_predicate(self):
        return lambda k: "expert_bias" not in k

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # The router is tiny and routing decisions are sensitive to it.
            if path.endswith("mlp.gate"):
                return False
            if path.endswith("lm_head") or path.endswith("embed_tokens"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def make_cache(self):
        return [
            (
                KVCache()
                if layer.is_full_attention
                else RotatingKVCache(max_size=self.args.sliding_window)
            )
            for layer in self.model.layers
        ]

    @property
    def layers(self):
        return self.model.layers
