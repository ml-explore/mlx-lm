# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, LinearAttentionCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    num_experts_per_tok: int
    num_local_experts: int
    shared_intermediate_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    vocab_size: int
    block_size: int = 256
    tie_word_embeddings: bool = False
    shared_moe_mode: str = "sigmoid"
    full_attn_alpha_factor: float = 3.5565588200778455
    full_attn_beta_factor: float = 1.0
    linear_attn_alpha_factor: float = 3.5565588200778455
    linear_attn_beta_factor: float = 1.0
    mlp_alpha_factor: float = 3.5565588200778455
    mlp_beta_factor: float = 1.0
    layer_types: List[str] = None
    head_dim: Optional[int] = None


class MiniMaxAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = (
            args.hidden_size // args.num_attention_heads
            if args.head_dim is None
            else args.head_dim
        )

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.rope = nn.RoPE(
            args.rotary_dim,
            traditional=False,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
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
        return self.o_proj(output)


class MiniMaxLightningAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = (
            args.hidden_size // args.num_attention_heads
            if args.head_dim is None
            else args.head_dim
        )
        self.num_hidden_layers = args.num_hidden_layers
        self.block_size = args.block_size

        self.norm = nn.RMSNorm(self.head_dim * self.num_attention_heads, eps=1e-6)
        self.qkv_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim * 3, bias=False
        )
        self.out_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.output_gate = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )

        self.rope = nn.RoPE(
            args.rotary_dim,
            traditional=False,
            base=args.rope_theta,
        )

        self.slope_rate = self.get_slope_rate()
        self.query_decay, self.key_decay, self.diagonal_decay = self.decay_factors(
            self.slope_rate
        )

    @property
    def ratio(self):
        return mx.exp(-self.slope_rate)

    def get_slope_rate(self):
        base = 1 / (2 ** (8 / self.num_attention_heads))
        exp = mx.arange(self.num_attention_heads) + 1
        factor = 1 - self.layer_idx / (self.num_hidden_layers - 1 + 1e-5) + 1e-5
        rate = base**exp
        rate = rate * factor
        return rate[:, None, None]

    def decay_factors(self, slope_rate):
        block_size_range = mx.arange(self.block_size) + 1
        query_decay = mx.exp(-slope_rate * block_size_range[:, None])
        key_decay = mx.exp(-slope_rate * (self.block_size - block_size_range[:, None]))

        diagonal_decay = block_size_range[:, None] - block_size_range[None, :]
        diagonal_decay = diagonal_decay[None, None, :, :]
        diagonal_decay = slope_rate * diagonal_decay
        diagonal_decay = mx.where(diagonal_decay >= 0, -diagonal_decay, float("-inf"))
        diagonal_decay = mx.exp(diagonal_decay)

        return query_decay, key_decay, diagonal_decay

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if isinstance(mask, str) or mask is None:
            mask = None
        B, L, D = x.shape

        qkv = nn.silu(self.qkv_proj(x))
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        k = k.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, :, None, None]
            v = v * mask

        if cache is not None and cache.kv is not None:
            q = self.rope(q, offset=cache.kv.offset)
            k = self.rope(k, offset=cache.kv.offset)
            k, v = cache.kv.update_and_fetch(k, v)
            state = cache.recurrent.cache
        else:
            q = self.rope(q)
            k = self.rope(k)
            state = None

        if state is None:
            state = mx.zeros(
                (B, self.num_attention_heads, self.head_dim, self.head_dim),
                dtype=k.dtype,
            )

        attn_output = []
        for i in range(L):
            qi = q[:, :, i : i + 1, :]
            ki = k[:, :, i : i + 1, :]
            vi = v[:, :, i : i + 1, :]

            kv_update = mx.matmul(ki.transpose(0, 1, 3, 2), vi)
            state = self.ratio * state + kv_update
            out_i = mx.matmul(qi, state)
            attn_output.append(out_i)

        y = mx.concatenate(attn_output, axis=2)
        y = y.transpose(0, 2, 1, 3).reshape(
            B, L, self.num_attention_heads * self.head_dim
        )
        y = self.norm(y)
        y = nn.sigmoid(self.output_gate(x)) * y

        if cache is not None and cache.kv is not None:
            cache.recurrent.cache = state

        return self.out_proj(y)


class MiniMaxSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok

        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.intermediate_size, args.num_local_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        inds = mx.stop_gradient(
            mx.argpartition(-gates, kth=self.num_experts_per_tok - 1, axis=-1)[
                ..., : self.num_experts_per_tok
            ]
        )
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y


class MiniMaxDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.mlp_alpha_factor = args.mlp_alpha_factor
        self.mlp_beta_factor = args.mlp_beta_factor

        if self.layer_type == "linear_attention":
            self.self_attn = MiniMaxLightningAttention(args, layer_idx=layer_idx)
            self.attn_alpha_factor = args.linear_attn_alpha_factor
            self.attn_beta_factor = args.linear_attn_beta_factor
        else:
            self.self_attn = MiniMaxAttention(args)
            self.attn_alpha_factor = args.full_attn_alpha_factor
            self.attn_beta_factor = args.full_attn_beta_factor

        self.block_sparse_moe = MiniMaxSparseMoeBlock(args)

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
        r = (
            self.input_layernorm(x) * self.attn_alpha_factor
            + self.self_attn(x, mask, cache) * self.attn_beta_factor
        )
        r = (
            self.block_sparse_moe(self.post_attention_layernorm(x))
            * self.mlp_alpha_factor
            + r * self.mlp_beta_factor
        )
        return r


class MiniMaxModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = [
            MiniMaxDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]

        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
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
        self.model = MiniMaxModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs=inputs, mask=mask, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in weights:
            return weights
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
            for orig_name, new_name in mapping.items():
                if f"{prefix}.block_sparse_moe.experts.0.{orig_name}.weight" in weights:
                    to_join = [
                        weights.pop(
                            f"{prefix}.block_sparse_moe.experts.{e}.{orig_name}.weight"
                        )
                        for e in range(self.args.num_local_experts)
                    ]
                    weights[
                        f"{prefix}.block_sparse_moe.switch_mlp.{new_name}.weight"
                    ] = mx.stack(to_join)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for l in self.layers:
            if l.layer_type == "linear_attention":
                caches.append(LinearAttentionCache())
            elif l.layer_type == "full_attention":
                caches.append(KVCache())
        return caches
