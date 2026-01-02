# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_linear

from .base import BaseModelArgs, create_causal_mask, scaled_dot_product_attention
from .cache import CacheList, KVCache, RotatingKVCache
from .rope_utils import initialize_rope


@partial(mx.compile, shapeless=True)
def _compute_gate(query: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    gate_logits = mx.sum(query * weight[None, :, None, :], axis=-1)
    gate_logits = gate_logits + bias[None, :, None]
    return mx.sigmoid(gate_logits)[..., None]


@partial(mx.compile, shapeless=True)
def _silu_mul(gate: mx.array, up: mx.array) -> mx.array:
    return nn.silu(gate) * up


@partial(mx.compile, shapeless=True)
def _mix_attention(
    gate: mx.array, attn_global: mx.array, attn_local: mx.array
) -> mx.array:
    return gate * attn_global + (1 - gate) * attn_local


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: int = 131072
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 500000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    loop_num: int = 2
    loop_window_size: int = 64

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class LoopGateProjection(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.weight = mx.zeros((num_heads, head_dim))
        self.bias = mx.zeros((num_heads,))

    def __call__(self, query: mx.array) -> mx.array:
        return _compute_gate(query, self.weight, self.bias)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        self.rope = initialize_rope(
            head_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def get_qkv(
        self, x: mx.array, offset: int = 0
    ) -> Tuple[mx.array, mx.array, mx.array]:
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        return queries, keys, values

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        offset = cache.offset if cache is not None else 0

        queries, keys, values = self.get_qkv(x, offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    def forward_with_kv(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

    def forward_with_precomputed_qkv(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, _, L, _ = queries.shape

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=args.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_silu_mul(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
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
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r

    def forward_loop2(
        self,
        x: mx.array,
        k1: mx.array,
        v1: mx.array,
        gate_proj: LoopGateProjection,
        mask: Optional[mx.array] = None,
        offset: int = 0,
        window_size: int = 64,
    ) -> mx.array:
        B, L, _ = x.shape
        h_norm = self.input_layernorm(x)

        q2, k2, v2 = self.self_attn.get_qkv(h_norm, offset)
        gate = gate_proj(q2)

        attn_global = self.self_attn.forward_with_kv(q2, k1, v1, mask)

        if L <= window_size:
            attn_local = self.self_attn.forward_with_kv(q2, k2, v2, mask)
        else:
            window_mask = create_causal_mask(L, offset, window_size=window_size)
            attn_local = self.self_attn.forward_with_kv(q2, k2, v2, window_mask)

        mixed = _mix_attention(gate, attn_global, attn_local)
        mixed = mixed.transpose(0, 2, 1, 3).reshape(B, L, -1)
        r = self.self_attn.o_proj(mixed)

        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class IQuestLoopCoderModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.gate_projections = [
            LoopGateProjection(args.num_attention_heads, args.head_dim)
            for _ in range(args.num_hidden_layers)
        ]
        self.loop_num = args.loop_num
        self.loop_window_size = args.loop_window_size

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Tuple[KVCache, KVCache]]] = None,
    ):
        h = self.embed_tokens(inputs)
        is_prefill = cache is None or cache[0][0].offset == 0

        if is_prefill:
            return self._forward_prefill(h, cache)
        else:
            return self._forward_generate(h, cache)

    def _forward_prefill(
        self,
        h: mx.array,
        cache: Optional[List[Tuple[KVCache, KVCache]]],
    ):
        L = h.shape[1]
        mask = create_causal_mask(L) if L > 1 else None

        loop1_kv = []
        for layer in self.layers:
            h_norm = layer.input_layernorm(h)
            q1, k1, v1 = layer.self_attn.get_qkv(h_norm, offset=0)
            loop1_kv.append((k1, v1))

            r = layer.self_attn.forward_with_precomputed_qkv(q1, k1, v1, mask, None)
            h = h + r
            r = layer.mlp(layer.post_attention_layernorm(h))
            h = h + r

        for _ in range(2, self.loop_num + 1):
            for layer, (k1, v1), gate_proj in zip(
                self.layers, loop1_kv, self.gate_projections
            ):
                h = layer.forward_loop2(
                    h,
                    k1,
                    v1,
                    gate_proj,
                    mask,
                    offset=0,
                    window_size=self.loop_window_size,
                )

        if cache is not None:
            for layer, c, (k1, v1) in zip(self.layers, cache, loop1_kv):
                h_norm = layer.input_layernorm(h)
                _, k_final, v_final = layer.self_attn.get_qkv(h_norm, offset=0)
                c[0].update_and_fetch(k1, v1)
                c[1].update_and_fetch(k_final, v_final)

        return self.norm(h)

    def _forward_generate(
        self,
        h: mx.array,
        cache: List[Tuple[KVCache, KVCache]],
    ):
        B, L, _ = h.shape
        offset = cache[0][0].offset

        for layer, c in zip(self.layers, cache):
            h_norm = layer.input_layernorm(h)
            q1, k1, v1 = layer.self_attn.get_qkv(h_norm, offset)
            k1_full, v1_full = c[0].update_and_fetch(k1, v1)

            out = scaled_dot_product_attention(
                q1, k1_full, v1_full, cache=c[0], scale=layer.self_attn.scale, mask=None
            )
            r = layer.self_attn.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))

            h = h + r
            r = layer.mlp(layer.post_attention_layernorm(h))
            h = h + r

        for _ in range(2, self.loop_num + 1):
            for layer, c, gate_proj in zip(self.layers, cache, self.gate_projections):
                h_norm = layer.input_layernorm(h)
                q2, k2, v2 = layer.self_attn.get_qkv(h_norm, offset)
                gate = gate_proj(q2)

                k1_full, v1_full = c[0].state
                attn_global = layer.self_attn.forward_with_kv(
                    q2, k1_full, v1_full, mask=None, cache=c[0]
                )

                k2_window, v2_window = c[1].update_and_fetch(k2, v2)
                attn_local = layer.self_attn.forward_with_kv(
                    q2, k2_window, v2_window, mask=None, cache=c[1]
                )

                mixed = _mix_attention(gate, attn_global, attn_local)
                r = layer.self_attn.o_proj(
                    mixed.transpose(0, 2, 1, 3).reshape(B, L, -1)
                )

                h = h + r
                r = layer.mlp(layer.post_attention_layernorm(h))
                h = h + r

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = IQuestLoopCoderModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()

        for i, layer in enumerate(self.model.layers):
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, "all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, "all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.n_heads //= N
            layer.self_attn.n_kv_heads //= N

            layer.mlp.gate_proj = shard_linear(
                layer.mlp.gate_proj, "all-to-sharded", group=group
            )
            layer.mlp.down_proj = shard_linear(
                layer.mlp.down_proj, "sharded-to-all", group=group
            )
            layer.mlp.up_proj = shard_linear(
                layer.mlp.up_proj, "all-to-sharded", group=group
            )

            gate_proj = self.model.gate_projections[i]
            heads_per_rank = gate_proj.num_heads // N
            start = rank * heads_per_rank
            end = start + heads_per_rank
            gate_proj.weight = gate_proj.weight[start:end, :]
            gate_proj.bias = gate_proj.bias[start:end]
            gate_proj.num_heads = heads_per_rank

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            CacheList(KVCache(), RotatingKVCache(max_size=self.args.loop_window_size))
            for _ in self.layers
        ]
