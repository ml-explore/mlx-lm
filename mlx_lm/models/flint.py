# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from ..kernels import expert_routing
from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    mlp_only_layers: List[int]
    layer_types: List[str]
    sliding_window: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float = 10000.0
    rope_parameters: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.rope_parameters and self.rope_parameters.get("rope_theta") is not None:
            self.rope_theta = self.rope_parameters["rope_theta"]


# Query rows per band, and the key-tile multiple band starts snap to. The fused
# kernel's key tile (16 or 32) and query tile (32 or 64) divide these, so a band
# sees the same tile grid and online-softmax order as the single call.
_BAND_QUERIES = 256
_BAND_KEY_ALIGN = 64
# Other head dims run the unfused fallback, whose softmax reduction depends on the key count.
_BAND_HEAD_DIMS = (64, 128)
# Queries <= 8 leave the full kernel (sdpa_vector or the unfused fallback).
_SDPA_VECTOR_MAX_QUERIES = 8


def _banded_sliding_attention(queries, keys, values, cache, scale, mask, sinks, window):
    """Sliding-window prefill attention that skips key tiles the window never reaches.

    For an ``L``-token chunk the rotating cache returns its retained keys in time
    order followed by the chunk, and ``mask`` is the model's window mask built with
    the same ``window``, so query ``i`` sees keys ``(kL - L + i - window, kL - L + i]``.
    Fully masked key tiles add exactly zero to the online softmax, so processing
    query bands over only the keys they can reach is bit-identical, whatever the
    cache's max_size. Returns None when the single call should be kept.
    """
    L, kL = queries.shape[2], keys.shape[2]
    if (
        not isinstance(cache, RotatingKVCache)
        or cache.keep != 0
        or not isinstance(mask, mx.array)
        or mask.shape != (L, kL)
        or queries.shape[-1] not in _BAND_HEAD_DIMS
        or L <= _SDPA_VECTOR_MAX_QUERIES
        or 0 < L % _BAND_QUERIES <= _SDPA_VECTOR_MAX_QUERIES
        or kL <= _BAND_QUERIES + window
    ):
        return None
    prior = kL - L
    outs = []
    for s in range(0, L, _BAND_QUERIES):
        n = min(_BAND_QUERIES, L - s)
        start = max(0, prior + s - window + 1)
        start -= start % _BAND_KEY_ALIGN
        stop = prior + s + n
        outs.append(
            mx.fast.scaled_dot_product_attention(
                queries[:, :, s : s + n],
                keys[:, :, start:stop],
                values[:, :, start:stop],
                scale=scale,
                mask=mask[s : s + n, start:stop],
                sinks=sinks,
            )
        )
    return mx.concatenate(outs, axis=2)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
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
        self.softmax_offset = mx.zeros((n_heads,))

        if args.layer_types[layer_idx] == "sliding_attention":
            self.rope = nn.RoPE(head_dim, traditional=False, base=args.rope_theta)
            self.sliding_window = args.sliding_window
        else:
            self.rope = None
            self.sliding_window = None

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

        output = None
        if self.sliding_window is not None and L > 1:
            output = _banded_sliding_attention(
                queries,
                keys,
                values,
                cache,
                self.scale,
                mask,
                self.softmax_offset,
                self.sliding_window,
            )
        if output is None:
            output = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=self.scale,
                mask=mask,
                sinks=self.softmax_offset,
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(dim, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = MLP(dim, args.shared_expert_intermediate_size)
        self._zero_bias = mx.zeros((args.num_experts,))

        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        logits = self.gate(x.astype(mx.float32))

        k = self.top_k
        # The routing kernel has no vjp and needs Metal.
        if self.training or not mx.metal.is_available():
            scores = mx.sigmoid(logits)
            inds = mx.stop_gradient(
                mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
            )
            weights = mx.take_along_axis(scores, inds, axis=-1)
            weights = weights / weights.sum(axis=-1, keepdims=True)
        else:
            inds, weights = expert_routing(
                logits,
                self._zero_bias,
                top_k=k,
                multiplier=1.0,
                score_eps=0.0,
                route_scale=1.0,
            )

        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        y = y + self.shared_expert(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.use_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.self_attn = Attention(args, layer_idx)
        if layer_idx in args.mlp_only_layers:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = SparseMoeBlock(args)
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
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class FlintModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # A config can use only one attention type.
        self.fa_idx = None
        self.swa_idx = None
        if "full_attention" in args.layer_types:
            self.fa_idx = args.layer_types.index("full_attention")
        if "sliding_attention" in args.layer_types:
            self.swa_idx = args.layer_types.index("sliding_attention")

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

        fa_mask = None
        swa_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.args.sliding_window
            )

        for layer, c in zip(self.layers, cache, strict=True):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FlintModel(args)
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
        # RecursiveSI/flint-int4 stores the packed weights as uint8.
        for k, v in weights.items():
            name = k.removesuffix(".weight")
            if v.dtype == mx.uint8 and f"{name}.scales" in weights:
                weights[k] = v.view(mx.uint32)
        for l in range(self.args.num_hidden_layers):
            if l in self.args.mlp_only_layers:
                continue
            prefix = f"model.layers.{l}.mlp"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                if f"{prefix}.experts.0.{n}.weight" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)
        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        R = group.rank()
        for layer in self.model.layers:
            attn = layer.self_attn
            attn.q_proj = shard_linear(attn.q_proj, "all-to-sharded", group=group)
            attn.k_proj = shard_linear(attn.k_proj, "all-to-sharded", group=group)
            attn.v_proj = shard_linear(attn.v_proj, "all-to-sharded", group=group)
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.n_heads //= N
            attn.n_kv_heads //= N
            attn.softmax_offset = attn.softmax_offset[
                attn.n_heads * R : attn.n_heads * (R + 1)
            ]

            if isinstance(layer.mlp, MLP):
                mlp = layer.mlp
                mlp.gate_proj = shard_linear(
                    mlp.gate_proj, "all-to-sharded", group=group
                )
                mlp.down_proj = shard_linear(
                    mlp.down_proj, "sharded-to-all", group=group
                )
                mlp.up_proj = shard_linear(mlp.up_proj, "all-to-sharded", group=group)
            else:
                moe = layer.mlp
                moe.sharding_group = group
                for m in [moe.switch_mlp, moe.shared_expert]:
                    shard_inplace(m.gate_proj, "all-to-sharded", group=group)
                    shard_inplace(m.down_proj, "sharded-to-all", group=group)
                    shard_inplace(m.up_proj, "all-to-sharded", group=group)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    @property
    def quant_predicate(self):
        def predicate(path, _):
            return not path.endswith("mlp.gate")

        return predicate
