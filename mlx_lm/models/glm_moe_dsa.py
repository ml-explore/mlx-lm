# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import CacheList, KVCache
from .deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32DecoderLayer,
    DeepseekV32Model,
)
from .deepseek_v32 import Model as DSV32Model


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    topk_method: str
    scoring_func: str
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    num_experts_per_tok: int
    moe_layer_freq: int
    first_k_dense_replace: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_parameters: Dict
    attention_bias: bool
    rope_scaling: Dict = None
    rope_theta: Optional[float] = None
    indexer_rope_interleave: bool = True
    indexer_types: Optional[List[str]] = None
    index_topk_pattern: Optional[Any] = None
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 2
    sparse_prefill: bool = True
    sparse_prefill_min_context: int = 16384

    def __post_init__(self):
        self.rope_scaling = self.rope_parameters
        self.rope_theta = self.rope_parameters["rope_theta"]

        if self.indexer_types is None:
            if self.index_topk_pattern is not None:
                pattern = self.index_topk_pattern
                if isinstance(pattern, str):
                    self.indexer_types = [
                        {"F": "full", "S": "shared"}[c] for c in pattern
                    ]
                else:
                    self.indexer_types = list(pattern)
            else:
                freq = max(self.index_topk_freq, 1)
                offset = self.index_skip_topk_offset
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                    for i in range(self.num_hidden_layers)
                ]


_SPARSE_QUERY_BLOCK = 1024


def _sparse_prefill(self, q_nope, q_pe, kv_latent, k_pe, topk, mask, B, L):
    # Absorbed form: 576-wide keys (512 latent + 64 rope), 512-wide values. B is 1.
    block = _SPARSE_QUERY_BLOCK
    if L > block:
        # Query blocks keep the gathered keys small, which is faster for long chunks.
        outs = []
        for s in range(0, L, block):
            e = min(L, s + block)
            m = None if mask is None else mask.reshape(L, -1)[s:e]
            q = (q_nope[:, :, s:e], q_pe[:, :, s:e])
            idx = topk[:, :, s:e]
            outs.append(_sparse_prefill(self, *q, kv_latent, k_pe, idx, m, B, e - s))
        return mx.concatenate(outs, axis=1)
    q_full = mx.concatenate([self.embed_q(q_nope), q_pe], axis=-1)[0]  # [H, L, 576]
    kv_full = mx.concatenate([kv_latent, k_pe], axis=-1)[0, 0]  # [N, 576]
    idx = topk[0, 0]  # [L, K]
    gathered = kv_full[idx]  # [L, K, 576], shared by all heads

    q_t = q_full.transpose(1, 0, 2)  # [L, H, 576]
    scores = (q_t @ gathered.swapaxes(-1, -2)).astype(mx.float32) * self.scale
    if mask is not None:
        valid = mx.take_along_axis(mask.reshape(L, -1), idx, axis=-1)  # [L, K]
        scores = mx.where(valid[:, None, :], scores, -mx.inf)
    probs = mx.softmax(scores, axis=-1).astype(gathered.dtype)
    latent_out = probs @ gathered[..., : self.kv_lora_rank]  # [L, H, 512]
    latent_out = latent_out.transpose(1, 0, 2)[None]  # [1, H, L, 512]
    out = self.unembed_out(latent_out)  # [1, H, L, v_head_dim]
    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(out)


def _blocked_topk(indexer, x, qr, mask, cache, block=4096):
    # Stock indexer scores, one key block at a time, without the [b, H, s, N] tensor.
    # Heads are added in the stock order. A running top-k keeps the sorts short and
    # selects the same keys as stock (up to the order of equal scores).
    b, s, _ = x.shape
    q = indexer.wq_b(qr)
    q = q.reshape(b, s, indexer.n_heads, indexer.head_dim).swapaxes(1, 2)
    k = indexer.k_norm(indexer.wk(x))
    k = mx.reshape(k, (b, 1, s, indexer.head_dim))
    offset = cache.offset if cache is not None else 0
    q = indexer.rope(q, offset=offset)
    k = indexer.rope(k, offset=offset)
    if cache is not None:
        k, _ = cache.update_and_fetch(k, k)
        cache.values = mx.zeros_like(cache.keys)
    n = k.shape[2]
    if n <= indexer.index_topk:
        return None

    weights = indexer.weights_proj(x) * (indexer.n_heads**-0.5 * indexer.softmax_scale)
    weights = weights.swapaxes(-1, -2)[..., None]  # [b, H, s, 1]

    best_v = best_i = None
    for s0 in range(0, n, block):
        s1 = min(n, s0 + block)
        sc = mx.maximum(q @ k[:, :, s0:s1].swapaxes(-1, -2), 0) * weights
        sc = sum(sc[:, h : h + 1] for h in range(sc.shape[1]))  # [b, 1, s, blk]
        if mask is not None:
            sc = mx.where(mask[..., s0:s1], sc, -float("inf"))
        ids = mx.broadcast_to(mx.arange(s0, s1), sc.shape)
        if best_v is not None:
            sc = mx.concatenate([best_v, sc], axis=-1)
            ids = mx.concatenate([best_i, ids], axis=-1)
        k_top = indexer.index_topk
        pos = mx.stop_gradient(mx.argpartition(sc, kth=-k_top, axis=-1)[..., -k_top:])
        best_v = mx.take_along_axis(sc, pos, axis=-1)
        best_i = mx.take_along_axis(ids, pos, axis=-1)
    return best_i


class GlmMoeDsaAttention(DeepseekV32Attention):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config)
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        self.sparse_prefill = config.sparse_prefill
        self.sparse_prefill_min_context = config.sparse_prefill_min_context
        if self.skip_topk:
            self.indexer = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        B, L, D = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None] * 2

        # Sparse prefill supports one sequence. Batches use the masked dense path.
        # Below the minimum context the dense path is faster.
        sparse = (
            self.sparse_prefill
            and B == 1
            and L > 1
            and kv_latent.shape[2] >= self.sparse_prefill_min_context
        )
        if self.indexer is not None:
            if sparse:
                topk_indices = _blocked_topk(self.indexer, x, qr, mask, cache[1])
            else:
                topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        else:
            topk_indices = prev_topk_indices

        if sparse and topk_indices is not None:
            if self.indexer is not None and cache[0] is not None:
                cache[0].keys = mx.depends(
                    cache[0].keys, (cache[1].keys, cache[1].values)
                )
            out = _sparse_prefill(
                self, q_nope, q_pe, kv_latent, k_pe, topk_indices, mask, B, L
            )
            return out, topk_indices

        if topk_indices is not None:
            if L == 1:
                idx = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk_indices, axis=-1)
            else:
                shape = list(topk_indices.shape)
                shape[-1] = kv_latent.shape[2]
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, topk_indices, mx.array(True), axis=-1
                )
                if mask is not None:
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask

        # Ensure the indexer cache is evaluated even if the topk_indices are unused
        # to keep the graph from getting too large
        if self.indexer is not None and cache is not None and cache[0] is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), topk_indices


class GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GlmMoeDsaAttention(config, layer_idx)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        r, topk_indices = self.self_attn(
            self.input_layernorm(x), mask, cache, prev_topk_indices
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, topk_indices


class GlmMoeDsaModel(DeepseekV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.layers = [
            GlmMoeDsaDecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        prev_topk_indices = None
        for layer, c in zip(self.pipeline_layers, cache):
            h, prev_topk_indices = layer(h, mask, c, prev_topk_indices)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(DSV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.model = GlmMoeDsaModel(config)

    def make_cache(self):
        # Shared layers run no indexer, so they get no indexer KVCache.
        caches = []
        for layer in self.layers:
            if getattr(layer.self_attn, "skip_topk", False):
                caches.append(CacheList(KVCache()))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches
