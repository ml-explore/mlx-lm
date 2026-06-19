# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import CacheList, KVCache
from .deepseek_v32 import DeepseekV32Attention
from .deepseek_v32 import DeepseekV32DecoderLayer as _DSV32DecoderLayer
from .deepseek_v32 import DeepseekV32Model as _DSV32Model
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
    # GLM-5.2 DeepSeek-Sparse-Attention layer typing. Each entry is "full"
    # (this layer owns an indexer) or "shared" (reuse the previous full
    # layer's topk indices; no indexer weights in the checkpoint). If absent
    # (vanilla DeepSeek-V3.2), every layer is treated as "full" so behavior is
    # identical to the parent.
    indexer_types: Optional[List[str]] = None
    index_share_for_mtp_iteration: bool = False
    index_topk_freq: int = 4
    index_skip_topk_offset: int = 3

    def __post_init__(self):
        self.rope_scaling = self.rope_parameters
        self.rope_theta = self.rope_parameters["rope_theta"]


def _is_full(config: ModelArgs, layer_idx: int) -> bool:
    """A layer owns an indexer iff it is typed "full". Missing config ==
    vanilla DeepSeek-V3.2 → all layers are full (parent behavior)."""
    if config.indexer_types is None:
        return True
    if layer_idx >= len(config.indexer_types):
        return True
    return config.indexer_types[layer_idx] == "full"


class GlmMoeDsaAttention(DeepseekV32Attention):
    """MLA + DSA attention with GLM-5.2's full/shared indexer typing.

    A "full" layer owns an ``Indexer`` (identical to the parent). A "shared"
    layer has NO indexer in the checkpoint: it reuses the previous full
    layer's ``topk_indices``. Below ``index_topk`` (2048) tokens the indexer
    short-circuits to ``None`` on every layer, so the whole model runs the
    dense path and ``topk_indices`` is ``None`` everywhere (Milestone 1).
    """

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config)
        self.layer_idx = layer_idx
        # Whether THIS layer owns an indexer.
        self.is_full = _is_full(config, layer_idx)
        # Whether the NEXT layer is "shared" (and therefore must reuse this
        # layer's topk_indices). Mirrors HF's `next_skip_topk`.
        if config.indexer_types is not None and layer_idx + 1 < len(
            config.indexer_types
        ):
            self.next_is_shared = config.indexer_types[layer_idx + 1] == "shared"
        else:
            self.next_is_shared = False

        if not self.is_full:
            # Shared layer: drop the indexer the parent created in super().
            # The checkpoint has no indexer.* weights here, so the param tree
            # must not contain them (0 missing / 0 unexpected on load).
            del self.indexer

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

        # cache layout:
        #   full   layer: CacheList(mla_kv_cache, indexer_kv_cache)
        #   shared layer: CacheList(mla_kv_cache)   (no indexer slot)
        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
            mla_cache = cache[0]
            indexer_cache = cache[1] if self.is_full else None
        else:
            mla_cache = None
            indexer_cache = None

        # === DSA indexer / topk selection ===
        if self.is_full:
            # Compute (or short-circuit to None below index_topk tokens).
            topk_indices = self.indexer(x, qr, mask, cache=indexer_cache)
        elif prev_topk_indices is not None:
            # Shared layer with a live topk from the previous full layer:
            # reuse it (sparse path). Below index_topk this is None, handled
            # by the branch below.
            topk_indices = prev_topk_indices
        else:
            # Shared layer, dense regime: no sparsity.
            topk_indices = None

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

        # Ensure the indexer cache is evaluated even if the topk_indices are
        # unused, to keep the graph from getting too large. Only do this on
        # full layers, which are the only ones with an indexer cache.
        if self.is_full and mla_cache is not None and indexer_cache is not None:
            mla_cache.keys = mx.depends(
                mla_cache.keys, (indexer_cache.keys, indexer_cache.values)
            )

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

        sdpa_cache = [mla_cache, None]
        output = scaled_dot_product_attention(
            q_nope, k, v, cache=sdpa_cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = self.o_proj(output)

        # Carry topk forward only if the next layer is shared and will reuse
        # it. Mirrors HF DecoderLayer: `topk_indices if next_skip_topk else None`.
        next_topk = topk_indices if self.next_is_shared else None
        return out, next_topk


class GlmMoeDsaDecoderLayer(_DSV32DecoderLayer):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config, layer_idx)
        # Replace the parent's attention with the indexer-typed one.
        self.self_attn = GlmMoeDsaAttention(config, layer_idx)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        r, next_topk = self.self_attn(
            self.input_layernorm(x),
            mask,
            cache,
            prev_topk_indices=prev_topk_indices,
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, next_topk


class GlmMoeDsaModel(_DSV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        # Rebuild the layer stack with indexer-typed decoder layers.
        self.layers = [
            GlmMoeDsaDecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]
        self.end_idx = len(self.layers)
        self.num_layers = self.end_idx

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        # Thread the indexer topk across layers. Below index_topk tokens this
        # stays None for the whole stack (Milestone 1 dense path); above it,
        # full layers compute it and shared layers reuse it (Milestone 2).
        prev_topk = None
        for i in range(self.num_layers):
            h, prev_topk = self.layers[self.start_idx + i](
                h, mask, cache[i], prev_topk_indices=prev_topk
            )

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(DSV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        # Swap in the indexer-typed model graph (parent built the plain one).
        self.model = GlmMoeDsaModel(config)

    def make_cache(self):
        # Full layers need both the MLA KV cache and the indexer KV cache.
        # Shared layers have no indexer, so they get a single MLA KV cache.
        # Whatever attention expects (cache[0] always; cache[1] only on full).
        caches = []
        for layer in self.layers:
            if getattr(layer.self_attn, "is_full", True):
                caches.append(CacheList(KVCache(), KVCache()))
            else:
                caches.append(CacheList(KVCache()))
        return caches
