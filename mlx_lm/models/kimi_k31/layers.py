# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import (
    blocked_topk,
    min_context,
    scaled_dot_product_attention,
    sparse_attention,
)
from mlx_lm.models.deepseek_v32.layers import Indexer
from mlx_lm.models.kimi_k3.layers import KimiK3DecoderLayer, KimiK3MLAAttention
from mlx_lm.models.kimi_k31.config import TextArgs

__all__ = [
    "KimiK31DecoderLayer",
    "KimiK31Indexer",
    "KimiK31MLAAttention",
]


class KimiK31Indexer(Indexer):
    """The DeepSeek-V3.2 indexer without RoPE (index_rope_dim is 0)."""

    def __init__(self, args: TextArgs):
        # The parent constructor reads the flat DeepSeek config and builds RoPE.
        nn.Module.__init__(self)
        dsa = args.dsa_config
        self.dim = args.hidden_size
        self.n_heads = dsa["index_num_heads"]
        self.head_dim = dsa["index_head_dim"]
        self.index_topk = dsa["index_topk"]
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=args.rms_norm_eps)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

    def rope(self, x, offset=0):
        return x


class KimiK31MLAAttention(KimiK3MLAAttention):
    """Kimi K3 gated NoPE MLA, attending to the indexer's top-k keys."""

    def __init__(self, args: TextArgs):
        super().__init__(args)
        self.indexer = KimiK31Indexer(args)
        self.sparse_prefill = args.sparse_prefill
        self.sparse_prefill_min_context = args.sparse_prefill_min_context
        self.sparse_prefill_fused = args.sparse_prefill_fused
        self.sparse_prefill_fused_min_context = args.sparse_prefill_fused_min_context

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = mx.expand_dims(self.kv_a_layernorm(compressed_kv), axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None] * 2

        gate = mx.sigmoid(self.g_proj(x)) if self.use_gate else None

        # Sparse prefill supports one sequence. Batches use the masked dense path.
        sparse = (
            self.sparse_prefill
            and B == 1
            and L > 1
            and kv_latent.shape[2] >= min_context(self, q_pe, kv_latent, k_pe)
        )
        if sparse:
            topk_indices = blocked_topk(self.indexer, x, qr, mask, cache[1])
        else:
            topk_indices = self.indexer(x, qr, mask, cache=cache[1])

        # Keep the indexer cache in the graph when the indices are not used.
        if cache[0] is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        if sparse and topk_indices is not None:
            return sparse_attention(
                self, q_nope, q_pe, kv_latent, k_pe, topk_indices, mask, B, L, gate
            )

        if topk_indices is not None:
            topk_indices = mx.stop_gradient(topk_indices)
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
                mask = sparse_mask if mask is None else sparse_mask & mask

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
            q_nope, k, v, cache=cache[0], scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if gate is not None:
            output = output * gate
        return self.o_proj(output)


class KimiK31DecoderLayer(KimiK3DecoderLayer):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__(args, layer_idx)
        dsa_layers = (args.dsa_config or {}).get("dsa_layers", [])
        # dsa_layers is 1-based.
        self.is_dsa = (layer_idx + 1) in dsa_layers
        if self.is_dsa:
            self.self_attn = KimiK31MLAAttention(args)
