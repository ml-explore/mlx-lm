# Copyright © 2026 Apple Inc.
#
# HiLS-Attention (Hierarchical Landmark Sparse attention) on an OLMo3
# backbone — port of tencent/HiLS-Attention-7B (arXiv:2607.02980,
# github.com/Tencent-Hunyuan/HiLS-Attention).
#
# Structure (values for the 7B checkpoint in parentheses):
#   * A landmark token (id = vocab_size, embedded via the standalone
#     `lmk_embed` parameter) is inserted after every chunk_size-1 (63)
#     real tokens. Landmarks share the position id of the following real
#     token, are masked out as attention targets everywhere, and their
#     hidden states act as chunk-retrieval queries.
#   * Layers with idx % full_attn_interleave == interleave-1 (every 4th)
#     use HiLS attention; the rest are sliding-window (512) attention
#     with landmark keys masked.
#   * HiLS layers factorize attention into a sliding-window part and a
#     top-k (32) retrieved-chunk part, fused by a softmax over
#     [per-chunk retrieval score + Taylor prior bias, window LSE].
#     Per-chunk landmark keys are attention-pooled from the chunk's keys
#     against the landmark query at the chunk's last position.
#   * HiLS layers use HoPE in-range RoPE: standard rope frequencies with
#     every pair whose period exceeds rope_context_length /
#     rope_period_multiplier zeroed (NoPE tail).
#
# Landmark insertion is handled inside `Model.__call__`: it takes real
# token ids and returns logits aligned to them, so the standard mlx-lm
# generation loop works unchanged. Batch size must be 1.
#
# Cache coordinates: the caches store keys/values in landmark-INSERTED
# coordinates but expose `offset`/`trim()` in REAL-token units, so
# generic cache machinery (prompt-lookup speculative decoding rollback,
# trim_prompt_cache, telemetry) composes without knowing about
# landmarks. Sliding-window layers keep only the window band (plus a
# rollback margin) — memory for those 24 layers is O(window), not O(T).
# Decode cost is bounded (topk*chunk_size + window keys per token) at
# any context length.

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import _BaseCache
from .olmo3 import Olmo3MLP


def _needs_qmm_row_pad():
    # mlx < 0.32 quantized_matmul is wrong for unaligned N (e.g. this
    # model's vocab, 100278) when 10 <= rows and rows % 32 != 0. Padding
    # is always numerically safe, so pad when the version is unparseable.
    try:
        return tuple(int(p) for p in mx.__version__.split(".")[:2]) < (0, 32)
    except ValueError:
        return True


_NEEDS_QMM_ROW_PAD = _needs_qmm_row_pad()


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    sliding_window: int
    rope_theta: float
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    # HiLS
    chunk_size: int = 64
    hils_topk: int = 32
    full_attn_interleave: int = 4
    num_swa_layers: int = 0
    hils_sliding_window: Optional[int] = None
    lmk_q_lora_dim: int = 256
    enable_lmk_q_proj: bool = True
    layerwise_qk_norm: bool = True
    layerwise_lmkq_norm: bool = True
    apply_hils_rope: bool = True
    enable_prior_query: bool = True
    enable_external_lmk_embed: bool = True
    mask_lmk_token: bool = True
    adjust_lmk_pos: bool = True
    enable_softmax1: bool = False
    use_hope: bool = True
    enable_inrange_rope: bool = True
    rope_context_length: int = 8192
    rope_period_multiplier: float = 2.0

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.num_key_value_heads != self.num_attention_heads:
            raise NotImplementedError("olmo_hils port assumes no GQA (h_kv == h_q)")
        unsupported = (
            self.enable_softmax1
            or not self.enable_prior_query
            or not self.enable_lmk_q_proj
            or not self.layerwise_qk_norm
            or not self.layerwise_lmkq_norm
            or not self.apply_hils_rope
            or not self.enable_external_lmk_embed
            or not self.mask_lmk_token
            or not self.use_hope
        )
        if unsupported:
            raise NotImplementedError(
                "olmo_hils port only supports the released HiLS-Attention-7B "
                "config flags"
            )
        if self.hils_sliding_window is None:
            self.hils_sliding_window = self.sliding_window


def _is_hils_layer(args: ModelArgs, layer_idx: int) -> bool:
    k = args.full_attn_interleave
    return (
        k > 0
        and layer_idx >= args.num_swa_layers
        and (layer_idx - args.num_swa_layers) % k == k - 1
    )


def _rotate_half(x):
    d = x.shape[-1] // 2
    return mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def _apply_rope(x, cos, sin):
    # x: (B, L, h, d); cos/sin: (L, d)
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return (x * cos + _rotate_half(x) * sin).astype(x.dtype)


class RopeTable:
    """cos/sin from explicit position ids (landmarks repeat positions)."""

    def __init__(self, head_dim, base, inrange_threshold=None):
        inv_freq = base ** (-mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        if inrange_threshold is not None:
            keep = inv_freq >= inrange_threshold
            keep[0] = True
            inv_freq = mx.where(keep, inv_freq, mx.zeros_like(inv_freq))
        self.inv_freq = inv_freq

    def __call__(self, position_ids):
        # position_ids: (L,) -> cos/sin (L, head_dim) fp32
        freqs = position_ids.astype(mx.float32)[:, None] * self.inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


class InsertedKVCache(_BaseCache):
    """KV cache in landmark-inserted coordinates with a real-token API.

    `update_and_fetch` takes inserted-space K/V (B, h, L_ins, d).
    `offset` and `trim(n)` are in REAL tokens so generic machinery
    (speculative rollback, trim_prompt_cache) works unchanged; the
    conversion relies on the insertion invariant that position p is a
    landmark iff (p + 1) % chunk_size == 0 and every call leaves the
    sequence landmark-complete.
    """

    step = 256

    def __init__(self, chunk_size: int):
        self.keys = None
        self.values = None
        self.chunk_size = chunk_size
        self.ins_offset = 0  # inserted-space total length
        self.start_pos = 0  # inserted-space position of buffer row 0

    # -- real-token API ---------------------------------------------------
    @property
    def offset(self):
        return self.ins_offset - self.ins_offset // self.chunk_size

    def _ins_len(self, n_real):
        return n_real + n_real // (self.chunk_size - 1)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(n, self.offset)
        new_ins = self._ins_len(self.offset - n)
        if new_ins < self.start_pos:
            raise RuntimeError(
                "olmo_hils cache: trim below retained window "
                f"(target {new_ins} < start {self.start_pos})"
            )
        self._on_trim(new_ins)
        self.ins_offset = new_ins
        return n

    def _on_trim(self, new_ins):
        pass

    # -- storage ----------------------------------------------------------
    def _rows(self):
        return self.ins_offset - self.start_pos

    def update_and_fetch(self, keys, values):
        B, H, L, D = keys.shape
        prev = self._rows()
        if self.keys is None or prev + L > self.keys.shape[2]:
            n_alloc = ((prev + L + self.step - 1) // self.step) * self.step
            new_k = mx.zeros((B, H, n_alloc, D), keys.dtype)
            new_v = mx.zeros((B, H, n_alloc, values.shape[3]), values.dtype)
            if self.keys is not None and prev > 0:
                new_k[:, :, :prev] = self.keys[:, :, :prev]
                new_v[:, :, :prev] = self.values[:, :, :prev]
            self.keys, self.values = new_k, new_v
        self.keys[:, :, prev : prev + L] = keys
        self.values[:, :, prev : prev + L] = values
        self.ins_offset += L
        return self.fetch()

    def fetch(self):
        n = self._rows()
        return self.keys[:, :, :n], self.values[:, :, :n]

    @property
    def state(self):
        if self.keys is None:
            return []
        n = self._rows()
        return [self.keys[:, :, :n], self.values[:, :, :n]]

    @state.setter
    def state(self, v):
        if v:
            self.keys, self.values = v[0], v[1]

    @property
    def meta_state(self):
        raise NotImplementedError(
            "olmo_hils caches do not support save_prompt_cache (landmark "
            "bookkeeping is not serialized)"
        )

    @meta_state.setter
    def meta_state(self, v):
        raise NotImplementedError("olmo_hils caches do not support load_prompt_cache")

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes

    def size(self):
        return self.offset

    def empty(self):
        return self.keys is None


class SWABandCache(InsertedKVCache):
    """Window-band KV cache for the sliding-window layers.

    Keeps only rows the window can still reach plus a rollback margin,
    so memory is O(window), not O(context). Fail-closed: front-trims
    only rows strictly older than any position a future query (or a
    speculative rollback + re-decode) can attend to.
    """

    def __init__(self, chunk_size: int, window: int, margin: int = 128):
        super().__init__(chunk_size)
        self.window = window
        self.base_margin = margin
        self.margin = margin

    def start_speculation(self, rollback_window: Optional[int] = None):
        if rollback_window is not None:
            # rollback_window is in real tokens; convert with landmark slack
            ins = rollback_window + rollback_window // (self.chunk_size - 1) + 2
            self.margin = max(self.base_margin, ins + self.window)

    def stop_speculation(self):
        self.margin = self.base_margin

    def trim(self, n):
        n = min(n, self.offset)
        new_ins = self._ins_len(self.offset - n)
        # fail closed: after the trim, re-decoding from new_ins must still
        # find its full attention window in the retained band
        if new_ins - (self.window - 1) < self.start_pos:
            raise RuntimeError(
                "olmo_hils SWA band cache: trim would truncate the attention "
                f"window (target {new_ins}, window start "
                f"{new_ins - self.window + 1} < retained start {self.start_pos}). "
                "Deep trims past the retained band are not supported; "
                "re-prefill instead."
            )
        return super().trim(n)

    def update_and_fetch(self, keys, values):
        k, v = super().update_and_fetch(keys, values)
        L = keys.shape[2]
        # oldest row any query in this segment (or later) can see is
        # seg_start - (window - 1); margin covers speculative rollback
        seg_start = self.ins_offset - L
        keep_from = seg_start - (self.window - 1) - self.margin
        cut = keep_from - self.start_pos
        if cut >= self.step:
            self.keys = self.keys[:, :, cut:]
            self.values = self.values[:, :, cut:]
            self.start_pos += cut
        n = self._rows()
        return self.keys[:, :, :n], self.values[:, :, :n]


class HiLSCache(InsertedKVCache):
    """Full KV plus the incremental per-chunk landmark-key/prior cache."""

    def __init__(self, chunk_size: int):
        super().__init__(chunk_size)
        self.lmk_k = None  # (B, N, h, d)
        self.prior_b = None  # (B, N, h) fp32
        self.num_pooled_chunks = 0

    def _on_trim(self, new_ins):
        new_pooled = new_ins // self.chunk_size
        if new_pooled < self.num_pooled_chunks:
            self.lmk_k = self.lmk_k[:, :new_pooled]
            self.prior_b = self.prior_b[:, :new_pooled]
            self.num_pooled_chunks = new_pooled

    @property
    def state(self):
        base = InsertedKVCache.state.fget(self)
        if self.lmk_k is not None:
            base = base + [self.lmk_k, self.prior_b]
        return base

    @state.setter
    def state(self, v):
        if v:
            self.keys, self.values = v[0], v[1]
            if len(v) > 2:
                self.lmk_k, self.prior_b = v[2], v[3]


class SWAAttention(nn.Module):
    """Banded sliding-window attention with landmark keys masked."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.window = args.sliding_window
        self.chunk_size = args.chunk_size

        d = args.hidden_size
        self.q_proj = nn.Linear(d, d, bias=args.attention_bias)
        self.k_proj = nn.Linear(d, d, bias=args.attention_bias)
        self.v_proj = nn.Linear(d, d, bias=args.attention_bias)
        self.o_proj = nn.Linear(d, d, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)

    def __call__(self, x, cos, sin, q_pos, cache, shared):
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x)).reshape(B, L, self.n_heads, -1)
        k = self.k_norm(self.k_proj(x)).reshape(B, L, self.n_heads, -1)
        v = self.v_proj(x).reshape(B, L, self.n_heads, -1)

        q = _apply_rope(q, cos, sin).transpose(0, 2, 1, 3)
        k = _apply_rope(k, cos, sin).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        k, v = cache.update_and_fetch(k, v)

        # every SWA cache sees the identical update sequence, so the band
        # mask is shared across SWA layers within one forward
        key = (cache.start_pos, k.shape[2], self.window)
        mask = shared.get(key)
        if mask is None:
            kv_pos = cache.start_pos + mx.arange(k.shape[2])
            mask = (
                (kv_pos[None, :] <= q_pos[:, None])
                & (kv_pos[None, :] >= q_pos[:, None] - self.window + 1)
                & ((kv_pos[None, :] + 1) % self.chunk_size != 0)
            )[None, None]
            shared[key] = mask

        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o)


class HiLSAttention(nn.Module):
    # hard cap on the sparse (gather) path: the per-(token, head) chunk
    # gather allocates O(L * topk * chunk_size) rows, so bound L to keep
    # the transient buffer small even when the KV-moved heuristic below
    # would prefer sparse
    SPARSE_MAX_L = 34

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.chunk_size = args.chunk_size
        self.topk = args.hils_topk
        self.window = args.hils_sliding_window

        d = args.hidden_size
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.q_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.lmk_q_proj = [
            nn.Linear(d, args.lmk_q_lora_dim, bias=False),
            nn.Linear(args.lmk_q_lora_dim, d, bias=False),
        ]
        self.lmk_q_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)

        self._last_in_chunk = mx.arange(args.chunk_size) == args.chunk_size - 1

    def _pool_new_chunks(self, cache, lmk_q, q_offset):
        """Append landmark keys / prior biases for newly completed chunks.

        Pools the cached (roped, normed) keys inside each new chunk against
        the landmark query at the chunk's last position, which is always
        inside the current input segment.
        """
        S = self.chunk_size
        full_chunks = cache.ins_offset // S
        new_chunks = full_chunks - cache.num_pooled_chunks
        if new_chunks <= 0:
            return
        c0 = cache.num_pooled_chunks
        k_new = cache.keys[:, :, c0 * S : full_chunks * S, :]
        B, H, _, D = k_new.shape
        k_chunks = k_new.reshape(B, H, new_chunks, S, D)

        boundary = [S * (c0 + j) - 1 - q_offset for j in range(1, new_chunks + 1)]
        mu_q = lmk_q[:, mx.array(boundary), :, :]  # (B, N, h, d)

        logits = (
            mx.einsum("bnhd,bhnsd->bhns", mu_q, k_chunks).astype(mx.float32)
            * self.scale
        )
        # mask the chunk's own landmark (last in-chunk position)
        neg_inf = mx.array(-mx.inf, mx.float32)
        logits = mx.where(self._last_in_chunk, neg_inf, logits)
        p = mx.softmax(logits, axis=-1)
        lmk_k = mx.einsum("bhns,bhnsd->bnhd", p.astype(k_chunks.dtype), k_chunks)
        log_p = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        log_p = mx.where(mx.isinf(log_p), mx.zeros_like(log_p), log_p)
        prior_b = -mx.sum(p * log_p, axis=-1).transpose(0, 2, 1)  # (B, N, h)

        if cache.lmk_k is None:
            cache.lmk_k, cache.prior_b = lmk_k, prior_b
        else:
            cache.lmk_k = mx.concatenate([cache.lmk_k, lmk_k], axis=1)
            cache.prior_b = mx.concatenate([cache.prior_b, prior_b], axis=1)
        cache.num_pooled_chunks = full_chunks

    def _swa_with_lse(self, q, cache, q_pos, q_offset):
        """Chunk-expanded sliding-window attention returning (o, lse).

        Window start for the query at global position p is
        floor((p - W + 1) / S) * S (expand_to_chunk), landmarks masked.
        Keys are restricted to the band [kv_lo, T), bounding the score
        matrix at L x (L + W + S).
        """
        S, W = self.chunk_size, self.window
        kv_lo = max(0, (q_offset - W + 1) // S * S)
        keys = cache.keys[:, :, kv_lo : cache.ins_offset, :]
        values = cache.values[:, :, kv_lo : cache.ins_offset, :]
        kv_pos = kv_lo + mx.arange(keys.shape[2])

        scores = mx.einsum("blhd,bhtd->bhlt", q, keys).astype(mx.float32) * self.scale
        start = mx.maximum(q_pos - W + 1, 0) // S * S
        keep = (
            (kv_pos[None, :] >= start[:, None])
            & (kv_pos[None, :] <= q_pos[:, None])
            & ((kv_pos[None, :] + 1) % S != 0)
        )
        scores = mx.where(keep[None, None], scores, mx.array(-mx.inf, mx.float32))
        lse = mx.logsumexp(scores, axis=-1)  # (B, h, L)
        p = mx.softmax(scores, axis=-1)
        o = mx.einsum("bhlt,bhtd->blhd", p.astype(values.dtype), values)
        return o, lse.transpose(0, 2, 1)  # lse: (B, L, h) fp32

    def _retrieve(self, lmk_q, cache, q_pos, lse_swa):
        """Top-k chunk retrieval.

        Returns (indices, scores): (B, L, h, K) each; indices are -1 and
        scores -inf for invalid (causally masked / unavailable) slots.
        Selection ranks chunks by prior-biased attention log-probability
        normalized against the total (window + chunks) LSE; returned
        scores are the raw scaled q·lmk_k logits.
        """
        N = cache.num_pooled_chunks
        logits = (
            mx.einsum("blhd,bshd->blhs", lmk_q, cache.lmk_k).astype(mx.float32)
            * self.scale
        )  # (B, L, h, N)
        # chunks still (partially) covered by the window are handled by SWA
        threshold = (q_pos - self.window + 1) // self.chunk_size
        j = mx.arange(N)
        masked = j[None, :] >= threshold[:, None]  # (L, N)
        neg_inf = mx.array(-mx.inf, mx.float32)
        logits = mx.where(masked[None, :, None, :], neg_inf, logits)

        prior = cache.prior_b.transpose(0, 2, 1)[:, None]  # (B, 1, h, N)
        select = logits + prior
        lse_hils = mx.logsumexp(select, axis=-1)
        lse_total = mx.logaddexp(lse_swa, lse_hils)
        log_probs = select - lse_total[..., None]

        K = min(self.topk, N)
        idx = mx.argpartition(-log_probs, kth=K - 1, axis=-1)[..., :K]
        idx = mx.sort(idx, axis=-1).astype(mx.int32)
        # prior-biased scores (raw + prior) drive the final chunk weights
        top_scores = mx.take_along_axis(select, idx, axis=-1)
        idx = mx.where(mx.isinf(top_scores), mx.array(-1, mx.int32), idx)
        return idx, top_scores

    def _chunk_weights(self, idx, scores, lse_swa):
        """Softmax over [selected prior-biased scores, window LSE]."""
        scores = mx.where(idx < 0, mx.array(-mx.inf, mx.float32), scores)
        cat = mx.concatenate([scores, lse_swa[..., None]], axis=-1)
        return mx.softmax(cat, axis=-1)  # (B, L, h, K+1)

    def _hils_sparse(self, q, cache, idx, w):
        """Decode/verify path: gather selected chunks per (token, head)."""
        S = self.chunk_size
        B, L, H, D = q.shape
        K = idx.shape[-1]
        N = cache.num_pooled_chunks
        k_chunks = cache.keys[:, :, : N * S, :].reshape(B, H, N, S, D)
        v_chunks = cache.values[:, :, : N * S, :].reshape(B, H, N, S, D)

        sel = mx.maximum(idx, 0).transpose(0, 2, 1, 3).reshape(B, H, L * K)
        gk = mx.take_along_axis(k_chunks, sel[..., None, None], axis=2)
        gv = mx.take_along_axis(v_chunks, sel[..., None, None], axis=2)
        gk = gk.reshape(B, H, L, K, S, D)
        gv = gv.reshape(B, H, L, K, S, D)

        qk = mx.einsum("blhd,bhlksd->bhlks", q, gk).astype(mx.float32) * self.scale
        qk = mx.where(self._last_in_chunk, mx.array(-mx.inf, mx.float32), qk)
        p = mx.softmax(qk, axis=-1)
        o_k = mx.einsum("bhlks,bhlksd->bhlkd", p.astype(gv.dtype), gv)
        w_eff = mx.where(idx < 0, mx.zeros_like(w), w)
        return mx.einsum("bhlkd,blhk->blhd", o_k.astype(mx.float32), w_eff)

    def _hils_dense(self, q, cache, idx, w):
        """Prefill path: scatter the selected-chunk weights into a dense
        (L, num_chunks) matrix, fold them into per-chunk softmax
        probabilities, and hit V with one matmul. Same math as the sparse
        path with O(L*T) compute but no per-token gather blowup.
        """
        S = self.chunk_size
        B, L, H, D = q.shape
        N = cache.num_pooled_chunks
        T = N * S
        keys = cache.keys[:, :, :T, :]
        values = cache.values[:, :, :T, :]

        # invalid slots -> extra column N, dropped after the scatter
        w_eff = mx.where(idx < 0, mx.zeros_like(w), w).astype(keys.dtype)
        scatter_idx = mx.where(idx < 0, mx.array(N, mx.int32), idx)
        w_dense = mx.zeros((B, L, H, N + 1), dtype=keys.dtype)
        w_dense = mx.put_along_axis(w_dense, scatter_idx, w_eff, axis=-1)
        # (B, h, L, N) in compute dtype; per-chunk probs also stay in the
        # compute dtype (softmax accumulates in fp32 via precise=True),
        # matching the reference kernel's fp32-accum / low-precision-GEMM
        w_dense = w_dense[..., :N].transpose(0, 2, 1, 3)

        neg_inf = mx.array(-mx.inf, keys.dtype)
        scale = mx.array(self.scale, keys.dtype)
        outs = []
        tile = 512
        for s0 in range(0, L, tile):
            s1 = min(s0 + tile, L)
            qt = q[:, s0:s1]
            sc = mx.einsum("blhd,bhtd->bhlt", qt, keys) * scale
            sc = sc.reshape(B, H, s1 - s0, N, S)
            sc = mx.where(self._last_in_chunk, neg_inf, sc)
            p = mx.softmax(sc, axis=-1, precise=True)  # per-chunk softmax
            p = p * w_dense[:, :, s0:s1, :, None]
            p = p.reshape(B, H, s1 - s0, T)
            o = mx.einsum("bhlt,bhtd->blhd", p, values)
            outs.append(o.astype(mx.float32))
        return mx.concatenate(outs, axis=1)

    def __call__(self, x, cos, sin, q_pos, cache):
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x))
        lmk_q = self.lmk_q_norm(self.lmk_q_proj[1](self.lmk_q_proj[0](x)) + q)
        q = _apply_rope(q.reshape(B, L, self.n_heads, -1), cos, sin)
        lmk_q = _apply_rope(lmk_q.reshape(B, L, self.n_heads, -1), cos, sin)
        k = self.k_norm(self.k_proj(x)).reshape(B, L, self.n_heads, -1)
        k = _apply_rope(k, cos, sin)
        v = self.v_proj(x).reshape(B, L, self.n_heads, -1)

        q_offset = cache.ins_offset
        cache.update_and_fetch(k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3))

        swa_o, lse_swa = self._swa_with_lse(q, cache, q_pos, q_offset)

        if cache.ins_offset >= self.chunk_size:
            self._pool_new_chunks(cache, lmk_q, q_offset)
            idx, scores = self._retrieve(lmk_q, cache, q_pos, lse_swa)
            w = self._chunk_weights(idx, scores, lse_swa)
            w_chunks, w_swa = w[..., :-1], w[..., -1]
            # The sparse path reads L*topk chunks (duplicated per token);
            # the dense path reads every chunk once, shared across tokens.
            # Pick whichever moves less KV.
            if L <= self.SPARSE_MAX_L and L * self.topk < cache.num_pooled_chunks:
                hils_o = self._hils_sparse(q, cache, idx, w_chunks)
            else:
                hils_o = self._hils_dense(q, cache, idx, w_chunks)
            o = hils_o + swa_o.astype(mx.float32) * w_swa[..., None]
        else:
            o = swa_o

        o = o.astype(x.dtype).reshape(B, L, -1)
        return self.o_proj(o)


class OlmoHilsDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_hils = _is_hils_layer(args, layer_idx)
        attn_cls = HiLSAttention if self.is_hils else SWAAttention
        self.self_attn = attn_cls(args, layer_idx)
        self.mlp = Olmo3MLP(args)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, cos, sin, hope_cos, hope_sin, q_pos, cache, shared):
        if self.is_hils:
            r = self.self_attn(x, hope_cos, hope_sin, q_pos, cache)
        else:
            r = self.self_attn(x, cos, sin, q_pos, cache, shared)
        h = x + self.post_attention_layernorm(r)
        return h + self.post_feedforward_layernorm(self.mlp(h))


class OlmoHilsModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.chunk_size = args.chunk_size
        self.window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.lmk_embed = mx.zeros((args.hidden_size,))
        self.layers = [
            OlmoHilsDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        head_dim = args.hidden_size // args.num_attention_heads
        self.rope = RopeTable(head_dim, args.rope_theta)
        thr = (
            2.0 * math.pi * args.rope_period_multiplier / args.rope_context_length
            if args.enable_inrange_rope
            else None
        )
        self.hope = RopeTable(head_dim, args.rope_theta, inrange_threshold=thr)

    def _insert_landmarks(self, inputs, offset):
        """Map real tokens onto the landmark-inserted segment for this call.

        A landmark follows every (chunk_size-1)-th real token, so in the
        inserted sequence position p is a landmark iff (p+1) % chunk_size
        == 0, and the real token with global index r sits at position
        r + r // (chunk_size - 1).
        """
        S = self.chunk_size
        L_real = inputs.shape[1]
        if L_real == 0:
            raise ValueError("olmo_hils got an empty input segment")
        real_off = offset - offset // S
        r = real_off + mx.arange(L_real)
        real_slots = r + r // (S - 1) - offset

        r_last = real_off + L_real - 1
        p_end = r_last + r_last // (S - 1) + (1 if (r_last + 1) % (S - 1) == 0 else 0)
        seg_len = p_end - offset + 1

        p_global = offset + mx.arange(seg_len)
        lmk_mask = (p_global + 1) % S == 0
        ids = mx.zeros((seg_len,), dtype=inputs.dtype)
        ids[real_slots] = inputs[0]
        return ids[None], lmk_mask, p_global, real_slots

    def __call__(self, inputs, cache=None):
        if inputs.shape[0] != 1:
            raise ValueError("olmo_hils supports batch size 1 only")
        if cache is None:
            cache = self.make_cache()

        offset = cache[0].ins_offset
        ids, lmk_mask, p_global, real_slots = self._insert_landmarks(inputs, offset)

        h = self.embed_tokens(ids)
        h = mx.where(lmk_mask[None, :, None], self.lmk_embed.astype(h.dtype), h)

        pos_ids = p_global - p_global // self.chunk_size
        cos, sin = self.rope(pos_ids)
        hope_cos, hope_sin = self.hope(pos_ids)
        cos, sin = cos.astype(h.dtype), sin.astype(h.dtype)
        hope_cos, hope_sin = hope_cos.astype(h.dtype), hope_sin.astype(h.dtype)

        shared = {}
        for layer, c in zip(self.layers, cache):
            h = layer(h, cos, sin, hope_cos, hope_sin, p_global, c, shared)

        h = self.norm(h)
        return h[:, real_slots, :]

    def make_cache(self, max_kv_size=None):
        if max_kv_size is not None:
            raise ValueError(
                "olmo_hils does not support max_kv_size: HiLS layers must "
                "retain full KV for chunk retrieval (sliding-window layers "
                "are already window-bounded)"
            )
        return [
            (
                HiLSCache(self.chunk_size)
                if _is_hils_layer(self.args, i)
                else SWABandCache(self.chunk_size, self.window)
            )
            for i in range(self.args.num_hidden_layers)
        ]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = OlmoHilsModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        if input_embeddings is not None:
            raise NotImplementedError(
                "olmo_hils does not support input_embeddings: landmark "
                "insertion requires token ids"
            )
        out = self.model(inputs, cache)
        # mlx 0.31.x quantized_matmul is wrong for unaligned N (the vocab,
        # 100278, is not a multiple of 32) when 10 <= rows and rows % 32
        # != 0 (fixed in mlx 0.32). Pad the row count to a multiple of 32
        # so batched-verify logits (speculative decoding) are correct.
        L = out.shape[1]
        pad = 32 - L % 32 if (_NEEDS_QMM_ROW_PAD and L >= 10 and L % 32) else 0
        if pad:
            out = mx.concatenate(
                [out, mx.zeros((out.shape[0], pad, out.shape[2]), out.dtype)],
                axis=1,
            )
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)
        return logits[:, :L] if pad else logits

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self, max_kv_size=None):
        return self.model.make_cache(max_kv_size=max_kv_size)
