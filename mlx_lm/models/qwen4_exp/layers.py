# Copyright © 2026 Apple Inc.

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.kernels import sparse_attention
from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.layers.recurrent import gated_delta_update, normalize_qk
from mlx_lm.models.qwen4_exp.config import TextArgs

__all__ = [
    "Attention",
    "DecoderLayer",
    "GatedDeltaNet",
    "GatedResidual",
    "MLP",
    "NGramEmbedding",
    "PLELayer",
    "QSAIndexer",
    "RMSNorm",
    "RMSNormGated",
    "SparseMoeBlock",
]

# splitmix64 and n-gram hash constants.
_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1, _M2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
_PRIME_1 = 10007

# The prefill top k scores this many key blocks at a time (as blocked_topk in
# mlx_lm/layers/attention.py), so that only one block of scores is alive.
_TOPK_KEY_BLOCK = 4096
# The gather prefill reads the keys of this many queries at a time.
_QUERY_BLOCK = 128


class RMSNorm(nn.Module):
    """RMSNorm, optionally per `group_size` slice; `weight` holds the folded `1 + w`."""

    def __init__(self, dim: int, *, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.group_size = group_size

    def __call__(self, x: mx.array) -> mx.array:
        if self.group_size is None:
            return mx.fast.rms_norm(x, self.weight, eps=self.eps)
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.group_size)
        x = mx.fast.rms_norm(x, None, eps=self.eps).reshape(shape)
        return x * self.weight


def _keep_last(x: mx.array, *, n: int, lengths: mx.array | None) -> mx.array:
    """The last `n` steps of `x`; a right padded row ends at its length."""
    if lengths is None:
        # A copy: a slice would keep all of `x` alive in the cache.
        return mx.contiguous(x[:, -n:])
    positions = mx.clip(lengths, 0, x.shape[1] - n)[:, None] + mx.arange(n)
    if x.ndim == 3:
        positions = positions[..., None]
    return mx.take_along_axis(x, positions, axis=1)


class RMSNormGated(nn.Module):
    """Gated RMSNorm; the checkpoint stores `weight` as is."""

    def __init__(self, dim: int, *, activation: str, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.activation = activation

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        act = mx.sigmoid if self.activation == "sigmoid" else nn.silu
        out = mx.fast.rms_norm(x, self.weight, eps=self.eps)
        return (out * act(gate.astype(mx.float32))).astype(x.dtype)


class GatedDeltaNet(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_v = args.linear_num_value_heads
        self.n_k = args.linear_num_key_heads
        self.dk = args.linear_key_head_dim
        self.dv = args.linear_value_head_dim
        self.key_dim = self.dk * self.n_k
        self.value_dim = self.dv * self.n_v
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        d = args.hidden_size

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )
        self.in_proj_qkv = nn.Linear(d, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(d, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(d, self.n_v, bias=False)
        self.in_proj_a = nn.Linear(d, self.n_v, bias=False)
        self.dt_bias = mx.ones(self.n_v)
        self.A_log = mx.zeros(self.n_v)
        self.norm = RMSNormGated(
            self.dv, activation=args.output_gate_type, eps=args.rms_norm_eps
        )
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, S, _ = x.shape
        mixed_qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, S, self.n_v, self.dv)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        conv_state = (
            cache[0]
            if (cache is not None and cache[0] is not None)
            else mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
        )
        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            cache[0] = _keep_last(conv_input, n=n_keep, lengths=cache.lengths)
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q.reshape(B, S, self.n_k, self.dk)
        k = k.reshape(B, S, self.n_k, self.dk)
        v = v.reshape(B, S, self.n_v, self.dv)

        q, k = normalize_qk(q, k, inv_scale=self.dk**-0.5, eps=1e-6)

        state = cache[1] if cache is not None else None
        out, state = gated_delta_update(
            q,
            k,
            v,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            state=state,
            mask=mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(S)

        return self.out_proj(self.norm(out, gate=z).reshape(B, S, -1))


class QSAIndexer(nn.Module):
    """Select, per query, a budget of compressed key blocks."""

    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.head_dim = args.indexer_head_dim
        self.token_budget = args.indexer_budget
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            args.hidden_size,
            (self.n_heads + args.indexer_kv_heads) * self.head_dim,
            bias=False,
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope = nn.RoPE(rotary_dim, base=args.rope_theta)
        # Block n starts at position n * compress_ratio.
        self.block_rope = nn.RoPE(
            rotary_dim, base=args.rope_theta, scale=self.compress_ratio
        )

    def _update_cache(
        self, raw_k: mx.array, cache: Any, pad: mx.array | None
    ) -> mx.array:
        """Add the raw keys to the cache and return all of them. The values buffer
        holds the key of each complete block at the block's last row; no other row is
        read. In token rows, the block keys follow trim, save/load and batching of the
        cache without other code."""
        S = raw_k.shape[1]
        r = self.compress_ratio
        # The values of the new rows are written below.
        keys = cache.update_and_fetch(raw_k[:, None], raw_k[:, None])[0][:, 0]
        n = keys.shape[1]
        # Each new row gets the mean of the r rows that end at it. Rows before row 0
        # are zeros: they are in no block.
        start = n - S - (r - 1)
        rows = keys[:, max(start, 0) :]
        if start < 0:
            rows = mx.pad(rows, [(0, 0), (-start, 0), (0, 0)])
        # Window i is rows i to i + r - 1 of `rows`, as a strided view.
        B, m, d = rows.shape
        windows = mx.as_strided(rows, (B, S, r, d), (m * d, d, d, 1))
        pooled = windows.astype(mx.float32).mean(axis=2).astype(keys.dtype)
        # Row t gets RoPE at t - pad - (r - 1): the first token of a block ending at t.
        offset = start if pad is None else start - pad
        block_keys = self.rope(self.k_layernorm(pooled), offset=offset)
        cache.values[..., n - S : n, :] = block_keys[:, None]
        return keys

    def __call__(
        self, x: mx.array, cache: Any | None = None
    ) -> tuple[mx.array, mx.array] | None:
        """The key positions of each query and their mask, [B, S, (k + 1) * r]: the
        picked blocks, then the query's own partial block. None when all keys fit in
        the budget (causal attention)."""
        B, S, _ = x.shape
        r = self.compress_ratio
        qk = self.index_qk_proj(x)
        split = self.n_heads * self.head_dim
        q = qk[..., :split].reshape(B, S, self.n_heads, self.head_dim)
        # One key head (`indexer_kv_heads` is 1).
        raw_k = qk[..., split:].reshape(B, S, self.head_dim)

        offset = cache.offset if cache is not None else 0
        # With left padding, each row's blocks start at its first real token.
        pad = getattr(cache, "left_padding", None)
        if cache is not None:
            raw_k = self._update_cache(raw_k, cache, pad)
        kv_len = raw_k.shape[1]

        # Every visible token fits in the budget: the causal mask is enough.
        if kv_len <= self.token_budget:
            return None

        n_blocks = kv_len // r
        if cache is None:
            blocks = raw_k[:, : n_blocks * r].reshape(B, n_blocks, r, self.head_dim)
            block_keys = blocks.astype(mx.float32).mean(axis=2).astype(raw_k.dtype)
            block_keys = self.block_rope(self.k_layernorm(block_keys))
        elif pad is None:
            block_keys = cache.values[:, 0, r - 1 : n_blocks * r : r]
        else:
            # A block that runs past the buffer is never visible.
            ends = pad[:, None] + mx.arange(r - 1, n_blocks * r, r)
            ends = mx.minimum(ends, kv_len - 1)[..., None]
            block_keys = mx.take_along_axis(cache.values[:, 0], ends, axis=1)

        q = self.q_layernorm(q).transpose(0, 2, 1, 3)
        q = self.rope(q, offset=offset).astype(mx.float32)

        # Only blocks fully in the query's past; positions exclude the padding.
        if isinstance(offset, int):
            n_complete = (offset + 1 + mx.arange(S)) // r
            all_visible = offset + 1 >= n_blocks * r
        else:
            q_pos = offset.reshape(-1, 1) + mx.arange(S)
            n_complete = mx.maximum(q_pos + 1, 0) // r
            all_visible = False
        n_complete = mx.broadcast_to(n_complete, (B, S))

        # Top k of the sum over heads of relu(q.k). Few queries (decode, speculative
        # verify) score all key blocks at once.
        k = min(self.block_topk, n_blocks)
        step = n_blocks if S <= 16 else max(_TOPK_KEY_BLOCK, k)
        best = ids = None
        for s0 in range(0, n_blocks, step):
            kb = block_keys[:, s0 : s0 + step]
            if best is not None:
                kb = mx.depends(kb, best)
            scores = q @ kb[:, None].astype(mx.float32).swapaxes(-1, -2)
            scores = mx.maximum(scores, 0).sum(axis=1) / math.sqrt(self.head_dim)
            cols = mx.arange(s0, s0 + kb.shape[1])
            if not all_visible:
                scores = mx.where(cols < n_complete[..., None], scores, -mx.inf)
            cols = mx.broadcast_to(cols, scores.shape)
            if best is not None:
                scores = mx.concatenate([best, scores], axis=-1)
                cols = mx.concatenate([ids, cols], axis=-1)
            top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]
            top = mx.stop_gradient(top)
            best = mx.take_along_axis(scores, top, axis=-1)
            ids = mx.take_along_axis(cols, top, axis=-1)

        # The picked blocks, then each query's own partial block (the reference's
        # `tail`): its positions up to the query.
        ids = mx.concatenate([ids, n_complete[..., None]], axis=-1)
        tok = (ids[..., None] * r + mx.arange(r)).reshape(B, S, -1)
        if pad is not None:
            tok = tok + pad[:, None, None]
        picked = mx.concatenate([best > -mx.inf, mx.ones((B, S, 1), mx.bool_)], axis=-1)
        valid = mx.repeat(picked, r, axis=-1) & (
            tok <= mx.arange(kv_len - S, kv_len)[:, None]
        )
        # The own block can run past the buffer: those positions are masked.
        return mx.minimum(tok, kv_len - 1), mx.stop_gradient(valid)


def _key_mask(tok: mx.array, valid: mx.array, kv_len: int) -> mx.array:
    """The selection as a dense mask [B, 1, S, kv_len]."""
    B, S, _ = tok.shape
    # The other positions go to a dropped column.
    tok = mx.where(valid, tok, kv_len)
    keep = mx.zeros((B, S, kv_len + 1), dtype=mx.bool_)
    return mx.put_along_axis(keep, tok, mx.array(True), axis=-1)[:, None, :, :kv_len]


class Attention(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        d = args.hidden_size
        # q_proj also carries the output gate.
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = nn.RoPE(
            int(args.head_dim * args.partial_rotary_factor), base=args.rope_theta
        )
        self.indexer = QSAIndexer(args)
        self.sparse_attention = args.sparse_attention
        self.sparse_prefill_min_context = args.sparse_prefill_min_context
        self.sparse_prefill_fused = args.sparse_prefill_fused

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, S, _ = x.shape
        # `cache` holds the attention keys and values, then the indexer keys.
        kv_cache, idx_cache = (
            (cache[0], cache[1]) if cache is not None else (None, None)
        )
        offset = kv_cache.offset if kv_cache is not None else 0

        q, gate = mx.split(self.q_proj(x).reshape(B, S, self.n_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, S, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)

        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(k, v)

        selection = self.indexer(x, idx_cache)
        # A hand-built quantized cache gives (packed keys, scales, biases) and uses
        # the masked path (quantized SDPA).
        quantized = hasattr(kv_cache, "bits")
        kv_len = (k[0] if quantized else k).shape[2]
        # Decode always reads only the picked keys. Sparse prefill supports one sequence;
        # below its minimum context the masked path is faster.
        sparse = (
            selection is not None
            and self.sparse_attention
            and not self.training
            and not quantized
            and (S == 1 or (B == 1 and kv_len >= self._min_context(q, k, v)))
        )

        if sparse:
            # A slice of the cache buffers is not contiguous: gather from them.
            keys, values = (
                (kv_cache.keys, kv_cache.values) if kv_cache is not None else (k, v)
            )
            out = self._sparse_attention(q, keys, values, *selection)
        else:
            if selection is not None:
                # The selection is causal and has no padding: it replaces `mask`.
                mask = _key_mask(*selection, kv_len)
            elif kv_cache is not None:
                # Nothing reads the indexer cache: keep its graph from growing.
                kv_cache.keys = mx.depends(
                    kv_cache.keys, (idx_cache.keys, idx_cache.values)
                )
            out = scaled_dot_product_attention(
                q, k, v, cache=kv_cache, scale=self.scale, mask=mask
            )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))

    def _sparse_attention(
        self,
        q: mx.array,
        keys: mx.array,
        values: mx.array,
        tok: mx.array,
        valid: mx.array,
    ) -> mx.array:
        """Attention of each query over its selected key positions only: one Metal
        kernel at prefill if it supports the inputs, else gather + SDPA."""
        if q.shape[2] > 1 and self.sparse_prefill_fused and not self.training:
            idx = mx.where(valid, tok, -1).squeeze(0)  # the kernel skips -1
            out = sparse_attention.indexed_attention(
                q.squeeze(0), keys.squeeze(0), values.squeeze(0), idx, self.scale
            )
            if out is not None:
                return out[None]
        B, H, S, _ = q.shape
        if S == 1:
            # Copy whole rows of the flat [B * kv_heads * T, D] buffers: (b, g, t) is
            # row (b * kv_heads + g) * T + t. Faster than take_along_axis for B > 1.
            T = keys.shape[2]
            first = mx.arange(0, B * self.n_kv_heads * T, T).reshape(B, -1, 1)
            rows = first + tok[:, :1]
            k = mx.take(keys.reshape(-1, keys.shape[-1]), rows, axis=0)
            v = mx.take(values.reshape(-1, values.shape[-1]), rows, axis=0)
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=valid[:, None]
            )
        # B is 1. One SDPA call per kv head: each query is a batch entry, with the G
        # heads as its rows. Query blocks run one after the other to keep the gathered
        # keys small.
        G = H // self.n_kv_heads
        outs = []
        for s0 in range(0, S, _QUERY_BLOCK):
            s1 = min(S, s0 + _QUERY_BLOCK)
            idx = tok[0, s0:s1]
            if outs:
                idx = mx.depends(idx, outs[-1])
            mask = valid[0, s0:s1, None, None]
            out = []
            for g in range(self.n_kv_heads):
                k = mx.take(keys[0, g], idx, axis=0)[:, None]
                v = mx.take(values[0, g], idx, axis=0)[:, None]
                qg = q[0, g * G : (g + 1) * G, s0:s1].transpose(1, 0, 2)[:, None]
                out.append(
                    mx.fast.scaled_dot_product_attention(
                        qg, k, v, scale=self.scale, mask=mask
                    )
                )
            outs.append(mx.concatenate(out, axis=2)[:, 0])
        return mx.concatenate(outs, axis=0).transpose(1, 0, 2)[None]

    def _min_context(self, q: mx.array, k: mx.array, v: mx.array) -> int:
        """Cache length from which prefill reads only the picked keys. With the Metal
        kernel, that is as soon as the indexer selects."""
        if (
            self.sparse_prefill_fused
            and not self.training
            and sparse_attention.supports(
                self.n_heads,
                self.head_dim,
                q.dtype,
                k.dtype,
                v.dtype,
                kv_heads=self.n_kv_heads,
            )
        ):
            return 0
        return self.sparse_prefill_min_context


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            input_dims=args.hidden_size,
            hidden_dims=args.moe_intermediate_size,
            num_experts=args.num_experts,
        )
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(-logits, self.top_k - 1, axis=-1)[..., : self.top_k]
        idx = mx.stop_gradient(idx)
        # The softmax of the top-k logits: the reference's renormalized top-k.
        w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1)
        out = (self.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        return out + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def _nth_prime_after(start: int, *, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


def _splitmix64(v: int) -> int:
    v = (v + _GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * _M1) & _MASK64
    v = ((v ^ (v >> 27)) * _M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def _ngram_multipliers(*, vocab_size: int, ngram_size: int, seed: int) -> list[int]:
    """Odd hash multipliers, small enough that `token * multiplier` fits in int64."""
    half = max(1, ((1 << 63) - 1) // max(vocab_size, 1) // 2)
    return [
        2 * (_splitmix64((seed + _GAMMA * (i + 1)) & _MASK64) % half) + 1
        for i in range(ngram_size)
    ]


class NGramEmbedding(nn.Module):
    """N-gram hash tables, one per head (~51B parameters)."""

    def __init__(self, args: TextArgs, *, ple_layer_index: int):
        super().__init__()
        self.ngram_size = args.ngram_size
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.eos_token_id = (
            args.eos_token_id[0]
            if isinstance(args.eos_token_id, list)
            else args.eos_token_id
        )
        head_dim = args.ple_embed_dim // self.ngram_heads

        sizes, offsets, total = [], [], 0
        for h in range(self.ngram_heads):
            g = ple_layer_index * self.ngram_heads + h
            s = _nth_prime_after(args.ngram_vocab_size_base - 1, count=g + 1)
            sizes.append(s)
            offsets.append(total)
            total += s
        self.head_vocab_sizes = sizes
        # One table per head keeps each checkpoint file small (~6 GB in bf16).
        self.ngram_embedding = [nn.Embedding(s, head_dim) for s in sizes]

        mults = _ngram_multipliers(
            vocab_size=args.vocab_size,
            ngram_size=self.ngram_size,
            seed=args.seed + _PRIME_1 * ple_layer_index,
        )
        # Not parameters: rebuilt from the config, checked in `sanitize`.
        self._mults = mx.array(mults, dtype=mx.int64)
        self._sizes = mx.array(sizes, dtype=mx.int64)
        self._offsets = mx.array(offsets, dtype=mx.int64)

    def __call__(self, ids: mx.array, cache: Any | None = None) -> mx.array:
        n, eos = self.ngram_size, self.eos_token_id
        # Slot 3 holds `token ^ eos`: XOR with eos again gives the token, and
        # the zeros that batching puts in a new row read as EOS (no tokens yet).
        if cache is not None and cache[3] is not None:
            prev = cache[3] ^ eos
        else:
            prev = mx.full((ids.shape[0], n - 1), eos, ids.dtype)
        history = mx.concatenate([prev, ids], axis=1)
        if cache is not None:
            cache[3] = _keep_last(history, n=n - 1, lengths=cache.lengths) ^ eos

        tokens = self._ngram_tokens(history.astype(mx.int64))

        # Step s gives the hash of the (s+1)-gram.
        mixed = tokens[0] * self._mults[0]
        rows = []
        for s in range(1, n):
            mixed = mx.bitwise_xor(mixed, tokens[s] * self._mults[s])
            lo = (s - 1) * self.heads_per_ngram
            rows.append(mixed[..., None] % self._sizes[lo : lo + self.heads_per_ngram])
        rows = mx.concatenate(rows, axis=-1)

        return mx.concatenate(
            [table(rows[..., h]) for h, table in enumerate(self.ngram_embedding)],
            axis=-1,
        )

    def _ngram_tokens(self, history: mx.array) -> list[mx.array]:
        """x[t], x[t-1], ... for the new tokens; `history` starts with the n-1
        tokens before them. Older tokens past an EOS read as EOS."""
        n, eos = self.ngram_size, self.eos_token_id
        new = history.shape[1] - (n - 1)
        # HF's `_shift_right_ignore_eos`, at the new positions only.
        tokens = [history[:, n - 1 :], history[:, n - 2 : n - 2 + new]]
        for s in range(2, n):
            older = history[:, n - 1 - s : n - 1 - s + new]
            tokens.append(mx.where(tokens[-1] == eos, eos, older))
        return tokens


class PLELayer(nn.Module):
    def __init__(self, args: TextArgs, *, ple_layer_index: int):
        super().__init__()
        self.d = args.hidden_size
        self.hc = args.hc_count
        hc_dim = self.d * self.hc
        self.ple_embedding = NGramEmbedding(args, ple_layer_index=ple_layer_index)
        k = args.ple_conv_kernel_size
        dilation = args.ngram_size
        self.short_conv_state_len = (k - 1) * dilation
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_dim, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, self.d, bias=False)
        self.norm_key = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.norm_query = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.norm_conv = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.conv1d = nn.Conv1d(
            hc_dim,
            hc_dim,
            kernel_size=k,
            groups=hc_dim,
            dilation=dilation,
            bias=False,
        )

    def __call__(
        self,
        hidden: mx.array,
        ids: mx.array,
        cache: Any | None = None,
        *,
        mask: mx.array | None = None,
    ) -> mx.array:
        if mask is not None:
            # Padding reads as EOS, like the start of a prompt.
            ids = mx.where(mask, ids, self.ple_embedding.eos_token_id)
        emb = self.ple_embedding(ids, cache)

        key = self.norm_key(self.key_proj(emb))
        key = key.reshape(*key.shape[:-1], self.hc, self.d)
        value = self.value_proj(emb)
        query = self.norm_query(hidden)
        query = query.reshape(*query.shape[:-1], self.hc, self.d)

        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(self.d)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.reshape(*gated.shape[:-2], -1)
        if mask is not None:
            gated = mx.where(mask[..., None], gated, 0)

        return gated + self._short_conv(self.norm_conv(gated), cache)

    def _short_conv(self, x: mx.array, cache: Any | None) -> mx.array:
        n = self.short_conv_state_len
        state = (
            cache[2]
            if (cache is not None and cache[2] is not None)
            else mx.zeros((x.shape[0], n, x.shape[-1]), dtype=x.dtype)
        )
        full = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[2] = _keep_last(full, n=n, lengths=cache.lengths)
        return nn.silu(self.conv1d(full))


class GatedResidual(nn.Module):
    def __init__(self, args: TextArgs, *, use_combine: bool = True):
        super().__init__()
        self.hc = args.hc_count
        self.d = args.hidden_size
        self.lowrank = args.hc_lowrank
        self.use_combine = use_combine
        hc_dim = self.hc * self.d
        self.hc_norm = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        # The last hc rows are the checkpoint's block inject: one matmul for both.
        down = self.lowrank + (self.hc if use_combine else 0)
        self.input_mix_weight_down = nn.Linear(hc_dim, down, bias=False)
        self.input_mix_weight_up = nn.Linear(self.lowrank, hc_dim, bias=False)

    def __call__(self, hyper: mx.array) -> mx.array | tuple[mx.array, mx.array]:
        normed = self.hc_norm(hyper)
        down = self.input_mix_weight_down(normed)
        w = nn.silu(down[..., : self.lowrank] / self.hc)
        w = mx.sigmoid(self.input_mix_weight_up(w))
        w = w.reshape(*w.shape[:-1], self.hc, self.d)
        mixed = (w * normed.reshape(*normed.shape[:-1], self.hc, self.d)).mean(axis=-2)
        if not self.use_combine:
            return mixed
        inject = 2 * mx.sigmoid(down[..., self.lowrank :] / self.hc)
        return mixed, inject


def _inject(hyper: mx.array, x: mx.array, inject: mx.array) -> mx.array:
    """Add `x` to every hyper-connection stream, scaled per stream."""
    return hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, *, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.mlp = SparseMoeBlock(args)
        # `ple_layer_ids` counts layers from 1.
        if layer_idx + 1 in args.ple_layer_ids:
            ple_index = args.ple_layer_ids.index(layer_idx + 1)
            self.ple = PLELayer(args, ple_layer_index=ple_index)
        else:
            self.ple = None
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    def __call__(
        self,
        h: mx.array,
        *,
        mask: mx.array | str | None,
        conv_mask: mx.array | None,
        cache: Any | None,
        ids: mx.array,
    ) -> mx.array:
        if self.ple is not None:
            h = h + self.ple(h, ids, cache, mask=conv_mask)

        x, inject = self.attn_hyper_connection(h)
        if self.layer_type == "linear_attention":
            x = self.linear_attn(x, conv_mask, cache)
        else:
            x = self.self_attn(x, mask, cache)
        h = _inject(h, x, inject)

        x, inject = self.mlp_hyper_connection(h)
        return _inject(h, self.mlp(x), inject)
