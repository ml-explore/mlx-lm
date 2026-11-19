# Copyright © 2026 Apple Inc.

# Qwen3.8-Flash-Next: qwen3_next with QSA sparse attention, hyper-connections
# and an n-gram embedding (PLE).

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, CacheList, KVCache
from .gated_delta import gated_delta_update, normalize_qk
from .switch_layers import SwitchGLU

# splitmix64 and n-gram hash constants.
_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1, _M2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
_PRIME_1 = 10007

# Norms stored as `w` for a `1 + w` scale; `linear_attn.norm` is not one.
_FOLD_ONE = (
    "q_layernorm.weight",
    "k_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "hc_norm.weight",
    "norm_key.weight",
    "norm_query.weight",
    "norm_conv.weight",
)

_NGRAM_CONSTANTS = (
    "ple_embedding.layer_multipliers",
    "ple_embedding.ngram_heads_vocab_sizes",
    "ple_embedding.ngram_heads_offsets",
)

_NGRAM_SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)\.")


@dataclass
class TextArgs(BaseModelArgs):
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_parameters: dict[str, Any]
    eos_token_id: int | list[int]
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    hc_count: int
    hc_lowrank: int
    indexer_n_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    ple_embed_dim: int
    ple_layer_ids: list[int]
    # Optional, with the reference's defaults. config.json has no `seed` (n-gram hash).
    ple_conv_kernel_size: int = 4
    seed: int = 1234
    layer_types: list[str] = field(default_factory=list)
    full_attention_interval: int = 4
    output_gate_type: str | None = None
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    rope_theta: float = field(init=False)
    partial_rotary_factor: float = field(init=False)

    def __post_init__(self) -> None:
        self.rope_theta = float(self.rope_parameters["rope_theta"])
        self.partial_rotary_factor = float(
            self.rope_parameters.get("partial_rotary_factor", 1.0)
        )
        self.output_gate_type = self.output_gate_type or self.hidden_act
        if not self.layer_types:
            k = self.full_attention_interval
            self.layer_types = [
                "full_attention" if (i + 1) % k == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict[str, Any]
    text: TextArgs = field(init=False)

    def __post_init__(self) -> None:
        self.text = TextArgs.from_dict(self.text_config)


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

    def __call__(self, x: mx.array, cache: Any | None = None) -> mx.array | None:
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
            raw_k, _ = cache.update_and_fetch(raw_k[:, None], raw_k[:, None])
            # Nothing reads the values: reset them, so their graph cannot grow.
            cache.values = mx.zeros_like(cache.keys)
            raw_k = raw_k[:, 0]
        kv_len = raw_k.shape[1]

        # Every visible token fits in the budget: the causal mask is enough.
        if kv_len <= self.token_budget:
            return None

        n_blocks = kv_len // r
        if pad is None:
            pad = mx.zeros((B,), dtype=mx.int32)
            blocks = raw_k[:, : n_blocks * r]
        else:
            # A block that runs past the buffer is never visible.
            pos = mx.minimum(pad[:, None] + mx.arange(n_blocks * r), kv_len - 1)
            blocks = mx.take_along_axis(raw_k, pos[..., None], axis=1)
        pooled = blocks.reshape(B, n_blocks, r, self.head_dim)
        pooled = pooled.astype(mx.float32).mean(axis=2).astype(raw_k.dtype)
        pooled = self.block_rope(self.k_layernorm(pooled))

        q = self.q_layernorm(q).transpose(0, 2, 1, 3)
        q = self.rope(q, offset=offset)

        # Per block: the sum over heads of relu(q.k).
        scores = mx.einsum(
            "bhsd,bnd->bsnh", q.astype(mx.float32), pooled.astype(mx.float32)
        )
        scores = mx.maximum(scores, 0).sum(axis=-1) / math.sqrt(self.head_dim)

        # Only blocks fully in the query's past; positions exclude the padding.
        q_pos = mx.array(offset).reshape(-1, 1) + mx.arange(S)
        n_complete = mx.maximum(q_pos + 1, 0) // r
        visible = mx.arange(n_blocks) < n_complete[..., None]
        scores = mx.where(visible, scores, -mx.inf)

        k = min(self.block_topk, n_blocks)
        top = mx.stop_gradient(mx.argpartition(-scores, k - 1, axis=-1)[..., :k])
        picked = mx.take_along_axis(visible, top, axis=-1)

        # The tokens of the picked blocks; the others go to a dropped column.
        tok = pad[:, None, None, None] + top[..., None] * r + mx.arange(r)
        tok = mx.where(picked[..., None], tok, kv_len).reshape(B, S, -1)
        keep = mx.zeros((B, S, kv_len + 1), dtype=mx.bool_)
        keep = mx.put_along_axis(keep, tok, mx.array(True), axis=-1)[..., :kv_len]

        # The reference also keeps each query's own partial block (its `tail`).
        own_start = n_complete * r + pad[:, None]
        cols = mx.arange(kv_len)
        q_cols = mx.arange(kv_len - S, kv_len)[:, None]
        own = (cols >= own_start[..., None]) & (cols <= q_cols)
        return (keep | own)[:, None]


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

        # The sparse mask is causal and has no padding: it replaces `mask`.
        sparse = self.indexer(x, idx_cache)
        if sparse is not None:
            mask = sparse

        out = scaled_dot_product_attention(
            q, k, v, cache=kv_cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))


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


class Qwen4ExpModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.hc = args.hc_count
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        # No final norm: the mixer normalizes the streams before it merges them.
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.first_attention = args.layer_types.index("full_attention")
        self.first_linear = args.layer_types.index("linear_attention")

    def __call__(
        self,
        ids: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = self.embed_tokens(ids) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)

        # Each kind of layer shares the mask of its first layer.
        attn_cache = cache[self.first_attention]
        mask = create_attention_mask(
            h, cache=attn_cache[0] if attn_cache is not None else None
        )
        # The deltanet and PLE must skip the padding.
        conv_mask = create_ssm_mask(h, cache=cache[self.first_linear])

        h = mx.tile(h, (1, 1, self.hc))
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask=mask, conv_mask=conv_mask, cache=c, ids=ids)
        return self.hyper_connection_mixer(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args.text)
        if not args.text.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.text.hidden_size, args.text.vocab_size, bias=False
            )

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.text.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.layers

    def make_cache(self) -> list[Any]:
        caches = []
        for layer in self.layers:
            if layer.layer_type == "full_attention":
                caches.append(CacheList(KVCache(), KVCache()))
            else:
                # The deltanet conv and recurrent state; a PLE layer adds its
                # conv state and n-gram context.
                caches.append(ArraysCache(2 if layer.ple is None else 4))
        return caches

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        # HF and mlx-vlm (flat layout, `shards.{i}` n-gram names) store these norms
        # as `w` for a `1 + w` scale; other checkpoints already hold `1 + w`.
        fold = any(
            k.startswith("model.language_model.")
            or (k.startswith("language_model.") and ".ngram_embedding.shards." in k)
            for k in weights
        )

        out = {}
        for k, v in weights.items():
            # Not in this text-only port: the MTP head and the vision tower.
            if k.startswith(("mtp.", "model.mtp.")):
                continue
            if k.startswith(("model.visual.", "visual.", "vision_tower.")):
                continue

            # Prefixes: HF `model.language_model.`, mlx-vlm `language_model.`.
            if k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model.") :]
            elif k.startswith("language_model."):
                k = k[len("language_model.") :]

            # The n-gram hash constants: check them against the config, then drop.
            if k.endswith(_NGRAM_CONSTANTS):
                self._check_ngram_constant(k, v)
                continue

            # Rename the n-gram shards to `shards.{i}`, as mlx-vlm saves them.
            k = _NGRAM_SHARD_RE.sub(r".ngram_embedding.shards.\1.", k)

            # The experts stack as (E, out, in); gate_up_proj holds gate, then up.
            if k.endswith("mlp.experts.gate_up_proj"):
                base = k[: -len("experts.gate_up_proj")]
                mid = v.shape[-2] // 2
                out[base + "switch_mlp.gate_proj.weight"] = v[..., :mid, :]
                out[base + "switch_mlp.up_proj.weight"] = v[..., mid:, :]
                continue
            if k.endswith("mlp.experts.down_proj"):
                base = k[: -len("experts.down_proj")]
                out[base + "switch_mlp.down_proj.weight"] = v
                continue

            # torch (C, 1, K) to mlx (C, K, 1); a converted weight has shape[1] > 1.
            if k.endswith("conv1d.weight") and v.ndim == 3 and v.shape[1] == 1:
                v = v.transpose(0, 2, 1)
            if fold and k.endswith(_FOLD_ONE):
                v = 1.0 + v
            out[k] = v

        # The FP8 checkpoint stores each routed expert alone, in e4m3 with a
        # scale per block.
        for k in [k for k in out if k.endswith("mlp.experts.0.down_proj.weight")]:
            base = k[: -len("experts.0.down_proj.weight")]
            for proj in ("gate_proj", "up_proj", "down_proj"):
                keys = [
                    f"{base}experts.{e}.{proj}.weight"
                    for e in range(self.args.text.num_experts)
                ]
                weight = mx.stack([out.pop(key) for key in keys])
                scale = mx.stack([out.pop(key + "_scale_inv") for key in keys])
                out[f"{base}switch_mlp.{proj}.weight"] = _from_fp8(weight, scale)

        # Merge the checkpoint's block inject into the mix down.
        for k in [k for k in out if k.endswith("block_inject_weight.weight")]:
            p = k[: -len("block_inject_weight.weight")]
            down = p + "input_mix_weight_down"
            _merge_linears(out, [down, p + "block_inject_weight"], target=down)

        # The checkpoint has equal row shards; the model keeps one table per head.
        for k in [k for k in out if k.endswith("ngram_embedding.shards.0.weight")]:
            p = k[: -len("shards.0.weight")]
            n = sum(q.startswith(p + "shards.") and q.endswith(".weight") for q in out)
            layer = int(k.split(".layers.")[1].split(".")[0])
            sizes = self.model.layers[layer].ple.ple_embedding.head_vocab_sizes
            # The FP8 checkpoint stores the table in e4m3 with one scale.
            if (scale := out.pop(p + "weight_scale", None)) is not None:
                for i in range(n):
                    key = f"{p}shards.{i}.weight"
                    out[key] = _from_fp8(out[key], scale.reshape(1, 1))
            # A quantized table splits its scales and biases by the same rows.
            for name in ("weight", "scales", "biases"):
                if f"{p}shards.0.{name}" in out:
                    shards = [out.pop(f"{p}shards.{i}.{name}") for i in range(n)]
                    for h, table in enumerate(_split_rows(shards, sizes)):
                        out[f"{p}{h}.{name}"] = table

        return out

    def _check_ngram_constant(self, key: str, value: mx.array) -> None:
        layer = int(key.split(".layers.")[1].split(".")[0])
        table = self.model.layers[layer].ple.ple_embedding
        rebuilt = {
            "layer_multipliers": table._mults,
            "ngram_heads_vocab_sizes": table._sizes,
            "ngram_heads_offsets": table._offsets,
        }[key.rsplit(".", 1)[1]]
        if not mx.array_equal(rebuilt, value.astype(rebuilt.dtype)):
            raise ValueError(
                f"{key}: the n-gram hash constants rebuilt from the config do "
                f"not match the checkpoint ({rebuilt.tolist()} vs "
                f"{value.tolist()}); check `seed` and the hash parameters"
            )

    @property
    def quant_predicate(self) -> Callable[[str, nn.Module], bool]:
        def predicate(path: str, _: nn.Module) -> bool:
            # The MoE router is not quantized.
            return not path.endswith("mlp.gate")

        return predicate


def _inject(hyper: mx.array, x: mx.array, inject: mx.array) -> mx.array:
    """Add `x` to every hyper-connection stream, scaled per stream."""
    return hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)


def _keep_last(x: mx.array, *, n: int, lengths: mx.array | None) -> mx.array:
    """The last `n` steps of `x`; a right padded row ends at its length."""
    if lengths is None:
        # A copy: a slice would keep all of `x` alive in the cache.
        return mx.contiguous(x[:, -n:])
    positions = mx.clip(lengths, 0, x.shape[1] - n)[:, None] + mx.arange(n)
    if x.ndim == 3:
        positions = positions[..., None]
    return mx.take_along_axis(x, positions, axis=1)


def _merge_linears(
    weights: dict[str, mx.array], parts: list[str], *, target: str
) -> None:
    """Concatenate the layers `parts` into `target` along the output axis,
    with their scales and biases when quantized."""
    for name in ("weight", "scales", "biases"):
        keys = [f"{p}.{name}" for p in parts]
        if all(k in weights for k in keys):
            weights[f"{target}.{name}"] = mx.concatenate(
                [weights.pop(k) for k in keys], axis=-2
            )


def _from_fp8(weight: mx.array, scale: mx.array) -> mx.array:
    """Dequantize an e4m3 `weight` to bfloat16. The last two axes of `scale`
    tile the last two of `weight` into equal blocks, one scale each."""
    *lead, rows, cols = weight.shape
    grid_rows, grid_cols = scale.shape[-2:]
    blocks = mx.from_fp8(weight, dtype=mx.float32).reshape(
        *lead, grid_rows, rows // grid_rows, grid_cols, cols // grid_cols
    )
    blocks = blocks * scale.astype(mx.float32)[..., :, None, :, None]
    return blocks.reshape(weight.shape).astype(mx.bfloat16)


def _split_rows(shards: list[mx.array], sizes: list[int]) -> list[mx.array]:
    """Split a table stored as equal row shards into tables of `sizes` rows."""
    rows = shards[0].shape[0]
    tables, start = [], 0
    for size in sizes:
        end = start + size
        first, last = start // rows, (end - 1) // rows
        pieces = [
            s[max(start - i * rows, 0) : min(end - i * rows, rows)]
            for i, s in enumerate(shards[first : last + 1], first)
        ]
        tables.append(mx.concatenate(pieces))
        start = end
    return tables


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
