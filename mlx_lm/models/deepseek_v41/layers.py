# Copyright © 2026 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.layers.common import swiglu
from mlx_lm.layers.moe import SwitchGLU, SwitchLinear
from mlx_lm.layers.rope import YarnRoPE
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.kernels import sinkhorn

__all__ = [
    "Attention",
    "Compressor",
    "DecoderLayer",
    "DeepseekV41SwiGLU",
    "Experts",
    "HyperConnection",
    "Indexer",
    "LayerCache",
    "MoE",
    "Router",
    "SharedMLP",
]


_FP4_LUT = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=mx.float32)


def _apply_rope(
    x: mx.array,
    rope: YarnRoPE,
    *,
    rope_dim: int,
    offset: int = 0,
    scale: float = 1.0,
    inverse: bool = False,
) -> mx.array:
    """Apply interleaved RoPE to the trailing feature slice."""
    if rope_dim == 0:
        return x
    head = x[..., :-rope_dim]
    tail = x[..., -rope_dim:].astype(mx.float32)
    if x.ndim == 4:
        tail = tail.transpose(0, 2, 1, 3)
        tail = rope(tail, offset=offset, scale=scale, inverse=inverse)
        tail = tail.transpose(0, 2, 1, 3)
    else:
        tail = rope(tail, offset=offset, scale=scale, inverse=inverse)
    return mx.concatenate([head, tail.astype(x.dtype)], axis=-1)


def _append(cached: Optional[mx.array], new: mx.array) -> mx.array:
    if cached is None:
        return new
    return mx.concatenate([cached, new], axis=1)


class LayerCache:
    """Cache for one decoder layer: the sliding window, the closed compressed
    groups and their index keys, and the rows of the group that is still open.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.offset = 0
        self.window = None
        self.compress_kv = None
        self.index_k = None
        self.pending_kv = None
        self.pending_gate = None

    def update_window(self, kv: mx.array) -> mx.array:
        """Return the window with kv appended, and keep its last rows."""
        window = _append(self.window, kv)
        self.window = window[:, -self.window_size :]
        self.offset += kv.shape[1]
        return window

    @property
    def state(self) -> tuple[Optional[mx.array], ...]:
        return (
            self.window,
            self.compress_kv,
            self.index_k,
            self.pending_kv,
            self.pending_gate,
            mx.array(self.offset),
        )

    def is_trimmable(self) -> bool:
        # Evicted window rows and closed compressed groups cannot be restored.
        return False

    def empty(self) -> bool:
        return self.offset == 0

    @property
    def nbytes(self) -> int:
        arrays = (
            self.window,
            self.compress_kv,
            self.index_k,
            self.pending_kv,
            self.pending_gate,
        )
        return sum(a.nbytes for a in arrays if a is not None)


class DeepseekV41SwiGLU(nn.Module):
    """DeepSeek-V4.1's clamped SwiGLU activation."""

    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, up: mx.array, gate: mx.array) -> mx.array:
        dtype = up.dtype
        gate = gate.astype(mx.float32)
        up = up.astype(mx.float32)
        if self.limit > 0:
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
        return swiglu(gate, up).astype(dtype)


class Compressor(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.ratio = int(args.compress_ratios[layer_id])
        self.kv_proj = nn.Linear(args.hidden_size, args.head_dim, bias=False)
        if self.ratio > 1:
            self.gate_proj = nn.Linear(args.hidden_size, args.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(args.head_dim, eps=args.rms_norm_eps)

    def __call__(
        self, x: mx.array, start_pos: int, cache: LayerCache
    ) -> tuple[Optional[mx.array], int]:
        """Pool each full group of ratio tokens into one latent. Return the
        latents, or None if no group is full, and the index of the first one."""
        if self.ratio == 1:
            return self.kv_norm(self.kv_proj(x)), start_pos

        h = x.astype(mx.float32)
        pending = 0 if cache.pending_kv is None else cache.pending_kv.shape[1]
        kv = _append(cache.pending_kv, self.kv_proj(h))
        gate = _append(cache.pending_gate, self.gate_proj(h))
        full = kv.shape[1] - kv.shape[1] % self.ratio
        cache.pending_kv = kv[:, full:]
        cache.pending_gate = gate[:, full:]
        # A group takes the position of its first token, and the RoPE scale
        # multiplies the offset by the ratio, so return the group index.
        first = (start_pos - pending) // self.ratio
        if full == 0:
            return None, first
        batch, _, dim = kv.shape
        kv = kv[:, :full].reshape(batch, -1, self.ratio, dim)
        gate = gate[:, :full].reshape(batch, -1, self.ratio, dim)
        latent = mx.sum(kv * mx.softmax(gate, axis=2), axis=2)
        return self.kv_norm(latent.astype(x.dtype)), first


def _round_fp4(x: mx.array) -> mx.array:
    mag = mx.abs(x)
    idx = mx.zeros(mag.shape, dtype=mx.int32)
    for threshold in (0.25, 1.25, 2.5, 5.0):
        idx = idx + (mag > threshold).astype(mx.int32)
    for threshold in (0.75, 1.75, 3.5):
        idx = idx + (mag >= threshold).astype(mx.int32)
    return mx.sign(x) * _FP4_LUT[idx]


def _fake_quant_fp4(
    x: mx.array, *, block_size: int, e4m3_scale: bool = False
) -> mx.array:
    if x.shape[-1] % block_size:
        return x
    dtype = x.dtype
    shape = x.shape
    blocks = x.astype(mx.float32).reshape(*shape[:-1], -1, block_size)
    floor = 6.0 * 2.0**-9 if e4m3_scale else 6.0 * 2.0**-126
    amax = mx.maximum(mx.max(mx.abs(blocks), axis=-1, keepdims=True), floor)
    if e4m3_scale:
        scale = mx.from_fp8(mx.to_fp8(amax / 6.0), mx.float32)
    else:
        scale = mx.power(2.0, mx.ceil(mx.log2(amax / 6.0)))
    q = _round_fp4(mx.clip(blocks / scale, -6.0, 6.0)) * scale
    return q.reshape(shape).astype(dtype)


class Indexer(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.ratio = int(args.compress_ratios[layer_id])
        self.is_candidate_source = layer_id == args.candidate_source_layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.num_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.rope_dim = args.qk_rope_head_dim
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, self.num_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(args.hidden_size, self.num_heads, bias=False)
        # Only a layer that compresses its own KV makes index keys.
        if layer_id in args.kv_source_layer_ids:
            self.k_proj = nn.Linear(args.head_dim, args.index_head_dim, bias=False)
            self.k_norm = nn.RMSNorm(args.index_head_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        q_residual: mx.array,
        *,
        index_k: Optional[mx.array],
        candidates: Optional[mx.array],
        rope: YarnRoPE,
        start_pos: int,
    ) -> tuple[mx.array, Optional[mx.array]]:
        """Return the top-k compressed entries of each query, -1 for none, and
        the candidate mask: the one this layer selects, or the given one."""
        batch, length, _ = x.shape
        if index_k is None or index_k.shape[1] == 0:
            return mx.zeros((batch, length, 0), dtype=mx.int32), candidates

        q = self.q_b_proj(q_residual).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        q = _apply_rope(q, rope, rope_dim=self.rope_dim, offset=start_pos)
        q = _fake_quant_fp4(q, block_size=32)
        keys = index_k.astype(mx.float32)
        weights = self.weights_proj(x).astype(mx.float32) * (self.num_heads**-0.5)
        scores = mx.matmul(q.astype(mx.float32), keys[:, None].swapaxes(-1, -2))
        scores = mx.maximum(scores, 0.0) * (self.head_dim**-0.5)
        scores = mx.sum(scores * weights[..., None], axis=2)
        # A query sees the groups that end at or before it.
        lens = ((start_pos + 1 + mx.arange(length)) // self.ratio)[:, None]
        visible = mx.arange(index_k.shape[1])[None, :] < lens
        scores = mx.where(visible[None], scores, -mx.inf)

        if self.is_candidate_source and self.candidate_topk_blocks > 0:
            block = self.candidate_block_size
            pad = (-scores.shape[-1]) % block
            padded = mx.pad(scores, [(0, 0), (0, 0), (0, pad)], constant_values=-mx.inf)
            block_scores = padded.reshape(batch, length, -1, block).max(axis=-1)
            # Always keep the newest block: it is only partly full, so an older
            # full block can have a higher score.
            last = mx.where(lens > 0, (lens - 1) // block, -1)
            num_blocks = block_scores.shape[-1]
            block_scores = mx.where(mx.arange(num_blocks) == last, mx.inf, block_scores)
            k_blocks = min(self.candidate_topk_blocks, num_blocks)
            chosen = mx.argpartition(-block_scores, k_blocks - 1, axis=-1)[
                ..., :k_blocks
            ]
            # With fewer reachable blocks than k_blocks, the extra picks are -inf.
            reachable = mx.take_along_axis(block_scores, chosen, axis=-1) > -mx.inf
            candidates = mx.zeros(block_scores.shape, dtype=mx.bool_)
            candidates = mx.put_along_axis(candidates, chosen, reachable, axis=-1)
            candidates = mx.repeat(candidates, block, axis=-1)[..., : scores.shape[-1]]
        elif candidates is not None:
            scores = mx.where(candidates, scores, -mx.inf)

        k = min(self.index_topk, index_k.shape[1])
        chosen = mx.argpartition(-scores, k - 1, axis=-1)[..., :k].astype(mx.int32)
        return mx.where(chosen < lens[None], chosen, -1), candidates


# The reference also rounds fp8/fp4 matmul inputs to fp8; skipping it is more precise.
def _fake_quant_fp8(x: mx.array, block_size: int = 32) -> mx.array:
    """Round-trip FP8 e4m3 with power-of-two block scales."""
    if x.shape[-1] % block_size:
        return x
    dtype = x.dtype
    shape = x.shape
    blocks = x.astype(mx.float32).reshape(*shape[:-1], -1, block_size)
    amax = mx.maximum(mx.max(mx.abs(blocks), axis=-1, keepdims=True), 1e-4)
    exponent = mx.ceil(mx.log2(amax / 448.0))
    scale = mx.power(2.0, exponent)
    quantized = mx.clip(blocks / scale, -448.0, 448.0)
    return (
        (mx.from_fp8(mx.to_fp8(quantized), mx.float32) * scale)
        .reshape(shape)
        .astype(dtype)
    )


def _gather_rows(values: mx.array, indices: mx.array) -> mx.array:
    batch, length, dim = values.shape
    safe = mx.maximum(indices, 0)
    base = (mx.arange(batch, dtype=mx.int32) * length).reshape(batch, 1, 1)
    flat = values.reshape(batch * length, dim)
    return flat[(safe + base).reshape(-1)].reshape(*indices.shape, dim)


def _sparse_attention(
    q: mx.array,
    kvs: list[mx.array],
    indices: list[mx.array],
    *,
    sinks: mx.array,
    scale: float,
) -> mx.array:
    """Attend each query to the rows its indices pick from each KV array.

    q is [B, S, H, D], each KV array [B, K, D] and its indices [B, S, k].
    An index of -1 picks nothing.
    """
    batch, length, heads, dim = q.shape
    rows = [_gather_rows(kv, idx) for kv, idx in zip(kvs, indices, strict=True)]
    # Each query has its own keys, so each query is one batch entry.
    keys = mx.concatenate(rows, axis=2).reshape(batch * length, 1, -1, dim)
    keys = keys.astype(mx.float32)
    mask = mx.concatenate(indices, axis=-1) >= 0
    # SDPA rounds the scores to the input dtype at this head size, so use fp32.
    out = mx.fast.scaled_dot_product_attention(
        q.reshape(batch * length, heads, 1, dim).astype(mx.float32),
        keys,
        keys,
        scale=scale,
        mask=mask.reshape(batch * length, 1, 1, -1),
        sinks=sinks.astype(mx.float32),
    )
    return out.reshape(batch, length, heads, dim).astype(q.dtype)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.o_groups = args.o_groups
        ratios = args.compress_ratios
        self.ratio = int(ratios[layer_id]) if layer_id < len(ratios) else 0
        self.is_kv_source = layer_id in args.kv_source_layer_ids
        self.is_index_source = layer_id in args.index_source_layer_ids
        self.q_a_proj = nn.Linear(args.hidden_size, args.q_lora_rank, bias=False)
        self.q_a_norm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, args.num_attention_heads * args.head_dim, bias=False
        )
        self.kv_proj = nn.Linear(args.hidden_size, args.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(args.head_dim, eps=args.rms_norm_eps)
        # wo_a is block diagonal: one projection per output group.
        self.o_a_proj = SwitchLinear(
            args.num_attention_heads * args.head_dim // args.o_groups,
            args.o_lora_rank,
            args.o_groups,
            bias=False,
        )
        self.o_b_proj = nn.Linear(
            args.o_groups * args.o_lora_rank, args.hidden_size, bias=False
        )
        self.sinks = mx.zeros((args.num_attention_heads,), dtype=mx.float32)
        self._group_ids = mx.arange(args.o_groups)[None]
        if self.is_kv_source:
            self.compressor = Compressor(args, layer_id)
        if self.is_index_source:
            self.indexer = Indexer(args, layer_id)
        if self.ratio:
            scaling = args.rope_scaling or {}
            original = scaling.get(
                "original_max_position_embeddings", args.max_position_embeddings
            )
            self.rope = YarnRoPE(
                dims=args.qk_rope_head_dim,
                traditional=True,
                base=args.compress_rope_theta,
                scaling_factor=float(scaling.get("factor", 1.0)),
                original_max_position_embeddings=int(original),
                beta_fast=int(scaling.get("beta_fast", 32)),
                beta_slow=int(scaling.get("beta_slow", 1)),
                # V4.1 uses attention_factor=1 for its YaRN RoPE.
                mscale=0.0,
                mscale_all_dim=0.0,
            )
        else:
            # A layer with only the window uses the base theta and no YaRN.
            self.rope = YarnRoPE(
                dims=args.qk_rope_head_dim, traditional=True, base=args.rope_theta
            )

    def __call__(
        self,
        x: mx.array,
        *,
        start_pos: int,
        window_idx: mx.array,
        cache: LayerCache,
        shared: dict[str, Optional[mx.array]],
    ) -> mx.array:
        batch, length, _ = x.shape
        q_residual = self.q_a_norm(self.q_a_proj(x))
        q = self.q_b_proj(q_residual).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        q = _apply_rope(q, self.rope, rope_dim=self.rope_dim, offset=start_pos)

        kv = self.kv_norm(self.kv_proj(x))
        kv = _apply_rope(kv, self.rope, rope_dim=self.rope_dim, offset=start_pos)
        kv = _fake_quant_fp8(kv, 32)
        kvs = [cache.update_window(kv)]
        indices = [window_idx]
        if self.ratio:
            if self.is_kv_source:
                self._compress(x, start_pos=start_pos, cache=cache, shared=shared)
            if self.is_index_source:
                topk, candidates = self.indexer(
                    x,
                    q_residual,
                    index_k=shared.get("index_k"),
                    candidates=shared.get("candidates"),
                    rope=self.rope,
                    start_pos=start_pos,
                )
                shared["topk_idx"] = topk
                if self.indexer.is_candidate_source:
                    shared["candidates"] = candidates
            compressed = shared.get("compress_kv")
            topk = shared.get("topk_idx")
            if compressed is not None and topk is not None and topk.shape[-1] > 0:
                kvs.append(compressed)
                indices.append(topk)

        output = _sparse_attention(
            q, kvs, indices, sinks=self.sinks, scale=self.head_dim**-0.5
        )
        output = _apply_rope(
            output, self.rope, rope_dim=self.rope_dim, offset=start_pos, inverse=True
        )
        output = output.reshape(batch * length, self.o_groups, 1, -1)
        output = self.o_a_proj(output, self._group_ids)
        return self.o_b_proj(output.reshape(batch, length, -1))

    def _compress(
        self,
        x: mx.array,
        *,
        start_pos: int,
        cache: LayerCache,
        shared: dict[str, Optional[mx.array]],
    ) -> None:
        """Add the new compressed KV and index keys to the cache, and share them."""
        latent, first = self.compressor(x, start_pos, cache)
        if latent is not None:
            kv = _apply_rope(
                latent,
                self.rope,
                rope_dim=self.rope_dim,
                offset=first,
                scale=self.ratio,
            )
            kv = _fake_quant_fp4(kv, block_size=16, e4m3_scale=True)
            cache.compress_kv = _append(cache.compress_kv, kv)
            if self.is_index_source:
                k = self.indexer.k_norm(self.indexer.k_proj(latent))
                k = _apply_rope(
                    k,
                    self.rope,
                    rope_dim=self.rope_dim,
                    offset=first,
                    scale=self.ratio,
                )
                k = _fake_quant_fp4(k, block_size=32)
                cache.index_k = _append(cache.index_k, k)
        shared["compress_kv"] = cache.compress_kv
        if self.is_index_source:
            shared["index_k"] = cache.index_k


class Router(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.weight = mx.zeros(
            (args.n_routed_experts, args.hidden_size), dtype=mx.float32
        )
        self.e_score_correction_bias = mx.zeros(
            (args.n_routed_experts,), dtype=mx.float32
        )
        self.top_k = args.num_experts_per_tok
        self.score_func = args.scoring_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.routed_scaling_factor

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        logits = logits / self.gate_temp
        if self.score_func == "sigmoid":
            scores = mx.sigmoid(logits)
        elif self.score_func == "softmax":
            scores = mx.softmax(logits, axis=-1)
        else:
            scores = mx.sqrt(nn.softplus(logits))
        biased = scores + self.e_score_correction_bias
        chosen = mx.argpartition(-biased, self.top_k - 1, axis=-1)[..., : self.top_k]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.norm_topk_prob and self.top_k > 1:
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        return chosen, weights * self.route_scale


class Experts(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=DeepseekV41SwiGLU(args.swiglu_limit),
        )

    def __call__(self, x: mx.array, indices: mx.array, weights: mx.array) -> mx.array:
        routed = self.switch_mlp(x, indices)
        return mx.sum(
            routed.astype(mx.float32) * weights[..., None].astype(mx.float32), axis=1
        )


class SharedMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.hidden_size, args.moe_intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            args.hidden_size, args.moe_intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            args.moe_intermediate_size, args.hidden_size, bias=False
        )
        self.activation = DeepseekV41SwiGLU(args.swiglu_limit)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.activation(up, gate).astype(x.dtype))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = Router(args)
        self.experts = Experts(args)
        self.shared_experts = SharedMLP(args)
        self.dim = args.hidden_size
        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        shape = x.shape
        flat = x.reshape(-1, self.dim)
        indices, weights = self.gate(flat)
        routed = self.experts(flat, indices, weights)
        out = routed + self.shared_experts(flat).astype(mx.float32)
        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        return out.reshape(shape).astype(x.dtype)


class HyperConnection(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hc = args.hc_mult
        self.iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.norm_eps = args.rms_norm_eps
        # The leading underscores keep these out of the parameters.
        self._eps = mx.array([args.hc_eps], dtype=mx.float32)
        # The scale of each mix column: pre, post, then comb.
        self._scale_ids = mx.array([0] * self.hc + [1] * self.hc + [2] * self.hc**2)
        mix = (2 + self.hc) * self.hc
        self.fn = mx.zeros((mix, self.hc * args.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def __call__(self, streams: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        # The reference normalizes in float32.
        flat = streams.reshape(*streams.shape[:2], -1).astype(mx.float32)
        flat = mx.fast.rms_norm(flat, weight=None, eps=self.norm_eps)
        mix = (flat @ self.fn.T) * self.scale[self._scale_ids] + self.base
        pre, post, comb = mx.split(mix, [self.hc, 2 * self.hc], axis=-1)
        pre = mx.sigmoid(pre) + self.hc_eps
        post = 2.0 * mx.sigmoid(post)
        comb = comb.reshape(*comb.shape[:-1], self.hc, self.hc)
        comb = mx.softmax(comb, axis=-1) + self.hc_eps
        return pre, post, sinkhorn(comb, iters=self.iters, eps=self._eps)


def _collapse(streams: mx.array, mix: mx.array) -> mx.array:
    out = mx.sum(mix[..., None] * streams.astype(mx.float32), axis=2)
    return out.astype(streams.dtype)


def _expand(
    x: mx.array, residual: mx.array, post: mx.array, comb: mx.array
) -> mx.array:
    mixed = mx.sum(comb[..., None] * residual[..., :, None, :], axis=2)
    out = post[..., None] * x[..., None, :] + mixed
    return out.astype(residual.dtype)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_idx = layer_id
        self.self_attn = Attention(args, layer_id)
        self.mlp = MoE(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.attn_hc = HyperConnection(args)
        self.ffn_hc = HyperConnection(args)

    def __call__(
        self,
        streams: mx.array,
        pre_mix: mx.array,
        *,
        start_pos: int,
        window_idx: mx.array,
        cache: LayerCache,
        shared: dict[str, Optional[mx.array]],
    ) -> tuple[mx.array, mx.array]:
        attn_pre, attn_post, attn_comb = self.attn_hc(streams)
        h = self.input_layernorm(_collapse(streams, pre_mix))
        h = self.self_attn(
            h, start_pos=start_pos, window_idx=window_idx, cache=cache, shared=shared
        )
        streams = _expand(h, streams, attn_post, attn_comb)

        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(streams)
        h = self.mlp(self.post_attention_layernorm(_collapse(streams, attn_pre)))
        streams = _expand(h, streams, ffn_post, ffn_comb)
        return streams, ffn_pre
