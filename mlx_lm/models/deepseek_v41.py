# Copyright © 2026 Apple Inc.

import re
from dataclasses import dataclass
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs
from .rope_utils import YarnRoPE
from .switch_layers import SwitchGLU

NEG_INF = -1e30


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v41"
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1280
    qk_rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    moe_intermediate_size: int = 2304
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-20

    sliding_window: int = 128
    compress_ratios: tuple = ()
    kv_source_layer_ids: tuple = ()
    index_source_layer_ids: tuple = ()
    compress_rope_theta: float = 160000.0
    candidate_source_layer_id: int = -1
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512

    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    max_position_embeddings: int = 1048576
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    engram_layer_ids: tuple = ()
    engram_num_embeddings: tuple = ()
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 99092
    num_nextn_predict_layers: int = 3
    dspark_target_layer_ids: tuple = ()
    vision_n_layers: int = 0


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


def _apply_rope(
    x: mx.array,
    rope: YarnRoPE,
    rope_dim: int,
    offset: Union[int, mx.array] = 0,
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


def _fake_quant_fp8(x: mx.array, block_size: int = 32) -> mx.array:
    """Round-trip FP8 e4m3 with power-of-two block scales."""
    if x.shape[-1] % block_size:
        return x
    dtype = x.dtype
    shape = x.shape
    blocks = x.astype(mx.float32).reshape(*shape[:-1], -1, block_size)
    amax = mx.maximum(mx.max(mx.abs(blocks), axis=-1, keepdims=True), 1e-4)
    # MLX exposes the same e4m3 conversion used by the reference runtime.
    exponent = mx.ceil(mx.log2(amax / 448.0))
    scale = mx.power(2.0, exponent)
    quantized = mx.clip(blocks / scale, -448.0, 448.0)
    return (
        (mx.from_fp8(mx.to_fp8(quantized), mx.float32) * scale)
        .reshape(shape)
        .astype(dtype)
    )


_FP4_LUT = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=mx.float32)


def _round_fp4(x: mx.array) -> mx.array:
    mag = mx.abs(x)
    idx = mx.zeros(mag.shape, dtype=mx.int32)
    for threshold in (0.25, 1.25, 2.5, 5.0):
        idx = idx + (mag > threshold).astype(mx.int32)
    for threshold in (0.75, 1.75, 3.5):
        idx = idx + (mag >= threshold).astype(mx.int32)
    return mx.sign(x) * _FP4_LUT[idx]


def _fake_quant_fp4(x: mx.array, block_size: int, e4m3_scale: bool = False) -> mx.array:
    if x.shape[-1] % block_size:
        return x
    dtype = x.dtype
    shape = x.shape
    blocks = x.astype(mx.float32).reshape(*shape[:-1], -1, block_size)
    floor = 6.0 * 2.0**-9 if e4m3_scale else 6.0 * 2.0**-126
    amax = mx.maximum(mx.max(mx.abs(blocks), axis=-1, keepdims=True), floor)
    if e4m3_scale:
        # e4m3 scale rounding is a small approximation on MLX versions without
        # a public float8 dtype; the quantization grid remains exact.
        scale = mx.power(2.0, mx.round(mx.log2(amax / 6.0)))
    else:
        scale = mx.power(2.0, mx.ceil(mx.log2(amax / 6.0)))
    q = _round_fp4(mx.clip(blocks / scale, -6.0, 6.0)) * scale
    return q.reshape(shape).astype(dtype)


class HyperConnection(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hc = args.hc_mult
        self.iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix = (2 + self.hc) * self.hc
        self.input_norm = UnweightedRMSNorm(args.rms_norm_eps)
        self.fn = mx.zeros((mix, self.hc * args.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def __call__(self, streams: mx.array):
        flat = self.input_norm(streams.reshape(*streams.shape[:2], -1)).astype(
            mx.float32
        )
        weights = flat @ self.fn.astype(mx.float32).T
        pre_w, post_w, comb_w = mx.split(weights, [self.hc, 2 * self.hc], axis=-1)
        pre_b, post_b, comb_b = mx.split(
            self.base.astype(mx.float32), [self.hc, 2 * self.hc], axis=0
        )
        pre_scale, post_scale, comb_scale = self.scale.astype(mx.float32)
        pre = mx.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2.0 * mx.sigmoid(post_w * post_scale + post_b)
        comb = comb_w.reshape(*comb_w.shape[:-1], self.hc, self.hc)
        comb = comb * comb_scale + comb_b.reshape(self.hc, self.hc)
        comb = mx.softmax(comb, axis=-1) + self.hc_eps
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + self.hc_eps)
        for _ in range(max(self.iters - 1, 0)):
            comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + self.hc_eps)
            comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + self.hc_eps)
        return pre, post, comb


def _collapse(streams: mx.array, mix: mx.array) -> mx.array:
    out = mx.sum(mix[..., None].astype(mx.float32) * streams.astype(mx.float32), axis=2)
    return out.astype(streams.dtype)


def _expand(
    x: mx.array, residual: mx.array, post: mx.array, comb: mx.array
) -> mx.array:
    mixed = mx.sum(comb[..., None] * residual[..., :, None, :], axis=2)
    out = post[..., None] * x[..., None, :] + mixed
    return out.astype(residual.dtype)


class LayerCache:
    """Dynamic cache for one decoder layer.

    The cache stores only the sliding window and completed compressed entries.
    It is deliberately separate from model parameters so ``make_prompt_cache``
    can construct it before the input batch size is known.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        self.args = args
        self.layer_id = layer_id
        self.offset = 0
        self.batch_size = None
        self.window = None
        self.comp_kv = None
        self.index_k = None
        self.pending_kv = None
        self.pending_gate = None
        self.pending_len = 0

    def ensure_batch(self, batch_size: int, dtype=mx.float32):
        if self.batch_size == batch_size:
            return
        self.batch_size = batch_size
        self.window = mx.zeros((batch_size, 0, self.args.head_dim), dtype=dtype)
        self.comp_kv = mx.zeros((batch_size, 0, self.args.head_dim), dtype=dtype)
        self.index_k = mx.zeros((batch_size, 0, self.args.index_head_dim), dtype=dtype)
        self.pending_kv = mx.zeros(
            (batch_size, 0, self.args.head_dim), dtype=mx.float32
        )
        self.pending_gate = mx.zeros(
            (batch_size, 0, self.args.head_dim), dtype=mx.float32
        )
        self.pending_len = 0

    def write_window(self, kv: mx.array):
        self.window = mx.concatenate([self.window, kv], axis=1)
        self.window = self.window[:, -self.args.sliding_window :]
        self.offset += kv.shape[1]

    @property
    def state(self):
        return (
            self.window,
            self.comp_kv,
            self.index_k,
            self.pending_kv,
            self.pending_gate,
            mx.array(self.offset),
        )

    def is_trimmable(self):
        return True

    def empty(self):
        return self.offset == 0

    @property
    def nbytes(self):
        arrays = (
            self.window,
            self.comp_kv,
            self.index_k,
            self.pending_kv,
            self.pending_gate,
        )
        return sum(a.nbytes for a in arrays if a is not None)

    def trim(self, n: int):
        n = min(n, self.offset)
        self.offset -= n
        if self.window is not None:
            self.window = self.window[:, : max(self.window.shape[1] - n, 0)]
        ratio = (
            self.args.compress_ratios[self.layer_id]
            if self.layer_id < len(self.args.compress_ratios)
            else 0
        )
        if ratio:
            keep = max((self.offset // ratio), 0)
            self.comp_kv = self.comp_kv[:, :keep]
            self.index_k = self.index_k[:, :keep]
        self.pending_kv = self.pending_kv[:, :0]
        self.pending_gate = self.pending_gate[:, :0]
        self.pending_len = 0
        return n

    def extract(self, idx: int):
        other = LayerCache(self.args, self.layer_id)
        other.batch_size = 1
        other.offset = self.offset
        for name in ("window", "comp_kv", "index_k", "pending_kv", "pending_gate"):
            value = getattr(self, name)
            setattr(other, name, value[idx : idx + 1] if value is not None else None)
        other.pending_len = self.pending_len
        return other

    def filter(self, keep):
        for name in ("window", "comp_kv", "index_k", "pending_kv", "pending_gate"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value[keep])
        self.batch_size = len(keep)


class Compressor(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.ratio = int(args.compress_ratios[layer_id])
        self.head_dim = args.head_dim
        self.kv_proj = nn.Linear(args.hidden_size, args.head_dim, bias=False)
        self.gate_proj = (
            nn.Linear(args.hidden_size, args.head_dim, bias=False)
            if self.ratio > 1
            else None
        )
        self.kv_norm = nn.RMSNorm(args.head_dim, args.rms_norm_eps)

    def __call__(self, x: mx.array, start_pos: int, cache: LayerCache):
        if self.ratio == 1:
            return self.kv_norm(self.kv_proj(x)), start_pos

        kv = x.astype(mx.float32) @ self.kv_proj.weight.astype(mx.float32).T
        gate = x.astype(mx.float32) @ self.gate_proj.weight.astype(mx.float32).T
        if cache.pending_len:
            kv = mx.concatenate([cache.pending_kv, kv], axis=1)
            gate = mx.concatenate([cache.pending_gate, gate], axis=1)
        total = kv.shape[1]
        groups = total // self.ratio
        rem = total % self.ratio
        if rem:
            cache.pending_kv = kv[:, -rem:]
            cache.pending_gate = gate[:, -rem:]
            cache.pending_len = rem
        else:
            cache.pending_kv = cache.pending_kv[:, :0]
            cache.pending_gate = cache.pending_gate[:, :0]
            cache.pending_len = 0
        if groups == 0:
            return None, mx.zeros((0,), dtype=mx.int32)
        kv = kv[:, : groups * self.ratio].reshape(kv.shape[0], groups, self.ratio, -1)
        gate = gate[:, : groups * self.ratio].reshape(
            gate.shape[0], groups, self.ratio, -1
        )
        latent = mx.sum(kv * mx.softmax(gate, axis=2), axis=2)
        latent = self.kv_norm(latent.astype(x.dtype))
        # pending_len has already been updated, so derive the first group from
        # the current absolute position and the number of newly emitted groups.
        group_start = start_pos - ((total - x.shape[1]) % self.ratio)
        return latent, group_start


def _window_indices(previous: int, length: int, window: int) -> mx.array:
    width = min(window, previous + length)
    end = previous + mx.arange(length)
    start = mx.maximum(end - window + 1, 0)
    indices = start[:, None] + mx.arange(width)[None, :]
    return mx.where(
        indices > end[:, None], mx.array(-1, mx.int32), indices.astype(mx.int32)
    )


def _gather_rows(values: mx.array, indices: mx.array) -> mx.array:
    batch, length, dim = values.shape
    safe = mx.maximum(indices, 0).astype(mx.int32)
    base = (mx.arange(batch, dtype=mx.int32) * length).reshape(batch, 1, 1)
    flat = values.reshape(batch * length, dim)
    return flat[(safe + base).reshape(-1)].reshape(*indices.shape, dim)


def _sparse_attention(
    q: mx.array, kv: mx.array, sinks: mx.array, indices: mx.array, scale: float
) -> mx.array:
    # q [B,S,H,D], kv [B,K,D], indices [B,S,Ksel].
    selected = _gather_rows(kv, indices).astype(mx.float32)
    qf = q.astype(mx.float32)
    logits = mx.matmul(qf, selected.swapaxes(-1, -2)) * scale
    valid = indices[:, :, None, :] >= 0
    logits = mx.where(valid, logits, NEG_INF)
    max_logit = mx.maximum(
        mx.max(logits, axis=-1, keepdims=True), sinks.reshape(1, 1, -1, 1)
    )
    weights = mx.exp(logits - max_logit)
    weights = mx.where(valid, weights, 0.0)
    denom = mx.sum(weights, axis=-1, keepdims=True) + mx.exp(
        sinks.reshape(1, 1, -1, 1) - max_logit
    )
    out = mx.matmul(weights, selected) / denom
    return out.astype(q.dtype)


class Indexer(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.ratio = int(args.compress_ratios[layer_id])
        self.owns_k = layer_id in args.kv_source_layer_ids
        self.is_candidate_source = layer_id == args.candidate_source_layer_id
        self.uses_candidates = 0 <= args.candidate_source_layer_id < layer_id
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
        if self.owns_k:
            self.k_proj = nn.Linear(args.head_dim, args.index_head_dim, bias=False)
            self.k_norm = nn.RMSNorm(args.index_head_dim, args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        q_residual: mx.array,
        positions: mx.array,
        index_k: mx.array,
        rope: YarnRoPE,
        start_pos: int,
        shared: dict,
    ):
        batch, length, _ = x.shape
        if index_k.shape[1] == 0:
            shared["topk_idx"] = mx.zeros((batch, length, 0), dtype=mx.int32)
            if self.is_candidate_source:
                shared["candidates"] = None
            return

        q = self.q_b_proj(q_residual).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        q = _apply_rope(q, rope, self.rope_dim, offset=start_pos)
        q = _fake_quant_fp4(q, 32)
        keys = index_k.astype(mx.float32)
        weights = self.weights_proj(x).astype(mx.float32) * (self.num_heads**-0.5)
        scores = mx.matmul(q.astype(mx.float32), keys[:, None, :, :].swapaxes(-1, -2))
        scores = mx.maximum(scores, 0.0) * (self.head_dim**-0.5)
        scores = mx.sum(scores * weights[..., None], axis=2)
        lens = ((positions + 1) // self.ratio).astype(mx.int32)[:, None]
        visible = mx.arange(index_k.shape[1])[None, :] < lens
        scores = mx.where(visible[None], scores, NEG_INF)

        if self.is_candidate_source and self.candidate_topk_blocks > 0:
            block = self.candidate_block_size
            pad = (-scores.shape[-1]) % block
            padded = mx.pad(scores, [(0, 0), (0, 0), (0, pad)], constant_values=NEG_INF)
            block_scores = padded.reshape(batch, length, -1, block).max(axis=-1)
            k_blocks = min(self.candidate_topk_blocks, block_scores.shape[-1])
            chosen = mx.argpartition(-block_scores, k_blocks - 1, axis=-1)[
                ..., :k_blocks
            ]
            candidate = mx.zeros(block_scores.shape, dtype=mx.bool_)
            candidate = mx.put_along_axis(candidate, chosen, True, axis=-1)
            shared["candidates"] = mx.repeat(candidate, block, axis=-1)[
                ..., : scores.shape[-1]
            ]
        elif self.uses_candidates and shared.get("candidates") is not None:
            scores = mx.where(shared["candidates"], scores, NEG_INF)

        k = min(self.index_topk, index_k.shape[1])
        chosen = mx.argpartition(-scores, k - 1, axis=-1)[..., :k].astype(mx.int32)
        valid = chosen < lens[None]
        shared["topk_idx"] = mx.where(valid, chosen, mx.array(-1, mx.int32))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ratio = (
            int(args.compress_ratios[layer_id])
            if layer_id < len(args.compress_ratios)
            else 0
        )
        self.is_kv_source = layer_id in args.kv_source_layer_ids
        self.is_index_source = layer_id in args.index_source_layer_ids
        self.q_a_proj = nn.Linear(args.hidden_size, args.q_lora_rank, bias=False)
        self.q_a_norm = nn.RMSNorm(args.q_lora_rank, args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, args.num_attention_heads * args.head_dim, bias=False
        )
        self.kv_proj = nn.Linear(args.hidden_size, args.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(args.head_dim, args.rms_norm_eps)
        self.o_a_proj = nn.Linear(
            args.num_attention_heads * args.head_dim // args.o_groups,
            args.o_groups * args.o_lora_rank,
            bias=False,
        )
        self.o_b_proj = nn.Linear(
            args.o_groups * args.o_lora_rank, args.hidden_size, bias=False
        )
        self.sinks = mx.zeros((args.num_attention_heads,), dtype=mx.float32)
        if self.is_kv_source:
            self.compressor = Compressor(args, layer_id)
        if self.is_index_source:
            self.indexer = Indexer(args, layer_id)
        self.rope = YarnRoPE(
            dims=args.qk_rope_head_dim,
            traditional=True,
            base=args.rope_theta,
            scaling_factor=1.0,
            mscale=0.0,
            mscale_all_dim=0.0,
        )
        rope = args.rope_scaling or {}
        self.compress_rope = YarnRoPE(
            dims=args.qk_rope_head_dim,
            traditional=True,
            base=args.compress_rope_theta,
            max_position_embeddings=args.max_position_embeddings,
            scaling_factor=float(rope.get("factor", 1.0)),
            original_max_position_embeddings=int(
                rope.get(
                    "original_max_position_embeddings", args.max_position_embeddings
                )
            ),
            beta_fast=int(rope.get("beta_fast", 32)),
            beta_slow=int(rope.get("beta_slow", 1)),
            # V4.1 uses attention_factor=1 for its YaRN RoPE.
            mscale=0.0,
            mscale_all_dim=0.0,
        )

    def __call__(
        self, x: mx.array, start_pos: int, cache: Optional[LayerCache], shared: dict
    ):
        batch, length, _ = x.shape
        positions = start_pos + mx.arange(length)
        compressed = bool(self.ratio)
        rope = self.compress_rope if compressed else self.rope

        q_residual = self.q_a_norm(self.q_a_proj(x))
        q = self.q_b_proj(q_residual).reshape(
            batch, length, self.args.num_attention_heads, self.args.head_dim
        )
        q = _apply_rope(q, rope, self.args.qk_rope_head_dim, offset=start_pos)

        kv = self.kv_norm(self.kv_proj(x))
        kv = _apply_rope(kv, rope, self.args.qk_rope_head_dim, offset=start_pos)
        kv = _fake_quant_fp8(kv, 32)
        previous = cache.window.shape[1] if cache is not None else 0
        window = (
            mx.concatenate([cache.window, kv], axis=1)
            if cache is not None and previous
            else kv
        )
        window_indices = _window_indices(previous, length, self.args.sliding_window)
        window_indices = mx.broadcast_to(
            window_indices[None], (batch, length, window_indices.shape[-1])
        )
        if cache is not None:
            cache.write_window(kv)

        selected_indices = window_indices
        if compressed:
            latent = None
            if self.is_kv_source:
                if cache is None:
                    # A temporary cache gives the compressor the same state API.
                    temp = LayerCache(self.args, self.layer_id)
                    temp.ensure_batch(batch, kv.dtype)
                    latent, group_positions = self.compressor(x, start_pos, temp)
                else:
                    latent, group_positions = self.compressor(x, start_pos, cache)
                shared["compress_kv"] = cache.comp_kv if cache is not None else None

            if self.is_kv_source and latent is not None:
                rotated = _apply_rope(
                    latent,
                    self.compress_rope,
                    self.args.qk_rope_head_dim,
                    offset=group_positions,
                    scale=self.ratio,
                )
                rotated = _fake_quant_fp4(rotated, 16, e4m3_scale=True)
                if cache is not None:
                    cache.comp_kv = mx.concatenate([cache.comp_kv, rotated], axis=1)
                    shared["compress_kv"] = cache.comp_kv
                else:
                    shared["compress_kv"] = rotated

            if self.is_index_source:
                index_k = None
                if self.indexer.owns_k and latent is not None:
                    index_latent = self.indexer.k_norm(self.indexer.k_proj(latent))
                    index_k = _apply_rope(
                        index_latent,
                        self.compress_rope,
                        self.args.qk_rope_head_dim,
                        offset=group_positions,
                        scale=self.ratio,
                    )
                    index_k = _fake_quant_fp4(index_k, 32)
                    if cache is not None:
                        cache.index_k = mx.concatenate([cache.index_k, index_k], axis=1)
                    shared["index_k"] = cache.index_k if cache is not None else index_k
                elif self.indexer.owns_k and cache is not None:
                    shared["index_k"] = cache.index_k

                index_k = shared.get("index_k")
                if index_k is None:
                    index_k = mx.zeros(
                        (batch, 0, self.args.index_head_dim), dtype=x.dtype
                    )
                self.indexer(
                    x,
                    q_residual,
                    positions,
                    index_k,
                    self.compress_rope,
                    start_pos,
                    shared,
                )
            elif "topk_idx" not in shared:
                shared["topk_idx"] = mx.zeros((batch, length, 0), dtype=mx.int32)

            compressed_kv = shared.get("compress_kv")
            topk = shared.get("topk_idx")
            if compressed_kv is not None and topk is not None and topk.shape[-1] > 0:
                # Compressed entries follow the window entries in the index space.
                topk = mx.where(
                    topk >= 0, topk + window.shape[1], mx.array(-1, mx.int32)
                )
                selected_indices = mx.concatenate([window_indices, topk], axis=-1)
                window = mx.concatenate([window, compressed_kv], axis=1)

        output = _sparse_attention(
            q, window, self.sinks, selected_indices, self.args.head_dim**-0.5
        )
        output = _apply_rope(
            output,
            rope,
            self.args.qk_rope_head_dim,
            offset=start_pos,
            inverse=True,
        )
        output = output.reshape(batch, length, self.args.o_groups, -1)
        grouped = self.o_a_proj.weight.reshape(
            self.args.o_groups, self.args.o_lora_rank, -1
        )
        output = mx.einsum(
            "bsgd,grd->bsgr", output.astype(mx.float32), grouped.astype(mx.float32)
        )
        return self.o_b_proj(output.reshape(batch, length, -1).astype(x.dtype))


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

    def __call__(self, x: mx.array):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.activation(up, gate).astype(x.dtype))


class Router(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.weight = mx.zeros(
            (args.n_routed_experts, args.hidden_size), dtype=mx.float32
        )
        self.e_score_correction_bias = mx.zeros(
            (args.n_routed_experts,), dtype=mx.float32
        )
        self.e_score_correction_bias_vl = mx.zeros(
            (args.n_routed_experts,), dtype=mx.float32
        )
        self.top_k = args.num_experts_per_tok
        self.score_func = args.scoring_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.routed_scaling_factor

    def __call__(self, x: mx.array):
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        logits = logits / self.gate_temp
        if self.score_func == "sigmoid":
            scores = mx.sigmoid(logits)
        elif self.score_func == "softmax":
            scores = mx.softmax(logits, axis=-1)
        else:
            scores = mx.sqrt(nn.softplus(logits))
        chosen = mx.argpartition(
            -(scores + self.e_score_correction_bias), self.top_k - 1, axis=-1
        )[..., : self.top_k].astype(mx.int32)
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.norm_topk_prob and self.top_k > 1:
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        return chosen, weights * self.route_scale


class Experts(nn.Module):
    """Sparse V4.1 experts backed by the shared SwitchGLU implementation."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=DeepseekV41SwiGLU(args.swiglu_limit),
        )

    def __call__(self, x: mx.array, indices: mx.array, weights: mx.array):
        routed = self.switch_mlp(x, indices)
        return mx.sum(
            routed.astype(mx.float32) * weights[..., None].astype(mx.float32), axis=1
        )


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = Router(args)
        self.experts = Experts(args)
        self.shared_experts = SharedMLP(args)
        self.dim = args.hidden_size

    def __call__(self, x: mx.array):
        shape = x.shape
        flat = x.reshape(-1, self.dim)
        indices, weights = self.gate(flat)
        routed = self.experts(flat, indices, weights)
        shared = self.shared_experts(flat)
        return (routed + shared.astype(mx.float32)).reshape(shape).astype(x.dtype)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_idx = layer_id
        self.self_attn = Attention(args, layer_id)
        self.mlp = MoE(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.attn_hc = HyperConnection(args)
        self.ffn_hc = HyperConnection(args)

    def __call__(
        self,
        streams: mx.array,
        pre_mix: mx.array,
        start_pos: int,
        cache: Optional[LayerCache],
        shared: dict,
    ):
        residual = streams
        attn_pre, attn_post, attn_comb = self.attn_hc(streams)
        collapsed = _collapse(streams, pre_mix)
        attn_output = self.self_attn(
            self.input_layernorm(collapsed), start_pos, cache, shared
        )
        streams = _expand(attn_output, residual, attn_post, attn_comb)

        residual = streams
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(streams)
        collapsed = _collapse(streams, attn_pre)
        ffn_output = self.mlp(self.post_attention_layernorm(collapsed))
        streams = _expand(ffn_output, residual, ffn_post, ffn_comb)
        return streams, ffn_pre


class TextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.args = args

    def make_cache(self):
        return [LayerCache(self.args, i) for i in range(self.args.num_hidden_layers)]

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)
        batch, length, _ = h.shape
        start_pos = cache[0].offset if cache is not None else 0
        if cache is not None:
            for c in cache:
                c.ensure_batch(batch, h.dtype)

        streams = mx.broadcast_to(
            h[:, :, None, :], (batch, length, self.args.hc_mult, self.args.hidden_size)
        )
        pre_mix = mx.zeros((batch, length, self.args.hc_mult), dtype=mx.float32)
        pre_mix[..., 0] = 1.0
        shared = {}
        for idx, layer in enumerate(self.layers):
            layer_cache = cache[idx] if cache is not None else None
            streams, pre_mix = layer(streams, pre_mix, start_pos, layer_cache, shared)
        return self.norm(_collapse(streams, pre_mix))


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return self.model.make_cache()

    def __call__(self, inputs: mx.array, cache=None):
        return self.lm_head(self.model(inputs, cache))

    def sanitize(self, weights: dict[str, mx.array]):
        """Drop non-text tensors and normalize Transformer checkpoint names."""
        clean = {}
        experts = {}
        for key, value in weights.items():
            if (
                key.startswith("vision.")
                or key.startswith("aligner.")
                or key.startswith("image_")
                or key.startswith("mtp.")
                or ".engram." in key
                or key.startswith("model.engram_tables.")
                or key.startswith("model.engram_hash_state.")
            ):
                continue

            expert = re.fullmatch(
                r"(model\.layers\.\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight", key
            )
            if expert is not None:
                experts.setdefault(expert.group(1), {}).setdefault(
                    expert.group(3), []
                ).append((int(expert.group(2)), value))
                continue

            if key == "head.weight":
                key = "lm_head.weight"
            elif key == "embed.weight":
                key = "model.embed_tokens.weight"
            key = key.replace(".attn.", ".self_attn.")
            key = key.replace(".ffn.", ".mlp.")
            key = key.replace(".attn_norm.", ".input_layernorm.")
            key = key.replace(".ffn_norm.", ".post_attention_layernorm.")
            key = key.replace(".hc_attn_fn", ".attn_hc.fn")
            key = key.replace(".hc_attn_base", ".attn_hc.base")
            key = key.replace(".hc_attn_scale", ".attn_hc.scale")
            key = key.replace(".hc_ffn_fn", ".ffn_hc.fn")
            key = key.replace(".hc_ffn_base", ".ffn_hc.base")
            key = key.replace(".hc_ffn_scale", ".ffn_hc.scale")
            key = key.replace(".attn_sink", ".sinks")
            key = key.replace(".wq_a.", ".q_a_proj.")
            key = key.replace(".wq_b.", ".q_b_proj.")
            key = key.replace(".wkv.", ".kv_proj.")
            key = key.replace(".wgate.", ".gate_proj.")
            key = key.replace(".wo_a.", ".o_a_proj.")
            key = key.replace(".wo_b.", ".o_b_proj.")
            key = key.replace(".q_norm.", ".q_a_norm.")
            key = key.replace(".compressor.norm.", ".compressor.kv_norm.")
            key = key.replace(".indexer.wk.", ".indexer.k_proj.")
            key = key.replace(".gate.bias_vl", ".gate.e_score_correction_bias_vl")
            key = key.replace(".gate.bias", ".gate.e_score_correction_bias")
            key = key.replace(".shared_experts.w1.", ".shared_experts.gate_proj.")
            key = key.replace(".shared_experts.w2.", ".shared_experts.down_proj.")
            key = key.replace(".shared_experts.w3.", ".shared_experts.up_proj.")
            clean[key] = value

        for prefix, parts in experts.items():
            for _name, rows in parts.items():
                rows.sort(key=lambda item: item[0])
            if all(
                name in parts and len(parts[name]) == self.args.n_routed_experts
                for name in ("w1", "w2", "w3")
            ):
                gate = mx.stack([value for _, value in parts["w1"]])
                up = mx.stack([value for _, value in parts["w3"]])
                down = mx.stack([value for _, value in parts["w2"]])
                clean[f"{prefix}.mlp.experts.switch_mlp.gate_proj.weight"] = gate
                clean[f"{prefix}.mlp.experts.switch_mlp.up_proj.weight"] = up
                clean[f"{prefix}.mlp.experts.switch_mlp.down_proj.weight"] = down

        for layer_id in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_id}.mlp.experts"
            gate_up_key = f"{prefix}.gate_up_proj"
            if gate_up_key not in clean:
                continue
            gate_up = clean.pop(gate_up_key)
            mid = gate_up.shape[-2] // 2
            clean[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
            clean[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
            clean[f"{prefix}.switch_mlp.down_proj.weight"] = clean.pop(
                f"{prefix}.down_proj"
            )
        return clean
