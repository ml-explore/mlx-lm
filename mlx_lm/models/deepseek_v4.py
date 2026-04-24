# Copyright © 2024 Apple Inc.

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoConfig, PreTrainedConfig

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchLinear


class DeepseekV4HFConfig(PreTrainedConfig):
    """Minimal HF config stub so transformers recognizes deepseek_v4 and
    accepts rope_scaling without warnings."""

    model_type = "deepseek_v4"

    def __init__(self, rope_scaling=None, **kwargs):
        self.rope_scaling = rope_scaling
        super().__init__(**kwargs)


AutoConfig.register("deepseek_v4", DeepseekV4HFConfig, exist_ok=True)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_hash_layers: int = 3
    num_nextn_predict_layers: int = 1
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    o_groups: int = 8
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2048
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    norm_topk_prob: bool = True
    sliding_window: int = 128
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    max_position_embeddings: int = 1048576
    attention_bias: bool = False


def _apply_inverse_rope(x: mx.array, rope_fn, offset: int) -> mx.array:
    """Conjugate RoPE: rotate by -theta instead of +theta.

    For traditional=True RoPE, pairs are consecutive (x[2i], x[2i+1]).
    inverse_rope(x) = conj(forward_rope(conj(x)))
    where conj negates the imaginary (odd-index) element of each pair.
    """
    sh = x.shape
    rd = sh[-1]
    pairs = x.reshape(*sh[:-1], rd // 2, 2)
    pairs_conj = pairs * mx.array([1.0, -1.0])
    y_conj = rope_fn(pairs_conj.reshape(sh), offset)
    y_pairs = y_conj.reshape(*sh[:-1], rd // 2, 2) * mx.array([1.0, -1.0])
    return y_pairs.reshape(sh)


def _hc_split_sinkhorn(
    mixes: mx.array,
    hc_fn_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int,
    n_iters: int,
    eps: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Split mixes into (pre, post, comb) matching the reference kernel exactly.

    mixes: [B, L, mix_hc] where mix_hc = (2 + hc_mult) * hc_mult
    pre:   sigmoid(mixes[:hc] * scale[0] + base[:hc]) + eps
    post:  2 * sigmoid(mixes[hc:2hc] * scale[1] + base[hc:2hc])
    comb:  row-softmax + eps → col-norm → (n_iters-1) × (row-norm + col-norm)
    """
    hc = hc_mult
    pre = mx.sigmoid(mixes[..., :hc] * hc_fn_scale[0] + hc_base[:hc]) + eps
    post = 2 * mx.sigmoid(mixes[..., hc : 2 * hc] * hc_fn_scale[1] + hc_base[hc : 2 * hc])
    comb_logits = (
        mixes[..., 2 * hc :].reshape(*mixes.shape[:-1], hc, hc) * hc_fn_scale[2]
        + hc_base[2 * hc :].reshape(hc, hc)
    )
    # Initialize comb: row-softmax + eps, then one column normalization
    comb = mx.softmax(comb_logits, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    # Remaining Sinkhorn iterations
    for _ in range(n_iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _hc_pre(
    x: mx.array,
    hc_fn: mx.array,
    hc_fn_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int,
    n_iters: int,
    eps: float,
    norm_eps: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Combine hc_mult hidden copies into a single state.

    x: [B, L, hc_mult, hidden_size]
    Returns: (combined [B, L, hidden_size], post, comb)
    """
    B, L, H, D = x.shape
    x_flat = x.reshape(B, L, H * D).astype(mx.float32)
    rsqrt = mx.rsqrt(mx.mean(x_flat * x_flat, axis=-1, keepdims=True) + norm_eps)
    mixes = (x_flat @ hc_fn.T) * rsqrt  # [B, L, mix_hc]
    pre, post, comb = _hc_split_sinkhorn(mixes, hc_fn_scale, hc_base, hc_mult, n_iters, eps)
    # Weighted sum over hc copies
    combined = mx.sum(pre[..., None] * x, axis=2)  # [B, L, hidden_size]
    return combined, post, comb


def _hc_post(
    x: mx.array,
    residual: mx.array,
    post: mx.array,
    comb: mx.array,
) -> mx.array:
    """Distribute output back into hc_mult copies.

    x:        [B, L, hidden_size]
    residual: [B, L, hc_mult, hidden_size]
    post:     [B, L, hc_mult]
    comb:     [B, L, hc_mult, hc_mult]
    Returns:  [B, L, hc_mult, hidden_size]
    """
    # post[k] * x + sum_j comb[k,j] * residual[j]
    out = post[..., None] * x[:, :, None, :] + mx.sum(
        comb[..., None] * residual[:, :, None, :, :], axis=3
    )
    return out.astype(x.dtype)


class Compressor(nn.Module):
    """Learned gated pooling that compresses groups of tokens into summary KV vectors."""

    def __init__(self, args: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        coff = 1 + (compress_ratio == 4)  # overlap factor for ratio==4
        self.ape = mx.zeros((compress_ratio, coff * head_dim))
        self.wkv = nn.Linear(args.hidden_size, coff * head_dim, bias=False)
        self.wgate = nn.Linear(args.hidden_size, coff * head_dim, bias=False)
        self.norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        """Compress x: [B, L, D] → [B, L//ratio, head_dim]."""
        B, L, _ = x.shape
        ratio = self.compress_ratio
        cutoff = (L // ratio) * ratio
        if cutoff == 0:
            return None
        x_cut = x[:, :cutoff]
        kv = self.wkv(x_cut.astype(mx.float32))    # [B, cutoff, coff*head_dim]
        score = self.wgate(x_cut.astype(mx.float32))
        # Reshape to groups and apply softmax gating
        kv = kv.reshape(B, cutoff // ratio, ratio, -1)
        score = (score.reshape(B, cutoff // ratio, ratio, -1) + self.ape).softmax(axis=2)
        compressed = (kv * score).sum(axis=2)  # [B, cutoff//ratio, coff*head_dim]
        # Take only first head_dim dims if overlap
        compressed = compressed[..., : self.head_dim]
        return self.norm(compressed.astype(x.dtype))


class Indexer(nn.Module):
    """Lightweight index module for sparse attention (simplified: no Hadamard rotation)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.wq_b = nn.Linear(args.q_lora_rank, args.index_n_heads * args.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(args.hidden_size, args.index_n_heads, bias=False)
        self.compressor = Compressor(args, compress_ratio=4, head_dim=args.index_head_dim)
        self.softmax_scale = args.index_head_dim ** -0.5


class Attention(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rd = args.qk_rope_head_dim  # rope head dim
        self.nope_dim = args.head_dim - args.qk_rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.scale = args.head_dim ** -0.5
        self.window = args.sliding_window
        self.eps = args.rms_norm_eps

        compress_ratio = args.compress_ratios[layer_id] if args.compress_ratios else 0
        self.compress_ratio = compress_ratio

        self.wq_a = nn.Linear(args.hidden_size, args.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.wq_b = nn.Linear(args.q_lora_rank, args.num_attention_heads * args.head_dim, bias=False)
        self.wkv = nn.Linear(args.hidden_size, args.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(args.head_dim, eps=args.rms_norm_eps)
        D_group = args.num_attention_heads * args.head_dim // args.o_groups
        self.wo_a = [
            nn.Linear(D_group, args.o_lora_rank, bias=False)
            for _ in range(args.o_groups)
        ]
        self.wo_b = nn.Linear(args.o_groups * args.o_lora_rank, args.hidden_size, bias=False)
        # Attention sink: per-head additive logit for position 0
        self.attn_sink = mx.zeros((args.num_attention_heads,))

        if compress_ratio:
            self.compressor = Compressor(args, compress_ratio, args.head_dim)
            if compress_ratio == 4:
                self.indexer = Indexer(args)

        # RoPE: layers with compression use YaRN + compress_rope_theta
        if compress_ratio and args.rope_scaling is not None:
            yarn_cfg = dict(args.rope_scaling)
            rope_base = args.compress_rope_theta
        else:
            yarn_cfg = None
            rope_base = args.rope_theta

        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=rope_base,
            traditional=True,
            scaling_config=yarn_cfg,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        rd = self.rd

        # Q: low-rank → q_norm → full heads → [B, n_heads, L, head_dim]
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        # Manual RMS norm on each Q head (reference model.py line 498)
        q = q * mx.rsqrt(mx.mean(q * q, axis=-1, keepdims=True) + self.eps)

        # KV: single projection (MQA: K == V), stays [B, L, head_dim]
        kv = self.kv_norm(self.wkv(x))

        offset = cache.offset if cache is not None else 0

        # RoPE: apply to rope-dim slices; q stays [B, n_heads, L, *]
        q_rope = self.rope(q[..., -rd:], offset)  # [B, n_heads, L, rd]
        q = mx.concatenate([q[..., :-rd], q_rope], axis=-1)

        # kv is [B, L, head_dim]; rope treats L as sequence dim (shape [B,1,L,rd])
        kv_rope = self.rope(kv[..., -rd:].reshape(B, 1, L, rd), offset)  # [B, 1, L, rd]
        kv = mx.concatenate([kv[..., :-rd], kv_rope.squeeze(1)], axis=-1)  # [B, L, head_dim]

        if cache is not None:
            kv, _ = cache.update_and_fetch(kv[:, None, :, :], kv[:, None, :, :])
            kv = kv.squeeze(1)

        # Broadcast single KV head across all query heads: [B, 1, L_kv, head_dim]
        k = kv[:, None, :, :]
        v = kv[:, None, :, :]

        # SDPA: q [B, n_heads, L, head_dim], k/v [B, 1, L_kv, head_dim]
        o = scaled_dot_product_attention(
            q, k, v, cache, scale=self.scale, mask=mask, sinks=self.attn_sink
        )
        # o is [B, n_heads, L, head_dim]; K==V so V carries RoPE — undo it on rope dims
        o_rope_inv = _apply_inverse_rope(o[..., -rd:], self.rope, offset)
        o = mx.concatenate([o[..., :-rd], o_rope_inv], axis=-1)
        o = o.transpose(0, 2, 1, 3)  # [B, L, n_heads, head_dim]

        # Grouped output projection: each group of heads uses a separate linear
        o = o.reshape(B, L, self.n_groups, -1)  # [B, L, n_groups, D_group]
        o = mx.concatenate(
            [self.wo_a[g](o[:, :, g, :]) for g in range(self.n_groups)], axis=-1
        )  # [B, L, n_groups * o_lora_rank]
        return self.wo_b(o)


class Gate(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.topk = args.num_experts_per_tok
        self.scoring_func = args.scoring_func
        self.route_scale = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.is_hash = layer_id < args.num_hash_layers
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        if self.is_hash:
            self.tid2eid = mx.zeros((args.vocab_size, args.num_experts_per_tok), dtype=mx.int32)
        else:
            self.bias = mx.zeros((args.n_routed_experts,))

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        scores = (x.astype(mx.float32) @ self.weight.T)
        if self.scoring_func == "softmax":
            scores = mx.softmax(scores, axis=-1)
        elif self.scoring_func == "sigmoid":
            scores = mx.sigmoid(scores)
        else:  # sqrtsoftplus
            scores = mx.sqrt(nn.softplus(scores))

        original_scores = scores
        if not self.is_hash and hasattr(self, "bias"):
            scores = scores + self.bias

        if self.is_hash and input_ids is not None:
            indices = self.tid2eid[input_ids.reshape(-1)]  # [T, topk]
        else:
            indices = mx.stop_gradient(mx.argpartition(-scores, kth=self.topk, axis=-1)[..., :self.topk])

        weights = mx.take_along_axis(original_scores, indices, axis=-1)
        if self.scoring_func != "softmax" and self.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-9)
        weights = (weights * self.route_scale).astype(x.dtype)
        return weights, indices


class Expert(nn.Module):
    """Single SwiGLU FFN expert."""

    def __init__(self, d_in: int, d_out: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_out, bias=False)
        self.w2 = nn.Linear(d_out, d_in, bias=False)
        self.w3 = nn.Linear(d_in, d_out, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.w1(x)
        up = self.w3(x)
        if self.swiglu_limit > 0:
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
            gate = mx.minimum(gate, self.swiglu_limit)
        return self.w2(nn.silu(gate) * up)


class Experts(nn.Module):
    """Stacked routed experts using SwitchLinear for efficient gather_mm + quantization support."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        n = args.n_routed_experts
        d, e = args.hidden_size, args.moe_intermediate_size
        self.w1 = SwitchLinear(d, e, n, bias=False)
        self.w2 = SwitchLinear(e, d, n, bias=False)
        self.w3 = SwitchLinear(d, e, n, bias=False)
        self.swiglu_limit = args.swiglu_limit

    def __call__(self, x: mx.array, indices: mx.array, weights: mx.array) -> mx.array:
        """x: [T, D], indices: [T, k], weights: [T, k] → [T, D]."""
        x = mx.expand_dims(x, (-2, -3))  # [T, 1, 1, D]
        gate = self.w1(x, indices)  # [T, k, 1, E]
        up = self.w3(x, indices)
        if self.swiglu_limit > 0:
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
            gate = mx.minimum(gate, self.swiglu_limit)
        hidden = nn.silu(gate) * up  # [T, k, 1, E]
        out = self.w2(hidden, indices)  # [T, k, 1, D]
        # Weight and sum over k expert slots
        out = (out.squeeze(-2) * weights[..., None]).sum(axis=1)  # [T, D]
        return out


class MoE(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.gate = Gate(layer_id, args)
        self.experts = Experts(args)
        self.shared_experts = Expert(
            args.hidden_size, args.moe_intermediate_size, args.swiglu_limit
        )

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None) -> mx.array:
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        weights, indices = self.gate(x_flat, input_ids)
        routed = self.experts(x_flat, indices, weights).reshape(B, L, D)
        shared = self.shared_experts(x)
        return (routed + shared).astype(x.dtype)


class DeepseekV4Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        hc = args.hc_mult
        hc_dim = hc * args.hidden_size
        mix_hc = (2 + hc) * hc
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim))
        self.hc_attn_base = mx.zeros((mix_hc,))
        self.hc_attn_scale = mx.zeros((3,))
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim))
        self.hc_ffn_base = mx.zeros((mix_hc,))
        self.hc_ffn_scale = mx.zeros((3,))
        self._args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        input_ids: Optional[mx.array] = None,
    ) -> mx.array:
        args = self._args

        # Attention with HC
        residual = x
        x_comb, post, comb = _hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            args.hc_mult, args.hc_sinkhorn_iters, args.hc_eps, args.rms_norm_eps,
        )
        attn_out = self.attn(self.attn_norm(x_comb), mask, cache)
        x = _hc_post(attn_out, residual, post, comb)

        # FFN with HC
        residual = x
        x_comb, post, comb = _hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            args.hc_mult, args.hc_sinkhorn_iters, args.hc_eps, args.rms_norm_eps,
        )
        ffn_out = self.ffn(self.ffn_norm(x_comb), input_ids)
        x = _hc_post(ffn_out, residual, post, comb)
        return x


class MTPBlock(nn.Module):
    """Multi-token prediction block (used only for training; weights loaded but forward skipped)."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.e_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.h_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        hc = args.hc_mult
        hc_dim = hc * args.hidden_size
        mix_hc = (2 + hc) * hc
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim))
        self.hc_attn_base = mx.zeros((mix_hc,))
        self.hc_attn_scale = mx.zeros((3,))
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim))
        self.hc_ffn_base = mx.zeros((mix_hc,))
        self.hc_ffn_scale = mx.zeros((3,))
        self.hc_head_fn = mx.zeros((hc, hc_dim))
        self.hc_head_base = mx.zeros((hc,))
        self.hc_head_scale = mx.zeros((1,))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DeepseekV4Block(i, args) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        hc = args.hc_mult
        hc_dim = hc * args.hidden_size
        self.hc_head_fn = mx.zeros((hc, hc_dim))
        self.hc_head_base = mx.zeros((hc,))
        self.hc_head_scale = mx.zeros((1,))

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed(inputs)  # [B, L, D]
        # Expand to hc_mult copies for Hyper-Connections
        h = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], self.args.hc_mult, h.shape[-1]))

        mask = create_attention_mask(h[:, :, 0, :].astype(mx.float32), cache)
        for i, layer in enumerate(self.layers):
            h = layer(h, mask, cache[i] if cache is not None else None, inputs)

        # Collapse hc copies for head
        B, L, hc, D = h.shape
        h_flat = h.reshape(B, L, hc * D).astype(mx.float32)
        rsqrt = mx.rsqrt(mx.mean(h_flat * h_flat, axis=-1, keepdims=True) + self.args.hc_eps)
        mixes = (h_flat @ self.hc_head_fn.T) * rsqrt  # [B, L, hc]
        pre = (mx.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.args.hc_eps)
        h_out = mx.sum(pre[..., None] * h, axis=2)  # [B, L, D]

        return self.head(self.norm(h_out.astype(h.dtype)))

    def sanitize(self, weights: dict) -> dict:
        """Dequantize FP8 weights, stack experts, drop MTP layers."""

        # float4_e2m1fn lookup table (nibble value 0-15 → bfloat16)
        _FP4_LUT = mx.array(
            [0., 0.5, 1., 1.5, 2., 3., 4., 6., 0., -0.5, -1., -1.5, -2., -3., -4., -6.],
            dtype=mx.bfloat16,
        )

        def to_uint8(w: mx.array) -> mx.array:
            # F8_E4M3 tensors may arrive as int8 or uint8; from_fp8 requires uint8.
            return w.view(mx.uint8) if w.dtype != mx.uint8 else w

        def is_fp4(w: mx.array, scale: mx.array) -> bool:
            # FP4 packed: logical in_features = w.shape[1]*2, block_size = 32
            # → scale_n = logical_in // 32 = w.shape[1]*2 // 32
            return scale.shape[1] * 32 == w.shape[1] * 2

        def dequant_fp4(w: mx.array, scale: mx.array) -> mx.array:
            """Dequantize float4_e2m1fn_x2 with per-(out, 32-in-block) E8M0 scales."""
            w = to_uint8(w)
            lo = _FP4_LUT[w & 0x0F]     # lower nibble → first element in pair
            hi = _FP4_LUT[(w >> 4)]      # upper nibble → second element
            M, N_half = w.shape
            # Interleave pairs: [M, N_half, 2] → [M, N_full]
            w_fp = mx.stack([lo, hi], axis=-1).reshape(M, N_half * 2)
            # Scale: [M, N_full//32], block size = 32 along in
            fp4_bs = 32
            N_full = N_half * 2
            pn = (-N_full) % fp4_bs
            if pn:
                w_fp = mx.pad(w_fp, ((0, 0), (0, pn)))
            w_fp = w_fp.reshape(M, -1, fp4_bs) * scale[:, :, None]
            return w_fp.reshape(M, -1)[:, :N_full].astype(mx.bfloat16)

        def dequant_fp8(w: mx.array, scale: mx.array) -> mx.array:
            """Block-wise FP8 dequantization (128×128 blocks)."""
            w = mx.from_fp8(to_uint8(w), dtype=mx.bfloat16)
            bs = 128
            m, n = w.shape
            pm, pn = (-m) % bs, (-n) % bs
            if pm or pn:
                w = mx.pad(w, ((0, pm), (0, pn)))
            M, N = w.shape
            w = w.reshape(M // bs, bs, N // bs, bs)
            w = (w * scale[:, None, :, None]).reshape(M, N)
            return w[:m, :n].astype(mx.bfloat16)

        # Dequantize all .weight / .scale pairs
        scale_keys = {k[: -len(".scale")] for k in weights if k.endswith(".scale")}
        new_weights = {}
        for k, v in weights.items():
            if k.endswith(".scale"):
                continue
            base = k[: -len(".weight")] if k.endswith(".weight") else None
            if base is not None and base in scale_keys:
                s = weights[base + ".scale"]
                if is_fp4(v, s):
                    new_weights[k] = dequant_fp4(v, s)
                else:
                    new_weights[k] = dequant_fp8(v, s)
            elif v.dtype in (mx.uint8, mx.int8) and k.endswith(".weight"):
                # FP8 weight without a block scale (e.g. gate routing matrix):
                # direct fp8→bfloat16 conversion, no block scaling.
                new_weights[k] = mx.from_fp8(to_uint8(v), dtype=mx.bfloat16)
            else:
                new_weights[k] = v
        weights = new_weights

        # Stack routed expert weights into SwitchLinear format [n_exp, out, in]
        n_exp = self.args.n_routed_experts
        for l in range(self.args.num_hidden_layers):
            pfx = f"layers.{l}.ffn.experts"
            for proj in ("w1", "w2", "w3"):
                src = f"{pfx}.0.{proj}.weight"
                if src in weights:
                    stacked = mx.stack([
                        weights.pop(f"{pfx}.{e}.{proj}.weight")
                        for e in range(n_exp)
                    ])
                    weights[f"{pfx}.{proj}.weight"] = stacked

        # Split wo_a weight [n_groups*o_lora_rank, D_group] into per-group linears
        n_groups = self.args.o_groups
        o_lora_rank = self.args.o_lora_rank
        for l in range(self.args.num_hidden_layers):
            key = f"layers.{l}.attn.wo_a.weight"
            if key in weights:
                w = weights.pop(key)  # [n_groups * o_lora_rank, D_group]
                for g in range(n_groups):
                    weights[f"layers.{l}.attn.wo_a.{g}.weight"] = w[
                        g * o_lora_rank : (g + 1) * o_lora_rank, :
                    ]

        # Drop MTP layers (training-only)
        return {k: v for k, v in weights.items() if not k.startswith("mtp.")}

    @property
    def layers_list(self):
        return self.layers

    @property
    def cast_predicate(self):
        """Exclude precision-sensitive float32 parameters from dtype casting."""
        excluded = {
            "hc_attn_fn", "hc_attn_base", "hc_attn_scale",
            "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
            "hc_head_fn", "hc_head_base", "hc_head_scale",
            "attn_sink", "bias",  # gate.bias
        }

        def predicate(k):
            return not any(e in k for e in excluded)

        return predicate
