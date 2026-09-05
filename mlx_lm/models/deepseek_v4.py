# Copyright © 2026 Apple Inc. / mlx-community
#
# DeepSeek-V4 (Pro / Flash) for mlx-lm.
# Architecture: Multi-head Latent Attention (num_kv_heads=1) + grouped low-rank output,
# sliding-window + compressed KV + indexer topk (sparse attention), hash-routed MoE
# with sqrtsoftplus scoring, Manifold-constrained Hyper-Connections (mHC) replacing
# residuals. Weights are native FP8 (e4m3) with 128x128 block scaling (ue8m0).
#
# Reference: deepseek-ai/DeepSeek-V4 (Apr 2026). mHC: arXiv:2512.24880.

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .pipeline import PipelineMixin
from .switch_layers import SwitchGLU

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1

    # Attention (MLA-style with single shared KV head)
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    attention_bias: bool = False
    sliding_window: int = 128
    compress_ratios: List[int] = field(default_factory=list)

    # Compressor / Indexer
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # The top-k indexer runs only once the pool exceeds max(index_topk,
    # index_gate_rows) rows; below that the full pool is attended (full
    # attention at short context, where the selector's overhead would not pay).
    index_gate_rows: int = 8192
    compress_rope_theta: float = 160000.0

    # MoE
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    num_hash_layers: int = 3
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0

    # Hyper-Connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # RoPE / YaRN
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    rms_norm_eps: float = 1e-6

    # Quantization (FP8 block)
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        # Auto-fill compress_ratios with V4 defaults if not specified, else
        # require one entry per hidden layer. Adapted from @eauchs c6a7828 (#1192).
        if not self.compress_ratios:
            n = self.num_hidden_layers
            self.compress_ratios = (
                [0]
                + [4 if i % 2 else 128 for i in range(max(n - 2, 0))]
                + ([0] if n >= 2 else [])
            )
        elif len(self.compress_ratios) < self.num_hidden_layers:
            raise ValueError(
                "`compress_ratios` must have at least one entry per hidden layer, "
                f"got {len(self.compress_ratios)} for {self.num_hidden_layers} layers."
            )
        # Extra trailing entries (the checkpoint lists them for its MTP layers) are
        # ignored, matching the dropped MTP weights.
        self.compress_ratios = list(self.compress_ratios[: self.num_hidden_layers])
        bad = [r for r in self.compress_ratios if r not in (0, 4, 128)]
        if bad:
            raise ValueError(f"Unsupported DeepSeek-V4 compress ratios: {bad}")


# --------------------------------------------------------------------------- #
# Manifold-constrained Hyper-Connections (mHC)                                #
# --------------------------------------------------------------------------- #
# mHC expands the residual into `hc_mult` parallel copies, mixes them by a
# Sinkhorn-normalized doubly-stochastic matrix (Birkhoff polytope), applies the
# block, and recombines. Sinkhorn + HyperConnection/HyperHead live here.


def _make_hc_split_sinkhorn_fused_kernel():
    """All-fused HC split + sinkhorn kernel (hc_mult=4 only).

    Combines pre-sigmoid, post-sigmoid, comb scaling, softmax, and Sinkhorn
    iterations into a single Metal dispatch. Replaces 4 dispatches with 1 on
    the hc_mult=4 fast path. Source ported from Blaizzy/mlx-lm#1192.
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint idx = thread_position_in_grid.x;
        constexpr int MIX  = (2 + HC) * HC;
        constexpr int BASE = 2 * HC;

        const device float* mix = (const device float*)mixes + idx * MIX;
        device float* pre_out   = (device float*)pre  + idx * HC;
        device float* post_out  = (device float*)post + idx * HC;
        device float* comb_out  = (device float*)comb + idx * HC * HC;

        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];
        const float epsv       = eps[0];

        {
            float4 z = *(const device float4*)mix * pre_scale
                     + *(const device float4*)base;
            *(device float4*)pre_out = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
        }
        {
            float4 z = *(const device float4*)(mix + HC) * post_scale
                     + *(const device float4*)(base + HC);
            *(device float4*)post_out = 2.0f * 1.0f / (1.0f + metal::fast::exp(-z));
        }

        float4 v0 = *(const device float4*)(mix  + BASE     ) * comb_scale + *(const device float4*)(base + BASE     );
        float4 v1 = *(const device float4*)(mix  + BASE +  4) * comb_scale + *(const device float4*)(base + BASE +  4);
        float4 v2 = *(const device float4*)(mix  + BASE +  8) * comb_scale + *(const device float4*)(base + BASE +  8);
        float4 v3 = *(const device float4*)(mix  + BASE + 12) * comb_scale + *(const device float4*)(base + BASE + 12);

        float m0 = metal::max(metal::max(v0.x, v0.y), metal::max(v0.z, v0.w));
        float m1 = metal::max(metal::max(v1.x, v1.y), metal::max(v1.z, v1.w));
        float m2 = metal::max(metal::max(v2.x, v2.y), metal::max(v2.z, v2.w));
        float m3 = metal::max(metal::max(v3.x, v3.y), metal::max(v3.z, v3.w));

        float4 e0 = metal::fast::exp(v0 - m0);
        float4 e1 = metal::fast::exp(v1 - m1);
        float4 e2 = metal::fast::exp(v2 - m2);
        float4 e3 = metal::fast::exp(v3 - m3);

        float4 r0 = e0 * 1.0f / (e0.x + e0.y + e0.z + e0.w) + epsv;
        float4 r1 = e1 * 1.0f / (e1.x + e1.y + e1.z + e1.w) + epsv;
        float4 r2 = e2 * 1.0f / (e2.x + e2.y + e2.z + e2.w) + epsv;
        float4 r3 = e3 * 1.0f / (e3.x + e3.y + e3.z + e3.w) + epsv;

        float4 col = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col; r1 *= col; r2 *= col; r3 *= col;

        for (int iter = 1; iter < ITERS; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);
            col = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col; r1 *= col; r2 *= col; r3 *= col;
        }

        *(device float4*)(comb_out)      = r0;
        *(device float4*)(comb_out +  4) = r1;
        *(device float4*)(comb_out +  8) = r2;
        *(device float4*)(comb_out + 12) = r3;
    """

    return mx.fast.metal_kernel(
        name="deepseek_v4_hc_split_sinkhorn_fused",
        input_names=["mixes", "scale", "base", "eps"],
        output_names=["pre", "post", "comb"],
        source=source,
    )


_hc_split_sinkhorn_fused_kernel = _make_hc_split_sinkhorn_fused_kernel()


def hc_split_sinkhorn(
    mixes: mx.array,  # [B*S, (2+hc)*hc] fp32
    hc_scale: mx.array,  # [3] fp32
    hc_base: mx.array,  # [(2+hc)*hc] fp32
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Split `mixes` into (pre, post, comb_logits); Sinkhorn-normalize comb to doubly stochastic.

    Returns:
        pre  [N, hc]        — sigmoid(mixes[:,:hc] * s0 + base[:hc]) + eps
        post [N, hc]        — 2*sigmoid(mixes[:,hc:2hc] * s1 + base[hc:2hc])
        comb [N, hc, hc]    — Sinkhorn-normalized (rows & cols ~= 1) from the last hc*hc logits.
    """
    # Fast path: all-fused single-dispatch kernel for hc_mult=4 (DeepSeek-V4 default).
    # Combines pre/post sigmoid, comb scaling+softmax, and Sinkhorn iters into 1 GPU dispatch
    # (was 4 dispatches: pre sigmoid, post sigmoid, comb scaling, sinkhorn kernel).
    if (
        hc_mult == 4
        and _hc_split_sinkhorn_fused_kernel is not None
        and mx.metal.is_available()
        and mixes.size > 0
    ):
        n_rows = mixes.size // ((2 + hc_mult) * hc_mult)
        eps_arr = mx.array([eps], dtype=mx.float32)
        return _hc_split_sinkhorn_fused_kernel(
            inputs=[mixes, hc_scale, hc_base, eps_arr],
            template=[("HC", hc_mult), ("ITERS", sinkhorn_iters)],
            grid=(n_rows, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[
                (*mixes.shape[:-1], hc_mult),
                (*mixes.shape[:-1], hc_mult),
                (*mixes.shape[:-1], hc_mult, hc_mult),
            ],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )

    # Fallback: original split path (general hc_mult, or no Metal).
    n = mixes.shape[0]
    mix = mixes
    s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]

    pre_log = mix[:, :hc_mult] * s0 + hc_base[:hc_mult]
    post_log = mix[:, hc_mult : 2 * hc_mult] * s1 + hc_base[hc_mult : 2 * hc_mult]
    comb_log = mix[:, 2 * hc_mult :].reshape(n, hc_mult, hc_mult) * s2 + hc_base[
        2 * hc_mult :
    ].reshape(hc_mult, hc_mult)

    pre = mx.sigmoid(pre_log) + eps
    post = 2 * mx.sigmoid(post_log)

    # Sinkhorn projection to a doubly-stochastic comb (pure MLX). The hc_mult=4
    # fast path above handles the shipping config on Metal; every other case
    # (non-default hc_mult, or no Metal) falls through to this reference loop.
    comb = mx.softmax(comb_log, axis=-1, precise=True) + eps
    col_sum = comb.sum(axis=1, keepdims=True) + eps
    comb = comb / col_sum
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(axis=2, keepdims=True) + eps
        comb = comb / row_sum
        col_sum = comb.sum(axis=1, keepdims=True) + eps
        comb = comb / col_sum

    return pre, post, comb


class HyperConnection(nn.Module):
    """Per-block mHC parameters: projects x -> (pre, post, comb) used in hc_pre/hc_post.

    Paper/ref stores the weights as:
        hc_fn    : [(2+hc)*hc, hc*dim]
        hc_scale : [3]
        hc_base  : [(2+hc)*hc]

    hc_pre reduces `hc_mult` parallel hidden states to 1 via `pre`.
    Block F is applied to the reduced state. hc_post expands 1 -> hc via `post` (the new
    contribution) added to `comb @ residual` (where `comb` is a doubly-stochastic mix
    that recombines the input `hc_mult` copies to stay on the Birkhoff manifold).
    """

    def __init__(
        self,
        dim: int,
        hc_mult: int,
        norm_eps: float,
        sinkhorn_iters: int,
        hc_eps: float,
    ):
        super().__init__()
        self.dim = dim
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim
        # All mHC params are fp32 in the checkpoint.
        self.fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.scale = mx.zeros((3,), dtype=mx.float32)

    def hc_pre(self, x: mx.array):
        B, S, hc, D = x.shape
        dtype = x.dtype
        xf = x.reshape(B, S, hc * D).astype(mx.float32)
        xf_norm = mx.fast.rms_norm(xf, weight=None, eps=self.norm_eps)
        # MLX fuses the transpose into the matmul, so no cached transpose is needed.
        mixes = (xf_norm @ self.fn.T).reshape(B * S, -1)
        pre, post, comb = hc_split_sinkhorn(
            mixes, self.scale, self.base, hc, self.sinkhorn_iters, self.hc_eps
        )
        pre = pre.reshape(B, S, hc)
        post = post.reshape(B, S, hc)
        comb = comb.reshape(B, S, hc, hc)
        y = (pre[..., None] * x.astype(mx.float32)).sum(axis=2)
        return y.astype(dtype), post, comb

    def hc_post(
        self, f_out: mx.array, residual: mx.array, post: mx.array, comb: mx.array
    ):
        # f_out    [B,S,D] (block output, reduced state)
        # residual [B,S,hc,D] (input to hc_pre)
        # post     [B,S,hc]
        # comb     [B,S,hc,hc]
        # returns  [B,S,hc,D]
        dtype = f_out.dtype
        # post.unsqueeze(-1) * f_out.unsqueeze(-2)  -> [B,S,hc,D]
        term_new = post[..., None] * f_out[:, :, None, :].astype(mx.float32)
        # comb is doubly-stochastic (not symmetric) and is consumed transposed,
        # matching the HF reference: comb^T @ residual -> [B,S,hc,D].
        term_res = mx.swapaxes(comb, -1, -2).astype(mx.float32) @ residual.astype(
            mx.float32
        )
        y = term_new + term_res
        return y.astype(dtype)


class HyperHead(nn.Module):
    """Final (head) mHC projection: reduces [B,S,hc,D] -> [B,S,D] via sigmoid-weighted sum.
    No Sinkhorn here — this is the simpler head variant from `ParallelHead.hc_head`.
    """

    def __init__(self, dim: int, hc_mult: int, norm_eps: float, hc_eps: float):
        super().__init__()
        self.dim = dim
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.fn = mx.zeros((hc_mult, hc_mult * dim), dtype=mx.float32)
        self.base = mx.zeros((hc_mult,), dtype=mx.float32)
        self.scale = mx.zeros((1,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        B, S, hc, D = x.shape
        dtype = x.dtype
        xf = x.reshape(B, S, hc * D).astype(mx.float32)
        inv = mx.rsqrt((xf * xf).mean(axis=-1, keepdims=True) + self.norm_eps)
        mixes = (xf @ self.fn.T) * inv  # [B,S,hc]
        pre = mx.sigmoid(mixes * self.scale[0] + self.base) + self.hc_eps
        y = (pre[..., None] * x.astype(mx.float32)).sum(axis=2)
        return y.astype(dtype)


# --------------------------------------------------------------------------- #
# Fused partial-RoPE Metal kernel                                             #
# --------------------------------------------------------------------------- #
# Collapses the many small rope ops per decode step into one Metal dispatch
# (one work item per (b, h, l); lane t handles the pair x[2t], x[2t+1]).
# Adapted from @0xClandestine's Blaizzy/mlx-lm#13.


def _make_partial_rope_kernel():
    # Env-var escape hatch so benchmarks can A/B kernel ON vs OFF without
    # monkey-patching: MLX_LM_DISABLE_PARTIAL_ROPE_KERNEL=1 -> falls back
    # to the pure-MLX path used pre-2026-04-25.
    import os

    if os.environ.get("MLX_LM_DISABLE_PARTIAL_ROPE_KERNEL", "0") == "1":
        return None
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid = thread_position_in_threadgroup.x;
        uint gid = threadgroup_position_in_grid.x;

        constexpr int DRH = D_ROPE / 2;
        int L_v = dims[0];
        int H_v = dims[1];
        uint l   = gid % (uint)L_v;
        uint tmp = gid / (uint)L_v;
        uint h   = tmp % (uint)H_v;
        uint b   = tmp / (uint)H_v;

        const auto xp = x  + ((uint64_t)b * H_v * L_v + h * L_v + l) * D_ROPE;
        auto       yp = y  + ((uint64_t)b * H_v * L_v + h * L_v + l) * D_ROPE;
        const auto cp = cos_s + l * DRH;
        const auto sp = sin_s + l * DRH;

        // Lane t handles one interleaved pair (x[2t], x[2t+1]).
        if ((int)tid < DRH) {
            float x0 = float(xp[2 * tid]);
            float x1 = float(xp[2 * tid + 1]);
            float c  = float(cp[tid]);
            float s  = float(sp[tid]);
            if (INVERSE) {
                store_elem(yp[2 * tid],     fma( x1, s, x0 * c));   //  x0*c + x1*s
                store_elem(yp[2 * tid + 1], fma(-x0, s, x1 * c));   // -x0*s + x1*c
            } else {
                store_elem(yp[2 * tid],     fma(-x1, s, x0 * c));   //  x0*c - x1*s
                store_elem(yp[2 * tid + 1], fma( x0, s, x1 * c));   //  x0*s + x1*c
            }
        }
    """
    return mx.fast.metal_kernel(
        name="ds4_partial_rope",
        input_names=["x", "cos_s", "sin_s", "dims"],
        output_names=["y"],
        header="template<typename T> inline void store_elem(device T& dst, float v) { dst = T(v); }",
        source=source,
    )


_partial_rope_kernel = _make_partial_rope_kernel()


class DeepseekV4RoPE(nn.Module):
    """DeepSeek-V4 rotary embedding.

    The reference implementation applies RoPE to the KV tensor before attention
    and applies the conjugate rotation to the attention output. The generic MLX
    RoPE layers do not expose an inverse path, so keep the small DeepSeek-specific
    implementation here.
    """

    def __init__(
        self,
        dims: int,
        base: float,
        scaling_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.dims = dims

        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        # YaRN attention scaling (mscale); applied to the queries by the caller.
        self.attention_scaling = 1.0
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")

        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            original_max_position_embeddings = scaling_config[
                "original_max_position_embeddings"
            ]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)

            def correction_dim(num_rotations):
                return (
                    dims
                    * math.log(
                        original_max_position_embeddings / (num_rotations * 2 * math.pi)
                    )
                    / (2 * math.log(base))
                )

            low = math.floor(correction_dim(beta_fast))
            high = math.ceil(correction_dim(beta_slow))
            low = max(low, 0)
            high = min(high, dims - 1)
            if low == high:
                high += 0.001

            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

            # mscale, matching HF _compute_yarn_parameters: the mscale/mscale_all_dim
            # ratio when both are set, else the default 0.1*ln(factor)+1 (factor > 1).
            if factor > 1:
                mscale = scaling_config.get("mscale")
                mscale_all_dim = scaling_config.get("mscale_all_dim")
                if mscale is not None and mscale_all_dim is not None:

                    def _get_mscale(scale, m):
                        return 0.1 * m * math.log(scale) + 1.0 if scale > 1 else 1.0

                    self.attention_scaling = _get_mscale(factor, mscale) / _get_mscale(
                        factor, mscale_all_dim
                    )
                else:
                    self.attention_scaling = 0.1 * math.log(factor) + 1.0
        elif rope_type not in (None, "default", "linear"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type {rope_type}")

        # This is derived from config, not a checkpoint parameter.
        self._inv_freq = (inv_freq,)

    @property
    def inv_freq(self):
        return self._inv_freq[0]

    def __call__(self, x: mx.array, offset: int = 0, inverse: bool = False):
        dtype = x.dtype
        T = x.shape[-2]
        if isinstance(offset, mx.array):
            if offset.size == 1:
                offset = offset.item()
            else:
                B = offset.shape[0]
                pos = offset[:, None] + mx.arange(T, dtype=mx.float32)[None, :]
                theta = pos[..., None] * self.inv_freq[None, None, :]
                if inverse:
                    theta = -theta
                # theta: [B, T, dims//2]. Reshape for x dims: [B,H,T,D] or [B,1,T,D]
                target_shape = (B,) + (1,) * (x.ndim - 3) + (T, self.dims // 2)
                cos = mx.cos(theta).reshape(target_shape).astype(dtype)
                sin = mx.sin(theta).reshape(target_shape).astype(dtype)
                rot = x[..., : self.dims].reshape(*x.shape[:-1], self.dims // 2, 2)
                x0 = rot[..., 0]
                x1 = rot[..., 1]
                r0 = x0 * cos - x1 * sin
                r1 = x0 * sin + x1 * cos
                rotated = mx.stack([r0, r1], axis=-1).reshape(*x.shape[:-1], self.dims)
                if self.dims < x.shape[-1]:
                    return mx.concatenate([rotated, x[..., self.dims :]], axis=-1)
                return rotated
        # Fast path: fused Metal kernel for the rope-only 4D case used by
        # V4Attention. Falls through to the pure-MLX path on CPU, on Mode-B
        # (x has a nope tail), or on non-4D inputs (e.g. Indexer rope).
        # The kernel itself handles inverse via formula sign-flip; theta is
        # always forward-direction (do NOT negate it here).
        if (
            _partial_rope_kernel is not None
            and x.shape[-1] == self.dims
            and x.ndim == 4
        ):
            B, H, L, _ = x.shape
            pos = mx.arange(offset, offset + T, dtype=mx.float32)
            theta = pos[:, None] * self.inv_freq[None, :]
            cos = mx.cos(theta).astype(mx.float32)
            sin = mx.sin(theta).astype(mx.float32)
            dims_arr = mx.array([L, H], dtype=mx.int32)
            # One lane per interleaved rope pair, so the threadgroup must cover
            # all dims/2 pairs (not a fixed 32) or the upper pairs go unwritten.
            drh = self.dims // 2
            return _partial_rope_kernel(
                inputs=[x, cos, sin, dims_arr],
                template=[("D_ROPE", self.dims), ("INVERSE", 1 if inverse else 0)],
                grid=(B * H * L * drh, 1, 1),
                threadgroup=(drh, 1, 1),
                output_shapes=[x.shape],
                output_dtypes=[x.dtype],
            )[0]

        pos = mx.arange(offset, offset + T, dtype=mx.float32)
        theta = pos[:, None] * self.inv_freq[None, :]
        if inverse:
            theta = -theta

        broadcast_shape = (1,) * (x.ndim - 2) + theta.shape
        cos = mx.cos(theta).reshape(broadcast_shape).astype(dtype)
        sin = mx.sin(theta).reshape(broadcast_shape).astype(dtype)

        rot = x[..., : self.dims].reshape(*x.shape[:-1], self.dims // 2, 2)
        x0 = rot[..., 0]
        x1 = rot[..., 1]
        y = mx.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), axis=-1)
        y = y.reshape(*x.shape[:-1], self.dims)
        if x.shape[-1] == self.dims:
            return y
        return mx.concatenate([y, x[..., self.dims :]], axis=-1)


# --------------------------------------------------------------------------- #
# Gate (hash + score-based)                                                   #
# --------------------------------------------------------------------------- #

# Pre-allocated scalar zero for sqrtsoftplus: avoids mx.zeros_like() allocation per call.
_SCORE_ZERO = mx.array(0.0)


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    # sqrtsoftplus: sqrt(softplus(x))  — used by V4
    # Scalar broadcast avoids allocating a zeros tensor every call.
    return mx.sqrt(mx.logaddexp(scores, _SCORE_ZERO))


class MoEGate(nn.Module):
    """Routing gate. First `num_hash_layers` layers use a deterministic hash
    (token-id -> expert-id table) instead of learned score-based topk. Remaining
    layers run sqrtsoftplus scoring + e_score_correction_bias + topk, then
    renormalize the selected top-k weights (matching the HF reference)."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_routed = args.n_routed_experts
        self.top_k = args.num_experts_per_tok
        self.hash = layer_idx < args.num_hash_layers
        self.score_func = args.scoring_func
        self.route_scale = args.routed_scaling_factor

        self.weight = mx.zeros((self.n_routed, args.hidden_size))
        # Cache transposed weight to avoid recomputing .T every forward call.
        if self.hash:
            # tid2eid: [vocab, top_k] int32 — predetermined expert routing per token id
            self.tid2eid = mx.zeros((args.vocab_size, self.top_k), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros((self.n_routed,), dtype=mx.float32)

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        # x: [B, S, D] or [N, D]. MLX fuses the weight transpose into the matmul.
        if self.hash:
            # x shape -> [B*S, D]; input_ids -> [B, S] flattened to [B*S]
            flat = x.reshape(-1, x.shape[-1])
            scores = flat.astype(mx.float32) @ self.weight.T.astype(mx.float32)
            scores = _score_func(scores, self.score_func)
            ids = input_ids.reshape(-1)
            inds = self.tid2eid[ids].astype(mx.int32)
            weights = mx.take_along_axis(scores, inds, axis=-1)
            # Reshape inds/weights back to match x's leading dims so SwitchGLU
            # can broadcast against x: [B, S, top_k] (mirrors non-hash branch).
            inds = inds.reshape(*x.shape[:-1], self.top_k)
            weights = weights.reshape(*x.shape[:-1], self.top_k)
        else:
            scores = x.astype(mx.float32) @ self.weight.T.astype(mx.float32)
            scores = _score_func(scores, self.score_func)
            orig = scores
            biased = scores + self.e_score_correction_bias
            inds = mx.argpartition(-biased, kth=self.top_k - 1, axis=-1)[
                ..., : self.top_k
            ]
            weights = mx.take_along_axis(orig, inds, axis=-1)

        # HF normalizes the selected top-k weights unconditionally (both TopK and
        # Hash routers), independent of score_func / norm_topk_prob.
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        weights = weights * self.route_scale
        return inds, weights


# --------------------------------------------------------------------------- #
# MoE                                                                          #
# --------------------------------------------------------------------------- #


def _swiglu_limited(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        up = mx.clip(up, -limit, limit)
        gate = mx.minimum(gate, limit)
    return nn.silu(gate) * up


class _LimitedSwiGLU(nn.Module):
    """SwiGLU activation for SwitchGLU routed experts, with V4's gate/up clamp.

    SwitchGLU invokes its activation as ``activation(up, gate)`` (up first), matching
    HF's per-expert ``clamp(gate)/clamp(up)`` then ``silu(gate) * up``.
    """

    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, up: mx.array, gate: mx.array) -> mx.array:
        return _swiglu_limited(gate, up, self.limit)


class DeepseekV4MLP(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, swiglu_limit: float = 0.0
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _swiglu_limited(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


class DeepseekV4MoE(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=_LimitedSwiGLU(args.swiglu_limit),
        )
        self.gate = MoEGate(args, layer_idx)
        if args.n_shared_experts:
            self.shared_experts = DeepseekV4MLP(
                args.hidden_size,
                args.moe_intermediate_size * args.n_shared_experts,
                swiglu_limit=args.swiglu_limit,
            )
        self.sharding_group = None

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        inds, weights = self.gate(x, input_ids)
        # Compute shared_experts before switch_mlp so MLX can overlap both
        # on the GPU — shared_experts doesn't depend on routing results.
        shared_y = self.shared_experts(x) if hasattr(self, "shared_experts") else None
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        if shared_y is not None:
            y = y + shared_y
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


def _rope_at(rope, x_pe: mx.array, positions: mx.array) -> mx.array:
    """Interleaved-pair RoPE on x_pe[..., S, dims] at explicit `positions` [S]."""
    dims = rope.dims
    theta = positions[:, None] * rope.inv_freq[None, :]
    cos = mx.cos(theta).astype(x_pe.dtype)
    sin = mx.sin(theta).astype(x_pe.dtype)
    shape = (1,) * (x_pe.ndim - 2) + cos.shape
    cos = cos.reshape(shape)
    sin = sin.reshape(shape)
    rot = x_pe[..., :dims].reshape(*x_pe.shape[:-1], dims // 2, 2)
    x0 = rot[..., 0]
    x1 = rot[..., 1]
    y = mx.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), axis=-1)
    return y.reshape(x_pe.shape)


def _rope_pool(
    rope, pool: mx.array, positions: mx.array, rope_head_dim: int
) -> mx.array:
    """RoPE the last `rope_head_dim` dims of compressed pool rows [..., n, head_dim]
    at absolute `positions` [n]. Rows are roped ONCE at emit time (row j at its true
    absolute position j*ratio) so a later top-k gather carries each selected row's
    correct positional signal — matching the HF DeepSeek-V4 reference, which ropes
    compressed rows at emit and never repositions them. (The old code roped after the
    gather at arange(n)*ratio, mis-positioning the unordered selected rows.)"""
    nope = pool.shape[-1] - rope_head_dim
    p_nope, p_pe = mx.split(pool, [nope], axis=-1)
    p_pe = _rope_at(rope, p_pe, positions)
    return mx.concatenate([p_nope, p_pe], axis=-1)


@lru_cache(maxsize=32)
def _compressed_col_mask(
    S_q: int, offset: int, n_comp: int, r: int, dtype: mx.Dtype
) -> mx.array:
    """Contiguous compressed-pool mask columns, [S_q, n_comp]. Block j summarizes
    raw positions [j*r, (j+1)*r), so a query at absolute position offset+i attends it
    iff offset+i >= (j+1)*r-1. Memoized (depends only on these scalars) so a
    multi-token forward builds one mask per (offset, compress-ratio) rather than
    rebuilding it in every compressed layer. bool dtype -> bool; else additive 0/-inf.
    """
    comp_end = mx.arange(n_comp) * r + (r - 1)
    keep = (offset + mx.arange(S_q))[:, None] >= comp_end[None, :]  # [S_q, n_comp]
    if dtype == mx.bool_:
        return keep
    return mx.where(keep, mx.array(0.0, dtype), mx.array(float("-inf"), dtype))


def _compressed_mask(
    S_q: int,
    offset: Union[int, mx.array],
    n_comp: int,
    r: int,
    dtype: mx.Dtype,
    block_ids: Optional[mx.array],
) -> mx.array:
    """Compressed-pool mask columns. Query row i sits at absolute position offset+i and
    attends compressed column g (covering raw block b, ending at b*r+r-1) iff
    offset+i >= b*r+r-1, where b = block_ids[g] for a gathered (top-k) pool or g for a
    contiguous pool. Scalar offset -> [S_q, n_comp] (contiguous case memoized). Batched
    (B>1) serving passes a per-batch offset (mx.array [B]) -> per-batch [B, 1, S_q,
    n_comp] mask (block_ids is None there: B>1 top-k is unsupported at the call site).
    """
    if isinstance(offset, mx.array) and offset.ndim > 0:
        comp_end = (
            mx.arange(n_comp) * r + (r - 1)
            if block_ids is None
            else block_ids.astype(mx.int32) * r + (r - 1)
        )  # [n_comp]
        q_pos = offset[:, None] + mx.arange(S_q)[None, :]  # [B, S_q]
        keep = (
            q_pos[:, None, :, None] >= comp_end[None, None, None, :]
        )  # [B,1,S_q,n_comp]
        if dtype == mx.bool_:
            return keep
        return mx.where(keep, mx.array(0.0, dtype), mx.array(float("-inf"), dtype))
    if block_ids is None:
        return _compressed_col_mask(S_q, offset, n_comp, r, dtype)
    comp_end = block_ids.astype(mx.int32) * r + (r - 1)  # [n_comp]
    keep = (offset + mx.arange(S_q))[:, None] >= comp_end[None, :]  # [S_q, n_comp]
    if dtype == mx.bool_:
        return keep
    return mx.where(keep, mx.array(0.0, dtype), mx.array(float("-inf"), dtype))


# --------------------------------------------------------------------------- #
# Attention: MLA (num_kv_heads=1) + sliding window + optional compressed KV   #
# --------------------------------------------------------------------------- #


class CompressedKVCache(KVCache):
    """Cache for compressed-attention layers: sliding-window local cache + compressed KV pool.

    During prefill, the compressor produces all compressed rows at once.
    During decode, tokens accumulate in a buffer; every `ratio` tokens the
    buffer is compressed and the result is appended to the pool.

    Inherits from KVCache so external engines (vllm-mlx) recognize it via
    isinstance checks. All state is proxied through self.local (RotatingKVCache).
    """

    def __init__(self, max_size: int = 128):
        # Skip KVCache.__init__ — we proxy everything through self.local
        self.local = RotatingKVCache(max_size=max_size, keep=0)
        self._pool = None
        self._buf = None
        self._buf_count = 0
        self._abs_pos = 0
        self._index_pool = None
        self._index_buf = None
        self._index_abs_pos = 0

    @property
    def offset(self):
        return self.local.offset

    @property
    def keys(self):
        return self.local.keys

    @keys.setter
    def keys(self, value):
        self.local.keys = value

    @property
    def values(self):
        return self.local.values

    @values.setter
    def values(self, value):
        self.local.values = value

    @property
    def pool(self):
        return self._pool

    def update_and_fetch(self, keys, values):
        return self.local.update_and_fetch(keys, values)

    @property
    def state(self):
        ls = self.local.state
        ls = ls if isinstance(ls, tuple) else (ls,)
        empty = mx.array([], dtype=mx.bfloat16)
        extra = tuple(
            a if a is not None else empty
            for a in (self._pool, self._buf, self._index_pool, self._index_buf)
        )
        return (*ls, *extra)

    @state.setter
    def state(self, value):
        *ls, pool, buf, ipool, ibuf = value
        self.local.state = tuple(ls) if len(ls) != 1 else ls[0]
        self._pool = pool if pool.size > 0 else None
        self._buf = buf if buf.size > 0 else None
        self._index_pool = ipool if ipool.size > 0 else None
        self._index_buf = ibuf if ibuf.size > 0 else None
        self._buf_count = 0 if self._buf is None else self._buf.shape[1]

    @property
    def nbytes(self):
        n = self.local.nbytes
        if self._pool is not None:
            n += self._pool.nbytes
        if self._buf is not None:
            n += self._buf.nbytes
        if self._index_pool is not None:
            n += self._index_pool.nbytes
        if self._index_buf is not None:
            n += self._index_buf.nbytes
        return n

    @property
    def meta_state(self):
        return (
            self.local.meta_state,
            str(self._buf_count),
            str(self._abs_pos),
            str(self._index_abs_pos),
        )

    @meta_state.setter
    def meta_state(self, value):
        self.local.meta_state = value[0]
        self._buf_count = int(value[1])
        self._abs_pos = int(value[2])
        self._index_abs_pos = int(value[3])

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        *ls, pool, buf, ipool, ibuf = state
        obj.local = RotatingKVCache.from_state(
            tuple(ls) if len(ls) != 1 else ls[0], meta_state[0]
        )
        obj._pool = pool if pool.size > 0 else None
        obj._buf = buf if buf.size > 0 else None
        obj._index_pool = ipool if ipool.size > 0 else None
        obj._index_buf = ibuf if ibuf.size > 0 else None
        obj._buf_count = int(meta_state[1])
        obj._abs_pos = int(meta_state[2])
        obj._index_abs_pos = int(meta_state[3])
        return obj

    def is_trimmable(self):
        # trim() rolls back only the sliding-window local cache. The compressed
        # pool / indexer pool cannot be un-compressed, and the raw hidden states
        # behind them are not retained, so a trim would leave stale compressed rows
        # for tokens that were rolled back — corrupting sparse attention on reuse.
        # Refuse to trim once any compressed state has accumulated; callers
        # (trim_prompt_cache) then skip trimming rather than corrupt the pool. A
        # fresh/empty cache stays trimmable (there is nothing to skew).
        if (
            self._pool is not None
            or self._buf is not None
            or self._index_pool is not None
            or self._index_buf is not None
        ):
            return False
        return self.local.is_trimmable()

    def trim(self, n):
        return self.local.trim(n)

    @classmethod
    def merge(cls, caches):
        """Merge multiple CompressedKVCaches into a single batched cache."""
        merged = cls.__new__(cls)

        # Merge local rotating caches (delegates to BatchRotatingKVCache)
        merged.local = caches[0].local.merge([c.local for c in caches])

        # Merge compressed pools: pad to max length, stack along B
        pools = [c._pool for c in caches]
        if all(p is None for p in pools):
            merged._pool = None
        else:
            head_dim = next(p.shape[-1] for p in pools if p is not None)
            dtype = next(p.dtype for p in pools if p is not None)
            max_len = max(p.shape[1] if p is not None else 0 for p in pools)
            padded = []
            for p in pools:
                if p is None:
                    padded.append(mx.zeros((1, max_len, head_dim), dtype=dtype))
                elif p.shape[1] < max_len:
                    pad = mx.zeros((1, max_len - p.shape[1], head_dim), dtype=dtype)
                    padded.append(mx.concatenate([p, pad], axis=1))
                else:
                    padded.append(p)
            merged._pool = mx.concatenate(padded, axis=0)

        # Merge buffers: pad to max buf_count, stack along B
        bufs = [c._buf for c in caches]
        buf_counts = [c._buf_count for c in caches]
        if all(b is None for b in bufs):
            merged._buf = None
            merged._buf_count = 0
        else:
            D = next(b.shape[-1] for b in bufs if b is not None)
            dtype = next(b.dtype for b in bufs if b is not None)
            max_bc = max(buf_counts)
            padded = []
            for b, bc in zip(bufs, buf_counts):
                if b is None:
                    padded.append(mx.zeros((1, max_bc, D), dtype=dtype))
                elif b.shape[1] < max_bc:
                    pad = mx.zeros((1, max_bc - b.shape[1], D), dtype=dtype)
                    padded.append(mx.concatenate([b, pad], axis=1))
                else:
                    padded.append(b)
            merged._buf = mx.concatenate(padded, axis=0)
            merged._buf_count = max_bc

        merged._index_pool = cls._merge_field([c._index_pool for c in caches])
        merged._index_buf = cls._merge_field([c._index_buf for c in caches])
        merged._abs_pos = max(c._abs_pos for c in caches)
        merged._index_abs_pos = max(c._index_abs_pos for c in caches)

        return merged

    def filter(self, batch_indices):
        if hasattr(self.local, "filter"):
            self.local.filter(batch_indices)
        for attr in ("_pool", "_buf", "_index_pool", "_index_buf"):
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, v[batch_indices])

    def extend(self, other):
        if hasattr(self.local, "extend"):
            self.local.extend(other.local)
        self._pool = self._extend_field(self._pool, other._pool)
        self._buf = self._extend_field(self._buf, other._buf)
        self._buf_count = max(self._buf_count, other._buf_count)
        self._index_pool = self._extend_field(self._index_pool, other._index_pool)
        self._index_buf = self._extend_field(self._index_buf, other._index_buf)
        self._abs_pos = max(self._abs_pos, other._abs_pos)
        self._index_abs_pos = max(self._index_abs_pos, other._index_abs_pos)

    def finalize(self):
        if hasattr(self.local, "finalize"):
            self.local.finalize()

    def extract(self, idx):
        extracted = CompressedKVCache.__new__(CompressedKVCache)
        extracted.local = (
            self.local.extract(idx) if hasattr(self.local, "extract") else self.local
        )

        def sl(a):
            return a[idx : idx + 1] if a is not None else None

        extracted._pool = sl(self._pool)
        extracted._buf = sl(self._buf)
        extracted._index_pool = sl(self._index_pool)
        extracted._index_buf = sl(self._index_buf)
        extracted._buf_count = self._buf_count
        extracted._abs_pos = self._abs_pos
        extracted._index_abs_pos = self._index_abs_pos
        return extracted

    @property
    def batch_size(self):
        if hasattr(self.local, "batch_size"):
            return self.local.batch_size
        return 1

    @staticmethod
    def _merge_field(arrays):
        # Pad batch-1 [1, n, D] arrays to max length and stack along batch.
        if all(a is None for a in arrays):
            return None
        present = next(a for a in arrays if a is not None)
        D, dtype = present.shape[-1], present.dtype
        max_len = max(a.shape[1] if a is not None else 0 for a in arrays)
        out = []
        for a in arrays:
            if a is None:
                out.append(mx.zeros((1, max_len, D), dtype=dtype))
            elif a.shape[1] < max_len:
                pad = mx.zeros((a.shape[0], max_len - a.shape[1], D), dtype=dtype)
                out.append(mx.concatenate([a, pad], axis=1))
            else:
                out.append(a)
        return mx.concatenate(out, axis=0)

    @staticmethod
    def _extend_field(a, b):
        # Concatenate two [B, n, D] fields along batch, padding to max length.
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        max_len = max(a.shape[1], b.shape[1])

        def pad(x):
            if x.shape[1] < max_len:
                p = mx.zeros(
                    (x.shape[0], max_len - x.shape[1], x.shape[2]), dtype=x.dtype
                )
                return mx.concatenate([x, p], axis=1)
            return x

        return mx.concatenate([pad(a), pad(b)], axis=0)

    @staticmethod
    def _accumulate_window(pool, buf, abs_pos, x, compressor, rope=None):
        """Emit compressed rows for completed windows; return (pool, buf, abs_pos).

        Shared by the main compressed KV pool and the indexer pool. Each row is
        produced by calling `compressor` over exactly the raw-token window a
        single full-sequence forward would see (retaining the previous window
        for ratio-4 overlap), so incremental decode is bit-for-bit consistent
        with prefill, including prefill remainders and chunk boundaries.

        When `rope` is given, each newly emitted row is RoPE'd at its true absolute
        position (row j -> j*r) before it enters the pool, so a later top-k gather
        carries the correct positional signal.
        """
        r = compressor.ratio
        overlap = compressor.overlap
        buf = x if buf is None else mx.concatenate([buf, x], axis=1)
        abs_pos += x.shape[1]
        raw_start = abs_pos - buf.shape[1]
        n_rows = 0 if pool is None else pool.shape[1]
        n_before = n_rows
        while (n_rows + 1) * r <= abs_pos:
            w = n_rows
            need_start = ((w - 1) * r) if (overlap and w > 0) else (w * r)
            s = need_start - raw_start
            e = (w + 1) * r - raw_start
            rows = compressor(buf[:, s:e])
            row = rows[:, -1:]
            pool = row if pool is None else mx.concatenate([pool, row], axis=1)
            n_rows += 1
        if rope is not None and n_rows > n_before:
            # RoPE just-emitted rows [n_before, n_rows) at positions [n_before*r, ...).
            pos = mx.arange(n_before, n_rows, dtype=mx.float32) * r
            roped = _rope_pool(rope, pool[:, n_before:], pos, compressor.rope_head_dim)
            pool = (
                roped
                if n_before == 0
                else mx.concatenate([pool[:, :n_before], roped], axis=1)
            )
        keep_abs = max(((n_rows - 1) * r) if overlap else (n_rows * r), 0)
        drop = keep_abs - raw_start
        if drop > 0:
            buf = buf[:, drop:]
        return pool, buf, abs_pos

    def accumulate(
        self, x: mx.array, compressor: "Compressor", rope=None
    ) -> Optional[mx.array]:
        """Main compressed-KV pool: buffer raw tokens, emit rows on boundaries. Pass
        `rope` (the compress RoPE) to pre-rope emitted rows at their true positions."""
        self._pool, self._buf, self._abs_pos = self._accumulate_window(
            self._pool, self._buf, self._abs_pos, x, compressor, rope
        )
        self._buf_count = 0 if self._buf is None else self._buf.shape[1]
        return self._pool

    def accumulate_index(
        self, x: mx.array, compressor: "Compressor", rope=None
    ) -> Optional[mx.array]:
        """Indexer pool: same windowing as accumulate, separate compressor/state.
        Pass `rope` to pre-rope emitted index rows at their true positions (the
        aligned indexer scores RoPE'd keys against RoPE'd queries).

        Lets the lightweight index compressor accumulate across decode steps so
        top-k block selection works during generation (it previously recomputed
        on the single decode token, produced no rows, and disabled top-k).
        """
        self._index_pool, self._index_buf, self._index_abs_pos = (
            self._accumulate_window(
                self._index_pool,
                self._index_buf,
                self._index_abs_pos,
                x,
                compressor,
                rope,
            )
        )
        return self._index_pool


class Compressor(nn.Module):
    """Learned gated pooling over `ratio` consecutive tokens for KV compression.

    At prefill, produces ~ seq/ratio compressed KV rows. At decode, accumulates
    tokens in a state buffer and emits a compressed row every `ratio` steps.
    Pure-MLX; a fused Metal kernel may replace this in a follow-up.
    """

    def __init__(self, args: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.dim = args.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.ratio = compress_ratio
        self.overlap = compress_ratio == 4
        out_dim = head_dim * (2 if self.overlap else 1)
        self.wkv = nn.Linear(self.dim, out_dim, bias=False)
        self.wgate = nn.Linear(self.dim, out_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, out_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

    def _overlap_transform(self, tensor: mx.array, value: float) -> mx.array:
        B, S, R, _ = tensor.shape
        D = self.head_dim
        out = mx.full((B, S, 2 * R, D), value, dtype=tensor.dtype)
        out[:, :, R:] = tensor[:, :, :, D:]
        out[:, 1:, :R] = tensor[:, :-1, :, :D]
        return out

    def __call__(self, x: mx.array) -> mx.array:
        # Prefill-only MVP: chunk x into windows of `ratio` tokens. Ratio-4
        # layers use the overlapping layout from the reference implementation.
        # Returns compressed KV: [B, S//ratio, head_dim] (bf16).
        B, S, _ = x.shape
        r = self.ratio
        keep = (S // r) * r
        if keep == 0:
            return mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        xf = x[:, :keep].astype(mx.float32)
        kv = self.wkv(xf).reshape(B, keep // r, r, -1)
        score = self.wgate(xf).reshape(B, keep // r, r, -1) + self.ape
        if self.overlap:
            kv = self._overlap_transform(kv, 0.0)
            score = self._overlap_transform(score, float("-inf"))
        weights = mx.softmax(score, axis=2, precise=True)
        kv = (kv * weights).sum(axis=2)
        return self.norm(kv.astype(x.dtype))


class V4Attention(nn.Module):
    """V4 attention block.

    Checkpoint shapes (Flash):
        n_heads=64, head_dim=512, rope_head_dim=64 (nope=448)
        q_lora_rank=1024,  wq_a: [dim, 1024], wq_b: [1024, n_heads*head_dim]
        wkv: [dim, head_dim]            (single shared K=V head, MQA-style)
        attn_sink: [n_heads] fp32
        wo_a: [n_heads*head_dim/n_groups, n_groups*o_lora_rank]
        wo_b: [n_groups*o_lora_rank, dim]
        For compress_ratio != 0: compressor.wkv/wgate/ape/norm; and if ratio==4, indexer.*

    Forward path (MVP):
        - Project Q (64 heads), K=V (1 head); apply RoPE to last `rope_head_dim` dims.
        - For ratio=0 layers: sliding window mask of size `sliding_window`.
        - For ratio!=0 layers: append compressed KV rows to attend to (no topk filtering
          yet — full compressed cache). Use attn_sink via SDPA `sinks=` argument.
        - Grouped low-rank output projection: wo_a per group -> concat -> wo_b.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = args.head_dim - args.qk_rope_head_dim
        self.n_groups = args.o_groups
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.window = args.sliding_window
        self.eps = args.rms_norm_eps

        ratios = args.compress_ratios or []
        self.compress_ratio = ratios[layer_idx] if layer_idx < len(ratios) else 0

        self.scale = self.head_dim**-0.5

        # q path
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )

        # kv path (single shared head)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=self.eps)

        # attention sink (per-head learnable bias added in softmax denominator)
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        # grouped low-rank output projection
        group_feat = (self.n_heads * self.head_dim) // self.n_groups
        self.wo_a = nn.Linear(group_feat, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(
            self.n_groups * self.o_lora_rank, self.dim, bias=args.attention_bias
        )

        # Two RoPEs, selected per layer type: sliding-window layers use plain
        # RoPE (no YaRN) at rope_theta; compressed layers use YaRN at
        # compress_rope_theta. Matches the HF reference (YaRN only on compressor
        # layers); applying YaRN to sliding layers drops CJK tokens (#1192).
        self.rope = DeepseekV4RoPE(
            self.rope_head_dim, args.rope_theta, scaling_config=None
        )
        self.compress_rope = DeepseekV4RoPE(
            self.rope_head_dim,
            args.compress_rope_theta,
            args.rope_scaling,
        )

        # Compressor / Indexer — present only when compress_ratio > 0
        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)

    def _grouped_output_projection(self, out: mx.array) -> mx.array:
        B, S = out.shape[:2]
        group_feat = (self.n_heads * self.head_dim) // self.n_groups
        out = out.reshape(B, S, self.n_groups, group_feat)

        if isinstance(self.wo_a, nn.QuantizedLinear):
            # Batched grouped quantized matmul: collapse the per-group Python
            # loop into one mx.quantized_matmul by treating the group dim as a
            # broadcast batch dim (adapted from @Blaizzy's
            # pc/add-deepseekv4flash-model branch): out [G,B,S,gf] against
            # weight/scales [G,1,r,.] -> [G,B,S,r], transposed back to [B,S,G*r].
            out_g = out.transpose(2, 0, 1, 3)
            weight = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)[
                :, None
            ]
            scales = self.wo_a.scales.reshape(self.n_groups, self.o_lora_rank, -1)[
                :, None
            ]
            biases = (
                None
                if self.wo_a.biases is None
                else self.wo_a.biases.reshape(self.n_groups, self.o_lora_rank, -1)[
                    :, None
                ]
            )
            out_g = mx.quantized_matmul(
                out_g,
                weight,
                scales=scales,
                biases=biases,
                transpose=True,
                group_size=self.wo_a.group_size,
                bits=self.wo_a.bits,
                mode=self.wo_a.mode,
            )
            out = out_g.transpose(1, 2, 0, 3).reshape(
                B, S, self.n_groups * self.o_lora_rank
            )
            if "bias" in self.wo_a:
                out = out + self.wo_a.bias
            return out

        wa = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, group_feat)
        out = mx.einsum("bsgd,grd->bsgr", out, wa)
        out = out.reshape(B, S, self.n_groups * self.o_lora_rank)
        if "bias" in self.wo_a:
            out = out + self.wo_a.bias
        return out

    def __call__(self, x: mx.array, mask=None, cache=None):
        B, S, _ = x.shape

        # --- Q (shared intermediate reused by indexer) ---
        qr = self.q_norm(self.wq_a(x))
        q = (
            self.wq_b(qr)
            .reshape(B, S, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        q = mx.fast.rms_norm(q, weight=None, eps=self.eps)

        # --- K = V (shared single-head) ---
        kv = self.kv_norm(self.wkv(x))
        kv = kv.reshape(B, 1, S, self.head_dim)

        offset = cache.offset if cache is not None else 0
        if isinstance(offset, mx.array):
            # Snapshot the pre-update offset. BatchRotatingKVCache stores offset as
            # an mx.array and update_and_fetch() (below) mutates it in place
            # (self.offset += S). This forward reuses `offset` AFTER that update —
            # for the inverse RoPE on the attention output and for _compressed_mask —
            # where the correct value is the query start position (pre-update). An
            # aliased array would be corrupted to the post-update value, misrotating
            # the output. Sequential caches expose an int (a value copy), so are immune.
            offset = mx.array(offset)

        # Apply RoPE only to the last rope_head_dim dims
        q_nope, q_pe = mx.split(q, [self.nope_head_dim], axis=-1)
        k_nope, k_pe = mx.split(kv, [self.nope_head_dim], axis=-1)
        attn_rope = self.compress_rope if self.compress_ratio else self.rope
        q_pe = attn_rope(q_pe, offset=offset)
        k_pe = attn_rope(k_pe, offset=offset)
        # HF scales both roped q and k by the YaRN mscale; fold both factors into
        # the query so the key (which is also the value here) stays unscaled.
        q_pe = q_pe * (attn_rope.attention_scaling**2)
        q = mx.concatenate([q_nope, q_pe], axis=-1)
        k = v = mx.concatenate([k_nope, k_pe], axis=-1)

        # --- Compressed sparse attention ---
        # Pool rows are RoPE'd at their true absolute positions when emitted (in the
        # cache) or below (cacheless prefill), so the top-k gather carries each
        # selected row's correct positional signal — matching the HF reference.
        rope_head_dim = self.head_dim - self.nope_head_dim
        compressed_k = compressed_v = None
        gathered_ids = None
        if self.compress_ratio:
            comp_cache = cache if isinstance(cache, CompressedKVCache) else None
            if comp_cache is not None:
                pool = comp_cache.accumulate(x, self.compressor, self.compress_rope)
            elif S > 1:
                pool = self.compressor(x)
                if pool.shape[1] > 0:
                    pos = (
                        mx.arange(pool.shape[1], dtype=mx.float32) * self.compress_ratio
                    )
                    pool = _rope_pool(self.compress_rope, pool, pos, rope_head_dim)
                else:
                    pool = None
            else:
                pool = None

            # Accumulate the indexer pool in lockstep so top-k block
            # selection works during decode (S==1), not just prefill. Index rows are
            # RoPE'd at emit (aligned indexer scores roped keys vs roped queries).
            index_pool = None
            if hasattr(self, "indexer"):
                irope = self.compress_rope
                if comp_cache is not None:
                    index_pool = comp_cache.accumulate_index(
                        x, self.indexer.compressor, irope
                    )
                elif S > 1:
                    index_pool = self.indexer.compressor(x)
                    if index_pool.shape[1] > 0:
                        ipos = (
                            mx.arange(index_pool.shape[1], dtype=mx.float32)
                            * self.compress_ratio
                        )
                        index_pool = _rope_pool(
                            irope,
                            index_pool,
                            ipos,
                            self.indexer.compressor.rope_head_dim,
                        )
                    else:
                        index_pool = None

            if pool is not None:
                ckv = pool
                # Gate: only run the top-k selector once the pool is large enough that
                # sparse attention pays for it (~index_gate_rows); below that, attend
                # the full pool (avoids the selector's overhead at short context).
                gate = max(self.args.index_topk, self.args.index_gate_rows)
                if hasattr(self, "indexer") and ckv.shape[1] > gate:
                    topk_idx = self.indexer(
                        x, qr, index_pool, offset, self.compress_rope
                    )
                    if topk_idx is not None:
                        gathered_ids = topk_idx  # [B, topk]
                        idx = mx.broadcast_to(
                            topk_idx[:, :, None],
                            (B, topk_idx.shape[1], self.head_dim),
                        )
                        ckv = mx.take_along_axis(ckv, idx, axis=1)
                compressed_k = ckv[:, None, :, :]  # rows already roped
                compressed_v = compressed_k

        # Update KV cache
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # Prepend compressed KV to cached KV for sparse attention
        if compressed_k is not None:
            k = mx.concatenate([compressed_k, k], axis=2)
            v = mx.concatenate([compressed_v, v], axis=2)
            n_comp = compressed_k.shape[2]
            if mask is not None:
                # Compressed pool columns are ATTENDED (bool True) and CAUSAL: query
                # row i (absolute position offset+i) attends compressed block b iff
                # offset+i >= b*r+r-1, where b is the true block id (gathered_ids after
                # top-k, else the contiguous index). Decode (S==1) has mask=None. The
                # offset term (previously omitted) is required for any forward at
                # offset>0 past the sliding window — e.g. chunked prefill.
                S_q = mask.shape[-2]
                if gathered_ids is not None and B > 1:
                    # Each batch selects its own top-k blocks, so a single shared
                    # compressed-column mask cannot be correct. B=1 generation and
                    # batched decode (mask=None) never reach here.
                    raise NotImplementedError(
                        "Batched (B>1) long-context prefill with indexer top-k is "
                        "not supported; run long prefill with batch size 1."
                    )
                block_ids = gathered_ids[0] if gathered_ids is not None else None
                comp_mask = _compressed_mask(
                    S_q, offset, n_comp, self.compress_ratio, mask.dtype, block_ids
                )
                comp_mask = mx.broadcast_to(
                    comp_mask, list(mask.shape[:-2]) + [S_q, n_comp]
                )
                mask = mx.concatenate([comp_mask, mask], axis=-1)

        out = scaled_dot_product_attention(
            q,
            k,
            v,
            cache=cache,
            scale=self.scale,
            mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )

        out_nope, out_pe = mx.split(out, [self.nope_head_dim], axis=-1)
        out_pe = attn_rope(out_pe, offset=offset, inverse=True)
        out = mx.concatenate([out_nope, out_pe], axis=-1)

        # Grouped low-rank projection: [B, n_heads, S, head_dim] -> [B, S, n_heads*head_dim]
        out = out.transpose(0, 2, 1, 3).reshape(B, S, self.n_heads * self.head_dim)
        out = self._grouped_output_projection(out)
        return self.wo_b(out)


class Indexer(nn.Module):
    """Top-k selector over compressed KV rows for ratio-4 sparse attention.

    Two-pass design: this module uses a lightweight compressor (index_head_dim,
    typically 128) to score all compressed rows cheaply, then returns topk
    indices used to gather from the main attention compressor's output
    (head_dim, typically 512). This reduces per-layer attention from O(S/4)
    to O(topk) compressed rows, a large reduction at long context (topk << S/4).

    Checkpoint params:
        wq_b: [q_lora_rank, n_heads * index_head_dim]
        weights_proj: [hidden_size, n_heads]
        compressor.{wkv, wgate, ape, norm}
    """

    def __init__(self, args: ModelArgs, compress_ratio: int):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.scale = args.index_head_dim**-0.5
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.compressor = Compressor(args, compress_ratio, self.head_dim)

    def __call__(
        self,
        x: mx.array,
        q_intermediate: mx.array,
        ck: Optional[mx.array] = None,
        offset: int = 0,
        rope=None,
    ) -> Optional[mx.array]:
        """Score compressed rows and return topk indices, aligned with the HF
        DeepSeek-V4 reference: RoPE'd queries (and RoPE'd keys `ck`, roped at emit)
        with the compress theta, ReLU(q·k) scoring, and raw signed head weights
        (no sigmoid) scaled by n_heads**-0.5, aggregated in fp32. The unroped/
        sigmoid form selects position-blind, degraded rows, so the reference
        RoPE'd, signed-weight form is used here.

        Args:
            x: [B, S, D] hidden state (fed to the lightweight compressor).
            q_intermediate: [B, S, q_lora_rank] post wq_a+q_norm (shared with main attn).
            ck: pre-computed index pool [B, n, index_head_dim], RoPE'd at emit.
            offset: absolute position of query row 0 (for query RoPE).
            rope: the compress RoPE; when None, queries are left unroped.

        Returns:
            topk_indices [B, topk] or None when there are too few compressed rows.
            Indices are shared across heads/queries (head-weighted scores aggregated).
        """
        B, S, _ = x.shape

        if ck is None:
            ck = self.compressor(x)
            if rope is not None and ck.shape[1] > 0:
                pos = mx.arange(ck.shape[1], dtype=mx.float32) * self.compressor.ratio
                ck = _rope_pool(rope, ck, pos, self.compressor.rope_head_dim)
        n_compressed = ck.shape[1]
        if n_compressed == 0:
            return None

        q = self.wq_b(q_intermediate)
        q = q.reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        if rope is not None:
            q_pos = mx.arange(offset, offset + S, dtype=mx.float32)
            q = _rope_pool(rope, q, q_pos, rope.dims)

        scores = mx.maximum(q @ ck[:, None].transpose(0, 1, 3, 2), 0.0) * self.scale
        w = (self.weights_proj(x) * (self.n_heads**-0.5)).transpose(0, 2, 1)[..., None]
        agg = (scores.astype(mx.float32) * w).sum(axis=1).sum(axis=1)  # [B, n_comp]

        topk = min(self.index_topk, n_compressed)
        return mx.argpartition(-agg, kth=topk - 1, axis=-1)[:, :topk]


# --------------------------------------------------------------------------- #
# Block                                                                       #
# --------------------------------------------------------------------------- #


class DeepseekV4Block(nn.Module):
    """V4 block: mHC-wrapped (attention-norm -> attention), mHC-wrapped (moe-norm -> moe).

    The block maintains `hc_mult` parallel hidden-state copies. Each sub-layer
    reduces them to 1 via hc_pre, applies its block, then expands back via hc_post.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn = V4Attention(args, layer_idx)
        self.hc_attn = HyperConnection(
            args.hidden_size,
            args.hc_mult,
            args.rms_norm_eps,
            args.hc_sinkhorn_iters,
            args.hc_eps,
        )

        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn = DeepseekV4MoE(args, layer_idx)
        self.hc_ffn = HyperConnection(
            args.hidden_size,
            args.hc_mult,
            args.rms_norm_eps,
            args.hc_sinkhorn_iters,
            args.hc_eps,
        )

    def __call__(self, h: mx.array, mask, cache, input_ids: mx.array) -> mx.array:
        # h: [B, S, hc, D]
        # Attention half
        residual = h
        y, post, comb = self.hc_attn.hc_pre(h)
        y = self.attn_norm(y)
        y = self.attn(y, mask=mask, cache=cache)
        h = self.hc_attn.hc_post(y, residual, post, comb)

        # FFN half
        residual = h
        y, post, comb = self.hc_ffn.hc_pre(h)
        y = self.ffn_norm(y)
        y = self.ffn(y, input_ids)
        h = self.hc_ffn.hc_post(y, residual, post, comb)
        return h


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #


class DeepseekV4Model(nn.Module, PipelineMixin):
    def __init__(self, args: ModelArgs):
        super().__init__()
        PipelineMixin.__init__(self)
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DeepseekV4Block(args, i) for i in range(args.num_hidden_layers)]
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.num_layers = self.end_idx
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Final HC head (reduces hc copies -> 1 before lm_head)
        self.hc_head = HyperHead(
            args.hidden_size, args.hc_mult, args.rms_norm_eps, args.hc_eps
        )

    def __call__(self, inputs: mx.array, cache=None, return_raw_hidden: bool = False):
        h = self.embed_tokens(inputs)  # [B, S, D]
        # Expand to hc_mult parallel copies
        h = mx.broadcast_to(
            h[:, :, None, :], (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2])
        )
        # Make it contiguous — broadcast_to gives a view
        h = mx.contiguous(h)

        if cache is None:
            cache = [None] * self.num_layers

        first_cache = cache[0]
        if isinstance(first_cache, CompressedKVCache):
            first_cache = first_cache.local
        elif isinstance(first_cache, (list, tuple)):
            first_cache = first_cache[0]
        mask = create_attention_mask(
            h[:, :, 0, :],
            first_cache if first_cache is not None else None,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for i in range(self.num_layers):
            h = self.layers[self.start_idx + i](h, mask, cache[i], inputs)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            last_cache = cache[-1]
            if last_cache is not None:
                lc = (
                    last_cache.local
                    if isinstance(last_cache, CompressedKVCache)
                    else last_cache
                )
                if hasattr(lc, "keys") and lc.keys is not None:
                    lc.keys = mx.depends(lc.keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        # Reduce [B,S,hc,D] -> [B,S,D] then RMSNorm
        out = self.norm(self.hc_head(h))
        if return_raw_hidden:
            return out, h
        return out


class Model(nn.Module):
    # Modules routed to the higher bit width by mixed-quant conversion.
    mixed_quant_extra_high = (
        "wq_a",
        "wq_b",
        "wkv",
        "wo_a",
        "wo_b",
        "compressor",
        "indexer",
        "shared_experts",
        "embed_tokens",
    )

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV4Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        return_hidden: bool = False,
    ):
        if return_hidden:
            h, h_raw = self.model(inputs, cache, return_raw_hidden=True)
            return self.lm_head(h), h_raw
        h = self.model(inputs, cache)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers[self.model.start_idx : self.model.end_idx]

    @property
    def cast_predicate(self):
        def pred(k: str):
            # Keep mHC params and gate biases in fp32
            if "hc_" in k or "e_score_correction_bias" in k or "attn_sink" in k:
                return False
            if k.endswith(".fn") or k.endswith(".base") or k.endswith(".scale"):
                return False
            return True

        return pred

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.attn.compress_ratio:
                caches.append(CompressedKVCache(max_size=self.args.sliding_window))
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches

    # ------------------------------------------------------------------- #
    # Weight loading                                                      #
    # ------------------------------------------------------------------- #

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Handle DeepSeek-V4 checkpoint conversion.

        Supports both raw HF checkpoints (FP8/FP4 block scales) and
        pre-quantized MLX checkpoints (e.g. mlx-community 8-bit).

        Checkpoint naming (from HF):
            layers.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}
            layers.N.attn.{q_norm,kv_norm,attn_sink}
            layers.N.attn.compressor.{wkv,wgate,ape,norm}
            layers.N.attn.indexer.{wq_b,weights_proj,compressor.*}
            layers.N.ffn.gate.{weight,bias,tid2eid}
            layers.N.ffn.experts.E.w{1,2,3}.{weight,scale}
            layers.N.ffn.shared_experts.w{1,2,3}.{weight,scale}
            layers.N.{attn_norm,ffn_norm}.weight
            layers.N.hc_{attn,ffn}_{fn,base,scale}
            embed.weight, head.weight, hc_head_{fn,base,scale}
            mtp.0.* (dropped)

        MLX-quantized naming (community 8-bit):
            embed.{weight,biases,scales}, head.{weight,biases,scales}
            layers.N.attn.wo_a.G.{weight,biases,scales} (per-group)
            layers.N.ffn.experts.w{1,2,3}.{weight,biases,scales} (pre-stacked)
        """
        n_layers = self.args.num_hidden_layers

        # 1) Drop MTP weights: this model does base-model inference only, so the
        # multi-token-prediction heads some checkpoints carry are not consumed
        # here. Base layers beyond num_hidden_layers are dropped below.
        new_weights = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    idx = int(parts[1])
                except ValueError:
                    new_weights[k] = v
                    continue
                if idx >= n_layers:
                    continue
            new_weights[k] = v
        weights = new_weights

        def _scale_to_float(scale: mx.array) -> mx.array:
            if scale.dtype == mx.uint8:
                return mx.exp((scale.astype(mx.float32) - 127.0) * math.log(2.0))
            return scale.astype(mx.float32)

        # 2) FP8/FP4 block dequant:
        #    `X.weight` + `X.scale` -> dequantized bf16 `X.weight`
        #    Routed experts in Flash are FP4-packed int8; other scaled matrices
        #    are FP8 e4m3 with 128x128 block scales.
        def _dequant_fp8_block(
            weight: mx.array, scale: mx.array, bs: int = 128
        ) -> mx.array:
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            scale = _scale_to_float(scale)
            m, n = weight.shape
            pad_b = (-m) % bs
            pad_s = (-n) % bs
            weight = mx.pad(weight, ((0, pad_b), (0, pad_s)))
            weight = weight.reshape(((m + pad_b) // bs, bs, (n + pad_s) // bs, bs))
            weight = (weight * scale[:, None, :, None]).reshape(m + pad_b, n + pad_s)
            return weight[:m, :n].astype(mx.bfloat16)

        def _dequant_fp4_block(
            weight: mx.array, scale: mx.array, bs: int = 32
        ) -> mx.array:
            table = mx.array(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ],
                dtype=mx.float32,
            )
            packed = weight.astype(mx.uint8)
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            unpacked = mx.stack([mx.take(table, low), mx.take(table, high)], axis=-1)
            unpacked = unpacked.reshape(weight.shape[0], weight.shape[1] * 2)
            scale = mx.repeat(_scale_to_float(scale), bs, axis=-1)
            return (unpacked * scale).astype(mx.bfloat16)

        new = {}
        for k, v in weights.items():
            if k.endswith(".scale"):
                wk = k[: -len(".scale")] + ".weight"
                weight = weights.get(wk)
                if (
                    weight is not None
                    and ".ffn.experts." in wk
                    and "shared_experts" not in wk
                    and weight.dtype in (mx.int8, mx.uint8)
                    and v.shape[-1] * 16 == weight.shape[-1]
                ):
                    new[wk] = _dequant_fp4_block(weight, v)
                elif weight is not None and weight.dtype in (mx.uint8,):
                    new[wk] = _dequant_fp8_block(weights[wk], v)
                else:
                    new[k] = v
            elif k not in new:
                new[k] = v
        weights = new

        # 3) Remap top-level names to our module structure
        #    Prefix-based remap handles both raw (.weight) and quantized
        #    (.weight, .biases, .scales) checkpoints.
        top_prefix_remap = {
            "embed.": "model.embed_tokens.",
            "head.": "lm_head.",
        }
        top_exact_remap = {
            "norm.weight": "model.norm.weight",
            "hc_head_fn": "model.hc_head.fn",
            "hc_head_base": "model.hc_head.base",
            "hc_head_scale": "model.hc_head.scale",
        }
        new = {}
        for k, v in weights.items():
            nk = k
            for old_pfx, new_pfx in top_prefix_remap.items():
                if nk.startswith(old_pfx):
                    nk = new_pfx + nk[len(old_pfx) :]
                    break
            if nk in top_exact_remap:
                nk = top_exact_remap[nk]
            new[nk] = v
        weights = new

        # 4) Remap layer-level names: layers.N.X -> model.layers.N.X
        #    Also remap gate.bias -> gate.e_score_correction_bias,
        #    hc_{attn,ffn}_{fn,base,scale} -> hc_{attn,ffn}.{fn,base,scale},
        #    shared_experts.w{1,2,3} -> shared_experts.{gate,down,up}_proj
        new = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            nk = k
            # Add model. prefix for main-model layers
            if nk.startswith("layers."):
                nk = "model." + nk

            # gate.bias -> gate.e_score_correction_bias
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")

            # hc_attn_fn -> hc_attn.fn (etc.) — raw HF checkpoint underscores
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".hc_{sub}.{param}")

            # attn_hc.X -> hc_attn.X (mlx-community/DeepSeek-V4-Flash-8bit
            # naming order: per-layer hyper-connections stored as <sub>_hc.<param>
            # rather than hc_<sub>.<param>). Apply after the underscore rename so
            # both naming orders converge to the model's hc_<sub>.<param> layout.
            for sub in ("attn", "ffn"):
                nk = nk.replace(f".{sub}_hc.", f".hc_{sub}.")

            # shared_experts.w1 -> shared_experts.gate_proj (etc.)
            for w_old, w_new in w_remap.items():
                nk = nk.replace(
                    f".shared_experts.{w_old}.", f".shared_experts.{w_new}."
                )

            new[nk] = v
        weights = new

        # 5) Stack expert weights: experts.E.w{1,2,3}.weight -> switch_mlp.{gate,down,up}_proj.weight
        #    Also handle pre-stacked experts (community quants): experts.w{1,2,3}.X -> switch_mlp.{proj}.X
        expert_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for l in range(n_layers):
            prefix = f"model.layers.{l}.ffn.experts"
            for src, dst in expert_remap.items():
                # Case A: per-expert weights need stacking (raw HF checkpoint)
                key0 = f"{prefix}.0.{src}.weight"
                if key0 in weights:
                    stack = [
                        weights.pop(f"{prefix}.{e}.{src}.weight")
                        for e in range(self.args.n_routed_experts)
                    ]
                    weights[f"model.layers.{l}.ffn.switch_mlp.{dst}.weight"] = mx.stack(
                        stack
                    )
                # Case B: already-stacked (community quant) — rename experts.w1.X -> switch_mlp.gate_proj.X
                for suffix in ("weight", "biases", "scales"):
                    old = f"{prefix}.{src}.{suffix}"
                    if old in weights:
                        weights[f"model.layers.{l}.ffn.switch_mlp.{dst}.{suffix}"] = (
                            weights.pop(old)
                        )

        # 6) Fuse split wo_a: community quants store wo_a.G.{weight,biases,scales}
        #    per-group; our model uses a single QuantizedLinear with grouped dequant.
        n_groups = self.args.o_groups
        for l in range(n_layers):
            prefix = f"model.layers.{l}.attn.wo_a"
            if f"{prefix}.0.weight" in weights:
                for suffix in ("weight", "biases", "scales"):
                    parts = []
                    for g in range(n_groups):
                        key = f"{prefix}.{g}.{suffix}"
                        if key in weights:
                            parts.append(weights.pop(key))
                    if parts:
                        weights[f"{prefix}.{suffix}"] = mx.concatenate(parts, axis=0)

        # 6b) Flatten pre-stacked grouped wo_a: some community quants store a
        #     single 3-D wo_a.{weight,scales,biases} of shape
        #     [o_groups, o_lora_rank, X]; our grouped QuantizedLinear expects it
        #     flattened to [o_groups * o_lora_rank, X] (the forward reshapes back).
        for l in range(n_layers):
            prefix = f"model.layers.{l}.attn.wo_a"
            if f"{prefix}.weight" in weights and weights[f"{prefix}.weight"].ndim == 3:
                for suffix in ("weight", "biases", "scales"):
                    key = f"{prefix}.{suffix}"
                    if key in weights and weights[key].ndim == 3:
                        weights[key] = weights[key].reshape(-1, weights[key].shape[-1])

        return weights

    # ------------------------------------------------------------------- #
    # Distributed sharding                                                 #
    # ------------------------------------------------------------------- #

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        R = group.rank()
        for layer in self.model.layers:
            a = layer.attn
            a.wq_b = shard_linear(a.wq_b, "all-to-sharded", group=group)
            a.wo_b = shard_linear(a.wo_b, "sharded-to-all", group=group)
            a.n_heads //= N
            # Slice attn_sink to local heads (mirrors gpt_oss.py:308-312).
            # Order matters: must run AFTER `a.n_heads //= N` so the stride is
            # the post-division (local) head count.
            a.attn_sink = a.attn_sink[a.n_heads * R : a.n_heads * (R + 1)]

            # wo_a: shape (n_groups * o_lora_rank, group_feat).
            # group_feat = n_heads * v_head_dim / n_groups. After sharding,
            # n_heads //= N and n_groups //= N cancel in the ratio, so group_feat
            # stays constant. Only the OUTPUT dim (n_groups axis) gets sharded —
            # each rank owns n_groups//N consecutive groups.
            # wo_b is "sharded-to-all" so its input = n_groups_local * o_lora_rank.
            old_n_groups = a.n_groups
            new_n_groups = old_n_groups // N
            gs = new_n_groups * R
            ge = new_n_groups * (R + 1)
            if isinstance(a.wo_a, nn.QuantizedLinear):
                gf = a.wo_a.weight.shape[-1]
                w = a.wo_a.weight.reshape(old_n_groups, a.o_lora_rank, gf)
                a.wo_a.weight = w[gs:ge].reshape(new_n_groups * a.o_lora_rank, gf)
                sc_gf = a.wo_a.scales.shape[-1]
                s = a.wo_a.scales.reshape(old_n_groups, a.o_lora_rank, sc_gf)
                a.wo_a.scales = s[gs:ge].reshape(new_n_groups * a.o_lora_rank, sc_gf)
                if getattr(a.wo_a, "biases", None) is not None:
                    b_gf = a.wo_a.biases.shape[-1]
                    b = a.wo_a.biases.reshape(old_n_groups, a.o_lora_rank, b_gf)
                    a.wo_a.biases = b[gs:ge].reshape(new_n_groups * a.o_lora_rank, b_gf)
            else:
                gf = a.wo_a.weight.shape[-1]
                w = a.wo_a.weight.reshape(old_n_groups, a.o_lora_rank, gf)
                a.wo_a.weight = w[gs:ge].reshape(new_n_groups * a.o_lora_rank, gf)
            a.n_groups = new_n_groups

            if isinstance(layer.ffn, DeepseekV4MoE):
                layer.ffn.sharding_group = group
                if hasattr(layer.ffn, "shared_experts"):
                    shard_inplace(
                        layer.ffn.shared_experts.gate_proj,
                        "all-to-sharded",
                        group=group,
                    )
                    shard_inplace(
                        layer.ffn.shared_experts.down_proj,
                        "sharded-to-all",
                        group=group,
                    )
                    shard_inplace(
                        layer.ffn.shared_experts.up_proj, "all-to-sharded", group=group
                    )
                shard_inplace(
                    layer.ffn.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.ffn.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.ffn.switch_mlp.up_proj, "all-to-sharded", group=group
                )
