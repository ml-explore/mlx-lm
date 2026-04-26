# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import BatchRotatingKVCache, RotatingKVCache
from .pipeline import PipelineMixin
from .switch_layers import SwitchGLU, _scatter_unsort


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 1.5
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    num_experts_per_tok: int = 6
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 512
    scoring_func: str = "sqrtsoftplus"
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1
    tie_word_embeddings: bool = False
    topk_method: str = "noaux_tc"

    def __post_init__(self):
        if not self.compress_ratios:
            n = self.num_hidden_layers
            self.compress_ratios = (
                [0]
                + [4 if i % 2 else 128 for i in range(max(n - 2, 0))]
                + ([0] if n >= 2 else [])
            )
        self.compress_ratios = list(self.compress_ratios[: self.num_hidden_layers])
        if len(self.compress_ratios) != self.num_hidden_layers:
            raise ValueError(
                "`compress_ratios` must have one entry per hidden layer, "
                f"got {len(self.compress_ratios)} for {self.num_hidden_layers} layers."
            )
        bad = [r for r in self.compress_ratios if r not in (0, 4, 128)]
        if bad:
            raise ValueError(f"Unsupported DeepSeek-V4 compress ratios: {bad}")


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    if func == "sqrtsoftplus":
        return mx.sqrt(mx.logaddexp(scores, 0))
    raise ValueError(f"Unsupported DeepSeek-V4 scoring function: {func}")


@mx.compile
def _expert_select(
    logits: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str,
) -> Tuple[mx.array, mx.array]:
    scores = _score_func(logits, scoring_func)
    biased = scores + e_score_correction_bias
    inds = mx.argpartition(-biased, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights


@mx.compile
def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class LimitedSwiGLU(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        return _limited_swiglu(gate, x, self.limit)


class DeepseekV4SwitchGLU(SwitchGLU):
    sort_threshold = 8

    def __call__(self, x, indices, scores) -> mx.array:
        out_shape = x.shape
        route_shape = indices.shape
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= self.sort_threshold
        inv_order = None
        if do_sort:
            flat_indices = indices.flatten()
            order = mx.argsort(flat_indices)
            inv_order = mx.argsort(order)
            x = x.flatten(0, -3)[order // route_shape[-1]]
            indices = flat_indices[order]
            scores = scores.flatten()[order]
        if self.training:
            indices = mx.stop_gradient(indices)
        x_up = self.up_proj(x, indices, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, indices, sorted_indices=do_sort)
        x = self.activation(x_up, x_gate)
        x = x * scores.astype(x.dtype)[..., None, None]
        x = self.down_proj(x, indices, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, route_shape)

        x = x.squeeze(-2)
        return x.sum(axis=-2).astype(x.dtype).reshape(out_shape)


class DeepseekV4RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float,
        scaling_config: Optional[Dict] = None,
        max_position_embeddings: int = 1048576,
    ):
        super().__init__()
        self.dims = dims

        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
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

            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001

            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

        elif rope_type not in (None, "default"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type: {rope_type}")

        self._inv_freq = (inv_freq,)

    @property
    def inv_freq(self):
        return self._inv_freq[0]

    def __call__(
        self,
        x: mx.array,
        offset: Any = 0,
        inverse: bool = False,
        positions: Optional[mx.array] = None,
    ):
        L = x.shape[-2]
        if positions is None:
            pos = mx.arange(L, dtype=mx.float32)
            if isinstance(offset, mx.array):
                offset = offset.astype(mx.float32)
                if offset.ndim > 0:
                    offset = offset[..., None]
            pos = pos + offset
        else:
            pos = positions.astype(mx.float32)
        return _rope_full(x, self.inv_freq, pos, inverse, 0)


@mx.compile
def _rope_full(
    x: mx.array,
    inv_freq: mx.array,
    pos: mx.array,
    inverse: bool,
    nope_dim: int,
) -> mx.array:
    freqs = pos[..., None] * inv_freq
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    if inverse:
        sin = -sin

    if nope_dim > 0:
        pe = x[..., nope_dim:]
    else:
        pe = x

    if pos.ndim == 1:
        broadcast_shape = (1,) * (pe.ndim - 2) + cos.shape
    else:
        broadcast_shape = (
            pos.shape[:-1] + (1,) * (pe.ndim - pos.ndim - 1) + cos.shape[-2:]
        )
    cos = cos.reshape(broadcast_shape).astype(pe.dtype)
    sin = sin.reshape(broadcast_shape).astype(pe.dtype)

    pe = pe.reshape(*pe.shape[:-1], pe.shape[-1] // 2, 2)
    x0, x1 = pe[..., 0], pe[..., 1]
    out = mx.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
    pe_out = out.reshape(*out.shape[:-2], out.shape[-2] * 2)

    if nope_dim > 0:
        return mx.concatenate([x[..., :nope_dim], pe_out], axis=-1)
    return pe_out


def _apply_partial_rope(
    x: mx.array,
    rope: DeepseekV4RoPE,
    offset: Any = 0,
    inverse: bool = False,
    positions: Optional[mx.array] = None,
) -> mx.array:
    nope_dim = x.shape[-1] - rope.dims if x.shape[-1] != rope.dims else 0
    L = x.shape[-2]
    if positions is None:
        pos = mx.arange(L, dtype=mx.float32)
        if isinstance(offset, mx.array):
            offset = offset.astype(mx.float32)
            if offset.ndim > 0:
                offset = offset[..., None]
        pos = pos + offset
    else:
        pos = positions.astype(mx.float32)
    return _rope_full(x, rope.inv_freq, pos, inverse, nope_dim)


@mx.compile
def _hc_split_sinkhorn_ops(
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    mixes = mixes.astype(mx.float32)
    scale = scale.astype(mx.float32)
    base = base.astype(mx.float32)
    pre_scale, post_scale, comb_scale = scale[0], scale[1], scale[2]

    pre = mx.sigmoid(mixes[..., :hc_mult] * pre_scale + base[:hc_mult]) + eps
    post = 2 * mx.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * post_scale + base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].reshape(
        *mixes.shape[:-1], hc_mult, hc_mult
    ) * comb_scale + base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _make_hc_split_sinkhorn_kernel():
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

        // Pre-sigmoid
        {
            float4 z = *(const device float4*)mix * pre_scale
                     + *(const device float4*)base;
            *(device float4*)pre_out = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
        }

        // Post-sigmoid
        {
            float4 z = *(const device float4*)(mix + HC) * post_scale
                     + *(const device float4*)(base + HC);
            *(device float4*)post_out = 2.0f * 1.0f / (1.0f + metal::fast::exp(-z));
        }

        // Comb: four float4 loads — all independent, GPU issues in parallel
        float4 v0 = *(const device float4*)(mix  + BASE     ) * comb_scale + *(const device float4*)(base + BASE     );
        float4 v1 = *(const device float4*)(mix  + BASE +  4) * comb_scale + *(const device float4*)(base + BASE +  4);
        float4 v2 = *(const device float4*)(mix  + BASE +  8) * comb_scale + *(const device float4*)(base + BASE +  8);
        float4 v3 = *(const device float4*)(mix  + BASE + 12) * comb_scale + *(const device float4*)(base + BASE + 12);

        // Per-row stable softmax: compute all maxes before any exp
        float m0 = metal::max(metal::max(v0.x, v0.y), metal::max(v0.z, v0.w));
        float m1 = metal::max(metal::max(v1.x, v1.y), metal::max(v1.z, v1.w));
        float m2 = metal::max(metal::max(v2.x, v2.y), metal::max(v2.z, v2.w));
        float m3 = metal::max(metal::max(v3.x, v3.y), metal::max(v3.z, v3.w));

        float4 e0 = metal::fast::exp(v0 - m0);
        float4 e1 = metal::fast::exp(v1 - m1);
        float4 e2 = metal::fast::exp(v2 - m2);
        float4 e3 = metal::fast::exp(v3 - m3);

        // Explicit adds instead of dot(e, 1) — avoids unnecessary fmul
        float4 r0 = e0 * 1.0f / (e0.x + e0.y + e0.z + e0.w) + epsv;
        float4 r1 = e1 * 1.0f / (e1.x + e1.y + e1.z + e1.w) + epsv;
        float4 r2 = e2 * 1.0f / (e2.x + e2.y + e2.z + e2.w) + epsv;
        float4 r3 = e3 * 1.0f / (e3.x + e3.y + e3.z + e3.w) + epsv;

        // Initial column normalization
        float4 col = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col; r1 *= col; r2 *= col; r3 *= col;

        // Sinkhorn iterations
        for (int iter = 1; iter < ITERS; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);
            col = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col; r1 *= col; r2 *= col; r3 *= col;
        }

        // Write comb output (four aligned 128-bit stores)
        *(device float4*)(comb_out)      = r0;
        *(device float4*)(comb_out +  4) = r1;
        *(device float4*)(comb_out +  8) = r2;
        *(device float4*)(comb_out + 12) = r3;
    """

    return mx.fast.metal_kernel(
        name="deepseek_v4_hc_split_sinkhorn",
        input_names=["mixes", "scale", "base", "eps"],
        output_names=["pre", "post", "comb"],
        source=source,
    )


_hc_split_sinkhorn_kernel = _make_hc_split_sinkhorn_kernel()


def hc_split_sinkhorn(
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    if _hc_split_sinkhorn_kernel is None or hc_mult != 4:
        return _hc_split_sinkhorn_ops(mixes, scale, base, hc_mult, sinkhorn_iters, eps)

    if not isinstance(eps, mx.array):
        eps = mx.array([eps], dtype=mx.float32)
    n_rows = mixes.size // ((2 + hc_mult) * hc_mult)
    return _hc_split_sinkhorn_kernel(
        inputs=[mixes, scale, base, eps],
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


@mx.compile
def _hc_collapse_op(pre: mx.array, x: mx.array) -> mx.array:
    return (pre[..., None] * x.astype(mx.float32)).sum(axis=2).astype(x.dtype)


def _make_hc_sinkhorn_collapse_kernel():
    """Fused sinkhorn + collapse: eliminates one dispatch per HC cycle.

    1. BRANCHLESS SINKHORN: all 32 lanes in simd group 0 execute identical
       instructions. Lanes >= HC use multiplicative mask (active=0) instead
       of divergent branches — eliminates SIMD serialization.
    2. PARALLEL SINKHORN: lanes 0-3 each own one comb row. Column norm
       via simd_sum() — free SIMD shuffle.
    3. NATIVE bfloat4 LOADS: single 64-bit load yields 4 bfloat16 values;
       cast to float4 is a free hardware conversion.
    4. FMA CHAINS: collapse uses fused multiply-add for 3 of 4 terms.
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid  = thread_position_in_threadgroup.x;
        uint row  = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg   = tid / 32;

        constexpr int MIX      = (2 + HC) * HC;
        constexpr int BASE_OFF = 2 * HC;

        const device float* mix      = (const device float*)mixes + row * MIX;
        device float*       post_out = (device float*)post + row * HC;
        device float*       comb_out = (device float*)comb + row * HC * HC;

        threadgroup float pre_shared[HC];

        // ================================================================
        // PHASE 1: Branchless sinkhorn on simd group 0
        //   All 32 lanes execute identical instructions. Lanes >= HC
        //   compute on clamped indices but multiply by active=0, so they
        //   contribute zero to simd_sum. No divergent branches in the loop.
        // ================================================================
        if (sg == 0) {
            const float pre_scale  = scale[0];
            const float post_scale = scale[1];
            const float comb_scale = scale[2];
            const float epsv       = eps[0];

            const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
            const uint  llane  = metal::min(lane, (uint)(HC - 1));

            // Pre/post sigmoids: all lanes compute, only active lanes write
            float pre_z  = mix[llane]      * pre_scale  + base[llane];
            float post_z = mix[HC + llane] * post_scale + base[HC + llane];
            float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + epsv;
            float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

            if (lane < (uint)HC) {
                pre_shared[lane] = pre_v;
                post_out[lane]   = post_v;
            }

            // Comb softmax: load + mask. Inactive lanes load row 0 (safe)
            // but multiply by active=0 so they hold zeros.
            float4 v = (*(const device float4*)(mix  + BASE_OFF + llane * HC)
                            * comb_scale
                      + *(const device float4*)(base + BASE_OFF + llane * HC))
                     * active;

            float row_max = metal::max(metal::max(v.x, v.y),
                                       metal::max(v.z, v.w));
            float4 e = metal::fast::exp(v - row_max) * active;
            float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + epsv))
                     + epsv * active;

            // Initial column normalization
            float4 col_inv = 1.0f / (float4(
                simd_sum(r.x), simd_sum(r.y),
                simd_sum(r.z), simd_sum(r.w)
            ) + epsv);
            r *= col_inv;

            // Sinkhorn iterations: zero branches in the loop body
            for (int iter = 1; iter < ITERS; ++iter) {
                // Row norm + re-clamp inactive lanes
                r *= (1.0f / (r.x + r.y + r.z + r.w + epsv)) * active;

                // Col norm via simd_sum
                col_inv = 1.0f / (float4(
                    simd_sum(r.x), simd_sum(r.y),
                    simd_sum(r.z), simd_sum(r.w)
                ) + epsv);
                r *= col_inv;
            }

            if (lane < (uint)HC) {
                *(device float4*)(comb_out + lane * HC) = r;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // PHASE 2: Collapse — all 256 threads, native bfloat4 vectorized
        // ================================================================
        const float p0 = pre_shared[0];
        const float p1 = pre_shared[1];
        const float p2 = pre_shared[2];
        const float p3 = pre_shared[3];

        const device bfloat16_t* x_row  = (const device bfloat16_t*)x_in
                                         + row * (HC * D);
        device bfloat16_t*       out_row = (device bfloat16_t*)collapsed
                                         + row * D;

        // Native bfloat4 pointers: single 64-bit load per vector
        using bf4 = vec<bfloat16_t, 4>;
        const device bf4* x_row0 = (const device bf4*)(x_row + 0*D);
        const device bf4* x_row1 = (const device bf4*)(x_row + 1*D);
        const device bf4* x_row2 = (const device bf4*)(x_row + 2*D);
        const device bf4* x_row3 = (const device bf4*)(x_row + 3*D);
        device bf4*       out4   = (device bf4*)out_row;

        constexpr uint D4 = (uint)D / 4;

        for (uint d4 = tid; d4 < D4; d4 += 256) {
            float4 x0 = float4(x_row0[d4]);
            float4 x1 = float4(x_row1[d4]);
            float4 x2 = float4(x_row2[d4]);
            float4 x3 = float4(x_row3[d4]);

            float4 result = fma(float4(p0), x0,
                            fma(float4(p1), x1,
                            fma(float4(p2), x2, float4(p3) * x3)));

            out4[d4] = bf4(result);
        }

        // Scalar tail for D not divisible by 4
        #if (D % 4) != 0
        for (uint d = D4 * 4 + tid; d < (uint)D; d += 256) {
            float val = p0*(float)x_row[0*D+d] + p1*(float)x_row[1*D+d]
                      + p2*(float)x_row[2*D+d] + p3*(float)x_row[3*D+d];
            out_row[d] = (bfloat16_t)val;
        }
        #endif
    """

    return mx.fast.metal_kernel(
        name="deepseek_v4_hc_sinkhorn_collapse",
        input_names=["mixes", "scale", "base", "eps", "x_in"],
        output_names=["post", "comb", "collapsed"],
        source=source,
    )


_hc_sinkhorn_collapse_kernel = _make_hc_sinkhorn_collapse_kernel()


@mx.compile
def _hc_expand_op(
    post: mx.array,
    block_out: mx.array,
    comb: mx.array,
    residual: mx.array,
) -> mx.array:
    y = post[..., None] * block_out[:, :, None, :].astype(mx.float32)
    y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))
    return y.astype(block_out.dtype)


@mx.compile
def _rms_rsqrt(flat: mx.array, eps: float) -> mx.array:
    return mx.rsqrt((flat * flat).mean(axis=-1, keepdims=True) + eps)


@mx.compile
def _hc_mixes(flat: mx.array, fn_T: mx.array, norm_eps: float) -> mx.array:
    """Fused RMS-rsqrt + matmul + scale into single compiled graph."""
    rsqrt = mx.rsqrt((flat * flat).mean(axis=-1, keepdims=True) + norm_eps)
    return (flat @ fn_T) * rsqrt


class HyperConnection(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self._hc_eps = (mx.array([config.hc_eps], dtype=mx.float32),)
        self.norm_eps = config.rms_norm_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = mx.zeros((mix, self.hc_mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)
        self._fn_T = None

    def compute_weights(self, x: mx.array):
        B, L, H, D = x.shape
        flat = x.reshape(B, L, H * D).astype(mx.float32)
        if self._fn_T is None:
            self._fn_T = self.fn.T
        if self.training:
            rsqrt = _rms_rsqrt(flat, self.norm_eps)
            mixes = (flat @ self._fn_T) * rsqrt
        else:
            mixes = _hc_mixes(flat, self._fn_T, self.norm_eps)
        split_sinkhorn = _hc_split_sinkhorn_ops if self.training else hc_split_sinkhorn
        return split_sinkhorn(
            mixes,
            self.scale,
            self.base,
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps if self.training else self._hc_eps[0],
        )

    def collapse(self, x: mx.array):
        if (
            not self.training
            and _hc_sinkhorn_collapse_kernel is not None
            and self.hc_mult == 4
            and x.dtype == mx.bfloat16
        ):
            return self._fused_collapse(x)
        pre, post, comb = self.compute_weights(x)
        return _hc_collapse_op(pre, x), post, comb

    def _fused_collapse(self, x: mx.array):
        """Fused sinkhorn + collapse in a single Metal kernel dispatch."""
        B, L, H, D = x.shape
        flat = x.reshape(B, L, H * D).astype(mx.float32)
        if self._fn_T is None:
            self._fn_T = self.fn.T
        mixes = _hc_mixes(flat, self._fn_T, self.norm_eps)

        eps = self._hc_eps[0]
        n_rows = B * L
        x_flat = mx.contiguous(x.reshape(n_rows, H, D))

        post, comb, collapsed = _hc_sinkhorn_collapse_kernel(
            inputs=[mixes, self.scale, self.base, eps, x_flat],
            template=[("HC", self.hc_mult), ("ITERS", self.sinkhorn_iters), ("D", D)],
            grid=(n_rows * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[
                (*mixes.shape[:-1], self.hc_mult),
                (*mixes.shape[:-1], self.hc_mult, self.hc_mult),
                (B, L, D),
            ],
            output_dtypes=[mx.float32, mx.float32, x.dtype],
        )
        return collapsed, post, comb

    def expand(
        self,
        block_out: mx.array,
        residual: mx.array,
        post: mx.array,
        comb: mx.array,
    ):
        return _hc_expand_op(post, block_out, comb, residual)


@mx.compile
def _hyper_head_op(
    x: mx.array,
    fn: mx.array,
    scale: mx.array,
    base: mx.array,
    norm_eps: float,
    hc_eps: float,
) -> mx.array:
    """Fused HyperHead: RMS-rsqrt + matmul + sigmoid + weighted sum."""
    B, L, H, D = x.shape
    flat = x.reshape(B, L, H * D).astype(mx.float32)
    rsqrt = mx.rsqrt((flat * flat).mean(axis=-1, keepdims=True) + norm_eps)
    mixes = (flat @ fn.T) * rsqrt
    pre = mx.sigmoid(mixes * scale[0] + base) + hc_eps
    return (pre[..., None] * x.astype(mx.float32)).sum(axis=2).astype(x.dtype)


class HyperHead(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.fn = mx.zeros(
            (self.hc_mult, self.hc_mult * config.hidden_size), dtype=mx.float32
        )
        self.base = mx.zeros((self.hc_mult,), dtype=mx.float32)
        self.scale = mx.ones((1,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        if not self.training:
            return _hyper_head_op(
                x, self.fn, self.scale, self.base, self.norm_eps, self.hc_eps
            )
        B, L, H, D = x.shape
        flat = x.reshape(B, L, H * D).astype(mx.float32)
        rsqrt = _rms_rsqrt(flat, self.norm_eps)
        mixes = (flat @ self.fn.T) * rsqrt
        pre = mx.sigmoid(mixes * self.scale[0] + self.base) + self.hc_eps
        return (pre[..., None] * x.astype(mx.float32)).sum(axis=2).astype(x.dtype)


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.hash = layer_idx < config.num_hash_layers
        self.scoring_func = config.scoring_func
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = mx.zeros((self.num_experts, self.hidden_dim))
        self._weight_T_f32 = None
        if self.hash:
            self.tid2eid = mx.zeros((config.vocab_size, self.top_k), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros(
                (self.num_experts,), dtype=mx.float32
            )

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        flat = x.reshape(-1, self.hidden_dim)
        if self._weight_T_f32 is None:
            self._weight_T_f32 = self.weight.T.astype(mx.float32)
        logits = flat.astype(mx.float32) @ self._weight_T_f32

        if self.hash:
            if input_ids is None:
                raise ValueError("DeepSeek-V4 hash routing requires input_ids.")
            scores = _score_func(logits, self.scoring_func)
            inds = self.tid2eid[input_ids.reshape(-1)].astype(mx.int32)
            weights = mx.take_along_axis(scores, inds, axis=-1)
            if self.scoring_func != "softmax" and self.norm_topk_prob:
                weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
            weights = weights * self.routed_scaling_factor
        else:
            inds, weights = _expert_select(
                logits,
                self.e_score_correction_bias,
                self.top_k,
                self.routed_scaling_factor,
                self.norm_topk_prob,
                self.scoring_func,
            )

        route_shape = (*x.shape[:-1], self.top_k)
        inds = inds.reshape(route_shape)
        weights = weights.reshape(route_shape)
        return inds, weights


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        intermediate_size: Optional[int] = None,
        swiglu_limit: float = 0.0,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


class DeepseekV4MoE(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.gate = MoEGate(config, layer_idx)
        self.switch_mlp = DeepseekV4SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=LimitedSwiGLU(config.swiglu_limit),
        )
        self.shared_experts = DeepseekV4MLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )
        self.sharding_group = None

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = self.gate(x, input_ids)
        y = self.switch_mlp(x, inds, scores)
        y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class DeepseekV4Cache:
    _state_keys = ("buffer_kv", "buffer_gate", "pooled")
    _length_keys = ("buffer_lengths", "pooled_lengths")

    def __init__(self, sliding_window: int):
        self.local = RotatingKVCache(max_size=sliding_window, keep=0)
        self.compressor_state = self._new_branch_state()
        self.indexer_state = self._new_branch_state()
        self._pending_lengths = None

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
    def state(self):
        local_state = None if self.local.empty() else self.local.state
        return (
            local_state,
            self._branch_state_tuple(self.compressor_state),
            self._branch_state_tuple(self.indexer_state),
        )

    @state.setter
    def state(self, value):
        local_state, compressor_state, indexer_state = value
        if local_state is None:
            self.local.keys = None
            self.local.values = None
        else:
            self.local.state = local_state
        self.compressor_state = self._state_from_tuple(compressor_state)
        self.indexer_state = self._state_from_tuple(indexer_state)

    @property
    def meta_state(self):
        return self.local.meta_state

    @meta_state.setter
    def meta_state(self, value):
        self.local.meta_state = value

    def update_and_fetch(self, keys, values):
        return self.local.update_and_fetch(keys, values)

    def make_mask(self, *args, **kwargs):
        return self.local.make_mask(*args, **kwargs)

    def is_trimmable(self):
        if not self.local.is_trimmable():
            return False
        for state in (self.compressor_state, self.indexer_state):
            pooled = state["pooled"]
            if pooled is not None and pooled.shape[1] > 0:
                return False
        return True

    def trim(self, n):
        trimmed = self.local.trim(n)
        batch_size = self._cache_batch_size(self.local)
        for state in (self.compressor_state, self.indexer_state):
            for key in ("buffer_kv", "buffer_gate"):
                value = state[key]
                if value is not None:
                    state[key] = value[:, : max(value.shape[1] - trimmed, 0)]
            lengths = state["buffer_lengths"]
            if lengths is not None:
                state["buffer_lengths"] = [
                    max(length - trimmed, 0)
                    for length in self._lengths_list(lengths, batch_size, 0)
                ]
        return trimmed

    def size(self):
        return self.local.size()

    def empty(self):
        return self.local.empty()

    @property
    def nbytes(self):
        total = self.local.nbytes
        for state in (self.compressor_state, self.indexer_state):
            for value in state.values():
                if value is not None and hasattr(value, "nbytes"):
                    total += value.nbytes
        return total

    def _branch_state(self, state_key: str):
        return (
            self.indexer_state
            if state_key == "indexer_state"
            else self.compressor_state
        )

    @classmethod
    def _new_branch_state(cls):
        return {key: None for key in cls._state_keys + cls._length_keys}

    @classmethod
    def _branch_state_tuple(cls, state):
        return tuple(state[k] for k in cls._state_keys + cls._length_keys)

    @classmethod
    def _state_from_tuple(cls, values):
        keys = cls._state_keys + cls._length_keys
        state = cls._new_branch_state()
        state.update(zip(keys, values))
        return state

    @staticmethod
    def _lengths_list(lengths, batch_size: int, default: Optional[int] = None):
        if lengths is None:
            if default is None:
                return None
            return [default] * batch_size
        if isinstance(lengths, mx.array):
            lengths = lengths.tolist()
        return [int(length) for length in lengths]

    @staticmethod
    def _filter_lengths(lengths, batch_indices):
        if lengths is None:
            return None
        if isinstance(lengths, mx.array):
            lengths = lengths.tolist()
        if isinstance(batch_indices, mx.array):
            batch_indices = batch_indices.tolist()
        if len(lengths) == 1 and any(idx != 0 for idx in batch_indices):
            lengths = lengths * (max(batch_indices) + 1)
        return [int(lengths[idx]) for idx in batch_indices]

    def accumulate_windows(
        self,
        kv: mx.array,
        gate: mx.array,
        state_key: str,
        ratio: int,
        start_pos: int,
    ):
        state = self._branch_state(state_key)
        buf_kv, buf_gate = state["buffer_kv"], state["buffer_gate"]
        B, L = kv.shape[:2]
        buf_lengths = self._lengths_list(state["buffer_lengths"], B)
        chunk_lengths = self._pending_lengths
        if buf_lengths is None and chunk_lengths is None:
            if buf_kv is not None and buf_kv.shape[1]:
                kv = mx.concatenate([buf_kv, kv], axis=1)
                gate = mx.concatenate([buf_gate, gate], axis=1)
            usable = (kv.shape[1] // ratio) * ratio
            state["buffer_kv"] = kv[:, usable:]
            state["buffer_gate"] = gate[:, usable:]
            state["buffer_lengths"] = None
            state["_new_pooled_lengths"] = None
            if isinstance(start_pos, mx.array):
                pool_base = mx.maximum(start_pos, 0)
            else:
                pool_base = max(0, start_pos)
            pool_base = pool_base - (buf_kv.shape[1] if buf_kv is not None else 0)
            return kv[:, :usable], gate[:, :usable], pool_base

        buf_lengths = self._lengths_list(
            state["buffer_lengths"],
            B,
            0 if buf_kv is None else buf_kv.shape[1],
        )
        chunk_lengths = self._lengths_list(chunk_lengths, B, L)
        total_lengths = [
            buf_length + min(chunk_length, L)
            for buf_length, chunk_length in zip(buf_lengths, chunk_lengths)
        ]
        usable_lengths = [(length // ratio) * ratio for length in total_lengths]
        buffer_lengths = [
            length - usable for length, usable in zip(total_lengths, usable_lengths)
        ]
        max_total = max(total_lengths, default=0)
        max_usable = max(usable_lengths, default=0)
        max_buffer = max(buffer_lengths, default=0)

        combined_kv = mx.zeros((B, max_total, kv.shape[-1]), dtype=kv.dtype)
        combined_gate = mx.zeros((B, max_total, gate.shape[-1]), dtype=gate.dtype)
        for i, (buf_length, chunk_length, total_length) in enumerate(
            zip(buf_lengths, chunk_lengths, total_lengths)
        ):
            parts_kv = []
            parts_gate = []
            if buf_length:
                parts_kv.append(buf_kv[i : i + 1, :buf_length])
                parts_gate.append(buf_gate[i : i + 1, :buf_length])
            if chunk_length:
                parts_kv.append(kv[i : i + 1, : min(chunk_length, L)])
                parts_gate.append(gate[i : i + 1, : min(chunk_length, L)])
            if parts_kv:
                row_kv = (
                    parts_kv[0]
                    if len(parts_kv) == 1
                    else mx.concatenate(parts_kv, axis=1)
                )
                row_gate = (
                    parts_gate[0]
                    if len(parts_gate) == 1
                    else mx.concatenate(parts_gate, axis=1)
                )
                combined_kv[i : i + 1, :total_length] = row_kv
                combined_gate[i : i + 1, :total_length] = row_gate

        ready_kv = combined_kv[:, :max_usable]
        ready_gate = combined_gate[:, :max_usable]
        state["buffer_kv"] = mx.zeros((B, max_buffer, kv.shape[-1]), dtype=kv.dtype)
        state["buffer_gate"] = mx.zeros(
            (B, max_buffer, gate.shape[-1]), dtype=gate.dtype
        )
        for i, (usable, buffer_length) in enumerate(
            zip(usable_lengths, buffer_lengths)
        ):
            if buffer_length:
                state["buffer_kv"][i : i + 1, :buffer_length] = combined_kv[
                    i : i + 1, usable : usable + buffer_length
                ]
                state["buffer_gate"][i : i + 1, :buffer_length] = combined_gate[
                    i : i + 1, usable : usable + buffer_length
                ]
        state["buffer_lengths"] = buffer_lengths
        state["_new_pooled_lengths"] = [usable // ratio for usable in usable_lengths]

        prev_lengths = mx.array(buf_lengths, dtype=mx.float32)
        if isinstance(start_pos, mx.array):
            pool_base = mx.maximum(start_pos, 0).astype(mx.float32)
        else:
            pool_base = mx.full((B,), max(0, start_pos), dtype=mx.float32)
        return ready_kv, ready_gate, pool_base - prev_lengths

    def update_pool(self, new_pooled: mx.array, state_key: str) -> mx.array:
        state = self._branch_state(state_key)
        new_lengths = state.pop("_new_pooled_lengths", None)
        pool = state["pooled"]
        if new_lengths is not None:
            B = new_pooled.shape[0]
            pool_lengths = self._lengths_list(
                state["pooled_lengths"],
                B,
                0 if pool is None else pool.shape[1],
            )
            total_lengths = [
                pool_length + new_length
                for pool_length, new_length in zip(pool_lengths, new_lengths)
            ]
            max_total = max(total_lengths, default=0)
            merged = mx.zeros(
                (B, max_total, new_pooled.shape[-1]), dtype=new_pooled.dtype
            )
            for i, (pool_length, new_length) in enumerate(
                zip(pool_lengths, new_lengths)
            ):
                if pool is not None and pool_length:
                    merged[i : i + 1, :pool_length] = pool[i : i + 1, :pool_length]
                if new_length:
                    merged[i : i + 1, pool_length : pool_length + new_length] = (
                        new_pooled[i : i + 1, :new_length]
                    )
            state["pooled"] = merged
            state["pooled_lengths"] = total_lengths
            return merged

        if new_pooled.shape[1] > 0:
            pool = (
                new_pooled
                if pool is None
                else mx.concatenate([pool, new_pooled], axis=1)
            )
            state["pooled"] = pool
            state["pooled_lengths"] = None
        if pool is None:
            pool = mx.zeros(
                (new_pooled.shape[0], 0, new_pooled.shape[-1]), new_pooled.dtype
            )
        return pool

    def pooled_lengths(self, state_key: str):
        return self._branch_state(state_key)["pooled_lengths"]

    def prepare(self, *args, **kwargs):
        lengths = kwargs.get("lengths")
        right_padding = kwargs.get("right_padding")
        self._pending_lengths = (
            list(lengths)
            if right_padding is not None and max(right_padding) > 0
            else None
        )
        if hasattr(self.local, "prepare"):
            self.local.prepare(*args, **kwargs)
            return

        left_padding = kwargs.get("left_padding")
        if left_padding is not None or (
            right_padding is not None and max(right_padding) > 0
        ):
            batch_size = (
                len(left_padding)
                if left_padding is not None
                else len(kwargs.get("lengths"))
            )
            self.local = BatchRotatingKVCache(self.local.max_size, [0] * batch_size)
            self.local.prepare(*args, **kwargs)

    def finalize(self):
        if hasattr(self.local, "finalize"):
            self.local.finalize()
        self._pending_lengths = None

    def filter(self, batch_indices):
        if hasattr(self.local, "filter"):
            self.local.filter(batch_indices)
        elif self.local.keys is not None:
            self.local.keys = self.local.keys[batch_indices]
            self.local.values = self.local.values[batch_indices]
        for state in (self.compressor_state, self.indexer_state):
            for key in self._state_keys:
                value = state[key]
                if value is not None:
                    state[key] = value[batch_indices]
            for key in self._length_keys:
                state[key] = self._filter_lengths(state[key], batch_indices)

    def extend(self, other):
        self_batch = self._cache_batch_size(self.local)
        other_batch = self._cache_batch_size(other.local)
        if hasattr(self.local, "extend"):
            other_local = other.local
            if not hasattr(other_local, "filter"):
                other_local = self._batch_rotating_from_local(other_local)
            self.local.extend(other_local)
        elif (
            not hasattr(other.local, "filter")
            and self.local.offset == other.local.offset
            and self.local._idx == other.local._idx
        ):
            if self.local.keys is not None or other.local.keys is not None:
                self.local.keys = self._concat_optional_local(
                    self.local.keys, other.local.keys
                )
                self.local.values = self._concat_optional_local(
                    self.local.values, other.local.values
                )
        else:
            self.local = self._batch_rotating_from_local(self.local)
            other_local = (
                other.local
                if hasattr(other.local, "filter")
                else self._batch_rotating_from_local(other.local)
            )
            self.local.extend(other_local)
        for self_state, other_state in (
            (self.compressor_state, other.compressor_state),
            (self.indexer_state, other.indexer_state),
        ):
            self_tensors = {key: self_state[key] for key in self._state_keys}
            other_tensors = {key: other_state[key] for key in self._state_keys}
            for key in self._state_keys:
                self_state[key] = self._concat_batch_state(
                    self_state[key], other_state[key], self_batch, other_batch
                )
            for key in self._length_keys:
                self_state[key] = self._concat_lengths(
                    self_state[key],
                    other_state[key],
                    self_tensors["pooled" if key.startswith("pooled") else "buffer_kv"],
                    other_tensors[
                        "pooled" if key.startswith("pooled") else "buffer_kv"
                    ],
                    self_batch,
                    other_batch,
                )

    def extract(self, idx):
        cache = DeepseekV4Cache(self.local.max_size)
        cache.local = (
            self.local.extract(idx)
            if hasattr(self.local, "extract")
            else self._extract_local(self.local, idx)
        )
        for cache_state, self_state in (
            (cache.compressor_state, self.compressor_state),
            (cache.indexer_state, self.indexer_state),
        ):
            for key in self._state_keys:
                value = self_state[key]
                cache_state[key] = (
                    None if value is None else mx.contiguous(value[idx : idx + 1])
                )
            for key in self._length_keys:
                lengths = self_state[key]
                if lengths is None:
                    cache_state[key] = None
                elif isinstance(lengths, mx.array):
                    cache_state[key] = [int(lengths.tolist()[idx])]
                else:
                    cache_state[key] = [int(lengths[idx])]
        return cache

    @classmethod
    def merge(cls, caches):
        if not all(c.local.max_size == caches[0].local.max_size for c in caches):
            raise ValueError(
                "DeepseekV4Cache can only merge caches with the same sliding window"
            )

        cache = cls(caches[0].local.max_size)
        cache.local = cls._merge_local([c.local for c in caches])
        for cache_state, state_name in (
            (cache.compressor_state, "compressor_state"),
            (cache.indexer_state, "indexer_state"),
        ):
            for key in cls._state_keys:
                cache_state[key] = cls._merge_batch_state(
                    [getattr(c, state_name)[key] for c in caches]
                )
            for key in cls._length_keys:
                tensor_key = "pooled" if key.startswith("pooled") else "buffer_kv"
                cache_state[key] = cls._merge_lengths(
                    [getattr(c, state_name)[key] for c in caches],
                    [getattr(c, state_name)[tensor_key] for c in caches],
                )
        return cache

    @staticmethod
    def _cache_batch_size(cache):
        offset = getattr(cache, "offset", 0)
        if isinstance(offset, mx.array) and offset.ndim:
            return offset.shape[0]
        if cache.keys is not None:
            return cache.keys.shape[0]
        return 1

    @staticmethod
    def _extract_local(local, idx):
        cache = RotatingKVCache(local.max_size, keep=getattr(local, "keep", 0))
        if local.keys is not None:
            keys = local._temporal_order(local.keys)
            values = local._temporal_order(local.values)
            cache.keys = mx.contiguous(keys[idx : idx + 1])
            cache.values = mx.contiguous(values[idx : idx + 1])
            cache._idx = cache.keys.shape[2]
        cache.offset = local.offset
        return cache

    @classmethod
    def _batch_rotating_from_local(cls, local):
        batch_size = cls._cache_batch_size(local)
        return BatchRotatingKVCache.merge(
            [cls._extract_local(local, idx) for idx in range(batch_size)]
        )

    @staticmethod
    def _concat_optional_local(a: Optional[mx.array], b: Optional[mx.array]):
        if a is None:
            return b
        if b is None:
            return a
        return mx.concatenate([a, b], axis=0)

    @classmethod
    def _merge_local(cls, locals):
        offsets = [local.offset for local in locals]
        sizes = [local.size() for local in locals]
        use_fast_local = (
            all(not isinstance(offset, mx.array) for offset in offsets)
            and all(offset == offsets[0] for offset in offsets)
            and all(size == sizes[0] for size in sizes)
        )
        if not use_fast_local:
            if hasattr(locals[0], "merge"):
                return locals[0].merge(locals)
            return BatchRotatingKVCache.merge(locals)

        cache = RotatingKVCache(locals[0].max_size, keep=getattr(locals[0], "keep", 0))
        cache.offset = offsets[0]
        if sizes[0] == 0:
            return cache

        cache.keys = mx.concatenate(
            [local._temporal_order(local.keys) for local in locals], axis=0
        )
        cache.values = mx.concatenate(
            [local._temporal_order(local.values) for local in locals], axis=0
        )
        cache._idx = cache.keys.shape[2]
        return cache

    @staticmethod
    def _concat_batch_state(
        a: Optional[mx.array],
        b: Optional[mx.array],
        a_batch: int,
        b_batch: int,
    ):
        if a is None and b is None:
            return None
        if a is None:
            shape = (a_batch, *b.shape[1:])
            a = mx.zeros(shape, dtype=b.dtype)
        if b is None:
            shape = (b_batch, *a.shape[1:])
            b = mx.zeros(shape, dtype=a.dtype)
        if a.shape[2:] != b.shape[2:]:
            raise ValueError(
                "Cannot extend DeepseekV4Cache entries with different state shapes"
            )
        if a.shape[1] != b.shape[1]:
            seq_len = max(a.shape[1], b.shape[1])
            if a.shape[1] != seq_len:
                padded = mx.zeros((a.shape[0], seq_len, *a.shape[2:]), dtype=a.dtype)
                padded[:, : a.shape[1]] = a
                a = padded
            if b.shape[1] != seq_len:
                padded = mx.zeros((b.shape[0], seq_len, *b.shape[2:]), dtype=b.dtype)
                padded[:, : b.shape[1]] = b
                b = padded
        return mx.concatenate([a, b], axis=0)

    @staticmethod
    def _full_lengths(lengths, value: Optional[mx.array], batch_size: int):
        if lengths is not None:
            if isinstance(lengths, mx.array):
                lengths = lengths.tolist()
            return [int(length) for length in lengths]
        length = 0 if value is None else value.shape[1]
        return [length] * batch_size

    @classmethod
    def _concat_lengths(
        cls,
        a_lengths,
        b_lengths,
        a_value: Optional[mx.array],
        b_value: Optional[mx.array],
        a_batch: int,
        b_batch: int,
    ):
        a_lengths = cls._full_lengths(a_lengths, a_value, a_batch)
        b_lengths = cls._full_lengths(b_lengths, b_value, b_batch)
        lengths = a_lengths + b_lengths
        max_length = max(lengths, default=0)
        return None if all(length == max_length for length in lengths) else lengths

    @classmethod
    def _merge_lengths(cls, lengths, values):
        batch_lengths = [
            cls._full_lengths(length, value, 1)[0]
            for length, value in zip(lengths, values)
        ]
        max_length = max(batch_lengths, default=0)
        return (
            None
            if all(length == max_length for length in batch_lengths)
            else batch_lengths
        )

    @staticmethod
    def _merge_batch_state(values: List[Optional[mx.array]]):
        present = [v for v in values if v is not None]
        if not present:
            return None
        if not all(v.shape[2:] == present[0].shape[2:] for v in present):
            raise ValueError(
                "Cannot batch DeepseekV4Cache entries with different state shapes"
            )
        seq_len = max(v.shape[1] for v in present)
        shape = present[0].shape
        dtype = present[0].dtype
        merged = []
        for value in values:
            if value is None:
                merged.append(mx.zeros((1, seq_len, *shape[2:]), dtype=dtype))
            else:
                if value.shape[1] != seq_len:
                    padded = mx.zeros(
                        (value.shape[0], seq_len, *value.shape[2:]), dtype=value.dtype
                    )
                    padded[:, : value.shape[1]] = value
                    value = padded
                merged.append(value)
        return mx.concatenate(merged, axis=0)


class Compressor(nn.Module):

    def __init__(self, config: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.overlap = compress_ratio == 4
        self.out_dim = head_dim * (2 if self.overlap else 1)
        self.wkv = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, self.out_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)

    def _overlap_transform(self, x: mx.array, fill_value: float):
        B, W, R, _ = x.shape
        second_half = x[:, :, :, self.head_dim :]  # (B, W, R, head_dim)
        fill_row = mx.full((B, 1, R, self.head_dim), fill_value, dtype=x.dtype)
        prev_first = mx.concatenate(
            [fill_row, x[:, :-1, :, : self.head_dim]], axis=1
        )  # (B, W, R, head_dim)
        return mx.concatenate([prev_first, second_half], axis=2)  # (B, W, 2R, head_dim)

    def __call__(
        self,
        x: mx.array,
        rope: DeepseekV4RoPE,
        cache: Optional[DeepseekV4Cache],
        start_pos: int,
        state_key: str = "compressor_state",
    ) -> mx.array:
        B, _, _ = x.shape
        kv = self.wkv(x)
        gate = self.wgate(x)
        if cache is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            ready_kv, ready_gate = kv[:, :usable], gate[:, :usable]
            pool_base = start_pos
        else:
            ready_kv, ready_gate, pool_base = cache.accumulate_windows(
                kv, gate, state_key, self.compress_ratio, start_pos
            )

        if ready_kv.shape[1] == 0:
            new_pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        else:
            W = ready_kv.shape[1] // self.compress_ratio
            kv = ready_kv.reshape(B, W, self.compress_ratio, self.out_dim)
            gate = ready_gate.reshape(
                B, W, self.compress_ratio, self.out_dim
            ) + self.ape.astype(ready_gate.dtype)
            if self.overlap:
                kv = self._overlap_transform(kv, 0.0)
                gate = self._overlap_transform(gate, -float("inf"))
            weights = mx.softmax(gate.astype(mx.float32), axis=2, precise=True).astype(
                kv.dtype
            )
            new_pooled = (kv * weights).sum(axis=2)
            new_pooled = self.norm(new_pooled.astype(x.dtype))
            positions = mx.arange(new_pooled.shape[1], dtype=mx.float32)
            positions = positions * self.compress_ratio
            if isinstance(pool_base, mx.array) and pool_base.ndim > 0:
                pool_base = pool_base[:, None]
            positions = positions + pool_base
            new_pooled = _apply_partial_rope(
                new_pooled[:, None], rope, positions=positions
            ).squeeze(1)

        if cache is not None:
            return cache.update_pool(new_pooled, state_key)
        return new_pooled


class Indexer(nn.Module):
    def __init__(self, config: ModelArgs, compress_ratio: int):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(config, compress_ratio, self.head_dim)
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        x: mx.array,
        q_residual: mx.array,
        rope: DeepseekV4RoPE,
        position_rope: DeepseekV4RoPE,
        cache: Optional[DeepseekV4Cache],
        start_pos: int,
    ):
        B, L, _ = x.shape
        pooled = self.compressor(x, rope, cache, start_pos, state_key="indexer_state")
        if pooled.shape[1] == 0:
            return None

        offset = start_pos
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = _apply_partial_rope(q, position_rope, offset)

        scores = q.astype(mx.float32) @ pooled[:, None].swapaxes(-1, -2).astype(
            mx.float32
        )
        scores = mx.maximum(scores, 0) * self.scale
        weights = self.weights_proj(x).astype(mx.float32) * (self.n_heads**-0.5)
        scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
        lengths = cache.pooled_lengths("indexer_state") if cache is not None else None
        if lengths is not None:
            lengths = mx.array(lengths)
            valid = mx.arange(pooled.shape[1]) < lengths[:, None]
            scores = mx.where(valid[:, None], scores, mx.finfo(scores.dtype).min)
        k = min(self.index_topk, pooled.shape[1])
        return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]


class V4Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)
        self._q_l2_norm_weight = (mx.ones((self.head_dim,)),)
        self._cached_dtype = None

        rope_theta = (
            config.compress_rope_theta if self.compress_ratio else config.rope_theta
        )
        rope_scaling = config.rope_scaling if self.compress_ratio else None
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            rope_theta,
            rope_scaling,
            config.max_position_embeddings,
        )
        self.compress_rope = self.rope
        if self.compress_ratio:
            self.compressor = Compressor(config, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(config, self.compress_ratio)

    def _ensure_cached(self, dtype):
        dtype_key = str(dtype)
        if self._cached_dtype is not None and self._cached_dtype == dtype_key:
            return
        self._cached_dtype = dtype_key
        self._attn_sink_cached = self.attn_sink.astype(dtype)
        self._q_norm_weight_cached = self._q_l2_norm_weight[0].astype(dtype)
        if isinstance(self.wo_a, nn.QuantizedLinear):
            self._wo_a_weight = self.wo_a.weight.reshape(
                self.o_groups, self.o_lora_rank, -1
            )[:, None]
            self._wo_a_scales = self.wo_a.scales.reshape(
                self.o_groups, self.o_lora_rank, -1
            )[:, None]
            self._wo_a_biases = (
                None
                if self.wo_a.biases is None
                else self.wo_a.biases.reshape(self.o_groups, self.o_lora_rank, -1)[
                    :, None
                ]
            )
        else:
            group_feat = (self.n_heads * self.head_dim) // self.o_groups
            self._wo_a_weight_reshaped = self.wo_a.weight.reshape(
                self.o_groups, self.o_lora_rank, group_feat
            )

    def _grouped_output_projection(self, out: mx.array) -> mx.array:
        B, L = out.shape[:2]
        group_feat = (self.n_heads * self.head_dim) // self.o_groups
        out = out.reshape(B, L, self.o_groups, group_feat)

        if isinstance(self.wo_a, nn.QuantizedLinear):
            out = out.transpose(2, 0, 1, 3)
            out = mx.quantized_matmul(
                out,
                self._wo_a_weight,
                scales=self._wo_a_scales,
                biases=self._wo_a_biases,
                transpose=True,
                group_size=self.wo_a.group_size,
                bits=self.wo_a.bits,
                mode=self.wo_a.mode,
            )
            out = out.transpose(1, 2, 0, 3).reshape(
                B, L, self.o_groups * self.o_lora_rank
            )
            if "bias" in self.wo_a:
                out = out + self.wo_a.bias
            return out

        out = mx.einsum("bsgd,grd->bsgr", out, self._wo_a_weight_reshaped)
        out = out.reshape(B, L, self.o_groups * self.o_lora_rank)
        if "bias" in self.wo_a:
            out = out + self.wo_a.bias
        return out

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        local_cache = cache
        if isinstance(cache, DeepseekV4Cache):
            local_cache = cache

        offset = local_cache.offset if local_cache is not None else 0
        if isinstance(offset, mx.array):
            offset = offset + 0
        q_residual = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        self._ensure_cached(q.dtype)
        q = mx.fast.rms_norm(q, self._q_norm_weight_cached, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        kv = self.kv_norm(self.wkv(x)).reshape(B, L, 1, self.head_dim)
        kv = kv.transpose(0, 2, 1, 3)

        q = _apply_partial_rope(q, self.rope, offset)
        kv = _apply_partial_rope(kv, self.rope, offset)

        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, kv)
        full_kv = kv
        local_kv_len = kv.shape[2]

        pooled_mask = None
        pooled_bias = None
        if self.compress_ratio:
            v4_cache = cache if isinstance(cache, DeepseekV4Cache) else None
            pooled = self.compressor(x, self.compress_rope, v4_cache, offset)
            if pooled.shape[1] > 0:
                lengths = (
                    v4_cache.pooled_lengths("compressor_state")
                    if v4_cache is not None
                    else None
                )
                use_indexer = hasattr(self, "indexer")
                select_all = use_indexer and (
                    L > 1
                    or (lengths is None and pooled.shape[1] <= self.indexer.index_topk)
                )
                if select_all:
                    pooled = pooled[:, None]
                    pooled_bias = math.log(L)
                    if lengths is not None:
                        lengths = mx.array(lengths)
                        pooled_mask = (
                            mx.arange(pooled.shape[2]) < lengths[:, None]
                        ).reshape(B, 1, 1, -1)
                elif use_indexer:
                    topk = self.indexer(
                        x, q_residual, self.compress_rope, self.rope, v4_cache, offset
                    )
                    if topk is not None:
                        if lengths is not None:
                            lengths = mx.array(lengths)
                            pooled_mask = (topk < lengths[:, None, None]).reshape(
                                B, 1, 1, -1
                            )
                        expanded = mx.broadcast_to(
                            pooled[:, None, None, :, :],
                            (B, 1, L, pooled.shape[1], self.head_dim),
                        )
                        idx = topk[:, None, :, :, None]
                        pooled = mx.take_along_axis(
                            expanded,
                            mx.broadcast_to(idx, idx.shape[:-1] + (self.head_dim,)),
                            axis=3,
                        ).reshape(B, 1, -1, self.head_dim)
                    else:
                        pooled = pooled[:, None]
                else:
                    if lengths is not None:
                        lengths = mx.array(lengths)
                        pooled_mask = (
                            mx.arange(pooled.shape[1]) < lengths[:, None]
                        ).reshape(B, 1, 1, -1)
                    pooled = pooled[:, None]
                full_kv = mx.concatenate([full_kv, pooled], axis=2)

        if mask is not None and mask.shape[-1] > local_kv_len:
            mask = mask[..., -local_kv_len:]

        if mask is not None and full_kv.shape[2] > mask.shape[-1]:
            pad_shape = mask.shape[:-1] + (full_kv.shape[2] - mask.shape[-1],)
            pad_pooled_mask = pooled_mask
            if (
                pad_pooled_mask is not None
                and pad_pooled_mask.shape[-1] != pad_shape[-1]
            ):
                pad_pooled_mask = pad_pooled_mask[..., -pad_shape[-1] :]
            if pooled_bias is not None:
                dtype = q.dtype
                if mask.dtype == mx.bool_:
                    mask = mx.where(
                        mask,
                        mx.array(0, dtype=dtype),
                        mx.full((), mx.finfo(dtype).min, dtype=dtype),
                    )
                pad = mx.full(pad_shape, pooled_bias, dtype=mask.dtype)
            elif mask.dtype == mx.bool_:
                pad = mx.ones(pad_shape, dtype=mask.dtype)
                if pad_pooled_mask is not None:
                    pad = pad & pad_pooled_mask
            else:
                pad = mx.zeros(pad_shape, dtype=mask.dtype)
                if pad_pooled_mask is not None:
                    pad = mx.where(
                        pad_pooled_mask,
                        pad,
                        mx.full(pad_shape, mx.finfo(mask.dtype).min, dtype=mask.dtype),
                    )
            mask = mx.concatenate([mask, pad], axis=-1)
        out = scaled_dot_product_attention(
            q,
            full_kv,
            full_kv,
            cache=local_cache,
            scale=self.scale,
            mask=mask,
            sinks=self._attn_sink_cached,
        )
        out = _apply_partial_rope(out, self.rope, offset, inverse=True)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * self.head_dim)
        out = self._grouped_output_projection(out)
        return self.wo_b(out)


class DeepseekV4Block(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn = V4Attention(config, layer_idx)
        self.ffn = DeepseekV4MoE(config, layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        input_ids: mx.array,
    ) -> mx.array:
        residual = h
        x, post, comb = self.attn_hc.collapse(h)
        x = self.attn(self.attn_norm(x), mask=mask, cache=cache)
        h = self.attn_hc.expand(x, residual, post, comb)

        residual = h
        x, post, comb = self.ffn_hc.collapse(h)
        x = self.ffn(self.ffn_norm(x), input_ids)
        return self.ffn_hc.expand(x, residual, post, comb)


class DeepseekV4Model(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV4Block(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = HyperHead(config)

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:
        h = self.embed_tokens(inputs)
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
        )
        h = mx.contiguous(h)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        first_cache = cache[0]
        mask_cache = (
            first_cache.local
            if isinstance(first_cache, DeepseekV4Cache)
            else first_cache
        )
        mask = create_attention_mask(
            h[:, :, 0, :],
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, layer_cache in zip(self.pipeline_layers, cache):
            h = layer(h, mask, layer_cache, inputs)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            cache_item = cache[-1]
            if isinstance(cache_item, DeepseekV4Cache):
                cache_item = cache_item.local
            if cache_item is not None:
                cache_item.keys = mx.depends(cache_item.keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(self.hc_head(h))


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None):
        return self.lm_head(self.model(inputs, cache))

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return not (
                "attn_sink" in k
                or "e_score_correction_bias" in k
                or ".attn_hc." in k
                or ".ffn_hc." in k
                or ".hc_head." in k
            )

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.attn.compress_ratio:
                caches.append(DeepseekV4Cache(self.args.sliding_window))
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        n_layers = self.args.num_hidden_layers

        new_weights = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    if int(parts[1]) >= n_layers:
                        continue
                except ValueError:
                    pass
            new_weights[k] = v
        weights = new_weights

        def scale_to_float(scale: mx.array) -> mx.array:
            if scale.dtype == mx.uint8:
                return mx.exp((scale.astype(mx.float32) - 127.0) * math.log(2.0))
            return scale.astype(mx.float32)

        def dequant_fp8(weight: mx.array, scale: mx.array, block_size: int = 128):
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            scale = scale_to_float(scale)
            m, n = weight.shape
            pad_m = (-m) % block_size
            pad_n = (-n) % block_size
            weight = mx.pad(weight, ((0, pad_m), (0, pad_n)))
            weight = weight.reshape(
                (m + pad_m) // block_size,
                block_size,
                (n + pad_n) // block_size,
                block_size,
            )
            weight = (weight * scale[:, None, :, None]).reshape(m + pad_m, n + pad_n)
            return weight[:m, :n].astype(mx.bfloat16)

        def dequant_fp4(weight: mx.array, scale: mx.array, block_size: int = 32):
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
            scale = mx.repeat(scale_to_float(scale), block_size, axis=-1)
            return (unpacked * scale).astype(mx.bfloat16)

        new_weights = {}
        for k, v in weights.items():
            if not k.endswith(".scale"):
                if k not in new_weights:
                    new_weights[k] = v
                continue

            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                new_weights[k] = v
                continue
            if (
                ".ffn.experts." in wk
                and ".shared_experts." not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            ):
                new_weights[wk] = dequant_fp4(weight, v)
            elif weight.dtype == mx.uint8:
                new_weights[wk] = dequant_fp8(weight, v)
            else:
                new_weights[k] = v
        weights = new_weights

        top_remap = {
            "embed.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "head.weight": "lm_head.weight",
            "hc_head_fn": "model.hc_head.fn",
            "hc_head_base": "model.hc_head.base",
            "hc_head_scale": "model.hc_head.scale",
        }
        for old, new in top_remap.items():
            if old in weights:
                weights[new] = weights.pop(old)

        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            nk = "model." + k if k.startswith("layers.") else k
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
            for old, new in w_remap.items():
                nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
            remapped[nk] = v
        weights = remapped

        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.ffn.experts"
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                key0 = f"{prefix}.0.{src}.weight"
                if key0 in weights:
                    stacked = [
                        weights.pop(f"{prefix}.{e}.{src}.weight")
                        for e in range(self.args.n_routed_experts)
                    ]
                    weights[f"model.layers.{layer_idx}.ffn.switch_mlp.{dst}.weight"] = (
                        mx.stack(stacked)
                    )

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        for layer in self.model.layers:
            layer.attn.wq_b = shard_linear(
                layer.attn.wq_b, "all-to-sharded", group=group
            )
            layer.attn.wo_b = shard_linear(
                layer.attn.wo_b, "sharded-to-all", group=group
            )
            layer.attn.n_heads //= N

            layer.ffn.sharding_group = group
            shard_inplace(
                layer.ffn.shared_experts.gate_proj, "all-to-sharded", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.down_proj, "sharded-to-all", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.up_proj, "all-to-sharded", group=group
            )
            shard_inplace(layer.ffn.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(layer.ffn.switch_mlp.down_proj, "sharded-to-all", group=group)
            shard_inplace(layer.ffn.switch_mlp.up_proj, "all-to-sharded", group=group)
