# Copyright © 2026 Apple Inc.

from typing import Dict

import mlx.core as mx

__all__ = [
    "attn_res_combine",
]


_ATTN_RES_SOURCE = """
    constexpr int NACC = K + 2;
    constexpr int NSIMD = THREADS / 32;

    auto n = threadgroup_position_in_grid.y;
    auto tid = thread_position_in_threadgroup.x;
    auto lane = thread_index_in_simdgroup;
    auto sg = simdgroup_index_in_threadgroup;

    auto partial_ = partial + n * D;

    float acc[NACC];
    for (int i = 0; i < NACC; ++i) {
      acc[i] = 0.0f;
    }
    for (uint d = tid; d < D; d += THREADS) {
      float w = static_cast<float>(w_eff[d]);
      float pv = static_cast<float>(partial_[d]);
      for (int k = 0; k < K; ++k) {
        acc[k] += static_cast<float>(raw[(k * N + n) * D + d]) * w;
      }
      acc[K] += pv * w;
      acc[K + 1] += pv * pv;
    }

    threadgroup float shm[NACC * NSIMD];
    for (int i = 0; i < NACC; ++i) {
      float s = simd_sum(acc[i]);
      if (lane == 0) {
        shm[i * NSIMD + sg] = s;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float weights[K + 1];
    if (tid == 0) {
      float tot[NACC];
      for (int i = 0; i < NACC; ++i) {
        tot[i] = 0.0f;
        for (int j = 0; j < NSIMD; ++j) {
          tot[i] += shm[i * NSIMD + j];
        }
      }
      float rinv = metal::rsqrt(tot[K + 1] / D + eps[0]);
      float logits[K + 1];
      float m = -1e30f;
      for (int k = 0; k < K; ++k) {
        logits[k] = tot[k] * inv_rms[k * N + n];
        m = metal::max(m, logits[k]);
      }
      logits[K] = tot[K] * rinv;
      m = metal::max(m, logits[K]);
      float denom = 0.0f;
      for (int k = 0; k <= K; ++k) {
        logits[k] = metal::exp(logits[k] - m);
        denom += logits[k];
      }
      for (int k = 0; k <= K; ++k) {
        weights[k] = logits[k] / denom;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float wp = weights[K];
    auto out_ = out + n * D;
    for (uint d = tid; d < D; d += THREADS) {
      float o = wp * static_cast<float>(partial_[d]);
      for (int k = 0; k < K; ++k) {
        o += weights[k] * static_cast<float>(raw[(k * N + n) * D + d]);
      }
      out_[d] = static_cast<InT>(o);
    }
"""

_attn_res_kernel = (
    mx.fast.metal_kernel(
        name="attnres_mix",
        input_names=["raw", "inv_rms", "partial", "w_eff", "eps", "N"],
        output_names=["out"],
        source=_ATTN_RES_SOURCE,
    )
    if mx.metal.is_available()
    else None
)

_ATTN_RES_THREADS = 512
_attn_res_eps_cache: Dict[float, mx.array] = {}


@mx.compile
def _attn_res_combine_reference(raw, inv_rms, partial_sum, w_eff, eps):
    pf = partial_sum.astype(mx.float32)
    p_logit = (pf @ w_eff) * mx.rsqrt((pf * pf).mean(axis=-1) + eps)
    logits = mx.concatenate([(raw.astype(mx.float32) @ w_eff) * inv_rms, p_logit[None]])
    p = mx.softmax(logits, axis=0, precise=True)
    out = (p[:-1, ..., None] * raw).sum(axis=0) + p[-1, ..., None] * partial_sum
    return out.astype(partial_sum.dtype)


def attn_res_combine(
    raw: mx.array,
    inv_rms: mx.array,
    partial_sum: mx.array,
    w_eff: mx.array,
    eps: float,
    use_kernel: bool = True,
) -> mx.array:
    """Combine with the Metal kernel when it can run, else with the reference."""
    if use_kernel and _attn_res_kernel is not None and mx.default_device() == mx.gpu:
        K, D = raw.shape[0], raw.shape[-1]
        N = raw.size // (K * D)
        eps_arr = _attn_res_eps_cache.get(eps)
        if eps_arr is None:
            eps_arr = _attn_res_eps_cache.setdefault(
                eps, mx.array([eps], dtype=mx.float32)
            )
        return _attn_res_kernel(
            inputs=[raw, inv_rms, partial_sum, w_eff, eps_arr, N],
            template=[
                ("InT", partial_sum.dtype),
                ("K", K),
                ("D", D),
                ("THREADS", _ATTN_RES_THREADS),
            ],
            grid=(_ATTN_RES_THREADS, N, 1),
            threadgroup=(_ATTN_RES_THREADS, 1, 1),
            output_shapes=[partial_sum.shape],
            output_dtypes=[partial_sum.dtype],
        )[0]
    return _attn_res_combine_reference(raw, inv_rms, partial_sum, w_eff, eps)
