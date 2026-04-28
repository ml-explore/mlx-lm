# Copyright © 2025 Apple Inc.

"""Sinkhorn normalization for doubly-stochastic matrices (HyperConnection mHC).

Shared module for models that use learned mixing weights on the Birkhoff
polytope (e.g., DeepSeek-V4's multi-HyperConnection). Provides both a
fused Metal kernel (register-resident, one thread per token) and a pure-MLX
reference path.
"""

import mlx.core as mx

_SINKHORN_KERNELS: dict = {}


def _make_sinkhorn_kernel(hc: int, iters: int, eps: float):
    """Build a fully-unrolled, register-resident Sinkhorn Metal kernel.

    Each thread owns one token's [hc, hc] matrix in registers (hc^2 floats).
    No threadgroup memory, no atomics, no global memory traffic between iters
    — kernel reads input once and writes output once.
    """
    key = (hc, iters)
    if key in _SINKHORN_KERNELS:
        return _SINKHORN_KERNELS[key]

    n_elem = hc * hc
    eps_lit = f"{eps:.8e}f"

    def row_softmax():
        out = []
        for r in range(hc):
            base = r * hc
            out.append(f"        {{ float mx = m[{base}];")
            for c in range(1, hc):
                out.append(f"          mx = metal::max(mx, m[{base + c}]);")
            out.append(f"          float s = 0.0f;")
            for c in range(hc):
                out.append(f"          m[{base + c}] = metal::exp(m[{base + c}] - mx); s += m[{base + c}];")
            out.append(f"          float inv = 1.0f / s;")
            for c in range(hc):
                out.append(f"          m[{base + c}] *= inv;")
            out.append("        }")
        return "\n".join(out)

    def row_norm():
        out = []
        for r in range(hc):
            base = r * hc
            terms = " + ".join(f"m[{base + c}]" for c in range(hc))
            out.append(f"        {{ float inv = 1.0f / ({terms} + {eps_lit});")
            for c in range(hc):
                out.append(f"          m[{base + c}] *= inv;")
            out.append("        }")
        return "\n".join(out)

    def col_norm():
        out = []
        for c in range(hc):
            terms = " + ".join(f"m[{r * hc + c}]" for r in range(hc))
            out.append(f"        {{ float inv = 1.0f / ({terms} + {eps_lit});")
            for r in range(hc):
                out.append(f"          m[{r * hc + c}] *= inv;")
            out.append("        }")
        return "\n".join(out)

    def add_eps():
        return "\n".join(f"        m[{i}] += {eps_lit};" for i in range(n_elem))

    iter_body = "\n".join([row_norm(), col_norm()])
    inner_iters = "\n".join([iter_body] * (iters - 1))

    source = f"""
        uint n = thread_position_in_grid.x;
        if (n >= n_tokens[0]) return;

        uint base = n * {n_elem};
        float m[{n_elem}];
{chr(10).join(f"        m[{i}] = comb_log[base + {i}];" for i in range(n_elem))}

        // Row softmax
{row_softmax()}

        // Add eps
{add_eps()}

        // Initial column normalization (matches reference: cols first after softmax)
{col_norm()}

        // Remaining (iters - 1) rounds of (row_norm, col_norm)
{inner_iters}

{chr(10).join(f"        comb[base + {i}] = m[{i}];" for i in range(n_elem))}
    """

    kernel = mx.fast.metal_kernel(
        name=f"sinkhorn_hc{hc}_it{iters}",
        input_names=["comb_log", "n_tokens"],
        output_names=["comb"],
        source=source,
    )
    _SINKHORN_KERNELS[key] = kernel
    return kernel


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
    mixes: mx.array,        # [B*S, (2+hc)*hc] fp32
    hc_scale: mx.array,     # [3] fp32
    hc_base: mx.array,      # [(2+hc)*hc] fp32
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

    pre_log  = mix[:, :hc_mult]             * s0 + hc_base[:hc_mult]
    post_log = mix[:, hc_mult:2 * hc_mult]  * s1 + hc_base[hc_mult:2 * hc_mult]
    comb_log = (
        mix[:, 2 * hc_mult:].reshape(n, hc_mult, hc_mult) * s2
        + hc_base[2 * hc_mult:].reshape(hc_mult, hc_mult)
    )

    pre  = mx.sigmoid(pre_log) + eps
    post = 2 * mx.sigmoid(post_log)

    use_kernel = (
        hc_mult <= 8
        and mx.metal.is_available()
        and comb_log.size > 0
    )
    if use_kernel:
        kernel = _make_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
        flat = comb_log.reshape(n, hc_mult * hc_mult).astype(mx.float32)
        n_tokens = mx.array([n], dtype=mx.uint32)
        (comb_flat,) = kernel(
            inputs=[flat, n_tokens],
            output_shapes=[(n, hc_mult * hc_mult)],
            output_dtypes=[mx.float32],
            grid=(n, 1, 1),
            threadgroup=(min(n, 256) or 1, 1, 1),
        )
        comb = comb_flat.reshape(n, hc_mult, hc_mult)
    else:
        comb = mx.softmax(comb_log, axis=-1, precise=True) + eps
        col_sum = comb.sum(axis=1, keepdims=True) + eps
        comb = comb / col_sum
        for _ in range(sinkhorn_iters - 1):
            row_sum = comb.sum(axis=2, keepdims=True) + eps
            comb = comb / row_sum
            col_sum = comb.sum(axis=1, keepdims=True) + eps
            comb = comb / col_sum

    return pre, post, comb
