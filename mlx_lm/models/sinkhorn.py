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
