# Copyright © 2025 Apple Inc.

from typing import Optional

import mlx.core as mx


def _make_fused_recurrent_gla_kernel():
    if not mx.metal.is_available():
        return None
    source = f"""
        auto bh = thread_position_in_grid.z; // ranges over B*H
        const int b = bh / H;
        const int h = bh % H;
        const int dv = thread_position_in_grid.y; // [0, Dv)

        // Local column of the recurrent state for this (b, h, dv)
        float h_col[Dk];

        // state_in: [B, H, Dk, Dv], or nullptr
        bool has_state = state_in != nullptr;
        int state_base = ((b * H + h) * Dk * Dv) + dv;
        if (has_state) {{
            for (int d = 0; d < Dk; ++d) {{
                h_col[d] = (float)state_in[state_base + d * Dv];
            }}
        }} else {{
            for (int d = 0; d < Dk; ++d) {{ h_col[d] = 0.0f; }}
        }}

        // Base offsets for batch b
        const int BH = B * H; // not used but kept for clarity
        const int q_base_b = b * H * T * Dk;
        const int k_base_b = b * H * T * Dk;
        const int v_base_b = b * H * T * Dv;
        const int y_base_b = b * H * T * Dv;
        const int g_base_b = b * H * T;

        for (int t = 0; t < T; ++t) {{
            // offsets for this time step and head h
            const int q_off = q_base_b + (h * T + t) * Dk;
            const int k_off = k_base_b + (h * T + t) * Dk;
            const int v_off = v_base_b + (h * T + t) * Dv;
            const int y_off = y_base_b + (h * T + t) * Dv;

            // 1) output = scale * dot(q_t, h_col)
            float acc = 0.0f;
            for (int d = 0; d < Dk; ++d) {{
                acc += (float)q[q_off + d] * h_col[d];
            }}
            y[y_off + dv] = (InT)(acc * (float)scale);

            // 2) update recurrent state column
            float gamma = exp((float)g[g_base_b + h * T + t]);
            float v_curr = (float)v[v_off + dv];
            for (int d = 0; d < Dk; ++d) {{
                h_col[d] = h_col[d] * gamma + (float)k[k_off + d] * v_curr;
            }}
        }}

        // Write back the final h_col into the state buffer if available
        if (has_state) {{
            for (int d = 0; d < Dk; ++d) {{
                state_out[state_base + d * Dv] = (InT)h_col[d];
            }}
        }}
    """
    inputs = ["q", "k", "v", "g", "B", "H_in", "T", "scale", "state_in"]
    return mx.fast.metal_kernel(
        name="fused_recurrent_gla_minimal",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


_fused_recurrent_gla_kernel = _make_fused_recurrent_gla_kernel()


def fused_recurrent_gla_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    scale: float,
    state: Optional[mx.array] = None,
) -> mx.array:
    """
    Reference ops implementation equivalent to fused_recurrent_simple_gla.
    q, k, v have shape [B, H, T, D], g has shape [B, H, T].
    Recurrence per (b, h):
        y_t = (q_t @ h) * scale
        h   = h * exp(g_t) + k_t^T @ v_t
    Returns y with shape [B, H, T, Dv].
    """
    B, Hq, L, K = q.shape
    Hv = k.shape[1]
    V = v.shape[-1]

    if state is None:
        h = mx.zeros((B, Hv, K, V), dtype=q.dtype)
    else:
        h = state

    outputs = []
    for t in range(L):
        q_t = q[:, :, t : t + 1] * scale
        k_t = k[:, :, t : t + 1]
        v_t = v[:, :, t : t + 1]
        g_t = g[:, :, t]
        o_t = mx.matmul(q_t, h)
        outputs.append(o_t)
        h = h * mx.exp(g_t)[:, :, None, None]
        h = h + mx.matmul(k_t.transpose(0, 1, 3, 2), v_t)

    return mx.concatenate(outputs, axis=2), h


def fused_recurrent_gla_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    scale: float,
    state: Optional[mx.array] = None,
    use_kernel: bool = True,
) -> mx.array:
    """
    Minimal fused recurrent GLA update: matches fused_recurrent_simple_gla.
    Expects q, k, v as [B, H, T, D] and g as [B, H, T]. Returns y [B, H, T, Dv].
    If `state` is provided, it should be of shape [B, H, Dk, Dv] and will be updated in-place.
    """
    if (
        (not use_kernel)
        or (mx.default_device() != mx.gpu)
        or (not mx.metal.is_available())
    ):
        return fused_recurrent_gla_ops(q, k, v, g, scale, state)

    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    input_type = q.dtype
    kernel = _fused_recurrent_gla_kernel

    # Prepare state buffer: if None, use zeros of shape [B, H, Dk, Dv]
    if state is None:
        state_buf = mx.zeros((B, H, Dk, Dv), dtype=input_type)
    else:
        state_buf = state

    # Launch one thread per (dv, b*h), iterate over time T inside the kernel.
    result = kernel(
        inputs=[q, k, v, g, B, H, T, scale, state_buf],
        template=[
            ("InT", input_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("H", H),
        ],
        grid=(1, Dv, B * H),
        threadgroup=(1, 1, 1),
        output_shapes=[(B, H, T, Dv), (B, H, Dk, Dv)],
        output_dtypes=[input_type, input_type],
    )
    # The state buffer is updated in-place; return output and updated state
    y, new_state = result
    return y, new_state
