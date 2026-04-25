# Copyright © 2024 Apple Inc.
"""Fused Metal kernels for TurboQuant KV-cache operations.

Two kernels, same pattern: one thread per token, all intermediate
values in float32 registers, single GPU dispatch.

  turbo_encode_metal  — WHT (optional) → L2 norm → codebook → bit pack
  wht_rotate_metal    — WHT → scale (used to pre-rotate Q before SDPA)
"""

import math

import mlx.core as mx


_ENCODE_HEADER = r"""
template <typename T, int D, int BITS, bool ROTATE>
inline void turbo_encode_impl(
    device const T* x_in,
    device uint32_t* packed_out,
    device T* scale_out,
    device const float* codebook,
    uint tok)
{
    constexpr int N_LEVELS = 1 << BITS;
    constexpr int N_U32    = D * BITS / 32;
    const float SQRT_D     = metal::sqrt(float(D));
    const float INV_SQRT_D = 1.0f / SQRT_D;

    device const T* src = x_in + tok * D;

    float buf[D];
    for (int i = 0; i < D; i++) buf[i] = float(src[i]);

    if constexpr (ROTATE) {
        for (int stride = 1; stride < D; stride <<= 1) {
            for (int i = 0; i < D; i += stride * 2) {
                for (int j = 0; j < stride; j++) {
                    float a = buf[i + j];
                    float b = buf[i + j + stride];
                    buf[i + j]          = a + b;
                    buf[i + j + stride] = a - b;
                }
            }
        }
        for (int i = 0; i < D; i++) buf[i] *= INV_SQRT_D;
    }

    float norm2 = 0.0f;
    for (int i = 0; i < D; i++) norm2 += buf[i] * buf[i];
    float norm = metal::sqrt(norm2);

    scale_out[tok] = T(norm * INV_SQRT_D);

    float inv_n_s = SQRT_D / (norm + 1e-8f);

    device uchar* out_bytes = (device uchar*)(packed_out + tok * N_U32);

    for (int p = 0; p < D / 8; p++) {
        uint val = 0u;
        for (int i = 0; i < 8; i++) {
            float xs = buf[p * 8 + i] * inv_n_s;
            int best = 0;
            float bd = (xs - codebook[0]) * (xs - codebook[0]);
            for (int k = 1; k < N_LEVELS; k++) {
                float d = xs - codebook[k];
                float dd = d * d;
                if (dd < bd) { bd = dd; best = k; }
            }
            val |= (uint(best) & uint(N_LEVELS - 1)) << uint(BITS * i);
        }
        if constexpr (BITS == 3) {
            out_bytes[p * 3]     = uchar(val & 0xFFu);
            out_bytes[p * 3 + 1] = uchar((val >> 8u) & 0xFFu);
            out_bytes[p * 3 + 2] = uchar((val >> 16u) & 0xFFu);
        } else {
            packed_out[tok * N_U32 + p] = val;
        }
    }
}
"""

_ENCODE_SOURCE = r"""
uint tok = thread_position_in_grid.x;
if (tok >= uint(N_tokens[0])) return;
turbo_encode_impl<T, D, BITS, ROTATE>(x, packed, scales, codebook, tok);
"""

# WHT butterfly in float32 registers; output cast back to T.
# Same precision as the Python _hadamard_transform fallback.
_WHT_HEADER = r"""
template <typename T, int D>
inline void wht_rotate_impl(
    device const T* x_in,
    device T* x_out,
    float scale,
    uint tok)
{
    float buf[D];
    for (int i = 0; i < D; i++) buf[i] = float(x_in[tok * D + i]);

    for (int stride = 1; stride < D; stride <<= 1) {
        for (int i = 0; i < D; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = buf[i + j];
                float b = buf[i + j + stride];
                buf[i + j]          = a + b;
                buf[i + j + stride] = a - b;
            }
        }
    }

    for (int i = 0; i < D; i++) x_out[tok * D + i] = T(buf[i] * scale);
}
"""

_WHT_SOURCE = r"""
uint tok = thread_position_in_grid.x;
if (tok >= uint(N_tokens[0])) return;
wht_rotate_impl<T, D>(x, y, scale[0], tok);
"""

_kernels: dict[str, mx.fast.metal_kernel] = {}


def _get_encode_kernel() -> mx.fast.metal_kernel:
    if "encode" not in _kernels:
        _kernels["encode"] = mx.fast.metal_kernel(
            name="turbo_encode",
            input_names=["x", "codebook", "N_tokens"],
            output_names=["packed", "scales"],
            header=_ENCODE_HEADER,
            source=_ENCODE_SOURCE,
        )
    return _kernels["encode"]


def _get_wht_kernel() -> mx.fast.metal_kernel:
    if "wht" not in _kernels:
        _kernels["wht"] = mx.fast.metal_kernel(
            name="wht_rotate",
            input_names=["x", "scale", "N_tokens"],
            output_names=["y"],
            header=_WHT_HEADER,
            source=_WHT_SOURCE,
        )
    return _kernels["wht"]


def turbo_encode_metal(
    x: mx.array,
    codebook: mx.array,
    bits: int,
    rotate: bool,
) -> tuple[mx.array, mx.array]:
    """Encode x to TurboQuant packed format via a single GPU dispatch.

    Args:
        x:        [*batch, D] float16 or bfloat16.
        codebook: [2**bits] float32.
        bits:     3 or 4.
        rotate:   Apply WHT before quantization (True for K, False for V).

    Returns:
        packed: [*batch, D*bits//32] uint32.
        scales: [*batch, 1] same dtype as x.
    """
    *batch, D = x.shape
    N_tokens = math.prod(batch) if batch else 1
    n_u32 = D * bits // 32
    # Threadgroup size: keep register pressure below spilling threshold.
    # Each thread holds float buf[D]; larger D → fewer threads per group.
    # Formula: max(32, 8192 // D) gives {64:128, 128:64, 256:32}.
    tg_size = min(N_tokens, max(32, 8192 // D))

    packed_flat, scales_flat = _get_encode_kernel()(
        inputs=[
            x.reshape(N_tokens, D),
            codebook.astype(mx.float32),
            mx.array([N_tokens], dtype=mx.int32),
        ],
        template=[("T", x.dtype), ("D", D), ("BITS", bits), ("ROTATE", rotate)],
        output_shapes=[(N_tokens, n_u32), (N_tokens, 1)],
        output_dtypes=[mx.uint32, x.dtype],
        grid=(N_tokens, 1, 1),
        threadgroup=(tg_size, 1, 1),
        stream=mx.gpu,
    )
    return packed_flat.reshape(*batch, n_u32), scales_flat.reshape(*batch, 1)


def wht_rotate_metal(x: mx.array, scale: float) -> mx.array:
    """Apply a normalized WHT via a single GPU dispatch.

    Runs the butterfly in float32 registers and casts back to x.dtype.
    Replaces the per-stage MLX op chain in _hadamard_transform — useful for
    models with many attention layers (40+ WHT calls per generation step).

    Args:
        x:     [*batch, D], any float dtype.
        scale: Multiplied into every output element (typically 1/sqrt(D)).

    Returns:
        [*batch, D] same dtype as x.
    """
    *batch, D = x.shape
    N_tokens = math.prod(batch) if batch else 1
    tg_size = min(N_tokens, max(32, 8192 // D))

    (y_flat,) = _get_wht_kernel()(
        inputs=[
            x.reshape(N_tokens, D),
            mx.array([scale], dtype=mx.float32),
            mx.array([N_tokens], dtype=mx.int32),
        ],
        template=[("T", x.dtype), ("D", D)],
        output_shapes=[(N_tokens, D)],
        output_dtypes=[x.dtype],
        grid=(N_tokens, 1, 1),
        threadgroup=(tg_size, 1, 1),
        stream=mx.gpu,
    )
    return y_flat.reshape(*batch, D)


def is_available() -> bool:
    """Return True if Metal kernels can be used."""
    return mx.metal.is_available()
