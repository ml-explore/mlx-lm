# Copyright © 2025 Apple Inc.

"""Fused 4-bit affine matmul for small batches of rows.

A serving step processes one row per sequence. MLX's ``quantized_matmul``
dequantizes a weight once per row, so a batched step costs almost as much as the
same rows decoded separately and aggregate throughput stalls: measured 30-58 GB/s
of weight traffic at M=4..8 against 92 GB/s at M=1.

This kernel dequantizes each weight once and reuses it for every row. It blocks
four output rows per thread, so each dequant and each activation load feeds four
times as many multiply-adds. Weights are read once per step, independent of M.

It is used for 4-8 rows. Single rows go to MLX's matrix-vector kernel and larger
row counts (prefill) to its GEMM; both are faster there.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

MIN_ROWS = 4
MAX_ROWS = 8
N_BLOCK = 4

_SOURCE = """
    uint lane = thread_position_in_grid.x;
    uint block = thread_position_in_grid.y;
    const uint row0 = block * NB;

    const uint k_pack = K / 8;
    const uint groups = K / GROUP;
    const uint u_per_group = GROUP / 8;

    float acc[NB][MAX_M];
    for (uint r = 0; r < NB; r++) {
        for (uint m = 0; m < M; m++) {
            acc[r][m] = 0.0f;
        }
    }

    // Two packed words per iteration: more independent loads in flight.
    auto accumulate = [&](uint i) {
        uint g = i / u_per_group;
        uint k_base = i * 8;
        uint u[NB];
        float sc[NB];
        float bi[NB];
        for (uint r = 0; r < NB; r++) {
            u[r] = w[(row0 + r) * k_pack + i];
            sc[r] = static_cast<float>(scales[(row0 + r) * groups + g]);
            bi[r] = static_cast<float>(biases[(row0 + r) * groups + g]);
        }
        for (uint j = 0; j < 8; j++) {
            float v[NB];
            for (uint r = 0; r < NB; r++) {
                v[r] = static_cast<float>((u[r] >> (4 * j)) & 0xF) * sc[r] + bi[r];
            }
            for (uint m = 0; m < M; m++) {
                float xv = static_cast<float>(x[m * K + k_base + j]);
                for (uint r = 0; r < NB; r++) {
                    acc[r][m] += v[r] * xv;
                }
            }
        }
    };

    for (uint i = lane; i < k_pack; i += 64) {
        accumulate(i);
        if (i + 32 < k_pack) {
            accumulate(i + 32);
        }
    }

    for (uint r = 0; r < NB; r++) {
        for (uint m = 0; m < M; m++) {
            float total = simd_sum(acc[r][m]);
            if (lane == 0) {
                out[m * N + row0 + r] = static_cast<T>(total);
            }
        }
    }
"""

_kernels = {}


def _kernel(dtype):
    kernel = _kernels.get(dtype)
    if kernel is None:
        name = "qgemm_4bit_" + str(dtype).removeprefix("mlx.core.")
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=["x", "w", "scales", "biases"],
            output_names=["out"],
            source=_SOURCE,
        )
        _kernels[dtype] = kernel
    return kernel


def qgemm_4bit(x, w, scales, biases, group_size):
    """``x @ dequantize(w).T`` for ``x`` of shape (M, K) and 4-bit affine weights.

    ``w`` is packed uint32 of shape (N, K/8); ``scales`` and ``biases`` are
    (N, K/group_size). Supported for 4-8 rows.
    """
    M, K = x.shape
    N = w.shape[0]
    out = _kernel(x.dtype)(
        inputs=[x, w, scales, biases],
        template=[
            ("T", x.dtype),
            ("M", M),
            ("MAX_M", MAX_ROWS),
            ("NB", N_BLOCK),
            ("K", K),
            ("N", N),
            ("GROUP", group_size),
        ],
        grid=(32, N // N_BLOCK, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]
    return out


def _supported(module):
    """Whether a quantized layer can use the kernel."""
    if type(module) not in (nn.QuantizedLinear, nn.QuantizedEmbedding):
        return False
    if "bias" in module:
        return False
    if module.bits != 4 or module.mode != "affine":
        return False
    weight = module["weight"]
    if weight.shape[0] % N_BLOCK or weight.shape[1] % N_BLOCK:
        return False
    in_features = weight.shape[1] * 32 // module.bits
    if in_features % module.group_size or module.group_size % 8:
        return False
    return True


def _fused(module, x):
    """Run ``x`` through the kernel, or return None when it does not apply."""
    rows = 1
    for dim in x.shape[:-1]:
        rows *= dim
    if not MIN_ROWS <= rows <= MAX_ROWS:
        return None
    out = qgemm_4bit(
        x.reshape(rows, x.shape[-1]),
        module["weight"],
        module["scales"],
        module["biases"],
        module.group_size,
    )
    return out.reshape(*x.shape[:-1], out.shape[-1])


class QgemmLinear(nn.QuantizedLinear):
    """Quantized linear that uses the fused kernel for small batches of rows."""

    def __call__(self, x):
        out = _fused(self, x)
        return super().__call__(x) if out is None else out


class QgemmEmbedding(nn.QuantizedEmbedding):
    """Quantized embedding whose tied output projection uses the fused kernel.

    Tied embeddings serve as the output projection, which is a large share of the
    traffic of a decode step.
    """

    def as_linear(self, x):
        out = _fused(self, x)
        return super().as_linear(x) if out is None else out


def qgemm_quantize(model):
    """Replace supported quantized layers with kernel-backed ones."""
    swaps = []
    for name, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if not _supported(module):
            continue
        out_features, packed_in = module["weight"].shape
        in_features = packed_in * 32 // module.bits
        if type(module) is nn.QuantizedLinear:
            layer = QgemmLinear(
                in_features,
                out_features,
                bias=False,
                group_size=module.group_size,
                bits=module.bits,
                mode=module.mode,
            )
        elif type(module) is nn.QuantizedEmbedding:
            layer = QgemmEmbedding(
                out_features,
                in_features,
                group_size=module.group_size,
                bits=module.bits,
                mode=module.mode,
            )
        else:
            continue
        layer.update(module.parameters())
        swaps.append((name, layer))
    if swaps:
        model.update_modules(tree_unflatten(swaps))
    return model
