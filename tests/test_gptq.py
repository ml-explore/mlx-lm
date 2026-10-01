# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.quant.gptq import gptq_quantize


class OneLinear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.proj = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        self.proj.weight = weight

    def __call__(self, x):
        return self.proj(x)


def reference_gptq(weight, data, bits, group_size):
    """Column-by-column GPTQ without the lazy per-group update.

    Every column's error is propagated to all remaining columns immediately,
    which is the form in the paper (Algorithm 1 without the block batching).
    The Hessian and its inverse are formed the same way as in gptq_quantize.
    """
    H = data.T @ data
    with mx.stream(mx.cpu):
        damp = 1e-2 * mx.mean(mx.diag(H))
        diag = mx.arange(H.shape[0])
        H[diag, diag] += damp
        H = mx.linalg.cholesky(H)
        H = mx.linalg.cholesky_inv(H)
        Hinv = mx.linalg.cholesky(H, upper=True)
    mx.eval(Hinv)

    n_bins = 2**bits - 1
    W = weight.astype(mx.float32)
    all_scales = []
    all_biases = []
    for i in range(0, W.shape[-1], group_size):
        j = i + group_size
        _, scales, biases = mx.quantize(W[..., i:j], bits=bits, group_size=group_size)
        all_scales.append(scales)
        all_biases.append(biases)
        for k in range(i, j):
            w = W[..., k : k + 1]
            q = mx.clip(mx.round((w - biases) / scales), 0.0, n_bins)
            q = scales * q + biases
            e = (w - q) / Hinv[k, k]
            W[..., k:] -= e @ Hinv[k : k + 1, k:]
            mx.eval(W)

    scales = mx.concatenate(all_scales, axis=-1)
    biases = mx.concatenate(all_biases, axis=-1)
    q = mx.unflatten(W, -1, (scales.shape[-1], -1))
    q = mx.clip(mx.round((q - biases[..., None]) / scales[..., None]), 0.0, n_bins)
    q = scales[..., None] * q + biases[..., None]
    return q.flatten(-2, -1)


class TestGPTQ(unittest.TestCase):
    def _check(self, in_features, group_size, bits=4, out_features=8):
        mx.random.seed(0)
        weight = mx.random.normal((out_features, in_features))
        data = mx.random.normal((256, in_features))
        # Correlated columns so that the error of one column has a visible
        # effect on the columns after it, across group boundaries.
        data = data + 0.5 * mx.sum(data, axis=-1, keepdims=True)
        mx.eval(weight, data)

        expected = reference_gptq(weight, data, bits, group_size)

        model, _ = gptq_quantize(
            OneLinear(weight),
            data,
            bits=bits,
            group_size=group_size,
            fallback_bits=bits,
            fallback_group_size=group_size,
            batch_size=32,
        )
        layer = model.proj
        actual = mx.dequantize(
            layer.weight, layer.scales, layer.biases, group_size, bits
        ).astype(mx.float32)
        mx.eval(expected, actual)
        self.assertTrue(
            mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item(),
            f"max abs diff {mx.abs(actual - expected).max().item()}",
        )

    def test_single_group(self):
        self._check(in_features=32, group_size=32)

    def test_multiple_groups(self):
        self._check(in_features=128, group_size=32)


if __name__ == "__main__":
    unittest.main()
