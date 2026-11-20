# Copyright © 2026 Apple Inc.

import math

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "HadamardQuantizedEmbedding",
    "HadamardQuantizedLinear",
    "hadamard_rotate",
]

_BLOCKS = (0, 512, 1024, 2048, 4096, 8192)


def hadamard_rotate(x, block, signs, inverse=False):
    """Signed, normalized blockwise Walsh-Hadamard rotation of the last axis."""
    shape, dtype = x.shape, x.dtype
    # The packs are calibrated against a float32 rotation.
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block))
    x = x.reshape(shape)
    if inverse:
        x = x * signs
    return x.astype(dtype)


class HadamardQuantizedLinear(nn.Module):
    """A 2-bit affine linear layer whose weights have a Hadamard rotation folded
    into the input axis, so the input activations get the same rotation."""

    def __init__(self, input_dims: int, output_dims: int, block: int):
        super().__init__()
        if input_dims % 128:
            raise ValueError("Packed input width must be divisible by 128")
        if block not in _BLOCKS or (block and input_dims % block):
            raise ValueError(f"Invalid Hadamard block {block} for width {input_dims}")
        self.block = block
        self.group_size = 128
        self.bits = 2
        self.weight = mx.zeros((output_dims, input_dims // 16), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // 128), dtype=mx.float16)
        self.biases = mx.zeros_like(self.scales)
        if block:
            self.signs = mx.ones((input_dims,), dtype=mx.float32)
        self.freeze()

    def __call__(self, x):
        if self.block:
            x = hadamard_rotate(x, self.block, self.signs)
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )


class HadamardQuantizedEmbedding(HadamardQuantizedLinear):
    """Dequantizes the selected rows and rotates them back to the embedding basis."""

    def __call__(self, indices):
        shape = indices.shape
        indices = indices.reshape(-1)
        out = mx.dequantize(
            self.weight[indices],
            self.scales[indices],
            self.biases[indices],
            group_size=self.group_size,
            bits=self.bits,
        )
        out = out.reshape(*shape, -1).astype(mx.float16)
        if self.block:
            out = hadamard_rotate(out, self.block, self.signs, inverse=True)
        return out

    def as_linear(self, x):
        return super().__call__(x)
