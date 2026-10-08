# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from . import qwen3_5

# Schema 1 is a text-only pack keyed like qwen3_5.TextModel, schema 2 a
# vision-language pack keyed like the mlx-vlm qwen3_5 model. The text weights
# and the transform are the same in both.
_NAMESPACES = {1: "mlx-lm-text", 2: "mlx-vlm-qwen3_5"}
_QUANTIZATION = {"bits": 2, "group_size": 128, "mode": "affine"}
_BLOCKS = (0, 512, 1024, 2048, 4096, 8192)


@dataclass
class ModelArgs(qwen3_5.ModelArgs):
    schema_version: int = 2
    tensor_namespace: str = ""
    gdn_activation_layout: str = "grouped"
    quantization: Dict[str, Any] = field(default_factory=dict)
    modules: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if _NAMESPACES.get(self.schema_version) != self.tensor_namespace:
            raise ValueError(
                f"Unsupported Hadamard pack: schema {self.schema_version}, "
                f"namespace {self.tensor_namespace!r}"
            )
        if self.gdn_activation_layout != "grouped":
            raise ValueError("Hadamard packs require grouped GDN activations")
        if self.quantization != _QUANTIZATION:
            raise ValueError(
                "Hadamard packs require 2-bit affine weights, group size 128"
            )
        if not self.modules:
            raise ValueError("Hadamard pack is missing its packed module manifest")


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


class Model(qwen3_5.Model):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        modules = dict(self.language_model.named_modules())
        replacements = []
        seen = set()
        for record in args.modules:
            path = record["path"]
            if path in seen:
                raise ValueError(f"Duplicate packed module: {path}")
            seen.add(path)
            original = modules.get(path)
            if not isinstance(original, (nn.Linear, nn.Embedding)):
                raise ValueError(f"Unknown or unsupported packed module: {path}")
            if record["embedding"] != isinstance(original, nn.Embedding):
                raise ValueError(f"Packed module kind mismatch: {path}")
            if record["dtype"] != "float16":
                raise ValueError(f"Unsupported packed activation dtype: {path}")
            if "bias" in original:
                raise ValueError(f"Packed linear with a bias is unsupported: {path}")
            cls = (
                HadamardQuantizedEmbedding
                if record["embedding"]
                else HadamardQuantizedLinear
            )
            rows, width = original.weight.shape
            replacements.append((path, cls(width, rows, record["block"])))
        self.language_model.update_modules(tree_unflatten(replacements))

    def sanitize(self, weights):
        weights = super().sanitize(weights)
        bad = []
        for record in self.args.modules:
            if not record["block"]:
                continue
            key = f"language_model.{record['path']}.signs"
            if key not in weights:
                raise ValueError(f"Missing Hadamard sign vector: {key}")
            signs = weights[key]
            bad.append(mx.any((signs != 1) & (signs != -1)))
        if bad and mx.any(mx.stack(bad)).item():
            raise ValueError("Hadamard signs must contain only -1 and +1")
        return weights
