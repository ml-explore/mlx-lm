# Copyright © 2026 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mlx_lm.models import qwen3_5
from mlx_lm.models.checkpoint import from_text_layout
from mlx_lm.models.prism_hadamard_qwen35.config import ModelArgs
from mlx_lm.models.prism_hadamard_qwen35.layers import (
    HadamardQuantizedEmbedding,
    HadamardQuantizedLinear,
)

__all__ = [
    "Model",
]


class Model(qwen3_5.Model):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        modules = dict(self.named_modules())
        replacements = []
        seen = set()
        for record in args.modules:
            # The packs name the modules as qwen3_5's text model did.
            path = from_text_layout(record["path"])
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
        self.update_modules(tree_unflatten(replacements))

    def sanitize(self, weights):
        weights = super().sanitize(weights)
        bad = []
        for record in self.args.modules:
            if not record["block"]:
                continue
            key = f"{from_text_layout(record['path'])}.signs"
            if key not in weights:
                raise ValueError(f"Missing Hadamard sign vector: {key}")
            signs = weights[key]
            bad.append(mx.any((signs != 1) & (signs != -1)))
        if bad and mx.any(mx.stack(bad)).item():
            raise ValueError("Hadamard signs must contain only -1 and +1")
        return weights
