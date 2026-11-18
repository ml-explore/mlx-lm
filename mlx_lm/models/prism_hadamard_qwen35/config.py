# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List

from mlx_lm.models import qwen3_5

__all__ = [
    "ModelArgs",
]


# Schema 1 is a text-only pack keyed like qwen3_5.TextModel, schema 2 a
# vision-language pack keyed like the mlx-vlm qwen3_5 model. The text weights
# and the transform are the same in both.
_NAMESPACES = {1: "mlx-lm-text", 2: "mlx-vlm-qwen3_5"}
_QUANTIZATION = {"bits": 2, "group_size": 128, "mode": "affine"}


@dataclass
class ModelArgs(qwen3_5.ModelArgs):
    schema_version: int = 2
    tensor_namespace: str = ""
    gdn_activation_layout: str = "grouped"
    quantization: Dict[str, Any] = field(default_factory=dict)
    modules: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
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
