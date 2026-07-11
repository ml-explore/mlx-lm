# Copyright © 2026 Apple Inc.

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .lora import QuantizedLoRALinear


def _pad_last_dim(x, size):
    padding = size - x.shape[-1]
    if padding < 0:
        raise ValueError(
            f"Input dimension {x.shape[-1]} exceeds packed dimension {size}"
        )
    if padding == 0:
        return x
    return mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, padding)])


class QuantizedLoRAAdapterPack(nn.Module):
    """Packed quantized LoRA weights for one target layer and many adapters.

    All adapters must use the same dimensions and quantization settings. The
    pack supports both homogeneous request-level selection and token-level
    mixed-adapter routing without materializing floating-point LoRA matrices.
    """

    @classmethod
    def from_weight_sets(
        cls,
        weight_sets: Sequence[Mapping[str, mx.array]],
        scales: Sequence[float],
        *,
        group_size: int,
        rank_group_size: int,
        bits: int,
        input_dims: int,
        output_dims: int,
        rank: int,
        padded_input_dims: int,
        padded_rank: int,
        mode: str = "affine",
    ):
        if not weight_sets:
            raise ValueError("Quantized LoRA adapter pack must not be empty")
        if len(scales) != len(weight_sets):
            raise ValueError("Expected one LoRA scale per packed adapter")
        if mode != "affine":
            raise ValueError("Adapter packs currently support affine mode")

        names = (
            "lora_a",
            "lora_a_scales",
            "lora_a_biases",
            "lora_b",
            "lora_b_scales",
            "lora_b_biases",
        )
        expected_shapes = {
            "lora_a": (rank, padded_input_dims * bits // 32),
            "lora_a_scales": (rank, padded_input_dims // group_size),
            "lora_a_biases": (rank, padded_input_dims // group_size),
            "lora_b": (output_dims, padded_rank * bits // 32),
            "lora_b_scales": (output_dims, padded_rank // rank_group_size),
            "lora_b_biases": (output_dims, padded_rank // rank_group_size),
        }
        missing = [name for name in names if name not in weight_sets[0]]
        if missing:
            raise ValueError("Packed adapter is missing weights: " + ", ".join(missing))
        reference_shapes = {name: weight_sets[0][name].shape for name in names}
        if reference_shapes != expected_shapes:
            raise ValueError("Packed adapter weight shapes do not match metadata")
        for weights in weight_sets[1:]:
            missing = [name for name in names if name not in weights]
            if missing:
                raise ValueError(
                    "Packed adapter is missing weights: " + ", ".join(missing)
                )
            shapes = {name: weights[name].shape for name in names}
            if shapes != reference_shapes:
                raise ValueError("Packed adapter weight shapes must match")

        pack = cls()
        pack.input_dims = input_dims
        pack.output_dims = output_dims
        pack.rank = rank
        pack.padded_input_dims = padded_input_dims
        pack.padded_rank = padded_rank
        if input_dims > pack.padded_input_dims or rank > pack.padded_rank:
            raise ValueError("Packed adapter padding does not match metadata")
        pack.group_size = group_size
        pack.rank_group_size = rank_group_size
        pack.bits = bits
        pack.mode = mode
        for name in names:
            setattr(pack, name, mx.stack([weights[name] for weights in weight_sets]))
        pack.scales = mx.array(scales)
        pack.freeze()
        mx.eval(pack.parameters())
        return pack

    @classmethod
    def from_layers(
        cls,
        layers: Sequence[QuantizedLoRALinear],
        scales: Sequence[float] | None = None,
    ):
        if not layers:
            raise ValueError("Quantized LoRA adapter pack must not be empty")
        if any(not isinstance(layer, QuantizedLoRALinear) for layer in layers):
            raise TypeError("All adapter pack layers must be QuantizedLoRALinear")

        first = layers[0]
        settings = (
            first.input_dims,
            first.output_dims,
            first.rank,
            first.padded_input_dims,
            first.padded_rank,
            first.group_size,
            first.rank_group_size,
            first.bits,
            first.mode,
        )
        for layer in layers[1:]:
            candidate = (
                layer.input_dims,
                layer.output_dims,
                layer.rank,
                layer.padded_input_dims,
                layer.padded_rank,
                layer.group_size,
                layer.rank_group_size,
                layer.bits,
                layer.mode,
            )
            if candidate != settings:
                raise ValueError(
                    "All packed adapters must share dimensions, rank, and "
                    "quantization settings"
                )
        if first.mode != "affine":
            raise ValueError("Adapter packs currently support affine mode")

        if scales is None:
            scales = [layer.scale for layer in layers]
        if len(scales) != len(layers):
            raise ValueError("Expected one LoRA scale per packed adapter")

        pack = cls()
        pack.input_dims = first.input_dims
        pack.output_dims = first.output_dims
        pack.rank = first.rank
        pack.padded_input_dims = first.padded_input_dims
        pack.padded_rank = first.padded_rank
        pack.group_size = first.group_size
        pack.rank_group_size = first.rank_group_size
        pack.bits = first.bits
        pack.mode = first.mode
        pack.lora_a = mx.stack([layer.lora_a for layer in layers])
        pack.lora_a_scales = mx.stack([layer.lora_a_scales for layer in layers])
        pack.lora_a_biases = mx.stack([layer.lora_a_biases for layer in layers])
        pack.lora_b = mx.stack([layer.lora_b for layer in layers])
        pack.lora_b_scales = mx.stack([layer.lora_b_scales for layer in layers])
        pack.lora_b_biases = mx.stack([layer.lora_b_biases for layer in layers])
        pack.scales = mx.array(scales)
        pack.freeze()
        return pack

    @property
    def num_adapters(self):
        return self.lora_a.shape[0]

    @property
    def adapter_bytes(self):
        arrays = (
            self.lora_a,
            self.lora_a_scales,
            self.lora_a_biases,
            self.lora_b,
            self.lora_b_scales,
            self.lora_b_biases,
            self.scales,
        )
        return sum(array.nbytes for array in arrays)

    def _first_projection(self, x, lora_a, scales, biases):
        x = _pad_last_dim(x, self.padded_input_dims)
        return mx.quantized_matmul(
            x,
            lora_a,
            scales,
            biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )

    def delta_for_adapter(self, x, adapter_index: int):
        """Return the scaled LoRA delta for one adapter and arbitrary rows."""
        if adapter_index < 0 or adapter_index >= self.num_adapters:
            raise IndexError(f"Adapter index out of range: {adapter_index}")
        z = self._first_projection(
            x,
            self.lora_a[adapter_index],
            self.lora_a_scales[adapter_index],
            self.lora_a_biases[adapter_index],
        )
        z = _pad_last_dim(z, self.padded_rank)
        delta = mx.quantized_matmul(
            z,
            self.lora_b[adapter_index],
            self.lora_b_scales[adapter_index],
            self.lora_b_biases[adapter_index],
            transpose=True,
            group_size=self.rank_group_size,
            bits=self.bits,
            mode=self.mode,
        )
        return self.scales[adapter_index].astype(delta.dtype) * delta

    def delta(self, x, adapter_indices):
        """Return scaled LoRA deltas routed by flattened token-row indices."""
        original_shape = tuple(x.shape)
        flat_x = x.reshape((-1, original_shape[-1]))
        adapter_indices = mx.array(adapter_indices, dtype=mx.int32).reshape(-1)
        if adapter_indices.size != flat_x.shape[0]:
            raise ValueError(
                f"Expected {flat_x.shape[0]} adapter indices, "
                f"received {adapter_indices.size}"
            )

        flat_x = _pad_last_dim(flat_x, self.padded_input_dims)
        z = mx.gather_qmm(
            mx.expand_dims(flat_x, -2),
            self.lora_a,
            self.lora_a_scales,
            self.lora_a_biases,
            rhs_indices=adapter_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=False,
        ).squeeze(-2)
        z = _pad_last_dim(z, self.padded_rank)
        delta = mx.gather_qmm(
            mx.expand_dims(z, -2),
            self.lora_b,
            self.lora_b_scales,
            self.lora_b_biases,
            rhs_indices=adapter_indices,
            transpose=True,
            group_size=self.rank_group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=False,
        ).squeeze(-2)
        scales = self.scales[adapter_indices].astype(delta.dtype).reshape((-1, 1))
        delta = scales * delta
        return delta.reshape(original_shape[:-1] + (self.output_dims,))


@dataclass(frozen=True)
class QuantizedLoRAAdapterBank:
    adapter_names: tuple[str, ...]
    packs: dict[str, QuantizedLoRAAdapterPack]

    @property
    def adapter_bytes(self):
        return sum(pack.adapter_bytes for pack in self.packs.values())

    def adapter_index(self, name: str):
        try:
            return self.adapter_names.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown adapter: {name}") from error


def load_quantized_lora_adapter_bank(
    adapters: Mapping[str, str | Path],
) -> QuantizedLoRAAdapterBank:
    """Load packed adapter files into layer-wise banks without a base model."""
    from .utils import (
        QUANTIZED_LORA_FORMAT,
        QUANTIZED_LORA_VERSION,
        _parse_quantized_lora_layers,
    )

    if not adapters:
        raise ValueError("Adapter bank must not be empty")

    adapter_names = tuple(adapters)
    loaded = []
    reference_quantization = None
    reference_layers = None
    for name, path in adapters.items():
        path = Path(path)
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        quantization = config.get("adapter_quantization")
        if quantization is None:
            raise ValueError(f"Adapter '{name}' is not a quantized LoRA adapter")
        if quantization.get("format") != QUANTIZED_LORA_FORMAT:
            raise ValueError(f"Adapter '{name}' uses an unsupported format")
        if quantization.get("version") != QUANTIZED_LORA_VERSION:
            raise ValueError(f"Adapter '{name}' uses an unsupported version")
        if quantization.get("mode") != "affine":
            raise ValueError("Adapter banks currently support affine mode")

        settings = {
            "group_size": int(quantization["group_size"]),
            "rank_group_size": int(quantization["rank_group_size"]),
            "mode": quantization["mode"],
        }
        layers = _parse_quantized_lora_layers(quantization)
        if reference_quantization is None:
            reference_quantization = settings
            reference_layers = layers
        elif settings != reference_quantization or layers != reference_layers:
            raise ValueError(
                "All adapter-bank entries must share layer and quantization metadata"
            )

        lora_parameters = config.get("lora_parameters", {})
        scale = float(lora_parameters.get("scale", 1.0))
        weights = mx.load(str(path / "adapters.safetensors"))
        weight_layers = {
            key[: -len(".lora_a")] for key in weights if key.endswith(".lora_a")
        }
        if weight_layers != set(layers):
            raise ValueError(
                "Adapter banks require every LoRA layer to use packed "
                f"quantization; adapter '{name}' has incompatible layers"
            )
        loaded.append((scale, weights))

    suffixes = (
        "lora_a",
        "lora_a_scales",
        "lora_a_biases",
        "lora_b",
        "lora_b_scales",
        "lora_b_biases",
    )
    packs = {}
    for layer_name, metadata in reference_layers.items():
        weight_sets = []
        for adapter_name, (_, weights) in zip(adapter_names, loaded):
            missing = [
                suffix for suffix in suffixes if f"{layer_name}.{suffix}" not in weights
            ]
            if missing:
                raise ValueError(
                    f"Adapter '{adapter_name}' layer '{layer_name}' is missing: "
                    + ", ".join(missing)
                )
            weight_sets.append(
                {suffix: weights[f"{layer_name}.{suffix}"] for suffix in suffixes}
            )
        packs[layer_name] = QuantizedLoRAAdapterPack.from_weight_sets(
            weight_sets,
            [scale for scale, _ in loaded],
            group_size=reference_quantization["group_size"],
            rank_group_size=reference_quantization["rank_group_size"],
            bits=metadata["bits"],
            input_dims=metadata["input_dims"],
            output_dims=metadata["output_dims"],
            rank=metadata["rank"],
            padded_input_dims=metadata["padded_input_dims"],
            padded_rank=metadata["padded_rank"],
            mode=reference_quantization["mode"],
        )
    return QuantizedLoRAAdapterBank(adapter_names=adapter_names, packs=packs)
