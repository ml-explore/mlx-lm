# Copyright © 2026 MLX Contributors
# SPDX-License-Identifier: MIT
"""Opt-in tiered OSCAR INT2 key/value cache for native ``mlx-lm``.

The cache keeps three tiers per attention head: an input-dtype sink, write-once
packed INT2 history, and an input-dtype recent window.  The packed state is
portable through :mod:`mlx_lm.models.cache` and can be used by the native
bounded-attention hook in :mod:`mlx_lm.models.base`.

Attribution: the OSCAR algorithm is from FutureMLS-Lab, *OSCAR: Offline
Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization*,
arXiv:2605.17757.  This file is a native ``mlx-lm`` implementation using the
MIT-licensed MLX cache APIs; it is not a copy of the oMLX integration.

Provenance: SGLang was consulted only for behavioral/API comparison while
designing the optional attention seam.  No SGLang source was copied or
adapted here.  Re-check this statement before any redistribution or NOTICE
change.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
import numpy as np

from .cache import KVCache, create_attention_mask

__all__ = [
    "OscarConfig",
    "OscarRotations",
    "OscarKVCache",
    "hadamard",
    "int2_quantize",
    "int2_dequantize",
    "relative_error",
    "load_rotations",
]


_STATE_NAMES = (
    "sink_k",
    "sink_v",
    "hist_k_packed",
    "hist_k_scales",
    "hist_k_zeros",
    "hist_v_packed",
    "hist_v_scales",
    "hist_v_zeros",
    "recent_k",
    "recent_v",
)
_STATE_SIZE = len(_STATE_NAMES)
_ZERO_POINT_LIMIT = 32768.0
_FP32_LEAST_NORMAL = 2.0**-126
_BYTE_WEIGHTS = mx.array([1, 4, 16, 64], dtype=mx.int32)


def _as_float32(value: mx.array) -> mx.array:
    """Use the array method for compatibility with older MLX releases."""
    return value.astype(mx.float32)


def _validate_grouping(dim: int, group_size: int) -> None:
    if group_size < 1:
        raise ValueError(f"OSCAR group_size must be positive, got {group_size}")
    if dim % group_size:
        raise ValueError(
            f"OSCAR head dimension {dim} is not divisible by group_size {group_size}"
        )
    if dim % 4:
        raise ValueError(f"OSCAR head dimension {dim} is not divisible by four")


def hadamard(size: int) -> mx.array | None:
    """Return a normalized Sylvester Hadamard matrix for a power-of-two size."""
    size = int(size)
    if size < 1 or size & (size - 1):
        return None
    result = mx.ones((1, 1), dtype=mx.float32)
    while result.shape[0] < size:
        result = mx.concatenate(
            [
                mx.concatenate([result, result], axis=1),
                mx.concatenate([result, -result], axis=1),
            ],
            axis=0,
        )
    return result / math.sqrt(size)


def int2_quantize(
    values: mx.array,
    group_size: int,
    clip_ratio: float | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize the last dimension to asymmetric four-level packed values."""
    values = _as_float32(values)
    dim = int(values.shape[-1])
    _validate_grouping(dim, int(group_size))
    groups = dim // int(group_size)

    grouped = values.reshape(*values.shape[:-1], groups, group_size)
    if clip_ratio is not None:
        ratio = float(clip_ratio)
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"OSCAR clip_ratio must be in (0, 1), got {ratio}")
        ordered = mx.sort(mx.abs(grouped), axis=-1)
        index = min(int(ratio * group_size), group_size - 1)
        limit = ordered[..., index : index + 1]
        grouped = mx.clip(grouped, -limit, limit)

    minimum = mx.min(grouped, axis=-1, keepdims=True)
    maximum = mx.max(grouped, axis=-1, keepdims=True)
    scale = mx.maximum(maximum - minimum, 1e-8) / 3.0
    scale = mx.maximum(scale, mx.abs(minimum) / _ZERO_POINT_LIMIT)
    zero = -minimum / scale
    levels = mx.floor(grouped / scale + zero + 0.5)
    levels = mx.clip(levels, 0.0, 3.0).astype(mx.int32)
    levels = levels.reshape(*values.shape[:-1], dim)
    packed = (levels.reshape(*values.shape[:-1], dim // 4, 4) * _BYTE_WEIGHTS).sum(
        axis=-1
    )
    return (
        packed.astype(mx.uint8),
        scale.squeeze(-1).astype(mx.float16),
        zero.squeeze(-1).astype(mx.float16),
    )


def int2_dequantize(
    packed: mx.array,
    scales: mx.array,
    zeros: mx.array,
    group_size: int,
) -> mx.array:
    """Dequantize an OSCAR INT2 state to float32."""
    dim = int(packed.shape[-1]) * 4
    _validate_grouping(dim, int(group_size))
    packed_i = packed.astype(mx.int32)
    nibbles = mx.stack(
        [
            packed_i & 3,
            (packed_i >> 2) & 3,
            (packed_i >> 4) & 3,
            (packed_i >> 6) & 3,
        ],
        axis=-1,
    )
    levels = nibbles.reshape(*nibbles.shape[:-2], dim).astype(mx.float32)
    scale = mx.repeat(scales.astype(mx.float32), group_size, axis=-1)
    zero = mx.repeat(zeros.astype(mx.float32), group_size, axis=-1)
    return (levels - zero) * scale


def relative_error(actual: mx.array, expected: mx.array) -> float:
    """Return the Frobenius relative error in float32."""
    actual = _as_float32(actual)
    expected = _as_float32(expected)
    numerator = mx.sqrt(mx.sum((actual - expected) ** 2))
    denominator = mx.maximum(mx.sqrt(mx.sum(actual**2)), _FP32_LEAST_NORMAL)
    return float((numerator / denominator).item())


@dataclass(frozen=True)
class OscarConfig:
    """Explicit settings for :func:`make_oscar_prompt_cache`.

    The configuration is never consulted by the ordinary
    :func:`mlx_lm.models.cache.make_prompt_cache` path.  Callers must pass it
    explicitly, which keeps existing models and prompt caches unchanged.
    """

    group_size: int = 128
    sink_tokens: int = 64
    recent_tokens: int = 256
    rotation_dir: Path | None = None
    absorb_v_rotation: bool = False
    k_clip_ratio: float | None = 0.96
    v_clip_ratio: float | None = 0.92
    bounded_attention: bool = False
    token_norm: bool = False

    def __post_init__(self):
        if int(self.group_size) < 1:
            raise ValueError("OSCAR group_size must be positive")
        if int(self.sink_tokens) < 0 or int(self.recent_tokens) < 0:
            raise ValueError("OSCAR sink_tokens and recent_tokens must be non-negative")
        for name in ("k_clip_ratio", "v_clip_ratio"):
            value = getattr(self, name)
            if value is not None and not 0.0 < float(value) < 1.0:
                raise ValueError(f"OSCAR {name} must be in (0, 1)")
        if self.token_norm:
            raise ValueError(
                "OSCAR token-normalized persistence is not part of the native "
                "ten-entry state contract"
            )


@dataclass(frozen=True)
class OscarRotations:
    """Per-cache-layer K and V rotations loaded from calibration artifacts."""

    rK: tuple[mx.array, ...]
    rV: tuple[mx.array, ...]

    @property
    def k(self) -> tuple[mx.array, ...]:
        return self.rK

    @property
    def v(self) -> tuple[mx.array, ...]:
        return self.rV


def _rotation_matrix(value: mx.array, side: str, name: str) -> mx.array:
    if value.ndim not in (2, 3):
        raise ValueError(
            f"OSCAR {side} rotation {name} must be [D,D] or [H,D,D], "
            f"got {tuple(value.shape)}"
        )
    if value.shape[-1] != value.shape[-2]:
        raise ValueError(f"OSCAR {side} rotation {name} must be square")
    matrix = _as_float32(value)
    mx.eval(matrix)
    return matrix


def _rotation_layers(
    tensors: dict[str, mx.array], num_layers: int, side: str
) -> tuple[mx.array, ...]:
    if "rotation" in tensors:
        matrix = _rotation_matrix(tensors["rotation"], side, "rotation")
        return (matrix,) * num_layers
    numbered = sorted(
        (
            int(key[len("layer_") :]),
            key,
        )
        for key in tensors
        if key.startswith("layer_") and key[len("layer_") :].isdigit()
    )
    if len(numbered) == num_layers:
        keys = [key for _, key in numbered]
    else:
        keys = [f"layer_{index}" for index in range(num_layers)]
    result = []
    for key in keys:
        if key not in tensors:
            raise ValueError(f"OSCAR {side} rotations are missing {key}")
        result.append(_rotation_matrix(tensors[key], side, key))
    return tuple(result)


def load_rotations(directory: str | Path, num_layers: int) -> OscarRotations:
    """Load and materialize calibrated rotations from a directory.

    Preferred files are ``k_rotation.safetensors`` and
    ``v_rotation.safetensors`` with ``layer_0`` ... keys.  A single
    ``rotation.safetensors`` is accepted as a shared K/V fallback.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"OSCAR rotation directory does not exist: {directory}")
    num_layers = int(num_layers)
    if num_layers < 1:
        raise ValueError("OSCAR num_layers must be positive")

    k_path = directory / "k_rotation.safetensors"
    v_path = directory / "v_rotation.safetensors"
    if k_path.exists() and v_path.exists():
        k = _rotation_layers(mx.load(str(k_path)), num_layers, "K")
        v = _rotation_layers(mx.load(str(v_path)), num_layers, "V")
    else:
        shared = directory / "rotation.safetensors"
        if not shared.exists():
            raise FileNotFoundError(
                f"OSCAR rotations require {k_path.name}/{v_path.name} or "
                f"{shared.name} in {directory}"
            )
        matrices = _rotation_layers(mx.load(str(shared)), num_layers, "shared")
        k = v = matrices
    return OscarRotations(k, v)


def _rotation_fingerprint(rotation: mx.array | None) -> str:
    if rotation is None:
        return ""
    values = np.ascontiguousarray(np.asarray(_as_float32(rotation)), dtype=np.float32)
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def _optional_state(value: Any) -> Any:
    """Encode an absent tier as an MLX-safe one-byte rank-1 sentinel."""
    if value is not None:
        return value
    # MLX safetensors rejects empty arrays. Valid OSCAR tiers are rank-4, so
    # this rank-1 sentinel is unambiguous when the state is read back.
    return mx.zeros((1,), dtype=mx.uint8)


def _is_empty_tensor(value: Any) -> bool:
    return (
        hasattr(value, "ndim")
        and int(value.ndim) == 1
        and hasattr(value, "size")
        and int(value.size) <= 1
    )


def _mlx_packed_uint32(packed: mx.array) -> mx.array | None:
    """Repack four persisted bytes into MLX's uint32 INT2 layout."""
    if int(packed.shape[-1]) % 4:
        # MLX's quantized_matmul stores sixteen 2-bit values per uint32. A
        # smaller OSCAR head is still valid, but uses the dense fallback in
        # ``attention`` because it cannot be represented by that kernel.
        return None
    values = packed.astype(mx.uint32)
    return (
        values[..., 0::4]
        | (values[..., 1::4] << 8)
        | (values[..., 2::4] << 16)
        | (values[..., 3::4] << 24)
    )


class OscarKVCache(KVCache):
    """Tiered OSCAR INT2 cache implementing the native cache protocol.

    ``state`` remains a fixed ten-entry tuple for inspection.  Use
    :meth:`serialized_state` when writing a prompt cache: native safetensors
    cannot encode Python ``None`` values, so absent tiers become one-byte
    rank-1 sentinels and are restored as ``None``.  The packed history is
    quantized once, when tokens age out of the recent tier.
    """

    _is_oscar_cache = True

    def __init__(
        self,
        rK: mx.array | None = None,
        rV: mx.array | None = None,
        group_size: int = 128,
        sink_tokens: int = 64,
        recent_tokens: int = 256,
        k_clip_ratio: float | None = 0.96,
        v_clip_ratio: float | None = 0.92,
        absorb_v_rotation: bool = False,
        bounded_attention: bool = False,
        token_norm: bool = False,
    ):
        if token_norm:
            raise ValueError(
                "OSCAR token-normalized persistence is not part of the native "
                "ten-entry state contract"
            )
        self.rK = rK
        self.rV = rV
        self.group_size = int(group_size)
        self.sink_tokens = int(sink_tokens)
        self.recent_tokens = int(recent_tokens)
        self.k_clip_ratio = k_clip_ratio
        self.v_clip_ratio = v_clip_ratio
        self.absorb_v_rotation = bool(absorb_v_rotation)
        self.bounded_attention = bool(bounded_attention)
        self._meta_offset_hint: int | None = None
        self._meta_k_rotation = ""
        self._meta_v_rotation = ""
        self._meta_has_v_rotation = True
        self._update_dtype_name = None
        self.offset = 0
        self._reset_storage()
        OscarConfig(
            group_size=self.group_size,
            sink_tokens=self.sink_tokens,
            recent_tokens=self.recent_tokens,
            k_clip_ratio=self.k_clip_ratio,
            v_clip_ratio=self.v_clip_ratio,
        )

    def _reset_storage(self):
        for name in _STATE_NAMES:
            setattr(self, f"_{name}", None)

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj.rK = None
        obj.rV = None
        obj.group_size = 128
        obj.sink_tokens = 64
        obj.recent_tokens = 256
        obj.k_clip_ratio = 0.96
        obj.v_clip_ratio = 0.92
        obj.absorb_v_rotation = False
        obj.bounded_attention = False
        obj._meta_offset_hint = None
        obj._meta_k_rotation = ""
        obj._meta_v_rotation = ""
        obj._meta_has_v_rotation = False
        obj._update_dtype_name = None
        obj.offset = 0
        obj._reset_storage()
        obj.meta_state = meta_state
        obj.state = state
        return obj

    @staticmethod
    def _rotate(values: mx.array, rotation: mx.array | None) -> mx.array:
        if rotation is None:
            return values
        if values.shape[-1] != rotation.shape[-1]:
            raise ValueError(
                f"OSCAR rotation dimension {rotation.shape[-1]} does not match "
                f"head dimension {values.shape[-1]}"
            )
        return mx.matmul(values, mx.swapaxes(rotation, -1, -2))

    @staticmethod
    def _inverse_rotate(values: mx.array, rotation: mx.array | None) -> mx.array:
        if rotation is None:
            return values
        if values.shape[-1] != rotation.shape[-1]:
            raise ValueError(
                f"OSCAR rotation dimension {rotation.shape[-1]} does not match "
                f"head dimension {values.shape[-1]}"
            )
        return mx.matmul(values, rotation)

    @staticmethod
    def _append(existing: mx.array | None, incoming: mx.array) -> mx.array:
        return incoming if existing is None else mx.concatenate([existing, incoming], axis=2)

    def _pack_history(self, keys: mx.array, values: mx.array) -> None:
        k_rot = self._rotate(_as_float32(keys), self.rK)
        v_rot = self._rotate(_as_float32(values), self.rV)
        packed_k = int2_quantize(k_rot, self.group_size, self.k_clip_ratio)
        packed_v = int2_quantize(v_rot, self.group_size, self.v_clip_ratio)
        for name, tensor in zip(
            (
                "_hist_k_packed",
                "_hist_k_scales",
                "_hist_k_zeros",
                "_hist_v_packed",
                "_hist_v_scales",
                "_hist_v_zeros",
            ),
            packed_k + packed_v,
        ):
            previous = getattr(self, name)
            setattr(
                self,
                name,
                tensor if previous is None else mx.concatenate([previous, tensor], axis=2),
            )

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Append K/V and return the native dense view (or bounded decode view)."""
        keys = mx.asarray(keys)
        values = mx.asarray(values)
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("OSCAR expects K/V with shape [batch, heads, tokens, dim]")
        if keys.shape[:3] != values.shape[:3]:
            raise ValueError("OSCAR K/V batch, head, and token dimensions must match")
        _validate_grouping(int(keys.shape[-1]), self.group_size)
        _validate_grouping(int(values.shape[-1]), self.group_size)
        dtype_name = str(keys.dtype)
        self._update_dtype_name = dtype_name.split(".")[-1]

        previous = self.offset
        total = previous + int(keys.shape[2])
        sink_end = min(self.sink_tokens, total)
        recent_start = max(sink_end, total - self.recent_tokens)

        if self._recent_k is not None:
            recent_begin = previous - int(self._recent_k.shape[2])
            age = min(max(recent_start - recent_begin, 0), int(self._recent_k.shape[2]))
            if age:
                self._pack_history(self._recent_k[..., :age, :], self._recent_v[..., :age, :])
                self._recent_k = self._recent_k[..., age:, :]
                self._recent_v = self._recent_v[..., age:, :]

        sink_take = max(0, sink_end - previous)
        if sink_take:
            self._sink_k = self._append(self._sink_k, keys[..., :sink_take, :])
            self._sink_v = self._append(self._sink_v, values[..., :sink_take, :])

        history_start = max(0, sink_end - previous)
        history_end = max(history_start, recent_start - previous)
        if history_end > history_start:
            self._pack_history(
                keys[..., history_start:history_end, :],
                values[..., history_start:history_end, :],
            )

        recent_start_local = max(previous, recent_start) - previous
        if int(keys.shape[2]) > recent_start_local:
            self._recent_k = self._append(self._recent_k, keys[..., recent_start_local:, :])
            self._recent_v = self._append(self._recent_v, values[..., recent_start_local:, :])

        self.offset = total
        if self.bounded_attention and int(keys.shape[2]) == 1:
            return None, None
        result = self._reconstruct()
        if result is None:
            raise RuntimeError("OSCAR produced an empty cache after a non-empty update")
        return result

    def _reconstruct(self):
        parts_k: list[mx.array] = []
        parts_v: list[mx.array] = []
        reference_dtype = None
        if self._update_dtype_name:
            reference_dtype = getattr(mx, self._update_dtype_name, None)
        if self._sink_k is not None and self._sink_k.shape[2]:
            parts_k.append(self._sink_k)
            parts_v.append(self._sink_v)
            reference_dtype = self._sink_k.dtype
        if self._hist_k_packed is not None and self._hist_k_packed.shape[2]:
            self._verify_rotations()
            history_k = int2_dequantize(
                self._hist_k_packed,
                self._hist_k_scales,
                self._hist_k_zeros,
                self.group_size,
            )
            history_v = int2_dequantize(
                self._hist_v_packed,
                self._hist_v_scales,
                self._hist_v_zeros,
                self.group_size,
            )
            history_k = self._inverse_rotate(history_k, self.rK)
            if not self.absorb_v_rotation:
                history_v = self._inverse_rotate(history_v, self.rV)
            parts_k.append(history_k.astype(reference_dtype) if reference_dtype else history_k)
            parts_v.append(history_v.astype(reference_dtype) if reference_dtype else history_v)
        if self._recent_k is not None and self._recent_k.shape[2]:
            parts_k.append(self._recent_k)
            parts_v.append(self._recent_v)
            if reference_dtype is None:
                reference_dtype = self._recent_k.dtype
        if not parts_k:
            return None
        return mx.concatenate(parts_k, axis=2), mx.concatenate(parts_v, axis=2)

    def _verify_rotations(self) -> None:
        if self._meta_k_rotation and _rotation_fingerprint(self.rK) != self._meta_k_rotation:
            raise ValueError("OSCAR K rotation fingerprint mismatch")
        if self._meta_v_rotation and _rotation_fingerprint(self.rV) != self._meta_v_rotation:
            raise ValueError("OSCAR V rotation fingerprint mismatch")
        if (
            self._meta_k_rotation
            and not self._meta_has_v_rotation
            and _rotation_fingerprint(self.rV) != self._meta_k_rotation
        ):
            raise ValueError(
                "OSCAR legacy metadata has no V rotation fingerprint; "
                "a distinct V rotation cannot be safely inferred"
            )

    def bind_rotations(self, rK: mx.array | None, rV: mx.array | None):
        """Attach calibrated rotations to a state restored from disk."""
        old_k, old_v = self.rK, self.rV
        self.rK = rK
        self.rV = rV
        try:
            self._verify_rotations()
        except Exception:
            self.rK, self.rV = old_k, old_v
            raise
        return self

    def _rotated_queries(self, queries: mx.array) -> mx.array:
        if self.rK is None or self.rK.ndim == 2:
            return self._rotate(queries, self.rK)
        query_heads = int(queries.shape[1])
        kv_heads = int(self.rK.shape[0])
        if query_heads == kv_heads:
            return self._rotate(queries, self.rK)
        if query_heads % kv_heads:
            raise ValueError("OSCAR query heads must be a multiple of rotation heads")
        repeats = query_heads // kv_heads
        grouped = queries.reshape(
            queries.shape[0], kv_heads, repeats, queries.shape[2], queries.shape[3]
        )
        rotation = mx.swapaxes(self.rK, -1, -2)[None, :, None, :, :]
        return mx.matmul(grouped, rotation).reshape(queries.shape)

    def _inverse_rotated_values(self, values: mx.array, query_heads: int) -> mx.array:
        if self.rV is None or self.rV.ndim == 2:
            return self._inverse_rotate(values, self.rV)
        kv_heads = int(self.rV.shape[0])
        if query_heads % kv_heads:
            raise ValueError("OSCAR query heads must be a multiple of rotation heads")
        repeats = query_heads // kv_heads
        grouped = values.reshape(values.shape[0], kv_heads, repeats, values.shape[2], values.shape[3])
        rotation = self.rV[None, :, None, :, :]
        return mx.matmul(grouped, rotation).reshape(values.shape)

    def attention(
        self,
        queries: mx.array,
        *,
        scale: float,
        mask: Any = None,
    ) -> mx.array:
        """Compute attention directly from packed history for bounded decode.

        K/V history is consumed by ``mx.quantized_matmul``.  Only the logits,
        probabilities, and current output are materialized; no dense history
        K/V tensor is built.  This method is called by the opt-in base SDPA
        seam after ``update_and_fetch`` has appended the current token.
        """
        if self.absorb_v_rotation:
            raise ValueError(
                "OSCAR bounded attention requires absorb_v_rotation=False; "
                "fold the V rotation into the output projection first"
            )
        if queries.ndim != 4:
            raise ValueError("OSCAR attention expects queries with shape [B, heads, tokens, dim]")
        query_heads = int(queries.shape[1])
        kv_heads = self._kv_heads()
        if query_heads % kv_heads:
            raise ValueError("OSCAR query heads must be a multiple of KV heads")
        repeats = query_heads // kv_heads
        rotated_queries = self._rotated_queries(queries)
        grouped_queries = rotated_queries.reshape(
            rotated_queries.shape[0], kv_heads, repeats, rotated_queries.shape[2], rotated_queries.shape[3]
        ) * scale

        score_parts: list[mx.array] = []
        value_specs: list[tuple[str, mx.array, int]] = []
        if self._sink_k is not None and self._sink_k.shape[2]:
            keys = self._rotate(_as_float32(self._sink_k), self.rK)
            score_parts.append(mx.matmul(grouped_queries, mx.swapaxes(keys[:, :, None, :, :], -1, -2)))
            value_specs.append(("dense", self._rotate(_as_float32(self._sink_v), self.rV), int(keys.shape[2])))
        if self._hist_k_packed is not None and self._hist_k_packed.shape[2]:
            packed_k = (
                _mlx_packed_uint32(self._hist_k_packed)
                if self.group_size in (32, 64, 128)
                else None
            )
            if packed_k is None:
                history_k = int2_dequantize(
                    self._hist_k_packed,
                    self._hist_k_scales,
                    self._hist_k_zeros,
                    self.group_size,
                )
                score_parts.append(
                    mx.matmul(
                        grouped_queries,
                        mx.swapaxes(history_k[:, :, None, :, :], -1, -2),
                    )
                )
                history_v = int2_dequantize(
                    self._hist_v_packed,
                    self._hist_v_scales,
                    self._hist_v_zeros,
                    self.group_size,
                )
                value_specs.append(("dense", history_v, int(history_k.shape[2])))
            else:
                score_parts.append(
                    mx.quantized_matmul(
                        grouped_queries,
                        packed_k[:, :, None, :, :],
                        self._hist_k_scales.astype(mx.float32)[:, :, None, :, :],
                        self._hist_k_zeros.astype(mx.float32)[:, :, None, :, :],
                        transpose=True,
                        group_size=self.group_size,
                        bits=2,
                    )
                )
                value_specs.append(("quantized", self._hist_v_packed, int(self._hist_k_packed.shape[2])))
        if self._recent_k is not None and self._recent_k.shape[2]:
            keys = self._rotate(_as_float32(self._recent_k), self.rK)
            score_parts.append(mx.matmul(grouped_queries, mx.swapaxes(keys[:, :, None, :, :], -1, -2)))
            value_specs.append(("dense", self._rotate(_as_float32(self._recent_v), self.rV), int(keys.shape[2])))
        if not score_parts:
            raise ValueError("OSCAR attention cannot run on an empty cache")

        scores = mx.concatenate(score_parts, axis=-1)
        if isinstance(mask, str):
            length = int(scores.shape[-1])
            q_length = int(scores.shape[-2])
            q_positions = mx.arange(length - q_length, length)
            mask = q_positions[:, None] >= mx.arange(length)[None, :]
        if mask is not None:
            while mask.ndim < scores.ndim:
                mask = mx.expand_dims(mask, axis=0)
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores = scores + mask
        probabilities = mx.softmax(scores, axis=-1, precise=True)

        outputs: list[mx.array] = []
        position = 0
        for kind, value, length in value_specs:
            weights = probabilities[..., position : position + length]
            if kind == "quantized":
                value = _mlx_packed_uint32(value)
                if value is None:
                    raise RuntimeError("OSCAR value packing did not fit MLX INT2 layout")
                value = value[:, :, None, :, :]
                values = mx.quantized_matmul(
                    weights,
                    value,
                    self._hist_v_scales.astype(mx.float32)[:, :, None, :, :],
                    self._hist_v_zeros.astype(mx.float32)[:, :, None, :, :],
                    transpose=False,
                    group_size=self.group_size,
                    bits=2,
                )
            else:
                values = mx.matmul(weights, value[:, :, None, :, :])
            outputs.append(values)
            position += length
        output = sum(outputs[1:], outputs[0])
        output = output.reshape(output.shape[0], query_heads, output.shape[-2], output.shape[-1])
        return self._inverse_rotated_values(output, query_heads).astype(queries.dtype)

    def _kv_heads(self) -> int:
        for name in ("_sink_k", "_hist_k_packed", "_recent_k"):
            value = getattr(self, name)
            if value is not None:
                return int(value.shape[1])
        if self.rK is not None and self.rK.ndim == 3:
            return int(self.rK.shape[0])
        raise ValueError("OSCAR cannot infer KV heads from an empty cache")

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.offset == 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, num_tokens: int) -> int:
        """Drop the newest ``num_tokens`` tokens, matching native MLX semantics."""
        count = min(max(int(num_tokens), 0), self.offset)
        if count == 0:
            return 0
        recent = int(self._recent_k.shape[2]) if self._recent_k is not None else 0
        history = int(self._hist_k_packed.shape[2]) if self._hist_k_packed is not None else 0
        from_recent = min(count, recent)
        from_history = min(count - from_recent, history)
        from_sink = count - from_recent - from_history

        def cut(name: str, keep: int):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value[..., :keep, :] if keep else None)

        if from_recent:
            cut("_recent_k", recent - from_recent)
            cut("_recent_v", recent - from_recent)
        if from_history:
            keep = history - from_history
            for name in (
                "_hist_k_packed",
                "_hist_k_scales",
                "_hist_k_zeros",
                "_hist_v_packed",
                "_hist_v_scales",
                "_hist_v_zeros",
            ):
                cut(name, keep)
        if from_sink:
            sink = int(self._sink_k.shape[2]) if self._sink_k is not None else 0
            cut("_sink_k", sink - from_sink)
            cut("_sink_v", sink - from_sink)
        self.offset -= count
        self._meta_offset_hint = None
        return count

    @property
    def state(self) -> tuple:
        return tuple(getattr(self, f"_{name}") for name in _STATE_NAMES)

    @state.setter
    def state(self, value):
        if value is None:
            return
        entries = tuple(None if _is_empty_tensor(item) else item for item in tuple(value))
        if len(entries) != _STATE_SIZE:
            raise ValueError(f"OSCAR state requires {_STATE_SIZE} entries")
        for item in entries:
            if item is not None and not hasattr(item, "shape"):
                raise ValueError("OSCAR state entries must be tensors or None")
        for name, item in zip(_STATE_NAMES, entries):
            setattr(self, f"_{name}", item)
        k_length = sum(
            int(getattr(self, name).shape[2])
            for name in ("_sink_k", "_hist_k_packed", "_recent_k")
            if getattr(self, name) is not None
        )
        v_length = sum(
            int(getattr(self, name).shape[2])
            for name in ("_sink_v", "_hist_v_packed", "_recent_v")
            if getattr(self, name) is not None
        )
        if k_length != v_length:
            raise ValueError(f"OSCAR K/V state lengths disagree: {k_length} != {v_length}")
        self.offset = k_length
        self._validate_state()
        self._verify_meta_offset()

    def serialized_state(self) -> tuple:
        return tuple(_optional_state(item) for item in self.state)

    def _validate_state(self) -> None:
        """Reject partial or shape-incompatible packed tiers before use."""
        paired = (("_sink_k", "_sink_v"), ("_recent_k", "_recent_v"))
        for left_name, right_name in paired:
            left = getattr(self, left_name)
            right = getattr(self, right_name)
            if (left is None) != (right is None):
                raise ValueError(f"OSCAR state requires paired {left_name}/{right_name} tiers")

        history = [getattr(self, f"_{name}") for name in _STATE_NAMES[2:8]]
        if any(item is not None for item in history) and not all(
            item is not None for item in history
        ):
            raise ValueError("OSCAR history requires all six packed K/V buffers")

        def shape(name: str, value: mx.array | None):
            if value is None:
                return None
            if value.ndim != 4:
                raise ValueError(f"OSCAR {name} must be rank 4, got {value.ndim}")
            return tuple(int(dimension) for dimension in value.shape)

        sink_k = shape("sink_k", self._sink_k)
        sink_v = shape("sink_v", self._sink_v)
        recent_k = shape("recent_k", self._recent_k)
        recent_v = shape("recent_v", self._recent_v)
        if sink_k and sink_v and sink_k[:3] != sink_v[:3]:
            raise ValueError("OSCAR sink K/V shapes disagree")
        if recent_k and recent_v and recent_k[:3] != recent_v[:3]:
            raise ValueError("OSCAR recent K/V shapes disagree")

        hist_k = hist_v = None
        if all(item is not None for item in history):
            hist_k = shape("hist_k_packed", self._hist_k_packed)
            hist_ks = shape("hist_k_scales", self._hist_k_scales)
            hist_kz = shape("hist_k_zeros", self._hist_k_zeros)
            hist_v = shape("hist_v_packed", self._hist_v_packed)
            hist_vs = shape("hist_v_scales", self._hist_v_scales)
            hist_vz = shape("hist_v_zeros", self._hist_v_zeros)
            assert hist_k and hist_ks and hist_kz and hist_v and hist_vs and hist_vz
            if hist_k[:3] != hist_v[:3]:
                raise ValueError("OSCAR history K/V token shapes disagree")
            for packed, scales, zeros, side in (
                (hist_k, hist_ks, hist_kz, "K"),
                (hist_v, hist_vs, hist_vz, "V"),
            ):
                if scales[:3] != packed[:3] or zeros[:3] != packed[:3]:
                    raise ValueError(f"OSCAR history {side} packed metadata axes disagree")
                if scales[3] != zeros[3]:
                    raise ValueError(f"OSCAR history {side} scale/zero widths disagree")
                if packed[3] * 4 % self.group_size:
                    raise ValueError(f"OSCAR history {side} width is incompatible with group_size")
                if scales[3] != packed[3] * 4 // self.group_size:
                    raise ValueError(f"OSCAR history {side} scale width is incompatible with group_size")
                if str(getattr(getattr(self, f"_hist_{side.lower()}_packed"), "dtype", "")) not in {
                    "uint8",
                    "mlx.core.uint8",
                }:
                    raise ValueError(f"OSCAR history {side} packed data must use uint8")
            # Compare packed channels with sink/recent channels in their
            # logical (unpacked) width below.
            hist_k = (*hist_k[:3], hist_k[3] * 4)
            hist_v = (*hist_v[:3], hist_v[3] * 4)
        for side, values in (("K", (sink_k, hist_k, recent_k)),
                             ("V", (sink_v, hist_v, recent_v))):
            populated = [value for value in values if value is not None]
            if populated:
                reference = populated[0]
                if any(value[:2] != reference[:2] or value[3] != reference[3] for value in populated[1:]):
                    raise ValueError(f"OSCAR {side} tier batch/head/dimension axes disagree")

    @property
    def meta_state(self) -> str:
        return json.dumps(
            {
                "version": 1,
                "offset": int(self.offset),
                "group_size": int(self.group_size),
                "sink_tokens": int(self.sink_tokens),
                "recent_tokens": int(self.recent_tokens),
                "absorb_v_rotation": bool(self.absorb_v_rotation),
                "k_rotation": _rotation_fingerprint(self.rK),
                "v_rotation": _rotation_fingerprint(self.rV),
                "k_clip_ratio": self.k_clip_ratio,
                "v_clip_ratio": self.v_clip_ratio,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @meta_state.setter
    def meta_state(self, value):
        if value is None or value == "":
            return
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                value = value[0]
            elif all(isinstance(item, str) for item in value):
                value = ",".join(value)
        if isinstance(value, dict):
            data = value
            has_v_rotation = "v_rotation" in data
        else:
            text = str(value)
            try:
                data = json.loads(text)
                has_v_rotation = isinstance(data, dict) and "v_rotation" in data
            except json.JSONDecodeError:
                parts = text.split(",")
                if len(parts) < 4:
                    raise ValueError("OSCAR meta_state is not valid JSON or legacy CSV")
                has_v_rotation = len(parts) > 8
                data = {
                    "offset": int(parts[0]),
                    "group_size": int(parts[1]),
                    "sink_tokens": int(parts[2]),
                    "recent_tokens": int(parts[3]),
                    "absorb_v_rotation": bool(int(parts[4])) if len(parts) > 4 else False,
                    "k_rotation": parts[5] if len(parts) > 5 else "",
                    "k_clip_ratio": None if len(parts) <= 6 or parts[6] == "none" else float(parts[6]),
                    "v_clip_ratio": None if len(parts) <= 7 or parts[7] == "none" else float(parts[7]),
                    "v_rotation": parts[8] if len(parts) > 8 else "",
                }
        if not isinstance(data, dict):
            raise ValueError("OSCAR meta_state must be an object")
        self.group_size = int(data.get("group_size", self.group_size))
        self.sink_tokens = int(data.get("sink_tokens", self.sink_tokens))
        self.recent_tokens = int(data.get("recent_tokens", self.recent_tokens))
        self.absorb_v_rotation = bool(data.get("absorb_v_rotation", self.absorb_v_rotation))
        self.k_clip_ratio = data.get("k_clip_ratio", self.k_clip_ratio)
        self.v_clip_ratio = data.get("v_clip_ratio", self.v_clip_ratio)
        self._meta_offset_hint = int(data.get("offset", self.offset))
        self._meta_k_rotation = str(data.get("k_rotation", ""))
        self._meta_v_rotation = str(data.get("v_rotation", ""))
        self._meta_has_v_rotation = has_v_rotation
        self._verify_meta_offset()

    def _verify_meta_offset(self):
        if not any(getattr(self, f"_{name}") is not None for name in _STATE_NAMES):
            return
        if self._meta_offset_hint is not None and self._meta_offset_hint != self.offset:
            raise ValueError(
                f"OSCAR meta_state offset {self._meta_offset_hint} does not match "
                f"state length {self.offset}"
            )

    def make_mask(self, N: int, window_size: int | None = None, return_array: bool = False):
        return create_attention_mask(N, offset=self.offset, return_array=return_array, window_size=window_size)

    def filter(self, batch_indices):
        for name in _STATE_NAMES:
            value = getattr(self, f"_{name}")
            if value is not None:
                setattr(self, f"_{name}", value[batch_indices])

    def extend(self, other: "OscarKVCache"):
        if not isinstance(other, OscarKVCache) or other.offset != self.offset:
            raise ValueError("OSCAR extend requires same-offset OscarKVCache instances")
        for name in _STATE_NAMES:
            left = getattr(self, f"_{name}")
            right = getattr(other, f"_{name}")
            if left is None and right is None:
                continue
            if left is None or right is None:
                raise ValueError("OSCAR extend requires matching populated tiers")
            setattr(self, f"_{name}", mx.concatenate([left, right], axis=0))

    def extract(self, index: int) -> "OscarKVCache":
        clone = OscarKVCache(
            rK=self.rK,
            rV=self.rV,
            group_size=self.group_size,
            sink_tokens=self.sink_tokens,
            recent_tokens=self.recent_tokens,
            k_clip_ratio=self.k_clip_ratio,
            v_clip_ratio=self.v_clip_ratio,
            absorb_v_rotation=self.absorb_v_rotation,
            bounded_attention=self.bounded_attention,
        )
        clone.state = tuple(
            value[index : index + 1] if value is not None else None
            for value in self.state
        )
        clone.offset = self.offset
        clone._meta_offset_hint = self._meta_offset_hint
        clone._meta_k_rotation = self._meta_k_rotation
        clone._meta_v_rotation = self._meta_v_rotation
        clone._meta_has_v_rotation = self._meta_has_v_rotation
        return clone

    @classmethod
    def merge(cls, caches: Sequence["OscarKVCache"]):
        if not caches or any(not isinstance(cache, cls) for cache in caches):
            raise ValueError("OSCAR merge requires at least one OscarKVCache")
        first = caches[0]
        if any(cache.offset != first.offset for cache in caches[1:]):
            raise ValueError("OSCAR merge requires same-offset caches")
        merged = first.extract(0)
        for cache in caches[1:]:
            merged.extend(cache)
        return merged

    @property
    def nbytes(self) -> int:
        return sum(int(value.nbytes) for value in self.state if value is not None)
