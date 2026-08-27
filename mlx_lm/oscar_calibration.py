# Copyright © 2026 MLX Contributors
# SPDX-License-Identifier: MIT
"""Offline OSCAR rotation calibration for native ``mlx-lm``.

This module intentionally consumes captured K/V tensors instead of importing
the runtime or a product-specific model loader.  A calibration job can run
offline, write portable safetensors rotations, and hand those artifacts to
``OscarConfig.rotation_dir``.

The rotation recipe follows FutureMLS-Lab OSCAR, arXiv:2605.17757.  The
implementation is authored for native ``mlx-lm``.  SGLang was used only as a
behavioral comparison; no SGLang source was copied or adapted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np

from .models.oscar import OscarRotations, hadamard

__all__ = [
    "CalibrationError",
    "bit_reversal_permutation",
    "derive_rotation",
    "calibrate_rotations",
    "save_rotations",
    "load_calibration_samples",
]


class CalibrationError(ValueError):
    """Raised when captured calibration statistics are unusable."""


def bit_reversal_permutation(size: int) -> mx.array:
    """Return the bit-reversal permutation matrix used by OSCAR."""
    size = int(size)
    if size < 1 or size & (size - 1):
        raise CalibrationError(f"head dimension must be a power of two, got {size}")
    bits = size.bit_length() - 1
    indices = [int(f"{index:0{bits}b}"[::-1], 2) for index in range(size)]
    result = np.zeros((size, size), dtype=np.float64)
    result[np.arange(size), indices] = 1.0
    return mx.array(result.astype(np.float32))


def _validate_samples(samples: mx.array, name: str) -> np.ndarray:
    array = np.asarray(samples.astype(mx.float32))
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4:
        raise CalibrationError(f"{name} must have shape [B,H,T,D] or [H,T,D]")
    if not np.isfinite(array).all():
        raise CalibrationError(f"{name} contains NaN or Inf")
    if array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise CalibrationError(f"{name} has an empty batch, head, or token axis")
    return array


def _orthogonal_rotation(covariance: np.ndarray, own_moment: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    vectors = vectors[:, np.argsort(values)[::-1]]
    own_energy = np.diag(vectors.T @ own_moment @ vectors)
    vectors = vectors[:, np.argsort(own_energy)[::-1]]
    size = covariance.shape[0]
    h_array = hadamard(size)
    if h_array is None:
        raise CalibrationError(f"head dimension must be a power of two, got {size}")
    h = np.asarray(h_array.astype(mx.float32), dtype=np.float64)
    permutation = np.asarray(bit_reversal_permutation(size), dtype=np.float64)
    composed = vectors @ h @ permutation
    # Canonicalize the same input-side sign convention as the reference.
    pivots = np.argmax(np.abs(composed.T), axis=0)
    signs = np.where(composed.T[pivots, np.arange(size)] >= 0, 1.0, -1.0)
    composed = composed.T * signs
    u, _, vt = np.linalg.svd(composed, full_matrices=False)
    rotation = u @ vt
    deviation = np.max(np.abs(rotation @ rotation.T - np.eye(size)))
    if not np.isfinite(deviation) or deviation > 1e-5:
        raise CalibrationError(
            f"derived rotation is not orthogonal (max deviation {deviation:.3g})"
        )
    return rotation.astype(np.float32)


def derive_rotation(samples: mx.array, own_samples: mx.array | None = None) -> mx.array:
    """Derive one ``[D,D]`` rotation from samples or covariance matrices.

    The sample form is ``[B,1,T,D]`` (or ``[1,T,D]``). For callers that have
    already reduced offline statistics, a square ``[D,D]`` covariance and an
    optional own-moment matrix are accepted as well.
    """
    if getattr(samples, "ndim", None) == 2:
        covariance = np.asarray(samples.astype(mx.float32), dtype=np.float64)
        own_moment = (
            covariance
            if own_samples is None
            else np.asarray(own_samples.astype(mx.float32), dtype=np.float64)
        )
        if covariance.shape[0] != covariance.shape[1] or own_moment.shape != covariance.shape:
            raise CalibrationError("covariance and own_moment must be matching square matrices")
        return mx.array(_orthogonal_rotation(covariance, own_moment))
    values = _validate_samples(samples, "samples")
    own = values if own_samples is None else _validate_samples(own_samples, "own_samples")
    if values.shape[1:] != own.shape[1:]:
        raise CalibrationError("samples and own_samples must have matching H/T/D shapes")
    covariance = np.einsum("bhtd,bhte->hde", values, values)
    own_moment = np.einsum("bhtd,bhte->hde", own, own)
    if covariance.shape[0] != 1:
        raise CalibrationError("derive_rotation expects one KV head; use calibrate_rotations for GQA")
    return mx.array(_orthogonal_rotation(covariance[0], own_moment[0]))


def _as_layers(value: mx.array | Sequence[mx.array], name: str) -> list[np.ndarray]:
    if isinstance(value, (list, tuple)):
        layers = [_validate_samples(item, f"{name}[{index}]") for index, item in enumerate(value)]
    else:
        layers = [_validate_samples(value, name)]
    if not layers:
        raise CalibrationError(f"{name} contains no layers")
    return layers


def calibrate_rotations(
    keys: mx.array | Sequence[mx.array],
    values: mx.array | Sequence[mx.array],
) -> OscarRotations:
    """Calibrate per-KV-head K/V rotations from offline captured tensors.

    Each input is either ``[B,H,T,D]`` for one attention layer or a sequence
    of such arrays.  The returned rotations contain one stacked ``[H,D,D]``
    matrix per layer and can be saved directly with :func:`save_rotations`.
    """
    key_layers = _as_layers(keys, "keys")
    value_layers = _as_layers(values, "values")
    if len(key_layers) != len(value_layers):
        raise CalibrationError("keys and values must contain the same number of layers")
    key_rotations, value_rotations = [], []
    for layer, (key, value) in enumerate(zip(key_layers, value_layers)):
        if key.shape != value.shape:
            raise CalibrationError(
                f"layer {layer}: keys and values have different shapes "
                f"{key.shape} and {value.shape}"
            )
        if key.shape[-1] < 4 or key.shape[-1] & (key.shape[-1] - 1):
            raise CalibrationError(
                f"layer {layer}: head dimension must be a power of two >= 4"
            )
        k_cov = np.einsum("bhtd,bhte->hde", key, key)
        v_cov = np.einsum("bhtd,bhte->hde", value, value)
        k_stack, v_stack = [], []
        for head in range(key.shape[1]):
            k_stack.append(_orthogonal_rotation(k_cov[head], k_cov[head]))
            v_stack.append(_orthogonal_rotation(v_cov[head], v_cov[head]))
        key_rotations.append(mx.array(np.stack(k_stack, axis=0)))
        value_rotations.append(mx.array(np.stack(v_stack, axis=0)))
    mx.eval(key_rotations, value_rotations)
    return OscarRotations(tuple(key_rotations), tuple(value_rotations))


def save_rotations(
    directory: str | Path,
    rotations: OscarRotations,
    *,
    layer_indices: Sequence[int] | None = None,
) -> Path:
    """Write calibrated rotations and a small provenance manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if len(rotations.rK) != len(rotations.rV):
        raise CalibrationError("K and V rotation counts differ")
    if layer_indices is None:
        layer_indices = list(range(len(rotations.rK)))
    if len(layer_indices) != len(rotations.rK):
        raise CalibrationError("layer_indices count does not match rotations")
    k = {f"layer_{index}": matrix for index, matrix in zip(layer_indices, rotations.rK)}
    v = {f"layer_{index}": matrix for index, matrix in zip(layer_indices, rotations.rV)}
    mx.save_safetensors(str(directory / "k_rotation.safetensors"), k)
    mx.save_safetensors(str(directory / "v_rotation.safetensors"), v)
    manifest = {
        "schema": "mlx-lm-oscar-rotations-v1",
        "algorithm": "FutureMLS-Lab OSCAR",
        "paper": "arXiv:2605.17757",
        "layer_indices": list(layer_indices),
        "provenance": {
            "implementation": "native mlx-lm",
            "sglang": "behavioral/API comparison only; no source copied or adapted",
        },
    }
    (directory / "metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return directory


def load_calibration_samples(path: str | Path) -> tuple[mx.array, mx.array]:
    """Load ``keys`` and ``values`` tensors from an MLX safetensors file."""
    tensors = mx.load(str(path))
    try:
        return tensors["keys"], tensors["values"]
    except KeyError as exc:
        raise CalibrationError("sample file must contain 'keys' and 'values'") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate native OSCAR INT2 rotations")
    parser.add_argument("--samples", required=True, help="safetensors file containing keys and values")
    parser.add_argument("--out", required=True, help="output directory for rotation artifacts")
    args = parser.parse_args(argv)
    try:
        keys, values = load_calibration_samples(args.samples)
        save_rotations(args.out, calibrate_rotations(keys, values))
    except (CalibrationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
