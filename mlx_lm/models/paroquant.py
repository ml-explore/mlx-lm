# Copyright © 2025 Apple Inc.

"""ParoQuant (Pairwise Rotation Quantization) inference layers.

ParoQuant [ICLR 2026, arXiv:2511.10645] applies a learned sequence of pairwise
Givens rotations to the activations before an affine-quantized matmul. The
rotation decorrelates channels so that low-bit (e.g. 4-bit) quantization keeps
more of the signal. At inference the rotation is a cheap Metal kernel followed by
the standard ``mx.quantized_matmul``.

This module provides the inference-time layers; weight loading and layer
swapping live in ``mlx_lm.utils`` (``_transform_paro_weights`` /
``_patch_paro_layers``), dispatched from ``load_model`` when a model's
``quantization_config`` has ``"quant_method": "paroquant"``.
"""

import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

_MAX_GROUP_SIZE = 128
_MAX_KROT = 16

_ROTATION_SOURCE = """
    constexpr int ROWS_PER_TILE = {ROWS_PER_TILE};
    constexpr int MAX_KROT      = {MAX_KROT};

    const int batch_size  = params[0];
    const int hidden_size = params[1];
    const int krot        = params[2];
    const int group_size  = params[3];

    const int half_gs     = group_size / 2;
    const int half_hidden = hidden_size / 2;

    const int tile_idx  = threadgroup_position_in_grid.x;
    const int group_idx = threadgroup_position_in_grid.y;
    const int tid       = thread_index_in_threadgroup;

    if (tid >= half_gs) return;

    // ---- Load rotation coefficients into registers ----
    float cos_vals[MAX_KROT], sin_vals[MAX_KROT];
    int   pair_vals[MAX_KROT];

    for (int k = 0; k < krot; k++) {{
        int idx = k * half_hidden + group_idx * half_gs + tid;
        cos_vals[k]  = float(cos_theta[idx]);
        sin_vals[k]  = float(sin_theta[idx]);
        pair_vals[k] = int(packed_pairs[idx]);
    }}

    // ---- Load activation tile into shared memory (fuse channel scales) ----
    threadgroup float tile[{MAX_GROUP_SIZE} * ROWS_PER_TILE];

    const int ch_lo = group_idx * group_size + tid;
    const int ch_hi = ch_lo + half_gs;
    float scale_lo = float(channel_scales[ch_lo]);
    float scale_hi = float(channel_scales[ch_hi]);

    for (int r = 0; r < ROWS_PER_TILE; r++) {{
        int row = tile_idx * ROWS_PER_TILE + r;
        if (row < batch_size) {{
            tile[tid * ROWS_PER_TILE + r]              = float(x[row * hidden_size + ch_lo]) * scale_lo;
            tile[(tid + half_gs) * ROWS_PER_TILE + r]  = float(x[row * hidden_size + ch_hi]) * scale_hi;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Apply pairwise Givens rotations in-place ----
    for (int k = 0; k < krot; k++) {{
        int i_local = pair_vals[k] & 0xFFFF;
        int j_local = pair_vals[k] >> 16;
        float c = cos_vals[k], s = sin_vals[k];

        for (int m = 0; m < ROWS_PER_TILE; m++) {{
            float a = tile[i_local * ROWS_PER_TILE + m];
            float b = tile[j_local * ROWS_PER_TILE + m];
            tile[i_local * ROWS_PER_TILE + m] = a * c + b * s;
            tile[j_local * ROWS_PER_TILE + m] = b * c - a * s;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // ---- Write results back ----
    for (int r = 0; r < ROWS_PER_TILE; r++) {{
        int row = tile_idx * ROWS_PER_TILE + r;
        if (row < batch_size) {{
            out[row * hidden_size + ch_lo] = tile[tid * ROWS_PER_TILE + r];
            out[row * hidden_size + ch_hi] = tile[(tid + half_gs) * ROWS_PER_TILE + r];
        }}
    }}
"""


def _check_kernel_limits(krot: int, group_size: int):
    """The Metal kernel statically sizes its register arrays (MAX_KROT) and
    threadgroup memory (MAX_GROUP_SIZE) at compile time, so these are hard upper
    bounds — exceeding them would silently corrupt memory. Fail loudly instead.
    """
    if krot > _MAX_KROT:
        raise ValueError(
            f"ParoQuant krot={krot} exceeds the kernel limit MAX_KROT={_MAX_KROT}; "
            "raise _MAX_KROT in paroquant.py and recompile."
        )
    if group_size > _MAX_GROUP_SIZE:
        raise ValueError(
            f"ParoQuant group_size={group_size} exceeds the kernel limit "
            f"MAX_GROUP_SIZE={_MAX_GROUP_SIZE}; raise _MAX_GROUP_SIZE in paroquant.py."
        )


@lru_cache(maxsize=None)
def _get_rotation_kernel(rows_per_tile: int):
    """Compile and cache the Metal rotation kernel for a given tile size."""
    return mx.fast.metal_kernel(
        name=f"paro_rotate_r{rows_per_tile}",
        input_names=[
            "x",
            "packed_pairs",
            "cos_theta",
            "sin_theta",
            "channel_scales",
            "params",
        ],
        output_names=["out"],
        source=_ROTATION_SOURCE.format(
            ROWS_PER_TILE=rows_per_tile,
            MAX_GROUP_SIZE=_MAX_GROUP_SIZE,
            MAX_KROT=_MAX_KROT,
        ),
    )


def pack_pairs(pairs: mx.array, group_size: int) -> mx.array:
    """Pack int16 pair indices into int32 (two per lane) for the Metal kernel."""
    krot, hidden = int(pairs.shape[0]), int(pairs.shape[1])
    p = pairs.reshape(krot, hidden // group_size, group_size).astype(mx.int32)
    return (p[:, :, 0::2] | (p[:, :, 1::2] << 16)).reshape(krot, -1)


def apply_rotation(
    x: mx.array,
    packed_pairs: mx.array,
    cos: mx.array,
    sin: mx.array,
    scales_flat: mx.array,
    dim: int,
    krot: int,
    group_size: int,
) -> mx.array:
    """Dispatch the pairwise-rotation kernel on a 2-D (batch, dim) tensor."""
    batch = x.shape[0]
    if batch == 0:
        return x
    tile = 1 if batch <= 1 else 4
    half_group = group_size // 2
    num_groups = dim // group_size
    params = mx.array([batch, dim, krot, group_size], dtype=mx.int32)
    grid = (math.ceil(batch / tile) * half_group, num_groups, 1)
    return _get_rotation_kernel(tile)(
        inputs=[x, packed_pairs, cos, sin, scales_flat, params],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=grid,
        threadgroup=(half_group, 1, 1),
    )[0]


class RotateQuantizedLinear(nn.Module):
    """Pairwise Givens rotation followed by an affine quantized matmul.

    The rotation parameters (``theta``, ``pairs``, ``channel_scales``) and the
    quantized weight (``weight``, ``scales``, ``biases``) are filled in by
    ``model.load_weights`` after this layer replaces an ``nn.Linear``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 128,
        bits: int = 4,
        krot: int = 8,
    ):
        super().__init__()
        _check_kernel_limits(krot, group_size)
        self.group_size = group_size
        self.bits = bits

        self.theta = mx.zeros((krot, input_dims // 2))
        self.pairs = mx.zeros((krot, input_dims), dtype=mx.int16)
        self.channel_scales = mx.ones((1, input_dims))

        self.weight = mx.zeros((output_dims, input_dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // group_size))
        self.biases = mx.zeros((output_dims, input_dims // group_size))

        if bias:
            self.bias = mx.zeros((output_dims,))

        self._cached = False
        # Match nn.QuantizedLinear / QuantizedSwitchLinear: quantized + rotation
        # params are not trainable, so tuner/LoRA skips them like any quant layer.
        self.freeze()

    def _cache_rotation(self):
        """Pre-compute sin/cos and pack pairs (called once on first forward)."""
        self._dim = self.theta.shape[1] * 2
        self._krot = int(self.theta.shape[0])
        self._cos = mx.cos(self.theta)
        self._sin = mx.sin(self.theta)
        self._packed_pairs = pack_pairs(self.pairs, self.group_size)
        self._scales_flat = self.channel_scales.reshape(-1)
        self._cached = True

    def __call__(self, x: mx.array) -> mx.array:
        if not self._cached:
            self._cache_rotation()

        shape = x.shape
        rotated = apply_rotation(
            x.reshape(-1, self._dim),
            self._packed_pairs,
            self._cos,
            self._sin,
            self._scales_flat,
            self._dim,
            self._krot,
            self.group_size,
        )

        y = mx.quantized_matmul(
            rotated.reshape(shape),
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        if "bias" in self:
            y = y + self.bias
        return y


class _CachedRotation:
    """Helper for modules that own one or more named rotations.

    A "rotation" is the triple (theta, pairs, channel_scales) plus the cached
    sin/cos and packed pairs derived from them. ``prefix`` namespaces multiple
    rotations on the same module (e.g. ``gate_up_rot`` and ``down_rot``).
    """

    def _init_rotation(self, krot: int, dim: int, group_size: int, prefix: str = ""):
        _check_kernel_limits(krot, group_size)
        pfx = f"{prefix}_" if prefix else ""
        setattr(self, f"{pfx}theta", mx.zeros((krot, dim // 2)))
        setattr(self, f"{pfx}pairs", mx.zeros((krot, dim), dtype=mx.int16))
        setattr(self, f"{pfx}channel_scales", mx.ones((1, dim)))
        self._rot_group_size = group_size

    def _cache_single_rotation(self, prefix: str = ""):
        pfx = f"{prefix}_" if prefix else ""
        theta = getattr(self, f"{pfx}theta")
        dim = int(theta.shape[1]) * 2
        krot = int(theta.shape[0])
        tag = f"_{prefix}" if prefix else ""
        setattr(self, f"_rot{tag}_dim", dim)
        setattr(self, f"_rot{tag}_krot", krot)
        setattr(self, f"_rot{tag}_cos", mx.cos(theta))
        setattr(self, f"_rot{tag}_sin", mx.sin(theta))
        setattr(
            self,
            f"_rot{tag}_packed_pairs",
            pack_pairs(getattr(self, f"{pfx}pairs"), self._rot_group_size),
        )
        setattr(
            self,
            f"_rot{tag}_scales_flat",
            getattr(self, f"{pfx}channel_scales").reshape(-1),
        )

    def _rotate(self, x: mx.array, prefix: str = "") -> mx.array:
        tag = f"_{prefix}" if prefix else ""
        dim = getattr(self, f"_rot{tag}_dim")
        shape = x.shape
        rotated = apply_rotation(
            x.reshape(-1, dim),
            getattr(self, f"_rot{tag}_packed_pairs"),
            getattr(self, f"_rot{tag}_cos"),
            getattr(self, f"_rot{tag}_sin"),
            getattr(self, f"_rot{tag}_scales_flat"),
            dim,
            getattr(self, f"_rot{tag}_krot"),
            self._rot_group_size,
        )
        return rotated.reshape(shape)


class RotateSwitchGLU(nn.Module, _CachedRotation):
    """``SwitchGLU`` with a shared pairwise rotation before each projection.

    All experts share one rotation per projection: ``gate_up_rot`` is applied to
    the input before gate/up, and ``down_rot`` is applied to the activation
    before the down projection. The expert weights themselves are quantized
    ``QuantizedSwitchLinear`` layers.
    """

    def __init__(self, glu: nn.Module, group_size: int, krot: int):
        super().__init__()
        self.gate_proj = glu.gate_proj
        self.up_proj = glu.up_proj
        self.down_proj = glu.down_proj
        self.activation = glu.activation

        self._init_rotation(krot, glu.gate_proj.input_dims, group_size, "gate_up_rot")
        self._init_rotation(krot, glu.down_proj.input_dims, group_size, "down_rot")
        self._cached = False
        # Rotation params (and the already-frozen quantized experts) are not
        # trainable; freeze to match the standard quantized switch layers.
        self.freeze()

    def _cache_rotation(self):
        self._cache_single_rotation("gate_up_rot")
        self._cache_single_rotation("down_rot")
        self._cached = True

    def __call__(self, x, indices) -> mx.array:
        if not self._cached:
            self._cache_rotation()

        from .switch_layers import _gather_sort, _scatter_unsort

        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)

        x = self._rotate(x, "gate_up_rot")

        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)

        act = self.activation(x_up, x_gate)
        act = self._rotate(act, "down_rot")

        x = self.down_proj(act, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
