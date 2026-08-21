# Copyright © 2023-2024 Apple Inc.

import math
from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn as nn


class SuScaledRoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float = 10000.0,
        max_position_embeddings: int = 131072,
        original_max_position_embeddings: int = 4096,
        short_factor: Union[List[float], float] = 1.0,
        long_factor: Union[List[float], float] = 1.0,
        short_mscale: float = None,
        long_mscale: float = None,
    ):
        """
        Su Scaled Rotary Embedding layer.

        Args:
            dims (int): The feature dimensions to be rotated.
            base (int, optional): Base for the exponential scaling.
            max_position_embeddings (int, optional): The maximum sequence
              length that this model was trained with. This is used to determine
              the size of the original RoPE embeddings when using long scaling.
              Default: ``131072``.
            original_max_position_embeddings (int, optional): The maximum
              sequence length that this model was trained with. This is used to
              determine the size of the original RoPE embeddings when using long
              scaling. Default: ``4096``.
            short_factor (float or list[float], optional): List of scaling
              factors for sequences of length lesser than
              ``original_max_position_embeddings``. Default: ``1.0``.
            long_factor (float or list[float], optional): List of scaling
              factors for sequences of length greater than
              ``original_max_position_embeddings``.  Default: ``1.0``.
            short_mscale (float, optional): Scale the input prior to embedding.
            long_mscale (float, optional): Scale the input prior to embedding.
        """
        super().__init__()
        self.original_max_position_embeddings = original_max_position_embeddings
        self.dim = dims

        freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        self._freqs = mx.array(long_factor, dtype=mx.float32) * freqs

        def default_scale(factor):
            return math.sqrt(
                1 + math.log(factor) / math.log(original_max_position_embeddings)
            )

        factor = max_position_embeddings / original_max_position_embeddings
        self._scale = long_mscale or (1.0 if factor <= 1.0 else default_scale(factor))

    def __call__(self, x, offset: Union[int, mx.array] = 0):
        x = x[...]
        x[..., : self.dim] = self._scale * x[..., : self.dim]
        return mx.fast.rope(
            x,
            self.dim,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class Llama3RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scaling_config: dict = None,
    ):
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional

        factor = scaling_config["factor"]
        low_freq_factor = scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = scaling_config.get("high_freq_factor", 4.0)
        old_context_len = scaling_config.get(
            "original_max_position_embeddings",
            8192,
        )

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        freqs = base ** (mx.arange(0, dims, 2) / dims)
        wavelens = 2 * mx.pi * freqs

        freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
        smooth_factors = (old_context_len / wavelens - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)

    def extra_repr(self):
        return (
            f"{self.dims}, traditional={self.traditional}, "
            f"max_position_embeddings={self.max_position_embeddings}"
        )

    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class YarnRoPE(nn.Module):
    def __init__(
        self,
        dims,
        traditional=False,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        super().__init__()

        def yarn_find_correction_dim(num_rotations):
            return (
                dims
                * math.log(
                    original_max_position_embeddings / (num_rotations * 2 * math.pi)
                )
            ) / (2 * math.log(base))

        def yarn_find_correction_range():
            low = math.floor(yarn_find_correction_dim(beta_fast))
            high = math.ceil(yarn_find_correction_dim(beta_slow))
            return max(low, 0), min(high, dims - 1)

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        def yarn_linear_ramp_mask(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001  # Prevent singularity

            linear_func = (mx.arange(dim, dtype=mx.float32) - min_val) / (
                max_val - min_val
            )
            return mx.clip(linear_func, 0, 1)

        self.mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(
            scaling_factor, mscale_all_dim
        )
        freq_extra = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        freq_inter = scaling_factor * freq_extra
        low, high = yarn_find_correction_range()
        freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dims // 2)
        self._freqs = (freq_inter * freq_extra) / (
            freq_inter * freq_mask + freq_extra * (1 - freq_mask)
        )
        self.dims = dims
        self.traditional = traditional

    def __call__(self, x, offset=0):
        if self.mscale != 1.0:
            x = x[...]
            x[..., : self.dims] = self.mscale * x[..., : self.dims]
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class ProportionalRoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        rotated_dims: int,
        traditional: bool = False,
        base: float = 10000.0,
        factor: float = 1.0,
    ):
        super().__init__()
        self.dims = dims
        self.traditional = traditional

        if rotated_dims > dims:
            raise ValueError("rotated_dims should be smaller than dims")

        exponents = mx.arange(0, rotated_dims, 2, dtype=mx.float32) / dims
        self._freqs = mx.concatenate(
            [
                factor * (base**exponents),
                mx.full(((dims - rotated_dims) // 2,), mx.inf),
            ]
        )

    def __call__(self, x, offset=0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class DynamicNTKScalingRoPE(nn.Module):

    def __init__(
        self,
        dims: int,
        max_position_embeddings: int,
        traditional: bool,
        base: float,
        factor: float,
    ):
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional
        self.base = base
        self.factor = factor

    def extra_repr(self) -> str:
        return (
            f"{self.dims}, traditional={self.traditional}, "
            f"max_position_embeddings={self.max_position_embeddings}, "
            f"factor={self.factor}"
        )

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        # x.shape: [batch, num_heads, seq_len, head_dim]
        seq_len = max(x.shape[-2] + offset, self.max_position_embeddings)
        base = self.base * (
            (self.factor * seq_len / self.max_position_embeddings) - (self.factor - 1)
        ) ** (self.dims / (self.dims - 2))
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=base,
            scale=1.0,
            offset=offset,
        )


class MRoPE(nn.Module):
    """Multimodal RoPE (M-RoPE), as used by the Qwen2-VL / Qwen3-VL family.

    Standard RoPE derives a token's rotation angle from a single scalar
    position. M-RoPE derives it from a 3-vector ``(t, h, w)`` -- temporal,
    height and width -- so that image patches carry their grid coordinates
    instead of their flattened index.

    For text-only input all three components equal the sequence index, and
    M-RoPE reduces exactly to 1-D RoPE. That is why text generation is
    unaffected by which of the two is used, and why this class falls back to
    the fused ``nn.RoPE`` kernel whenever ``position_ids`` is ``None``: the
    text path keeps its current speed and stays bit-identical.

    The two only diverge when image or video tokens are present, which is the
    case when embeddings produced by a vision encoder are supplied through the
    ``input_embeddings`` argument of ``generate_step``.

    Args:
        dims (int): Number of feature dimensions to rotate. Any remaining
          dimensions are passed through unchanged (partial rotary).
        base (float): Base for the exponential frequency scaling.
        mrope_section (list[int]): Per-axis frequency budget ``[t, h, w]``.
        traditional (bool): Traditional (interleaved-pair) rotation. Only
          used on the ``position_ids is None`` fallback path.
    """

    def __init__(
        self,
        dims: int,
        base: float = 10000.0,
        mrope_section: Optional[List[int]] = None,
        traditional: bool = False,
    ):
        super().__init__()
        self.dims = dims
        self.mrope_section = list(mrope_section or [])
        # Fallback used for text-only input; keeps the fused Metal kernel.
        self._rope = nn.RoPE(dims, traditional=traditional, base=base)

        half = dims // 2
        self._inv_freq = base ** (-mx.arange(0, half, dtype=mx.float32) / half)
        # selector[i] says which of (t, h, w) supplies the position for
        # frequency i. Qwen interleaves them as t,h,w,t,h,w,... over the span
        # covered by mrope_section, with the tail falling back to t.
        selector = [0] * half
        for axis, start in enumerate((1, 2), start=1):
            for i in range(start, min(self.mrope_section[axis] * 3, half), 3):
                selector[i] = axis
        self._selector = mx.array(selector, dtype=mx.int32)

    def __call__(
        self,
        x: mx.array,
        offset: int = 0,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if position_ids is None:
            # Text-only: identical to the behaviour before this class existed.
            return self._rope(x, offset=offset)

        # position_ids: (3, B, L) -> per-frequency positions (B, L, dims // 2)
        freqs = mx.take(position_ids, self._selector, axis=0).transpose(1, 2, 0)
        freqs = freqs.astype(mx.float32) * self._inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.expand_dims(mx.cos(emb), axis=1).astype(x.dtype)
        sin = mx.expand_dims(mx.sin(emb), axis=1).astype(x.dtype)

        # Partial rotary: rotate the leading `dims` features, pass the rest on.
        rot, passthrough = x[..., : self.dims], x[..., self.dims :]
        half = self.dims // 2
        rotated = mx.concatenate([-rot[..., half:], rot[..., :half]], axis=-1)
        out = rot * cos + rotated * sin
        if passthrough.shape[-1] > 0:
            out = mx.concatenate([out, passthrough], axis=-1)
        return out


def initialize_rope(
    dims,
    base,
    traditional,
    scaling_config: Optional[dict] = None,
    max_position_embeddings: Optional[int] = None,
):
    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get(
            "rope_type", "default"
        )
    else:
        rope_type = "default"

    # M-RoPE is signalled by the presence of `mrope_section`, not by the rope
    # "type": Qwen3-VL / Qwen3.5 configs still report their type as "default"
    # while carrying an mrope_section, so keying off the type alone misses them.
    mrope_section = (scaling_config or {}).get("mrope_section")
    if mrope_section or rope_type == "mrope":
        assert (
            mrope_section is not None and len(mrope_section) == 3
        ), f"MRoPE currently only supports 3 sections, got {mrope_section}."
        return MRoPE(
            dims,
            base=base,
            mrope_section=mrope_section,
            traditional=traditional,
        )

    if rope_type in ["default", "linear"]:
        scale = 1 / scaling_config["factor"] if rope_type == "linear" else 1.0
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)

    elif rope_type == "dynamic":
        return DynamicNTKScalingRoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional,
            base=base,
            factor=scaling_config["factor"],
        )

    elif rope_type == "llama3":
        return Llama3RoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional,
            base=base,
            scaling_config=scaling_config,
        )
    elif rope_type in ("yarn", "deepseek_yarn", "telechat3-yarn"):
        scaling_factor = scaling_config["factor"]
        rope_kwargs = {
            key: scaling_config[key]
            for key in [
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ]
            if key in scaling_config
        }
        return YarnRoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional,
            scaling_factor=scaling_factor,
            base=base,
            **rope_kwargs,
        )
    elif rope_type == "longrope":
        return SuScaledRoPE(
            dims=dims,
            base=base,
            max_position_embeddings=max_position_embeddings,
            original_max_position_embeddings=scaling_config[
                "original_max_position_embeddings"
            ],
            short_factor=scaling_config["short_factor"],
            long_factor=scaling_config["long_factor"],
        )
    elif rope_type == "proportional":
        return ProportionalRoPE(
            dims=dims,
            rotated_dims=int(dims * scaling_config.get("partial_rotary_factor", 1.0)),
            traditional=traditional,
            base=base,
            factor=scaling_config.get("factor", 1.0),
        )
    else:
        raise ValueError(f"Unsupported RoPE type {rope_type}")
