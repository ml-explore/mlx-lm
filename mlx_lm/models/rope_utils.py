# Copyright © 2023-2024 Apple Inc.

from typing import Optional, Union, List

import mlx.core as mx
import mlx.nn as nn


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


class MiniCPMLongRoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float = 10000.0,
        max_position_embeddings: int = 131072,
        original_max_position_embeddings: int = 4096,
        short_factor: Union[List[float], float] = 1.0,
        long_factor: Union[List[float], float] = 1.0,
    ):
        super().__init__()
        self.dim = dims
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.short_factor = short_factor
        self.long_factor = long_factor

        self.inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float16) / dims))
        
        # Calculate scaling factor using Dynamic NTK approach
        scale = max_position_embeddings / original_max_position_embeddings
        self.scaling_factor = mx.sqrt(
            1 + mx.log(scale) / mx.log(original_max_position_embeddings)
        )
    
    def extra_repr(self):
        return (
            f"dim={self.dim}, base={self.base}, "
            f"max_position_embeddings={self.max_position_embeddings}, "
            f"original_max_position_embeddings={self.original_max_position_embeddings}"
        )
    
    def __call__(self, x, offset: int = 0, seq_len: int = None):
        x = self.scaling_factor * x
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len > self.original_max_position_embeddings:
            freq_factor = mx.array(self.long_factor, dtype=mx.float32)
        else:
            freq_factor = mx.array(self.short_factor, dtype=mx.float32)
        inv_freqs = freq_factor * self.inv_freq
        return mx.fast.rope(
            x,
            self.dim,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=inv_freqs,
        )


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

    if rope_type == "default" or rope_type == "linear":
        scale = 1 / scaling_config["factor"] if rope_type == "linear" else 1.0
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)
    elif rope_type == "long":
        return MiniCPMLongRoPE(
            dims=dims,
            base=base,
            max_position_embeddings=scaling_config.get("max_position_embeddings", 131072),
            original_max_position_embeddings=scaling_config.get("original_max_position_embeddings", 4096),
            short_factor=scaling_config.get("short_factor", 1.0),
            long_factor=scaling_config.get("long_factor", 1.0),
        )
    elif rope_type == "llama3":
        return Llama3RoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional,
            base=base,
            scaling_config=scaling_config,
        )
    else:
        raise ValueError(f"Unsupported RoPE type {rope_type}")