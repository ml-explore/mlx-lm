"""RPSLinear: QuantizedLinear with optional residual precision boost."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class RPSLinear(nn.Module):
    """Drop-in replacement for QuantizedLinear that supports residual correction.

    The key insight: instead of dequantizing both base and residual and
    materializing a full-precision weight matrix, we perform two separate
    quantized matmuls and add the results:

        x @ (W_base + W_residual) = x @ W_base + x @ W_residual

    This avoids the memory cost of a full-precision weight matrix while
    achieving equivalent quality to higher-precision quantization.
    """

    def __init__(self, base_layer: nn.QuantizedLinear):
        super().__init__()
        self.weight = base_layer.weight
        self.scales = base_layer.scales
        self.biases = base_layer.biases
        self.group_size = base_layer.group_size
        self.bits = base_layer.bits

        if hasattr(base_layer, "bias") and base_layer.bias is not None:
            self.bias = base_layer.bias
        else:
            self.bias = None

        # Residual parameters (None until attached)
        self.r_weight: Optional[mx.array] = None
        self.r_scales: Optional[mx.array] = None
        self.r_biases: Optional[mx.array] = None
        self.r_bits: int = 2
        self.r_group_size: int = 64

        self.freeze()

    def attach_residual(
        self,
        r_weight: mx.array,
        r_scales: mx.array,
        r_biases: mx.array,
        r_bits: int = 2,
        r_group_size: int = 64,
    ):
        """Attach a quantized residual to boost precision."""
        self.r_weight = r_weight
        self.r_scales = r_scales
        self.r_biases = r_biases
        self.r_bits = r_bits
        self.r_group_size = r_group_size

    def detach_residual(self):
        """Remove residual for graceful degradation."""
        self.r_weight = None
        self.r_scales = None
        self.r_biases = None

    @property
    def has_residual(self) -> bool:
        return self.r_weight is not None

    def __call__(self, x: mx.array) -> mx.array:
        # Base quantized matmul (always available)
        out = mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )

        # Add residual contribution if attached
        if self.r_weight is not None:
            out = out + mx.quantized_matmul(
                x,
                self.r_weight,
                scales=self.r_scales,
                biases=self.r_biases,
                transpose=True,
                group_size=self.r_group_size,
                bits=self.r_bits,
            )

        if self.bias is not None:
            out = out + self.bias

        return out
