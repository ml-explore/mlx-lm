# Copyright © 2025 Apple Inc.

from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm.kernels.bitlinear import bitlinear_matmul

__all__ = [
    "BitLinear",
    "SwiGLU",
    "SwigluOAI",
    "XieLU",
    "ZeroCenteredRMSNorm",
    "bitnet_quantize",
    "precise_swiglu",
    "splice",
    "swiglu",
    "swiglu_oai",
    "xielu",
]


@partial(mx.compile, shapeless=True)
def swiglu(gate, x):
    return nn.silu(gate) * x


@partial(mx.compile, shapeless=True)
def precise_swiglu(h, gate, x):
    gate = nn.silu(gate.astype(mx.float32))
    x = x.astype(mx.float32)
    return (gate * x).astype(h.dtype)


@partial(mx.compile, shapeless=True)
def swiglu_oai(
    gate: mx.array, x: mx.array, alpha: float = 1.702, limit: float = 7.0
) -> mx.array:
    # GPT-OSS variant: clip the gate from above only, and add 1 to the linear part.
    gate = mx.minimum(gate, limit)
    x = mx.clip(x, -limit, limit)
    return (x + 1.0) * (gate * mx.sigmoid(alpha * gate))


class SwigluOAI(nn.Module):
    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self._alpha = alpha
        self._limit = limit

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return swiglu_oai(gate, x, alpha=self._alpha, limit=self._limit)


@partial(mx.compile, shapeless=True)
def xielu(x, alpha_p, alpha_n, beta, eps):
    alpha_p = nn.softplus(alpha_p)
    alpha_n = beta + nn.softplus(alpha_n)
    return mx.where(
        x > 0,
        alpha_p * mx.square(x) + beta * x,
        (mx.expm1(mx.minimum(x, eps)) - x) * alpha_n + beta * x,
    )


class XieLU(nn.Module):
    def __init__(
        self,
        alpha_p_init=0.8,
        alpha_n_init=0.8,
        beta=0.5,
        eps=-1e-6,
    ):
        super().__init__()
        alpha_p_tensor = mx.array(alpha_p_init)
        alpha_n_tensor = mx.array(alpha_n_init - beta)
        self.alpha_p = mx.log(mx.exp(alpha_p_tensor) - 1)
        self.alpha_n = mx.log(mx.exp(alpha_n_tensor) - 1)

        self.beta = mx.array(beta)
        self.eps = mx.array(eps)

    def __call__(self, x: mx.array) -> mx.array:
        return xielu(x, self.alpha_p, self.alpha_n, self.beta, self.eps)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm with the weight stored as an offset from 1."""

    def __init__(self, dims: int, *, eps: float):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def _extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


def bitnet_quantize(model, quantization_config: dict):
    quantize_layers = []
    modules_to_not_convert = quantization_config.get("modules_to_not_convert", [])
    invert_weight_scales = (
        quantization_config.get("linear_class", "") != "autobitlinear"
    )

    for name, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):

        # Replace nn.Linear layers, but skip any layer from the `modules_to_not_convert` list
        if name not in modules_to_not_convert and isinstance(module, nn.Linear):
            old_weight = module.weight
            out_features, in_features = old_weight.shape
            bias = "bias" in module
            new_layer = BitLinear(
                in_features,
                out_features,
                bias=bias,
                invert_weight_scales=invert_weight_scales,
            )
            quantize_layers.append((name, new_layer))
    if len(quantize_layers) > 0:
        model.update_modules(tree_unflatten(quantize_layers))
    return model


class BitLinear(nn.Module):
    """
    BitLinear module with memory-efficient weight handling.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        invert_weight_scales=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Calculate packed dimensions - the first dimension gets packed 4:1
        # The weights are ternary so can be represented with 2 bits, and they
        # are packed in uint8 tensors, hence the number of values per item is 4
        packed_out_features = (out_features + 3) // 4
        self.weight = mx.zeros((packed_out_features, in_features), dtype=mx.uint8)

        self.invert_weight_scales = invert_weight_scales
        self.weight_scale = mx.array([1.0])

        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

    def execute_matmul_kernel(self, x, packed_weights):
        return bitlinear_matmul(
            x,
            packed_weights,
            self.weight_scale,
            out_features=self.out_features,
            invert_weight_scales=self.invert_weight_scales,
        )

    def __call__(self, x):
        y = self.execute_matmul_kernel(x, self.weight)

        if self.bias is not None:
            y = mx.add(y, self.bias)
        return y


def splice(
    embeddings: mx.array,
    inputs: mx.array,
    *,
    rows: mx.array,
    token_id: int,
) -> mx.array:
    """Put the rows, in order, at the positions of the token in the inputs."""
    # A shape check only, so no host sync.
    if rows.shape[0] == 0:
        return embeddings
    mask = inputs == token_id
    # The row of each masked token: the number of masked tokens before it.
    idx = mx.cumsum(mask.flatten()).reshape(mask.shape) - 1
    idx = mx.clip(idx, 0, rows.shape[0] - 1)
    selected = rows[idx].astype(embeddings.dtype)
    return mx.where(mask[..., None], selected, embeddings)
