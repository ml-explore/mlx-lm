# Copyright © 2023-2026 Apple Inc.

from functools import partial

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def swiglu(gate, x):
    return nn.silu(gate) * x


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


# TODO rename
@partial(mx.compile, shapeless=True)
def swiglu(x_linear, x_glu, alpha: float = 1.702, limit: float = 7.0):
    # Clamp the input values
    x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
    x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)

    glu_scaled = alpha * x_glu
    sig = mx.sigmoid(glu_scaled)

    out_glu = x_glu * sig
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(x, gate)


@partial(mx.compile, shapeless=True)
def _silu_mul(gate: mx.array, up: mx.array) -> mx.array:
    return nn.silu(gate) * up


@partial(mx.compile, shapeless=True)
def relu_squared(x):
    return nn.relu(x).square()


@partial(mx.compile, shapeless=True)
def gegelu_impl(a_gelu, a_linear, limit):
    a_gelu = mx.where(
        mx.isinf(a_gelu),
        a_gelu,
        mx.clip(a_gelu, a_min=None, a_max=limit),
    )
    a_linear = mx.where(
        mx.isinf(a_linear),
        a_linear,
        mx.clip(a_linear, a_min=-limit, a_max=limit),
    )
    out_gelu = a_gelu * mx.sigmoid(1.702 * a_gelu)
    return out_gelu * (a_linear + 1.0)


def gegelu(x, limit):
    a_gelu, a_linear = x[..., ::2], x[..., 1::2]
    return gegelu_impl(a_gelu, a_linear, limit)


@partial(mx.compile, shapeless=True)
def gelu_topk(inputs, std_multiplier):
    inputs_mean = mx.mean(inputs, axis=-1, keepdims=True)
    inputs_std = mx.std(inputs, axis=-1, keepdims=True)
    cutoff_x = inputs_mean + inputs_std * std_multiplier.astype(inputs_std.dtype)
    return nn.gelu_approx(mx.maximum(0, inputs - cutoff_x))
