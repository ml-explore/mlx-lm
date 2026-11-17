# Copyright © 2024 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.common import swiglu
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.mamba.config import ModelArgs

__all__ = [
    "Mamba",
    "MambaBlock",
    "ResidualBlock",
]


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.intermediate_size
        self.time_step_rank = int(args.time_step_rank)
        self.use_conv_bias = args.use_conv_bias
        self.use_bcdt_rms = args.use_bcdt_rms

        self.in_proj = nn.Linear(
            self.hidden_size, self.intermediate_size * 2, bias=args.use_bias
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            bias=self.use_conv_bias,
            padding=0,
        )

        self.x_proj = nn.Linear(
            self.intermediate_size,
            self.time_step_rank + 2 * self.ssm_state_size,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        A = mx.repeat(
            mx.arange(1.0, self.ssm_state_size + 1.0).reshape([1, self.ssm_state_size]),
            repeats=self.intermediate_size,
            axis=0,
        )
        self.A_log = mx.log(A)
        self.D = mx.ones([self.intermediate_size])

        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.use_bias
        )

    def ssm_step(self, x, A, state=None):
        D = self.D
        deltaBC = self.x_proj(x)
        delta, B, C = mx.split(
            deltaBC,
            [self.time_step_rank, self.time_step_rank + self.ssm_state_size],
            axis=-1,
        )
        if self.use_bcdt_rms:
            eps = self.args.mixer_rms_eps
            delta = mx.fast.rms_norm(delta, weight=None, eps=eps)
            B = mx.fast.rms_norm(B, weight=None, eps=eps)
            C = mx.fast.rms_norm(C, weight=None, eps=eps)
        delta = nn.softplus(self.dt_proj(delta))
        new_state = mx.expand_dims(delta * x, -1) * mx.expand_dims(B, 1)
        if state is not None:
            new_state += state * mx.exp(mx.expand_dims(delta, -1) * A)
        y = (new_state @ mx.expand_dims(C, -1)).squeeze(2)
        y = y + D * x
        return y, new_state

    def _process_sequence(self, x, conv_cache, state_cache):
        B, T, D = x.shape
        xz = self.in_proj(x)
        x, z = xz.split(indices_or_sections=2, axis=-1)
        K = self.conv_kernel_size
        if conv_cache is not None:
            x_full = mx.concatenate([conv_cache, x], axis=1)
        else:
            x_full = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)])
        conv_out = self.conv1d(x_full)
        new_conv_cache = x_full[:, -(K - 1) :, :]
        x = nn.silu(conv_out)
        A = -mx.exp(self.A_log)
        current_state = state_cache
        y = []
        for t in range(T):
            y_t, current_state = self.ssm_step(x[:, t], A, current_state)
            y.append(y_t)
        y = mx.stack(y, axis=1)
        z = self.out_proj(swiglu(z, y))
        return z, (new_conv_cache, current_state)

    def __call__(self, x, cache):
        if cache is None:
            conv_cache, state_cache = None, None
        else:
            conv_cache, state_cache = cache[0], cache[1]

        output, (new_conv_cache, new_state_cache) = self._process_sequence(
            x, conv_cache, state_cache
        )

        if isinstance(cache, ArraysCache):
            cache[0] = new_conv_cache
            cache[1] = new_state_cache

        return output


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache):
        return self.mixer(self.norm(x), cache) + x


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache):
        x = self.embeddings(x)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            x = layer(x, c)
        return self.norm_f(x)
