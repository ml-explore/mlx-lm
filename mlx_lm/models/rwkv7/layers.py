# Copyright © 2025 Apple Inc.

from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.rwkv7.config import ModelArgs
from mlx_lm.models.rwkv7.kernels import wkv7_kernel

__all__ = [
    "LayerNormPerHead",
    "LoRA",
    "Rwkv7ChannelMixing",
    "Rwkv7Layer",
    "Rwkv7TimeMixing",
    "TokenShift",
    "addcmul",
    "l2_norm",
]


@partial(mx.compile, shapeless=True)
def addcmul(x, y, z):
    return x + y * z


class TokenShift(nn.Module):
    def __call__(self, x, state):
        B, L, D = x.shape
        if state is None:
            state = mx.zeros((B, 1, D), x.dtype)
        if L == 1:
            return state
        else:
            return mx.concatenate([state, x[:, :-1, :]], axis=1)


class LayerNormPerHead(nn.Module):
    def __init__(self, head_dim, num_heads, eps):
        super().__init__()
        self.weight = mx.zeros((num_heads, head_dim))
        self.bias = mx.zeros((num_heads, head_dim))
        self.eps = eps

    def __call__(self, x):
        return self.weight * mx.fast.layer_norm(x, None, None, self.eps) + self.bias


class LoRA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: Optional[bool] = True,
        activation: Optional[str] = "tanh",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias

        if activation is None:
            self.activation = nn.Identity()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation type: {activation}.")

        self.lora = [
            nn.Linear(self.input_dim, self.low_rank_dim, bias=False),
            self.activation,
            nn.Linear(self.low_rank_dim, self.output_dim, bias=self.bias),
        ]

    def __call__(self, x) -> mx.array:
        return self.lora[2](self.lora[1](self.lora[0](x)))


@partial(mx.compile, shapeless=True)
def l2_norm(x):
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-7)


@mx.compile
def _wkv7_step_ops(r, w, k, v, a, b, state):
    sab = (state @ a[..., None]) @ b[..., None, :]
    state = state * w[:, :, None, :] + v[..., None] @ k[..., None, :] + sab
    y = state @ r[..., None]
    return y, state


class Rwkv7TimeMixing(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.args = args
        self.hidden_size = args.hidden_size
        self.head_dim = args.head_dim
        self.num_heads = self.hidden_size // self.head_dim
        self.a_low_rank_dim = args.a_low_rank_dim
        self.v_low_rank_dim = args.v_low_rank_dim
        self.gate_low_rank_dim = args.gate_low_rank_dim
        self.decay_low_rank_dim = args.decay_low_rank_dim

        self.token_shift = TokenShift()

        self.x_r = mx.zeros((1, 1, self.hidden_size))
        self.x_w = mx.zeros((1, 1, self.hidden_size))
        self.x_k = mx.zeros((1, 1, self.hidden_size))
        self.x_v = mx.zeros((1, 1, self.hidden_size))
        self.x_a = mx.zeros((1, 1, self.hidden_size))
        self.x_g = mx.zeros((1, 1, self.hidden_size))

        self.k_k = mx.zeros((self.num_heads, self.head_dim))
        self.k_a = mx.zeros((self.num_heads, self.head_dim))
        self.r_k = mx.zeros((self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.g_norm = LayerNormPerHead(self.head_dim, self.num_heads, eps=64e-5)

        self.w_lora = LoRA(
            self.hidden_size,
            self.hidden_size,
            low_rank_dim=self.decay_low_rank_dim,
            activation="tanh",
        )

        if self.layer_idx > 0:
            self.v_lora = LoRA(
                self.hidden_size,
                self.hidden_size,
                low_rank_dim=self.v_low_rank_dim,
                activation=None,
            )

        self.a_lora = LoRA(
            self.hidden_size,
            self.hidden_size,
            low_rank_dim=self.a_low_rank_dim,
            activation=None,
        )

        self.g_lora = LoRA(
            self.hidden_size,
            self.hidden_size,
            low_rank_dim=self.gate_low_rank_dim,
            activation="sigmoid",
            bias=False,
        )

    def __call__(self, x, v_first, cache):
        if cache is None:
            token_shift_cache, state_cache = None, None
        else:
            token_shift_cache, state_cache = cache[0], cache[1]

        B, L, D = x.shape
        x_prev = self.token_shift(x, token_shift_cache)
        xx = x_prev - x

        xr = addcmul(x, xx, self.x_r)
        xw = addcmul(x, xx, self.x_w)
        xk = addcmul(x, xx, self.x_k)
        xv = addcmul(x, xx, self.x_v)
        xa = addcmul(x, xx, self.x_a)
        xg = addcmul(x, xx, self.x_g)

        key = self.k_proj(xk).reshape(B, L, self.num_heads, self.head_dim)
        value = self.v_proj(xv).reshape(B, L, self.num_heads, self.head_dim)
        receptance = self.r_proj(xr).reshape(B, L, self.num_heads, self.head_dim)
        iclr = mx.sigmoid(self.a_lora(xa)).reshape(B, L, self.num_heads, self.head_dim)
        gate = self.g_lora(xg)

        if self.layer_idx == 0:
            v_first = value
        else:
            vv = mx.sigmoid(self.v_lora(xv)).reshape(
                B, L, self.num_heads, self.head_dim
            )
            value = addcmul(value, v_first - value, vv)

        decay = mx.sigmoid(
            self.w_lora(xw).reshape(B, L, self.num_heads, self.head_dim)
        ).astype(mx.float32)
        decay = mx.exp(-0.606531 * decay).astype(receptance.dtype)
        kk = l2_norm((key * self.k_k))
        key = key * (1 + (iclr - 1) * self.k_a)
        a = -kk
        b = kk * iclr

        out, new_state_cache = self._wkv7(
            receptance, decay, key, value, a, b, state_cache
        )
        out = self.g_norm(out.reshape(B, L, self.num_heads, self.head_dim))
        out = (
            out + (receptance * key * self.r_k).sum(axis=-1, keepdims=True) * value
        ).reshape([B, L, D])

        if cache is not None:
            cache[0] = x[:, -1:, :]
            cache[1] = new_state_cache

        out = self.o_proj(out * gate)
        return out, v_first

    def _wkv7(self, r, w, k, v, a, b, state):
        B, L, _, _ = r.shape
        if state is None:
            state = mx.zeros(
                (B, self.num_heads, self.head_dim, self.head_dim), dtype=r.dtype
            )

        if mx.default_device() == mx.gpu and mx.metal.is_available():
            return wkv7_kernel(r, w, k, v, a, b, state)
        else:
            ys = []
            for t in range(L):
                y, state = _wkv7_step_ops(
                    r[:, t], w[:, t], k[:, t], v[:, t], a[:, t], b[:, t], state
                )
                ys.append(y)

            y = mx.stack(ys, axis=1).astype(r.dtype)
            return y, state


class Rwkv7ChannelMixing(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = args.hidden_size
        intermediate_size = args.intermediate_size

        self.key = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_dim, bias=False)

        self.x_k = mx.zeros((hidden_dim))

        self.token_shift = TokenShift()

    def __call__(self, x, cache) -> mx.array:
        state = cache[2] if cache is not None else None
        x_prev = self.token_shift(x, state)
        xx = addcmul(x, x_prev - x, self.x_k)
        if cache is not None:
            cache[2] = x[:, -1:, :]
        return self.value(nn.relu2(self.key(xx)))


class Rwkv7Layer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        if self.layer_idx == 0:
            self.pre_norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps)
        self.attn = Rwkv7TimeMixing(args, layer_idx=self.layer_idx)
        self.ffn = Rwkv7ChannelMixing(args)
        self.attn_norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(self, x, v_first, cache):
        if self.layer_idx == 0:
            x = self.pre_norm(x)

        h, v_first = self.attn(self.attn_norm(x), v_first, cache)
        h = x + h
        out = h + self.ffn(self.ffn_norm(h), cache)
        return out, v_first
