from dataclasses import dataclass
from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import RwkvCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    norm_eps: float
    head_dim: int
    num_hidden_layers: int
    a_low_rank_dim: int
    v_low_rank_dim: int
    gate_low_rank_dim: int
    decay_low_rank_dim: int
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if not hasattr(self, "num_hidden_layers") and hasattr(self, "n_layer"):
            self.num_hidden_layers = self.n_layer
        if not hasattr(self, "num_hidden_layers") and hasattr(self, "n_layers"):
            self.num_hidden_layers = self.n_layers


@partial(mx.compile, shapeless=True)
def relu_squared(x):
    return nn.relu(x).square()


@partial(mx.compile, shapeless=True)
def addcmul(x, y, z):
    return x + y * z


@partial(mx.compile, shapeless=True)
def l2_norm(x):
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-7)


@mx.compile
def _wkv7_step_ops(r, w, k, v, a, b, state):
    state = state * w + v @ k + (state @ a) @ b
    y = state @ r
    return y, state


class LoRA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: Optional[bool] = True,
        activation: Optional[str] = 'tanh'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias

        if activation is None:
            self.activation = nn.Identity()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation type: {activation}.")

        self.lora = [
            nn.Linear(self.input_dim, self.low_rank_dim, bias=False),
            self.activation,
            nn.Linear(self.low_rank_dim, self.output_dim, bias=self.bias)
        ]

    def __call__(self, x) -> mx.array:
        return self.lora[2](self.lora[1](self.lora[0](x)))


class TokenShift(nn.Module):
    def __call__(self, x, state) -> mx.array:
        B, L, D = x.shape
        if state is None:
            state = mx.zeros((B, 1, D), x.dtype)
        if L == 1:
            return state
        else:
            return mx.concatenate([state, x[:, :-1, :]], axis=1)


class Rwkv7ChannelMixing(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size

        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, dim, bias=False)

        self.x_k = mx.zeros((dim))

        self.token_shift = TokenShift()

    def __call__(self, x, cache) -> mx.array:
        token_shift_cache = cache[2] if cache is not None else None
        x_prev = self.token_shift(x, token_shift_cache)
        xx = x_prev - x
        xx = addcmul(x, xx, self.x_k)
        if isinstance(cache, RwkvCache):
            cache[2] = x[:, -1, :]
        return self.value(relu_squared(self.key(xx)))


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

        self.k_k = mx.zeros((self.hidden_size))
        self.k_a = mx.zeros((self.hidden_size))
        self.r_k = mx.zeros((self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.o_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False
        )

        self.g_norm = nn.GroupNorm(
            self.num_heads, self.hidden_size, eps=64e-5, affine=True, pytorch_compatible=True
        )

        self.w_lora = LoRA(
            self.hidden_size, self.hidden_size, low_rank_dim=self.decay_low_rank_dim, activation='tanh'
        )

        if self.layer_idx > 0:
            self.v_lora = LoRA(
                self.hidden_size, self.hidden_size, low_rank_dim=self.v_low_rank_dim, activation=None
            )

        self.a_lora = LoRA(
            self.hidden_size, self.hidden_size, low_rank_dim=self.a_low_rank_dim, activation=None
        )

        self.g_lora = LoRA(
            self.hidden_size, self.hidden_size, low_rank_dim=self.gate_low_rank_dim, activation='sigmoid', bias=False
        )
    
    def _wkv7(self, r, w, k, v, a, b, state):
        B, L, _ = r.shape
        if state is None:
            state = mx.zeros((B, self.num_heads, self.head_dim, self.head_dim), dtype=mx.float32)

        r = r.reshape([B, L, self.num_heads, self.head_dim, 1])
        w = w.reshape([B, L, self.num_heads, 1, self.head_dim])
        k = k.reshape([B, L, self.num_heads, 1, self.head_dim])
        v = v.reshape([B, L, self.num_heads, self.head_dim, 1])
        a = a.reshape([B, L, self.num_heads, self.head_dim, 1])
        b = b.reshape([B, L, self.num_heads, 1, self.head_dim])

        ys = []
        for t in range(L):
            y, state = _wkv7_step_ops(
                r[:, t],
                w[:, t],
                k[:, t],
                v[:, t],
                a[:, t],
                b[:, t],
                state
            )
            ys.append(y)

        y = mx.stack(ys, axis=1).astype(r.dtype)
        return y, state

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

        key = self.k_proj(xk)
        value = self.v_proj(xv)
        receptance = self.r_proj(xr)
        a = mx.sigmoid(self.a_lora(xa))
        g = self.g_lora(xg)

        if self.layer_idx == 0:
            v_first = value
        else:
            value = value + (v_first - value) * mx.sigmoid(self.v_lora(xv))

        decay = mx.exp(-0.606531 * mx.sigmoid(self.w_lora(xw)).astype(mx.float32))
        kk = l2_norm((key * self.k_k).reshape([B, L, self.num_heads, self.head_dim])).reshape([B, L, D])
        key = key * (1 + (a - 1) * self.k_a)
        b = kk * a
        a = -kk

        out, new_state_cache = self._wkv7(receptance, decay, key, value, a, b, state_cache)
        out = self.g_norm(out.reshape([B, L, D]))
        residual = (receptance * key * self.r_k.reshape([D])).reshape([B, L, self.num_heads, self.head_dim])
        residual = (residual.sum(axis=-1, keepdims=True) * value.reshape(B, L, self.num_heads, self.head_dim)).reshape([B, L, D])
        out = out + residual

        if isinstance(cache, RwkvCache):
            cache[0] = x[:, -1, :]
            cache[1] = new_state_cache

        output = self.o_proj(out * g)
        return output, v_first


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


class Rwkv7Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Rwkv7Layer(args, layer_idx=i) for i in range(args.num_hidden_layers)]
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(self, x: mx.array, cache):
        x = self.embeddings(x)
        if cache is None:
            cache = [None] * len(self.layers)

        v_first = None
        for layer, c in zip(self.layers, cache):
            x, v_first = layer(x, v_first, c)
        return self.norm(x)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Rwkv7Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        B, T = inputs.shape

        x = self.model(inputs, cache)

        if self.args.tie_word_embeddings:
            logits = self.model.embeddings.as_linear(x)
        else:
            logits = self.lm_head(x)

        return logits

    def make_cache(self):
        return [RwkvCache() for _ in range(len(self.layers))]

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        for k, v in weights.items():
            if v.dtype == mx.bfloat16:
                weights[k] = v.astype(mx.float16)
        return weights
