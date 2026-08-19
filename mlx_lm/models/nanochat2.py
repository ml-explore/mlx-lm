# Copyright © 2026 Lattice (oli-mebberson)

"""Complete nanochat architecture for Lattice Quark models.

This is the full Quark build: value embeddings with an input-dependent
per-head gate, smear (previous-token embedding mixing), backout at the
mid-layer, learned per-layer resid/x0 lambdas, weightless QK-norm with a
1.2x scale, the nanochat -theta rotary convention (precomputed cos/sin),
logit softcap 15 and the padded-vocab crop.

It mirrors the torch gpt.py forward pass exactly and matches the converted
weights published at lattice-research/lattice-quark-1.5b-mlx (model type
"nanochat2"). The older "nanochat" model type in this package is an
incomplete port (missing value embeddings, smear, backout and lambdas) and
must not be used to load these weights.
"""

from dataclasses import dataclass
from typing import List

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import ConcatenateKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "nanochat2"
    hidden_size: int = 1536
    num_hidden_layers: int = 26
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    vocab_size: int = 32768
    padded_vocab_size: int = 32768
    max_position_embeddings: int = 2048
    intermediate_size: int = 6144
    rope_theta: float = 100000.0
    rms_norm_eps: float = 1e-5
    logits_softcap: float = 15.0


def _rms_norm(x, eps=1e-5):
    # Weightless RMS norm: every norm in this architecture is affine-free.
    mean_squares = mx.mean(mx.square(x), axis=-1, keepdims=True)
    return x * mx.rsqrt(mean_squares + eps)


def _apply_rotary(x, cos, sin):
    # nanochat rotates by -theta (the transpose of the textbook convention);
    # cos/sin are precomputed (1, T, 1, head_dim/2) and match the training code.
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)


class NanoChat2Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_head = args.num_attention_heads
        self.n_kv_head = args.num_key_value_heads
        self.head_dim = dim // self.n_head
        self.ve_gate_channels = 12
        self.scale = self.head_dim**-0.5

        self.c_q = nn.QuantizedLinear(
            dim, self.n_head * self.head_dim, bias=False, group_size=64, bits=4
        )
        self.c_k = nn.QuantizedLinear(
            dim, self.n_kv_head * self.head_dim, bias=False, group_size=64, bits=4
        )
        self.c_v = nn.QuantizedLinear(
            dim, self.n_kv_head * self.head_dim, bias=False, group_size=64, bits=4
        )
        self.c_proj = nn.QuantizedLinear(dim, dim, bias=False, group_size=64, bits=4)
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)

    def __call__(self, x, ve, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            # Value residual: input-dependent gate per head, range (0, 3)
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * mx.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate[..., None] * ve

        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        q = _rms_norm(q)
        k = _rms_norm(k)
        q = q * 1.2
        k = k * 1.2

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        start = cache.offset if cache is not None else 0
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if self.n_kv_head < self.n_head:
            reps = self.n_head // self.n_kv_head
            k = mx.repeat(k, reps, axis=1)
            v = mx.repeat(v, reps, axis=1)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        L = k.shape[2]
        Tq = q.shape[2]
        qpos = mx.arange(Tq)[:, None]
        kpos = mx.arange(L)[None, :]
        blocked = kpos > (start + qpos)
        mask = mx.where(blocked, mx.array(-float("inf")), mx.array(0.0))
        scores = scores + mask

        probs = mx.softmax(scores, axis=-1)
        y = (probs @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.c_proj(y)


class NanoChat2MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.c_fc = nn.QuantizedLinear(
            dim, 4 * dim, bias=False, group_size=64, bits=4
        )
        self.c_proj = nn.QuantizedLinear(
            4 * dim, dim, bias=False, group_size=64, bits=4
        )

    def __call__(self, x):
        x = self.c_fc(x)
        x = nn.relu2(x)
        return self.c_proj(x)


class NanoChat2Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = NanoChat2Attention(args)
        self.mlp = NanoChat2MLP(args)

    def __call__(self, x, ve, cos, sin, cache=None):
        eps = 1e-5
        h = x + self.attn(_rms_norm(x, eps), ve, cos, sin, cache)
        return h + self.mlp(_rms_norm(h, eps))


class NanoChat2KV(ConcatenateKVCache):
    """Concatenating KV cache that also carries the previous token embedding
    for the smear mechanism."""

    def __init__(self):
        super().__init__()
        self.prev_embedding = None


class NanoChat2Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.wte = nn.Embedding(args.padded_vocab_size, args.hidden_size)
        self.h = [NanoChat2Block(args) for _ in range(args.num_hidden_layers)]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        # Top-level layout matches the converted safetensors exactly:
        # transformer.{wte,h}, lm_head, value_embeds.N, smear_gate and the
        # four learned lambdas are root-level parameters.
        self.transformer = NanoChat2Transformer(args)
        self.lm_head = nn.QuantizedLinear(
            args.hidden_size, args.padded_vocab_size, bias=False, group_size=64, bits=4
        )
        self.value_embeds = [
            nn.Embedding(args.padded_vocab_size, args.hidden_size)
            for _ in range(args.num_hidden_layers // 2)
        ]
        self.smear_gate = nn.Linear(24, 1, bias=False)
        self.resid_lambdas = mx.ones(args.num_hidden_layers)
        self.x0_lambdas = mx.zeros(args.num_hidden_layers)
        self.smear_lambda = mx.zeros(1)
        self.backout_lambda = mx.zeros(1)
        # Precomputed rotary embeddings, identical to the training code
        # (nanochat_rope.safetensors in the released repos).
        head_dim = args.hidden_size // args.num_attention_heads
        self._rope = self._precompute_rotary(
            args.max_position_embeddings, head_dim, args.rope_theta
        )

    @staticmethod
    def _precompute_rotary(seq_len, head_dim, base):
        channel_range = mx.arange(0, head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
        cos = mx.cos(freqs)[None, :, None, :].astype(mx.bfloat16)
        sin = mx.sin(freqs)[None, :, None, :].astype(mx.bfloat16)
        return cos, sin

    @staticmethod
    def _has_ve(layer_idx, n_layer):
        return layer_idx % 2 == (n_layer - 1) % 2

    def make_cache(self) -> List[NanoChat2KV]:
        return [NanoChat2KV() for _ in range(len(self.transformer.h))]

    def __call__(self, inputs, cache=None):
        if cache is None:
            cache = [None] * len(self.transformer.h)

        n_layer = self.args.num_hidden_layers
        T = inputs.shape[1]
        T0 = cache[0].offset if cache is not None else 0
        cos = self._rope[0][:, T0 : T0 + T]
        sin = self._rope[1][:, T0 : T0 + T]

        x = self.transformer.wte(inputs)
        x = _rms_norm(x)

        # Smear: mix the previous token's embedding into the current position
        first_cache = cache[0]
        x_pre_smear = first_cache.prev_embedding if first_cache is not None else None
        if first_cache is not None:
            first_cache.prev_embedding = x[:, -1:, :]
        if T > 1:
            gate = self.smear_lambda * mx.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = mx.concatenate([x[:, :1], x[:, 1:] + gate * x[:, :-1]], axis=1)
        elif x_pre_smear is not None:
            gate = self.smear_lambda * mx.sigmoid(self.smear_gate(x[:, :, :24]))
            x = x + gate * x_pre_smear

        x0 = x
        backout_layer = n_layer // 2
        x_backout = None
        for i, layer in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = (
                self.value_embeds[(i - 1) // 2](inputs)
                if self._has_ve(i, n_layer)
                else None
            )
            x = layer(x, ve, cos, sin, cache[i])
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda * x_backout
        hidden = _rms_norm(x)

        logits = self.lm_head(hidden)
        logits = logits[..., : self.args.vocab_size]
        cap = self.args.logits_softcap
        return cap * mx.tanh(logits / cap)

    @property
    def layers(self):
        return self.transformer.h
