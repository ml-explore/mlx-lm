# Copyright © 2026 Apple Inc.

import math
from functools import partial
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.train.args import ModelArgs
from mlx_lm.train.layers.norms import RMSNormGated


class YarnRoPE(nn.Module):
    def __init__(
        self,
        dims,
        traditional=False,
        base=500_000,
        scaling_factor=8.0,
        original_max_position_embeddings=8192,
        beta_fast=32,
        beta_slow=1,
    ):
        super().__init__()

        def find_correction_dim(num_rotations):
            return (
                dims
                * math.log(
                    original_max_position_embeddings / (num_rotations * 2 * math.pi)
                )
            ) / (2 * math.log(base))

        def find_correction_range():
            low = math.floor(find_correction_dim(beta_fast))
            high = math.ceil(find_correction_dim(beta_slow))
            return max(low, 0), min(high, dims - 1)

        def linear_ramp_mask(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001  # Prevent singularity
            ramp = (mx.arange(dim, dtype=mx.float32) - min_val) / (max_val - min_val)
            return mx.clip(ramp, 0, 1)

        self.mscale = 0.1 * math.log(scaling_factor) + 1.0
        freq_extra = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        freq_inter = scaling_factor * freq_extra
        low, high = find_correction_range()
        freq_mask = 1.0 - linear_ramp_mask(low, high, dims // 2)
        self._freqs = (freq_inter * freq_extra) / (
            freq_inter * freq_mask + freq_extra * (1 - freq_mask)
        )
        self.dims = dims
        self.traditional = traditional

    def __call__(self, x, offset=0):
        x = self.mscale * x
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class Attention(nn.Module):
    """Quadratic attention with grouped query heads and QK norm."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        head_dim = args.head_dim
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.scale = head_dim**-0.5
        self.output_gate = args.attn_output_gate

        # With an output gate q_proj emits a gate alongside every query head.
        q_dim = self.n_heads * head_dim * (2 if self.output_gate else 1)
        self.q_proj = nn.Linear(dim, q_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        rope_dims = int(head_dim * args.partial_rotary_factor)
        if args.rope_scaling_factor > 1:
            self._rope = YarnRoPE(
                rope_dims,
                base=args.rope_theta,
                scaling_factor=args.rope_scaling_factor,
                original_max_position_embeddings=args.original_max_position_embeddings,
            )
        else:
            self._rope = nn.RoPE(rope_dims, base=args.rope_theta, traditional=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1)
        gate = None
        if self.output_gate:
            queries, gate = mx.split(queries, 2, axis=-1)

        queries = self.q_norm(queries)
        keys = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1))
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self._rope(queries, offset=cache.offset)
            keys = self._rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self._rope(queries)
            keys = self._rope(keys)

        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if gate is not None:
            out = out * mx.sigmoid(gate.reshape(B, L, -1))
        return self.o_proj(out)


@partial(mx.compile, shapeless=True)
def _compute_g(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


@mx.compile
def _gated_delta_step(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """One step of the recurrence.

    Shapes: q, k [B, H, Dk]; v [B, H, Dv]; g, beta [B, H]; state [B, H, Dv, Dk].
    """
    old_state = state
    state = state * g[..., None, None]
    kv_mem = (state * k[..., None, :]).sum(axis=-1)
    delta = (v - kv_mem) * beta[..., None]
    state = state + k[..., None, :] * delta[..., None]
    y = (state * q[..., None, :]).sum(axis=-1)
    if mask is not None:
        state = mx.where(mask[:, None, None, None], state, old_state)
    return y.astype(q.dtype), state


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Run the gated delta rule over a sequence.

    The recurrence is a sequential scan over the T steps, which keeps it
    differentiable.

    Shapes: q, k [B, T, Hk, Dk]; v [B, T, Hv, Dv]; a, b [B, T, Hv];
    mask [B, T]; state [B, Hv, Dv, Dk].
    Returns y [B, T, Hv, Dv] and the state after the last step.
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    beta = mx.sigmoid(b)
    g = _compute_g(A_log, a, dt_bias)

    # The key heads are grouped, so each one serves several value heads.
    if (repeats := Hv // Hk) > 1:
        q = mx.repeat(q, repeats, -2)
        k = mx.repeat(k, repeats, -2)

    ys = []
    for t in range(T):
        y, state = _gated_delta_step(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            None if mask is None else mask[:, t],
        )
        ys.append(y)
    return mx.stack(ys, axis=1), state


class GatedDeltaNet(nn.Module):
    """Linear attention with a gated delta rule, as in Qwen3.5."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.inv_scale = self.head_k_dim**-0.5

        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"linear_num_value_heads ({self.num_v_heads}) must be divisible "
                f"by linear_num_key_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)
        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = RMSNormGated(self.head_v_dim, eps=args.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = x.shape

        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=x.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache is not None else None

        q = (self.inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = self.inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


ATTENTION_TYPES = {
    "full": Attention,
    "gated_delta": GatedDeltaNet,
}


def build_attention(args: ModelArgs, attn_type: str) -> nn.Module:
    if attn_type not in ATTENTION_TYPES:
        raise ValueError(
            f"unknown attention {attn_type!r}; expected one of "
            f"{', '.join(ATTENTION_TYPES)}"
        )
    return ATTENTION_TYPES[attn_type](args)
