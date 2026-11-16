# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.activations import precise_swiglu
from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.gated_delta import gated_delta_update, normalize_qk
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.olmo_hybrid.config import ModelArgs

__all__ = [
    "Attention",
    "FullAttentionDecoderLayer",
    "GatedDeltaNet",
    "LinearAttentionDecoderLayer",
    "MLP",
    "RMSNormGated",
]


class RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(hidden_size)
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        normed = mx.fast.rms_norm(x, self.weight, self.eps)
        return precise_swiglu(x, gate, normed)


class GatedDeltaNet(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.allow_neg_eigval = args.linear_allow_neg_eigval

        self.q_proj = nn.Linear(args.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.g_proj = nn.Linear(args.hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, args.hidden_size, bias=False)

        self.q_conv1d = nn.Conv1d(
            in_channels=self.key_dim,
            out_channels=self.key_dim,
            kernel_size=self.conv_kernel_size,
            bias=False,
            groups=self.key_dim,
            padding=0,
        )
        self.k_conv1d = nn.Conv1d(
            in_channels=self.key_dim,
            out_channels=self.key_dim,
            kernel_size=self.conv_kernel_size,
            bias=False,
            groups=self.key_dim,
            padding=0,
        )
        self.v_conv1d = nn.Conv1d(
            in_channels=self.value_dim,
            out_channels=self.value_dim,
            kernel_size=self.conv_kernel_size,
            bias=False,
            groups=self.value_dim,
            padding=0,
        )

        self.A_log = mx.zeros(self.num_v_heads)
        self.dt_bias = mx.zeros(self.num_v_heads)

        self.o_norm = RMSNormGated(self.head_v_dim, eps=args.layer_norm_epsilon)

    def _apply_conv1d(
        self,
        x: mx.array,
        conv: nn.Conv1d,
        conv_state: Optional[mx.array],
        cache_slot: Optional[Any],
        slot_idx: int,
    ) -> mx.array:
        B, S, D = x.shape
        n_keep = self.conv_kernel_size - 1

        if conv_state is not None:
            padded = mx.concatenate([conv_state, x], axis=1)
        else:
            padded = mx.concatenate(
                [mx.zeros((B, n_keep, D), dtype=x.dtype), x], axis=1
            )

        if cache_slot is not None:
            cache_slot[slot_idx] = mx.contiguous(padded[:, -n_keep:, :])

        return nn.silu(conv(padded))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        conv_q_state = cache[0] if cache is not None else None
        conv_k_state = cache[1] if cache is not None else None
        conv_v_state = cache[2] if cache is not None else None

        q = self._apply_conv1d(q, self.q_conv1d, conv_q_state, cache, 0)
        k = self._apply_conv1d(k, self.k_conv1d, conv_k_state, cache, 1)
        v = self._apply_conv1d(v, self.v_conv1d, conv_v_state, cache, 2)

        q = q.reshape(B, S, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, S, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, S, self.num_v_heads, self.head_v_dim)

        q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)

        state = cache[3] if (cache is not None and cache[3] is not None) else None
        out, state = gated_delta_update(
            q,
            k,
            v,
            self.a_proj(x),
            self.b_proj(x),
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
            allow_neg_eigval=self.allow_neg_eigval,
        )

        if cache is not None:
            cache[3] = state
            cache.advance(S)

        gate = self.g_proj(x)
        out = out.reshape(-1, self.head_v_dim)
        gate = gate.reshape(-1, self.head_v_dim)
        out = self.o_norm(out, gate)
        out = out.reshape(B, S, -1)

        return self.o_proj(out)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        total_q_dim = args.num_attention_heads * self.head_dim
        total_kv_dim = args.num_key_value_heads * self.head_dim

        self.q_proj = nn.Linear(args.hidden_size, total_q_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(
            args.hidden_size, total_kv_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            args.hidden_size, total_kv_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(total_q_dim, args.hidden_size, bias=args.attention_bias)

        self.q_norm = nn.RMSNorm(total_q_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(total_kv_dim, eps=args.rms_norm_eps)

        self.rope = (
            initialize_rope(
                self.head_dim,
                base=args.rope_theta,
                traditional=False,
                max_position_embeddings=args.max_position_embeddings,
            )
            if args.rope_theta is not None
            else None
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_norm(self.q_proj(x))
        k = self.k_norm(self.k_proj(x))
        v = self.v_proj(x)

        q = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        k = k.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            if self.rope is not None:
                q = self.rope(q, offset=cache.offset)
                k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        elif self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        output = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LinearAttentionDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.linear_attn = GatedDeltaNet(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.linear_attn(self.input_layernorm(x), mask=mask, cache=cache)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class FullAttentionDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.post_attention_layernorm(self.self_attn(x, mask=mask, cache=cache))
        out = h + self.post_feedforward_layernorm(self.mlp(h))
        return out
