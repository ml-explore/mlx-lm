# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.common import swiglu
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.models.jamba.config import ModelArgs

__all__ = [
    "JambaAttention",
    "JambaDecoderLayer",
    "JambaMLP",
    "JambaMambaMixer",
    "JambaSparseMoeBlock",
    "fma",
]


class JambaMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class JambaAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


@mx.compile
def fma(a, b, c):
    return a * b + c


class JambaMambaMixer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.mamba_d_state
        self.conv_kernel_size = args.mamba_d_conv
        self.intermediate_size = args.mamba_expand * args.hidden_size
        self.time_step_rank = args.mamba_dt_rank
        self.use_conv_bias = args.mamba_conv_bias
        self.use_bias = args.mamba_proj_bias

        self.in_proj = nn.Linear(
            self.hidden_size, self.intermediate_size * 2, bias=self.use_bias
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
            self.time_step_rank + self.ssm_state_size * 2,
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
            self.intermediate_size, self.hidden_size, bias=self.use_bias
        )

        self.dt_layernorm = nn.RMSNorm(self.time_step_rank, eps=args.rms_norm_eps)
        self.b_layernorm = nn.RMSNorm(self.ssm_state_size, eps=args.rms_norm_eps)
        self.c_layernorm = nn.RMSNorm(self.ssm_state_size, eps=args.rms_norm_eps)

    def ssm_step(self, x, A, state=None):
        T = x.shape[1]
        D = self.D
        deltaBC = self.x_proj(x)
        delta, B, C = mx.split(
            deltaBC,
            [self.time_step_rank, self.time_step_rank + self.ssm_state_size],
            axis=-1,
        )
        delta, B, C = self.dt_layernorm(delta), self.b_layernorm(B), self.c_layernorm(C)
        delta = nn.softplus(self.dt_proj(delta))
        new_state = mx.expand_dims(delta * x, -1) * mx.expand_dims(B, -2)
        dtA = mx.exp(mx.expand_dims(delta, -1) * A)

        # TODO, speed up prefill with chunked scan
        for t in range(T):
            if state is not None:
                new_state[:, t] = fma(state, dtA[:, t], new_state[:, t])
            state = new_state[:, t]
        y = (new_state @ mx.expand_dims(C, -1)).squeeze(-1)
        y = y + D * x
        return y, new_state[:, -1]

    def _process_sequence(self, x, conv_state, ssm_state):
        xz = self.in_proj(x)
        x, z = xz.split(indices_or_sections=2, axis=-1)
        K = self.conv_kernel_size
        if conv_state is not None:
            x_full = mx.concatenate([conv_state, x], axis=1)
        else:
            x_full = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)])
        conv_out = self.conv1d(x_full)
        conv_state = x_full[:, -(K - 1) :, :]
        x = nn.silu(conv_out)
        A = -mx.exp(self.A_log)
        y, ssm_state = self.ssm_step(x, A, ssm_state)
        z = self.out_proj(swiglu(z, y))
        return z, (conv_state, ssm_state)

    def __call__(self, x, cache):
        if cache is None:
            conv_state, ssm_state = None, None
        else:
            conv_state, ssm_state = cache[0], cache[1]

        output, (conv_state, ssm_state) = self._process_sequence(
            x, conv_state, ssm_state
        )

        if cache is not None:
            cache[0] = conv_state
            cache[1] = ssm_state

        return output


class JambaSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok

        self.router = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.intermediate_size, args.num_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.router(x)
        k = self.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y


class JambaDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str, layer_idx: int):
        super().__init__()
        self.is_attn = layer_type == "attention"
        if self.is_attn:
            self.self_attn = JambaAttention(args)
        else:
            self.mamba = JambaMambaMixer(args)
        if (
            args.num_experts > 1
            and (layer_idx + args.expert_layer_offset) % args.expert_layer_period == 0
        ):
            ffn_layer_class = JambaSparseMoeBlock
        else:
            ffn_layer_class = JambaMLP
        self.feed_forward = ffn_layer_class(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_ff_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_attn:
            h = self.self_attn(self.input_layernorm(x), mask, cache)
        else:
            h = self.mamba(self.input_layernorm(x), cache)
        r = x + h
        out = r + self.feed_forward(self.pre_ff_layernorm(r))
        return out
