# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, MambaCache


@dataclass()
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_bias: bool
    mamba_num_heads: int
    mamba_head_dim: int
    mamba_proj_bias: bool
    ssm_state_size: int
    conv_kernel: int
    n_groups: int
    time_step_limit: Tuple[float, float]
    mlp_bias: bool
    layer_norm_epsilon: float
    rms_norm_eps: float
    use_bias: bool
    use_conv_bias: bool
    residual_in_fp32: bool
    head_dim: Optional[int] = None
    hybrid_override_pattern: Optional[List[str]] = None


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array = None) -> mx.array:
        if gate is not None:
            hidden_states = hidden_states * nn.silu(gate)
        return mx.fast.rms_norm(hidden_states, self.weight, self.eps)


class NemotronHMamba2Mixer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.mamba_num_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.ssm_state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.mamba_num_heads * args.mamba_head_dim
        self.n_groups = args.n_groups
        self.head_dim = args.mamba_head_dim
        self.time_step_limit = args.time_step_limit
        self.heads_per_group = self.num_heads // self.n_groups

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.conv_kernel,
            padding=0,
            groups=self.conv_dim,
            bias=args.use_conv_bias,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size, projection_size, bias=args.mamba_proj_bias
        )

        self.dt_bias = mx.ones(self.num_heads)
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)

        self.norm = MambaRMSNormGated(
            self.intermediate_size, eps=args.layer_norm_epsilon
        )
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.mamba_proj_bias
        )

    def _apply_conv(
        self, conv_input: mx.array, cache: Optional[MambaCache] = None
    ) -> mx.array:
        if cache is not None:
            if cache[0] is None:
                conv_state = mx.zeros(
                    (conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim),
                    dtype=conv_input.dtype,
                )
            else:
                conv_state = cache[0]
            padded_input = mx.concatenate([conv_state, conv_input], axis=1)
            cache[0] = padded_input[:, -(self.conv_kernel_size - 1) :, :]
        else:
            padded_input = mx.pad(
                conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)]
            )

        conv_output = self.conv1d(padded_input)
        return nn.silu(conv_output)

    def _ssm(
        self,
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        cache: Optional[MambaCache] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        dt = nn.softplus(dt + self.dt_bias)
        dt = mx.clip(dt, self.time_step_limit[0], self.time_step_limit[1])

        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        B = mx.repeat(B, self.heads_per_group, axis=2)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = mx.repeat(C, self.heads_per_group, axis=2)

        A = -mx.exp(self.A_log.astype(mx.float32)).astype(hidden_states.dtype)

        if cache is not None and cache[1] is not None:
            h = cache[1]
        else:
            h = mx.zeros(
                (batch_size, self.num_heads, self.head_dim, self.ssm_state_size),
                dtype=hidden_states.dtype,
            )

        outputs = []
        for t in range(seq_len):
            dt_t = dt[:, t, :]
            dA = mx.exp(dt_t * A)[..., None, None]
            dB = (dt_t[..., None] * B[:, t])[..., None, :]

            h = dA * h + dB * hidden_states[:, t, :, :, None]
            y_t = (h @ C[:, t, :, :, None]).squeeze(-1) + self.D[
                :, None
            ] * hidden_states[:, t]
            outputs.append(y_t)

        if cache is not None:
            cache[1] = h

        y = mx.stack(outputs, axis=1)
        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[MambaCache] = None,
    ) -> mx.array:

        projected = self.in_proj(hidden_states)

        gate, conv_input, dt = mx.split(
            projected,
            [self.intermediate_size, self.intermediate_size + self.conv_dim],
            axis=-1,
        )

        conv_output = self._apply_conv(conv_input, cache)

        hidden_states_ssm, B, C = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )
        y = self._ssm(hidden_states_ssm, B, C, dt, cache)
        y = self.norm(y, gate)
        return self.out_proj(y)


class NemotronHAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = (
            args.head_dim
            if args.head_dim is not None
            else (args.hidden_size // args.num_attention_heads)
        )
        self.num_key_value_heads = args.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=args.attention_bias
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = (
            self.k_proj(x)
            .reshape(B, L, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(x)
            .reshape(B, L, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


@partial(mx.compile, shapeless=True)
def relu2(x):
    return mx.square(nn.relu(x))


class NemotronHMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.up_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=args.mlp_bias
        )
        self.down_proj = nn.Linear(
            args.intermediate_size, args.hidden_size, bias=args.mlp_bias
        )

    def __call__(self, x):
        return self.down_proj(relu2(self.up_proj(x)))


class NemotronHBlock(nn.Module):
    def __init__(self, args: ModelArgs, block_type: str):
        super().__init__()
        self.residual_in_fp32 = args.residual_in_fp32
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.block_type = block_type

        if self.block_type == "M":
            self.mixer = NemotronHMamba2Mixer(args)
        elif self.block_type == "*":
            self.mixer = NemotronHAttention(args)
        elif self.block_type == "-":
            self.mixer = NemotronHMLP(args)

    def __call__(
        self,
        x,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        hidden_states = self.norm(x)
        if self.block_type == "M":
            hidden_states = self.mixer(hidden_states, cache=cache)
        elif self.block_type == "*":
            hidden_states = self.mixer(hidden_states, mask=mask, cache=cache)
        else:
            hidden_states = self.mixer(hidden_states)

        return x + hidden_states


class NemotronHModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            NemotronHBlock(args, block_type)
            for block_type in args.hybrid_override_pattern
        ]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx = 0
        for b in args.hybrid_override_pattern:
            if b == "*":
                break
            elif b == "M":
                self.fa_idx += 1

    def __call__(
        self,
        inputs,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        hidden_states = self.embeddings(inputs)

        if mask is None:
            attn_mask = create_attention_mask(
                hidden_states, cache[self.fa_idx : self.fa_idx + 1]
            )

        if cache is None:
            cache = [None] * len(self.layers)

        cache_counter = 0
        for layer in self.layers:
            if layer.block_type == "M" or layer.block_type == "*":
                c = cache[cache_counter]
                cache_counter += 1
            else:
                c = None

            if layer.block_type == "*":
                mask = attn_mask
            else:
                mask = None
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm_f(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.backbone = NemotronHModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        out = self.backbone(inputs, mask=mask, cache=cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self):
        caches = []
        for l in self.layers:
            if l.block_type == "M":
                caches.append(MambaCache())
            elif l.block_type == "*":
                caches.append(KVCache())
        return caches

    def sanitize(self, weights):
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return weights
