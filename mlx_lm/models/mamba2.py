import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import MambaCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    num_heads: int
    head_dim: int
    vocab_size: int
    hidden_size: int
    state_size: int
    num_hidden_layers: int
    layer_norm_epsilon: float
    expand: int
    conv_kernel: int
    n_groups: int
    use_bias: bool
    use_conv_bias: bool
    initializer_range: float
    residual_in_fp32: bool
    chunk_size: int
    tie_word_embeddings: bool
    time_step_limit: Tuple[float, float]
    time_step_rank: Union[int, str]
    time_step_min: float
    time_step_max: float
    time_step_floor: float
    ssm_state_size: int = None
    norm_before_gate: bool = True
    max_position_embeddings: int = 2056

    def __post_init__(self):
        if not hasattr(self, "intermediate_size"):
            self.intermediate_size = int(self.expand * self.hidden_size)
        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_heads
        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)
        if self.ssm_state_size is None:
            self.ssm_state_size = self.state_size


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array = None) -> mx.array:
        if gate is not None:
            hidden_states = hidden_states * nn.silu(gate)

        variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
        return self.weight * hidden_states * mx.rsqrt(variance + self.eps)


class Mamba2Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = args.num_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.ssm_state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.num_heads * args.head_dim
        self.use_conv_bias = args.use_conv_bias
        self.n_groups = args.n_groups
        self.head_dim = args.head_dim
        self.time_step_limit = args.time_step_limit
        self.heads_per_group = self.num_heads // self.n_groups
        self.use_bias = args.use_bias

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
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=args.use_bias)

        self.dt_bias = mx.ones(self.num_heads)
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)

        self.norm = MambaRMSNormGated(
            self.intermediate_size, eps=args.layer_norm_epsilon
        )
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.use_bias
        )

    def _apply_conv(
        self, conv_input: mx.array, cache: Optional[MambaCache] = None
    ) -> mx.array:
        batch_size, seq_len, in_ch = conv_input.shape

        if self.conv_kernel_size == 1:
            return nn.silu(self.conv1d(conv_input))

        if cache is not None:
            current_conv_state = cache[0]
            if current_conv_state is None:
                current_conv_state = cache[0] = mx.zeros(
                    (batch_size, self.conv_kernel_size, in_ch)
                )

            if seq_len >= self.conv_kernel_size:
                cache[0] = conv_input[:, -self.conv_kernel_size :, :]
            else:
                cache[0] = mx.concatenate(
                    [current_conv_state[:, seq_len:, :], conv_input], axis=1
                )

            left_padding = current_conv_state[:, -(self.conv_kernel_size - 1) :, :]
            padded_input = mx.concatenate([left_padding, conv_input], axis=1)
        else:
            padded_input = mx.pad(
                conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)]
            )

        conv_output = self.conv1d(padded_input)[:, :seq_len, :]
        return nn.silu(conv_output)

    def _ssm_vectorized(
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

        B = mx.repeat(
            B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size),
            self.heads_per_group,
            axis=2,
        )
        C = mx.repeat(
            C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size),
            self.heads_per_group,
            axis=2,
        )
        A = -mx.exp(self.A_log.astype(mx.float32))

        if cache is not None:
            h = (
                cache[1]
                if cache[1] is not None
                else mx.zeros(
                    (batch_size, self.num_heads, self.head_dim, self.ssm_state_size)
                )
            )
        else:
            h = mx.zeros(
                (batch_size, self.num_heads, self.head_dim, self.ssm_state_size)
            )

        outputs = []

        for t in range(seq_len):
            dt_t = dt[:, t, :, None, None]
            dA = mx.exp(
                mx.einsum(
                    "bh,bhds->bhds",
                    dt_t.squeeze(-1).squeeze(-1) * A[None, :],
                    mx.ones_like(h),
                )
            )
            dB_x = mx.einsum("bh,bhs,bhd->bhds", dt[:, t], B[:, t], hidden_states[:, t])
            h = mx.einsum("bhds,bhds->bhds", dA, h) + dB_x
            y_t = mx.einsum("bhs,bhds->bhd", C[:, t], h) + mx.einsum(
                "h,bhd->bhd", self.D, hidden_states[:, t]
            )
            outputs.append(y_t)

        if cache is not None:
            cache[1] = h

        y = mx.stack(outputs, axis=1)
        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(
        self, hidden_states: mx.array, cache: Optional[MambaCache] = None
    ) -> mx.array:
        projected = self.in_proj(hidden_states)
        gate, conv_input, dt = mx.split(
            projected,
            [self.intermediate_size, self.intermediate_size + self.conv_dim],
            axis=-1,
        )
        conv_output = self._apply_conv(conv_input, cache)
        hidden_states, B, C = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )
        y = self._ssm_vectorized(hidden_states, B, C, dt, cache)
        y = self.norm(y, gate)
        return self.out_proj(y)


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.residual_in_fp32 = args.residual_in_fp32
        self.mixer = Mamba2Block(args, layer_idx)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache: Optional[MambaCache] = None) -> mx.array:
        if self.residual_in_fp32:
            x = x.astype(mx.float32)

        output = self.mixer(self.norm(x), cache)
        return output + x


class Mamba2(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args, i) for i in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(
        self, x: mx.array, cache: Optional[list[MambaCache]] = None
    ) -> mx.array:
        x = self.embeddings(x)
        hidden = x

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            hidden = layer(hidden, layer_cache)

        return self.norm_f(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Mamba2(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self, inputs: mx.array, cache: Optional[list[MambaCache]] = None
    ) -> mx.array:
        hidden = self.backbone(inputs, cache)

        if self.args.tie_word_embeddings:
            logits = self.backbone.embeddings.as_linear(hidden)
        else:
            logits = self.lm_head(hidden)

        return logits

    def make_cache(self, batch_size: int = 1) -> list[MambaCache]:
        return [MambaCache() for _ in range(self.args.num_hidden_layers)]

    @property
    def layers(self):
        return self.backbone.layers

    def sanitize(self, weights):
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return weights
