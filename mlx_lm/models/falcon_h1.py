from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import CacheList, KVCache, MambaCache
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    attention_bias: bool = False
    attention_in_multiplier: float = 1.0
    attention_out_multiplier: float = 0.9375
    embedding_multiplier: float = 5.656854249492381
    head_dim: int = 64
    hidden_act: str = "silu"
    hidden_size: int = 1024
    initializer_range: float = 0.02
    intermediate_size: int = 2048
    key_multiplier: float = 0.390625
    lm_head_multiplier: float = 0.0390625
    mamba_chunk_size: int = 128
    mamba_conv_bias: bool = True
    mamba_d_conv: int = 4
    mamba_d_head: int = 64
    mamba_d_ssm: int = 1536
    mamba_d_state: int = 128
    mamba_expand: int = 2
    mamba_n_groups: int = 1
    mamba_n_heads: int = 24
    mamba_norm_before_gate: bool = False
    mamba_proj_bias: bool = False
    mamba_rms_norm: bool = False
    mamba_use_mlp: bool = True
    max_position_embeddings: int = 131072
    mlp_bias: bool = False
    mlp_expansion_factor: int = 8
    mlp_multipliers: List[float] = field(
        default_factory=lambda: [0.8838834764831844, 0.5859375]
    )
    model_type: str = "falcon_h1"
    num_attention_heads: int = 8
    num_hidden_layers: int = 36
    num_key_value_heads: int = 2
    num_logits_to_keep: int = 1
    pad_token_id: int = 0
    projectors_bias: bool = False
    rms_norm_eps: float = 1e-05
    rope_traditional: bool = False
    rope_scaling: Optional[float] = None
    rope_theta: float = 100000000000.0
    ssm_in_multiplier: float = 1.25
    ssm_multipliers: List[float] = field(
        default_factory=lambda: [
            0.3535533905932738,
            0.25,
            0.3535533905932738,
            0.5,
            0.3535533905932738,
        ]
    )
    ssm_out_multiplier: float = 0.23570226039551587
    tie_word_embeddings: bool = False
    vocab_size: int = 32784


class FalconH1RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, n_groups=1, norm_before_gate=True):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps
        self.n_groups = n_groups
        self.norm_before_gate = norm_before_gate

    def __call__(self, hidden_states, gate=None):
        if not self.norm_before_gate and gate is not None:
            hidden_states = hidden_states * nn.silu(gate)

        hidden_states = mx.fast.rms_norm(
            hidden_states, self.weight, self.variance_epsilon
        )

        if self.norm_before_gate and gate is not None:
            hidden_states = hidden_states * nn.silu(gate)
        return hidden_states


def compute_mup_vector(args):
    intermediate_size = args.mamba_d_ssm
    groups_time_state_size = args.mamba_n_groups * args.mamba_d_state
    num_heads = args.mamba_n_heads
    sizes = [
        intermediate_size,
        intermediate_size,
        groups_time_state_size,
        groups_time_state_size,
        num_heads,
    ]
    return mx.concatenate(
        [
            mx.broadcast_to(mx.array(m), (s,))
            for s, m in zip(sizes, args.ssm_multipliers)
        ]
    )


class FalconH1Attention(nn.Module):

    def __init__(self, args, layer_idx: int):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.layer_idx = layer_idx
        self.key_multiplier = args.key_multiplier

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=args.attention_bias
        )

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = (
            keys.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
            * self.key_multiplier
        )
        values = values.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # TODO what's that about..
        if mask is not None:
            kv_seq_len = keys.shape[-2]

            if mask.ndim == 2:
                mask = mask[None, None, :, :]

            if kv_seq_len > L:
                if mask.shape[-1] < kv_seq_len:
                    num_heads_dim = mask.shape[1] if mask.shape[1] > 1 else 1

                    pad_length = kv_seq_len - mask.shape[-1]

                    pad_shape = (B, num_heads_dim, L, pad_length)
                    padding = mx.ones(pad_shape, dtype=mask.dtype)

                    mask = mx.concatenate([padding, mask], axis=-1)

        output = scaled_dot_product_attention(
            queries, keys, values, mask=mask, scale=self.scale, cache=cache
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output)


def pad_tensor_by_size(tensor, pad_size):
    if pad_size > 0:
        pad_shape = list(tensor.shape)
        pad_shape[1] = pad_size
        padding = mx.zeros(pad_shape, dtype=tensor.dtype)
        tensor = mx.concatenate([tensor, padding], axis=1)
    return tensor


def reshape_into_chunks(tensor, pad_size, chunk_size):

    if pad_size > 0:
        tensor = pad_tensor_by_size(tensor, pad_size)

    batch_size, seq_len = tensor.shape[:2]
    num_chunks = seq_len // chunk_size
    new_shape = [batch_size, num_chunks, chunk_size] + list(tensor.shape[2:])
    return mx.reshape(tensor, new_shape)


def segment_sum(input_tensor):
    chunk_size = input_tensor.shape[-1]

    input_tensor = mx.expand_dims(input_tensor, axis=-1)
    input_tensor = mx.broadcast_to(
        input_tensor, input_tensor.shape[:-1] + (chunk_size,)
    )

    mask = mx.tri(chunk_size, k=-1, dtype=mx.bool_)
    input_tensor = mx.where(mask, input_tensor, 0)

    tensor_segsum = mx.cumsum(input_tensor, axis=-2)

    mask = mx.tri(chunk_size, k=0, dtype=mx.bool_)
    tensor_segsum = mx.where(mask, tensor_segsum, -mx.inf)

    return tensor_segsum


class FalconH1Mixer(nn.Module):
    def __init__(self, args, layer_idx: int, mup_vector: mx.array):
        super().__init__()
        self.num_heads = args.mamba_n_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.mamba_d_state
        self.conv_kernel_size = args.mamba_d_conv
        self.intermediate_size = args.mamba_d_ssm
        self.layer_idx = layer_idx
        self.use_conv_bias = args.mamba_conv_bias

        self.layer_norm_epsilon = args.rms_norm_eps
        self.groups_time_state_size = args.mamba_n_groups * self.ssm_state_size

        self.n_groups = args.mamba_n_groups
        self.head_dim = args.mamba_d_head
        self.chunk_size = args.mamba_chunk_size

        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=args.mamba_proj_bias,
        )

        self.dt_bias = mx.ones(self.num_heads)

        A = mx.arange(1, self.num_heads + 1)
        self.A_log = mx.log(A)

        self.mamba_rms_norm = args.mamba_rms_norm
        if self.mamba_rms_norm:
            self.norm = FalconH1RMSNormGated(
                self.intermediate_size,
                eps=self.layer_norm_epsilon,
                n_groups=self.n_groups,
                norm_before_gate=args.mamba_norm_before_gate,
            )

        self.D = mx.ones(self.num_heads)

        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.projectors_bias
        )

        self.ssm_in_multiplier = args.ssm_in_multiplier
        self._mup_vector = mup_vector

    def __call__(self, input_states, cache=None, mask=None):
        batch_size, seq_len, _ = input_states.shape

        input_states = input_states * self.ssm_in_multiplier
        projected_states = self.in_proj(input_states)

        projected_states = (projected_states * self._mup_vector).astype(
            projected_states.dtype
        )

        gate, hidden_states_B_C, dt = mx.split(
            projected_states,
            (self.intermediate_size, self.intermediate_size + self.conv_dim),
            axis=-1,
        )

        if cache is not None:
            conv_state, ssm_state = cache[0], cache[1]
        else:
            conv_state, ssm_state = None, None
        use_precomputed_states = seq_len == 1 and conv_state is not None

        if conv_state is None:
            conv_state = mx.zeros(
                (batch_size, self.conv_kernel_size - 1, hidden_states_B_C.shape[-1]),
                hidden_states_B_C.dtype,
            )
        hidden_states_B_C = mx.concatenate([conv_state, hidden_states_B_C], axis=1)
        cache[0] = hidden_states_B_C[:, -self.conv_kernel_size + 1 :]
        hidden_states_B_C = nn.silu(self.conv1d(hidden_states_B_C))

        hidden_states, B, C = mx.split(
            hidden_states_B_C,
            (
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ),
            axis=-1,
        )

        A = -mx.exp(self.A_log)

        if use_precomputed_states:
            dt = dt[:, 0, :][:, None, ...]
            dt = mx.transpose(dt, [0, 2, 1])
            dt = mx.broadcast_to(dt, (batch_size, dt.shape[1], self.head_dim))

            dt_bias = mx.expand_dims(self.dt_bias, axis=-1)
            dt_bias = mx.broadcast_to(dt_bias, (self.dt_bias.shape[0], self.head_dim))

            dt = nn.softplus(dt + dt_bias)
            dt = mx.clip(dt, self.time_step_limit[0], self.time_step_limit[1])

            A = mx.expand_dims(mx.expand_dims(A, axis=-1), axis=-1)
            A = mx.broadcast_to(A, (self.num_heads, self.head_dim, self.ssm_state_size))

            dA = mx.exp(mx.expand_dims(dt, axis=-1) * A)

            B = mx.reshape(B, (batch_size, self.n_groups, -1))
            B = mx.expand_dims(B, axis=2)
            B = mx.broadcast_to(
                B,
                (
                    batch_size,
                    self.n_groups,
                    self.num_heads // self.n_groups,
                    B.shape[-1],
                ),
            )
            B = mx.reshape(B, (batch_size, -1, B.shape[-1]))

            dB = mx.expand_dims(dt, axis=-1) * mx.expand_dims(B, axis=2)

            hidden_states = mx.reshape(hidden_states, (batch_size, -1, self.head_dim))
            dBx = dB * mx.expand_dims(hidden_states, axis=-1)

            new_ssm_state = cache[1] * dA + dBx
            cache[1] = new_ssm_state

            C = mx.reshape(C, (batch_size, self.n_groups, -1))
            C = mx.expand_dims(C, axis=2)
            C = mx.broadcast_to(
                C,
                (
                    batch_size,
                    self.n_groups,
                    self.num_heads // self.n_groups,
                    C.shape[-1],
                ),
            )
            C = mx.reshape(C, (batch_size, -1, C.shape[-1]))

            ssm_states = cache[1]

            ssm_states_reshaped = mx.reshape(
                ssm_states,
                (batch_size * self.num_heads, self.head_dim, self.ssm_state_size),
            )
            C_reshaped = mx.reshape(
                C, (batch_size * self.num_heads, self.ssm_state_size, 1)
            )

            y = ssm_states_reshaped @ C_reshaped
            y = mx.reshape(y, (batch_size, self.num_heads, self.head_dim))

            D = mx.expand_dims(self.D, axis=-1)
            D = mx.broadcast_to(D, (self.D.shape[0], self.head_dim))
            y = y + hidden_states * D

            y = mx.reshape(y, (batch_size, -1))
            y = mx.expand_dims(y, axis=1)
        else:
            dt = nn.softplus(dt + self.dt_bias)
            dt = mx.clip(dt, self.time_step_limit[0], self.time_step_limit[1])
            hidden_states = mx.reshape(
                hidden_states, (batch_size, seq_len, -1, self.head_dim)
            )
            B = mx.reshape(B, (batch_size, seq_len, -1, self.ssm_state_size))
            C = mx.reshape(C, (batch_size, seq_len, -1, self.ssm_state_size))

            B = mx.repeat(B, self.num_heads // self.n_groups, axis=2)
            C = mx.repeat(C, self.num_heads // self.n_groups, axis=2)

            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            D_residual = mx.expand_dims(self.D, axis=-1) * pad_tensor_by_size(
                hidden_states, pad_size
            )

            hidden_states = hidden_states * mx.expand_dims(dt, axis=-1)
            A = A * dt

            hidden_states, A, B, C = [
                reshape_into_chunks(t, pad_size, self.chunk_size)
                for t in (hidden_states, A, B, C)
            ]

            A = mx.transpose(A, [0, 3, 1, 2])
            A_cumsum = mx.cumsum(A, axis=-1)

            L = mx.exp(segment_sum(A))

            C_expanded = mx.expand_dims(C, axis=3)
            B_expanded = mx.expand_dims(B, axis=2)
            G_intermediate = C_expanded * B_expanded
            G = mx.sum(G_intermediate, axis=-1)

            L_permuted = mx.transpose(L, [0, 2, 3, 4, 1])
            M_intermediate = mx.expand_dims(G, axis=-1) * mx.expand_dims(
                L_permuted, axis=-1
            )
            M = mx.sum(M_intermediate, axis=-1)

            hidden_states_expanded = mx.expand_dims(hidden_states, axis=2)
            M_expanded = mx.expand_dims(M, axis=-1)
            Y_diag = mx.sum(M_expanded * hidden_states_expanded, axis=3)

            decay_states = mx.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
            decay_states_permuted = mx.transpose(decay_states, [0, 2, 3, 1])
            B_decay = B * mx.expand_dims(decay_states_permuted, axis=-1)

            B_decay_expanded = mx.expand_dims(B_decay, axis=-2)
            hidden_states_expanded = mx.expand_dims(hidden_states, axis=-1)
            states = mx.sum(B_decay_expanded * hidden_states_expanded, axis=2)

            if use_precomputed_states:
                previous_states = mx.expand_dims(cache[1], axis=1)
            else:
                previous_states = mx.zeros_like(states[:, :1])

            states = mx.concatenate([previous_states, states], axis=1)

            A_cumsum_last = A_cumsum[:, :, :, -1]
            pad_width = [
                (0, 0),
                (0, 0),
                (1, 0),
            ]
            padded = mx.pad(A_cumsum_last, pad_width)
            decay_chunk = mx.exp(segment_sum(padded))
            decay_chunk = mx.transpose(decay_chunk, [0, 3, 2, 1])

            decay_expanded = mx.expand_dims(
                mx.expand_dims(decay_chunk, axis=-1), axis=-1
            )
            states_expanded = mx.expand_dims(states, axis=2)
            new_states = mx.sum(decay_expanded * states_expanded, axis=1)

            states = new_states[:, :-1]
            ssm_state = new_states[:, -1]

            state_decay_out = mx.exp(A_cumsum)
            C_expanded = mx.expand_dims(C, axis=-2)
            states_expanded = mx.expand_dims(states, axis=2)
            C_times_states = C_expanded * states_expanded

            state_decay_out_permuted = mx.transpose(state_decay_out, [0, 2, 3, 1])
            C_times_states_sum = mx.sum(C_times_states, axis=-1)
            Y_off = C_times_states_sum * mx.expand_dims(
                state_decay_out_permuted, axis=-1
            )

            y = Y_diag + Y_off

            y = mx.reshape(y, (batch_size, -1, self.num_heads, self.head_dim))
            y = y + D_residual

            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = mx.reshape(y, (batch_size, seq_len, -1))

            if cache is not None:
                cache[1] = ssm_state

        if self.mamba_rms_norm:
            scan_output = self.norm(y, gate)
        else:
            scan_output = y * nn.silu(gate)

        contextualized_states = self.out_proj(scan_output)
        return contextualized_states


class FalconH1MLP(nn.Module):

    def __init__(self, args):
        super().__init__()

        hidden_size = args.hidden_size
        intermediate_size = args.intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=args.mlp_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=args.mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=args.mlp_bias)
        self.gate_multiplier, self.down_multiplier = args.mlp_multipliers

    def __call__(self, x):
        y = self.up_proj(x) * nn.silu(self.gate_proj(x) * self.gate_multiplier)
        y = self.down_proj(y) * self.down_multiplier
        return y


class FalconH1DecoderLayer(nn.Module):
    def __init__(self, args, layer_idx: int, mup_vector: mx.array):
        super().__init__()
        self.feed_forward = FalconH1MLP(args)

        head_dim = args.head_dim
        self.channels_attn = (
            args.num_attention_heads * head_dim
            + 2 * args.num_key_value_heads * head_dim
        )

        self.mamba = FalconH1Mixer(
            args=args, layer_idx=layer_idx, mup_vector=mup_vector
        )

        self.self_attn = FalconH1Attention(args, layer_idx)

        self.attention_in_multiplier = args.attention_in_multiplier
        self.ssm_out_multiplier = args.ssm_out_multiplier
        self.attn_out_multiplier = args.attention_out_multiplier

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_ff_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        cache,
        mask: mx.array,
        mamba_mask: mx.array,
        **kwargs,
    ) -> mx.array:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        mamba_hidden_states = self.mamba(
            input_states=hidden_states, cache=cache[0], mask=mamba_mask
        )

        attention_hidden_states = self.self_attn(
            hidden_states * self.attention_in_multiplier,
            mask=mask,
            cache=cache[1],
        )

        # TODO maybe compile that
        mamba_hidden_states = mamba_hidden_states * self.ssm_out_multiplier
        attention_hidden_states = attention_hidden_states * self.attn_out_multiplier
        hidden_states = residual + mamba_hidden_states + attention_hidden_states

        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class FalconH1Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        mup_vector = compute_mup_vector(args)
        self.layers = [
            FalconH1DecoderLayer(args, layer_idx=layer_idx, mup_vector=mup_vector)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.final_layernorm = nn.RMSNorm(self.hidden_size, eps=args.rms_norm_eps)

    def _update_mamba_mask(self, attention_mask, cache):
        mamba_mask = attention_mask
        if (cache is not None and len(cache[0].key_cache) > 0) or (
            attention_mask is not None and mx.all(attention_mask == 1)
        ):
            mamba_mask = None
        return mamba_mask

    def __call__(self, inputs, mask=None, cache=None):

        h = self.embed_tokens(inputs)

        h = h * self.args.embedding_multiplier

        if mask is None:
            c = [cache[0][1]] if cache is not None else None
            mask = create_attention_mask(h, c, return_array=True)

        if cache is None:
            cache = [None] * len(self.layers)

        #        mamba_mask = self._update_mamba_mask(mask, cache)
        mamba_mask = None  # self._update_mamba_mask(mask, cache)

        for layer, c in zip(self.layers, cache):
            h = layer(
                h,
                cache=c,
                mask=mask,
                mamba_mask=mamba_mask,
            )

        return self.final_layernorm(h)


class Model(nn.Module):
    """Falcon-H1 model with language modeling head"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FalconH1Model(args=args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, mask=None, cache=None):
        hidden_states = self.model(inputs, mask=mask, cache=cache)
        logits = self.lm_head(hidden_states)
        return logits * self.args.lm_head_multiplier

    def sanitize(self, weights):
        sanitized_weights = {}
        for name, param in weights.items():
            if "conv1d.weight" in name:
                # MLX Conv1d expects [out_channels, in_channels, kernel_size] format
                if param.shape[-1] > param.shape[1]:
                    param = param.transpose(0, 2, 1)

            sanitized_weights[name] = param
        return sanitized_weights

    def make_cache(self):
        return [
            CacheList(MambaCache(), KVCache())
            for _ in range(self.args.num_hidden_layers)
        ]

    @property
    def layers(self):
        return self.model.layers
