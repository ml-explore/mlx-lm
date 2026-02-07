# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_attention_mask, create_ssm_mask

from .activations import swiglu
from .cache import ArraysCache, KVCache
from .ssm import ssm_update


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "plamo2"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    hidden_size_per_head: int = 128
    max_position_embeddings: int = 2048
    attention_window_size: int = 2048
    full_attention_idx: Optional[list[int]] = None
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_num_heads: int = 64
    mamba_step: int = 2
    mamba_chunk_size: int = 256
    mamba_enabled: bool = True
    intermediate_size: int = 13312
    vocab_size: int = 32000


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        offset: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = mx.zeros(hidden_size)
        self.variance_epsilon = eps
        self.offset = offset

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return mx.fast.rms_norm(
            hidden_states, self.weight + self.offset, self.variance_epsilon
        )


class Mamba(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.d_state = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.chunk_size = config.mamba_chunk_size
        self.num_heads = config.mamba_num_heads
        self.hidden_size_per_head = config.hidden_size_per_head

        self.intermediate_size = self.num_heads * self.hidden_size_per_head

        self.in_proj = nn.Linear(
            self.hidden_size, 2 * self.intermediate_size, bias=False
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=0,
        )
        self.dt_dim = max(64, self.hidden_size // 16)
        self.bcdt_proj = nn.Linear(
            self.intermediate_size,
            self.dt_dim + 2 * self.d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_dim, self.num_heads, bias=False)

        self.dt_bias = mx.zeros(shape=(self.num_heads,))
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))

        self.D = mx.ones(self.num_heads)

        self.dt_norm_weight = mx.ones(self.dt_dim)
        self.B_norm_weight = mx.ones(self.d_state)
        self.C_norm_weight = mx.ones(self.d_state)

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def _conv(
        self,
        conv_input: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
    ) -> mx.array:
        if mask is not None:
            conv_input = mx.where(mask[..., None], conv_input, 0)

        if cache is not None:
            if cache[0] is None:
                conv_state = mx.zeros(
                    (
                        conv_input.shape[0],
                        self.conv_kernel_size - 1,
                        self.intermediate_size,
                    ),
                    dtype=conv_input.dtype,
                )
            else:
                conv_state = cache[0]
            padded_input = mx.concatenate([conv_state, conv_input], axis=1)
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                t = padded_input.shape[1]
                ends = mx.clip(cache.lengths, 0, t - n_keep)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(padded_input, positions, axis=1)
            else:
                cache[0] = padded_input[:, -n_keep:, :]
        else:
            padded_input = mx.pad(
                conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)]
            )

        conv_output = self.conv1d(padded_input)
        return nn.silu(conv_output)

    def _ssm(
        self,
        x: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        cache: Optional[Any],
        mask: Optional[mx.array],
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape

        x = x.reshape(batch_size, seq_len, self.num_heads, self.hidden_size_per_head)
        B = B.reshape(batch_size, seq_len, 1, self.d_state)
        C = C.reshape(batch_size, seq_len, 1, self.d_state)
        if cache:
            state = cache[1]
            lengths = cache.lengths
        else:
            state, lengths = None, None

        y, state = ssm_update(
            x,
            self.A_log,
            B,
            C,
            self.D,
            dt,
            self.dt_bias,
            state,
            mask=mask,
            lengths=lengths,
        )
        if cache:
            cache[1] = state
        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        bsize, length, _ = hidden_states.shape

        zx = self.in_proj(hidden_states)
        zx = zx.reshape(bsize, length, self.num_heads, -1)
        # z: (bsize, length, num_heads, hidden_size_per_head)
        # x: (bsize, length, num_heads, hidden_size_per_head)
        z, x = mx.split(
            zx,
            [
                self.hidden_size_per_head,
            ],
            axis=-1,
        )

        x = x.reshape(bsize, -1, self.num_heads * self.hidden_size_per_head)
        x = self._conv(x, cache, mask)

        BCdt = self.bcdt_proj(x)
        B, C, dt = mx.split(BCdt, [self.d_state, self.d_state * 2], axis=-1)

        A = -mx.exp(self.A_log.astype(mx.float32))  # (num_heads,)
        dt = mx.fast.rms_norm(dt, self.dt_norm_weight, self.config.rms_norm_eps)
        B = mx.fast.rms_norm(B, self.B_norm_weight, self.config.rms_norm_eps)
        C = mx.fast.rms_norm(C, self.C_norm_weight, self.config.rms_norm_eps)

        # (bsize, length, num_heads)
        dt = self.dt_proj(dt)
        out = self._ssm(
            x,
            B,
            C,
            dt,
            cache,
            mask,
        )
        if cache:
            cache.advance(out.shape[1])

        out = swiglu(z.flatten(-2), out)
        return self.out_proj(out)


class Attention(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        head_dim = config.hidden_size_per_head
        self.max_position_embeddings = config.max_position_embeddings
        self.scale = head_dim**-0.5

        self.q_num_heads = config.num_attention_heads
        self.qk_dim = self.v_dim = head_dim
        self.k_num_heads = self.v_num_heads = config.num_key_value_heads
        assert self.q_num_heads % self.k_num_heads == 0
        self.n_group = self.q_num_heads // self.k_num_heads

        self.q_proj_dim = self.q_num_heads * self.qk_dim
        self.k_proj_dim = self.k_num_heads * self.qk_dim
        self.v_proj_dim = self.k_num_heads * self.v_dim
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.q_proj_dim + self.k_proj_dim + self.v_proj_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.q_num_heads * self.v_dim, self.hidden_size, bias=False
        )

        self.q_weight = mx.ones((self.q_num_heads, self.qk_dim))
        self.k_weight = mx.ones((self.k_num_heads, self.qk_dim))

        self.rope = nn.RoPE(self.qk_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        B, T, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = mx.split(
            qkv, [self.q_proj_dim, self.q_proj_dim + self.k_proj_dim], axis=-1
        )
        q = q.reshape(B, T, self.q_num_heads, self.qk_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.k_num_heads, self.qk_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.v_num_heads, self.v_dim).transpose(0, 2, 1, 3)

        q = mx.fast.rms_norm(q, weight=None, eps=1e-6) * self.q_weight[:, None]
        k = mx.fast.rms_norm(k, weight=None, eps=1e-6) * self.k_weight[:, None]

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        output = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            B, T, self.q_num_heads * self.v_dim
        )
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size * 2, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.gate_up_proj(x)
        hs = mx.split(h, 2, axis=-1)
        return self.down_proj(swiglu(hs[0], hs[1]))


class PlamoDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, is_mamba: bool) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.is_mamba = is_mamba
        self.mixer: nn.Module
        if is_mamba:
            self.mixer = Mamba(config)
        else:
            self.mixer = Attention(config)
        self.mlp = MLP(config)
        self.pre_mixer_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0
        )
        self.post_mixer_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / 5
        )
        self.pre_mlp_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0
        )
        self.post_mlp_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / (5**1.5)
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        residual = hidden_states
        hidden_states = self.pre_mixer_norm(hidden_states)

        hidden_states_sa = self.mixer(
            hidden_states=hidden_states,
            mask=mask,
            cache=cache,
        )

        hidden_states_sa = self.post_mixer_norm(hidden_states_sa)
        hidden_states = residual + hidden_states_sa

        residual = hidden_states
        hidden_states = self.pre_mlp_norm(hidden_states)

        # Fully Connected
        hidden_states_mlp = self.mlp(hidden_states)

        # Residual
        hidden_states_mlp = self.post_mlp_norm(hidden_states_mlp)
        return residual + hidden_states_mlp


def is_mamba(config: ModelArgs, i: int) -> bool:
    if not config.mamba_enabled:
        return False
    assert config.mamba_step > 1
    assert i < config.num_hidden_layers

    if config.num_hidden_layers <= (config.mamba_step // 2):
        # use attention in last layer
        return i != config.num_hidden_layers - 1
    return (i % config.mamba_step) != (config.mamba_step // 2)


class PlamoDecoder(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()

        self.layers = [
            PlamoDecoderLayer(config, is_mamba=is_mamba(config, i))
            for i in range(config.num_hidden_layers)
        ]
        self.ssm_idx = 0 if config.mamba_enabled else None
        self.fa_idx = config.mamba_step // 2

    def __call__(self, x: mx.array, cache):
        if cache is None:
            cache = [None] * len(self.layers)

        attn_mask = create_attention_mask(x, cache[self.fa_idx])
        if self.ssm_idx is not None:
            mamba_mask = create_ssm_mask(x, cache[self.ssm_idx])
        else:
            mamba_mask = None

        for (
            l,
            c,
        ) in zip(self.layers, cache):
            x = l(
                x,
                mask=mamba_mask if l.is_mamba else attn_mask,
                cache=c,
            )
        return x


class PlamoModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = PlamoDecoder(config)  # type: ignore
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        batch_size, seq_length = inputs.shape

        h = self.embed_tokens(inputs)

        out = self.layers(
            h,
            cache,
        )

        return self.norm(out)


class Model(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = PlamoModel(config)

        self.vocab_size = config.vocab_size

        if not config.tie_word_embeddings:
            self.lm_head: nn.Module = nn.Linear(
                config.hidden_size, self.vocab_size, bias=False
            )

    def sanitize(self, weights: dict[Any, Any]) -> dict[Any, Any]:
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return weights

    def make_cache(self):
        # TODO use RotatingKVCache is not full_attn
        # full_attn = self.layer_idx in self.config.full_attention_idx
        return [ArraysCache(size=2) if l.is_mamba else KVCache() for l in self.layers]

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        outputs = self.model(
            inputs=inputs,
            cache=cache,
        )
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(outputs)
        else:
            logits = self.lm_head(outputs)

        return logits

    @property
    def layers(self):
        return self.model.layers.layers
