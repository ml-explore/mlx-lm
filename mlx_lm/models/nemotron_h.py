# Copyright © 2023-2025 Apple Inc.

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, MambaCache


class UnifiedCache:
    def __init__(self, attn_cache: list[KVCache], mamba_cache: list[MambaCache]):
        self.attn_cache = attn_cache
        self.mamba_cache = mamba_cache


@dataclass(kw_only=True)
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
    mamba_hidden_act: str
    mamba_proj_bias: bool
    ssm_state_size: int
    chunk_size: int
    conv_kernel: int
    n_groups: int
    time_step_rank: int
    time_step_min: float
    time_step_max: float
    time_step_floor: float
    time_step_limit: Tuple[float, float]
    mlp_bias: bool
    mlp_hidden_act: str
    layer_norm_epsilon: float
    rms_norm_eps: float
    use_bias: bool
    use_conv_bias: bool
    residual_in_fp32: bool
    rescale_prenorm_residual: bool
    tie_word_embeddings: bool
    head_dim: Optional[int] = None
    num_heads: Optional[int] = None
    hybrid_override_pattern: Optional[List[str]] = None
    expand: Optional[int] = None

    def __post_init__(self):
        if self.expand is None:
            self.expand = self.intermediate_size // self.hidden_size
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_heads is None:
            self.num_heads = self.num_attention_heads


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array = None) -> mx.array:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        if gate is not None:
            hidden_states = hidden_states * nn.silu(gate.astype(mx.float32))
        variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).astype(input_dtype)


class NemotronHMamba2Mixer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
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
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=args.use_bias)

        self.dt_bias = mx.ones(self.num_heads)
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)

        self.norm = MambaRMSNormGated(self.intermediate_size, eps=args.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=args.use_bias)

    def _apply_conv(self, conv_input: mx.array, cache: Optional[MambaCache] = None) -> mx.array:
        if cache is not None:
            conv_state = cache[0] if cache[0] is not None else mx.zeros((conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim))
            padded_input = mx.concatenate([conv_state, conv_input], axis=1)
            new_conv_state = padded_input[:, -(self.conv_kernel_size - 1):, :]
            cache[0] = new_conv_state
        else:
            padded_input = mx.pad(conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)])

        conv_output = self.conv1d(padded_input)
        conv_output = conv_output[:, :conv_input.shape[1], :]
        return nn.silu(conv_output)

    def _ssm(self, hidden_states: mx.array, B: mx.array, C: mx.array, dt: mx.array, 
             cache: Optional[MambaCache] = None) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        dt = nn.softplus(dt + self.dt_bias)
        dt = mx.clip(dt, self.time_step_limit[0], self.time_step_limit[1])

        hidden_states = hidden_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        B = mx.repeat(B, self.heads_per_group, axis=2)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size) 
        C = mx.repeat(C, self.heads_per_group, axis=2)

        A = -mx.exp(self.A_log.astype(mx.float32))

        if cache is not None:
            h = cache[1] if cache[1] is not None else mx.zeros((batch_size, self.num_heads, self.head_dim, self.ssm_state_size))
        else:
            h = mx.zeros((batch_size, self.num_heads, self.head_dim, self.ssm_state_size))

        outputs = []
        for t in range(seq_len):
            dt_t = dt[:, t, :, None, None]
            dA = mx.exp(dt_t * A[None, :, None, None])
            dB = dt_t * B[:, t, :, None, :]

            h = dA * h + dB * hidden_states[:, t, :, :, None]
            y_t = mx.sum(C[:, t, :, None, :] * h, axis=-1) + self.D[None, :, None] * hidden_states[:, t]
            outputs.append(y_t)

        if cache is not None:
            cache[1] = h

        y = mx.stack(outputs, axis=1)
        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(self, hidden_states: mx.array, cache: Optional[MambaCache] = None,
                 attention_mask: Optional[mx.array] = None) -> mx.array:
        
        projected = self.in_proj(hidden_states)

        gate, conv_input, dt = mx.split(
            projected, 
            [self.intermediate_size, self.intermediate_size + self.conv_dim], 
            axis=-1
        )

        conv_output = self._apply_conv(conv_input, cache)

        # Split conv output
        hidden_states_ssm, B, C = mx.split(
            conv_output,
            [self.intermediate_size, self.intermediate_size + self.n_groups * self.ssm_state_size],
            axis=-1
        )

        # Apply SSM
        y = self._ssm(hidden_states_ssm, B, C, dt, cache)

        # Apply gated normalization and output projection
        y = self.norm(y, gate)
        return self.out_proj(y)


class NemotronHAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.num_key_value_heads = args.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=args.attention_bias)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, 
                 cache: Optional[KVCache] = None) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class NemotronHMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.mlp_bias)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.up_proj(x)))


class NemotronHBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_type = args.hybrid_override_pattern[layer_idx]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        if self.block_type == "M":
            self.mixer = NemotronHMamba2Mixer(args, layer_idx)
        elif self.block_type == "*":
            self.mixer = NemotronHAttention(args, layer_idx)
        elif self.block_type == "-":
            self.mixer = NemotronHMLP(args)

    def __call__(self, x, attention_mask=None, cache: Optional[UnifiedCache] = None):
        residual = x
        hidden_states = self.norm(x)

        if self.block_type == "M":
            layer_cache = cache.mamba_cache[self.layer_idx] if cache else None
            hidden_states = self.mixer(hidden_states, cache=layer_cache, attention_mask=attention_mask)
        elif self.block_type == "*":
            layer_cache = cache.attn_cache[self.layer_idx] if cache else None
            hidden_states = self.mixer(hidden_states, mask=attention_mask, cache=layer_cache)
        else:  # mlp
            hidden_states = self.mixer(hidden_states)

        return residual + hidden_states


class NemotronHModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [NemotronHBlock(args, idx) for idx in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache: Optional[UnifiedCache] = None):
        hidden_states = self.embeddings(inputs)
        
        # Create attention mask for attention layers
        attention_mask = None
        if any(layer.block_type == "*" for layer in self.layers):
            attn_cache = cache.attn_cache if cache else None
            attention_mask = create_attention_mask(hidden_states, attn_cache)

        for layer in self.layers:
            mask = attention_mask if layer.block_type == "*" else None
            hidden_states = layer(hidden_states, attention_mask=mask, cache=cache)

        return self.norm_f(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.backbone = NemotronHModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Optional[UnifiedCache] = None):
        out = self.backbone(inputs, cache=cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self, batch_size: int = 1, dtype=mx.float32):
        attn_cache = [KVCache() for _ in range(self.args.num_hidden_layers)]
        mamba_cache = [MambaCache() for _ in range(self.args.num_hidden_layers)]
        return UnifiedCache(attn_cache, mamba_cache)

    def sanitize(self, weights):
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return weights