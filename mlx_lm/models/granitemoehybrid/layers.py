# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.activations import swiglu
from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.layers.ssm import ssm_update
from mlx_lm.layers.switch import SwitchGLU
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.granitemoehybrid.config import ModelArgs

__all__ = [
    "GraniteMoeHybridAttention",
    "GraniteMoeHybridLayer",
    "GraniteMoeHybridMLP",
    "GraniteMoeHybridMamba2Mixer",
    "GraniteMoeHybridMoE",
    "GraniteMoeHybridRMSNormGated",
    "GraniteMoeHybridSharedMLP",
    "GraniteMoeHybridTopKGating",
]


class GraniteMoeHybridRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array = None) -> mx.array:
        if gate is not None:
            hidden_states = swiglu(gate, hidden_states)
        return mx.fast.rms_norm(hidden_states, self.weight, self.eps)


class GraniteMoeHybridMamba2Mixer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.mamba_n_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.mamba_d_state
        self.conv_kernel_size = args.mamba_d_conv
        self.intermediate_size = args.mamba_n_heads * args.mamba_d_head
        self.n_groups = args.mamba_n_groups
        self.head_dim = args.mamba_d_head
        self.time_step_limit = args.time_step_limit
        self.heads_per_group = self.num_heads // self.n_groups

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.mamba_d_conv,
            padding=0,
            groups=self.conv_dim,
            bias=args.mamba_conv_bias,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size, projection_size, bias=args.mamba_proj_bias
        )

        self.dt_bias = mx.ones(self.num_heads)
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)

        self.norm = GraniteMoeHybridRMSNormGated(
            self.intermediate_size, eps=args.rms_norm_eps
        )
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.mamba_proj_bias
        )

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
                    (conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim),
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
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        if cache:
            state = cache[1]
            lengths = cache.lengths
        else:
            state, lengths = None, None

        y, state = ssm_update(
            hidden_states,
            self.A_log,
            B,
            C,
            self.D.astype(hidden_states.dtype),
            dt,
            self.dt_bias,
            state,
            self.time_step_limit,
            mask,
            lengths,
        )
        if cache:
            cache[1] = state

        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array],
        cache: Optional[ArraysCache] = None,
    ) -> mx.array:

        projected = self.in_proj(hidden_states)

        gate, conv_input, dt = mx.split(
            projected,
            [self.intermediate_size, self.intermediate_size + self.conv_dim],
            axis=-1,
        )
        conv_output = self._conv(conv_input, cache, mask)
        hidden_states_ssm, B, C = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )
        y = self._ssm(hidden_states_ssm, B, C, dt, cache, mask)
        if cache:
            cache.advance(y.shape[1])
        y = self.norm(y, gate)
        return self.out_proj(y)


class GraniteMoeHybridAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads

        self.scale = args.attention_multiplier
        attention_bias = args.attention_bias
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        # Check if RoPE should be used based on position_embedding_type
        # If position_embedding_type is "nope", don't use RoPE
        use_rope = args.position_embedding_type != "nope"
        if use_rope:
            self.rope = initialize_rope(
                self.head_dim,
                args.rope_theta,
                False,
                None,  # rope_scaling
                args.max_position_embeddings,
            )
        else:
            self.rope = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE only if enabled
        if self.rope is not None:
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class GraniteMoeHybridTopKGating(nn.Module):
    def __init__(self, input_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.top_k = top_k
        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def __call__(self, hidden_states: mx.array):
        logits = self.layer(hidden_states)
        top_k_idx = mx.stop_gradient(
            mx.argpartition(logits, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        )
        top_k_logits = mx.take_along_axis(logits, top_k_idx, axis=-1)
        top_k_gates = mx.softmax(top_k_logits, precise=True, axis=-1)
        return top_k_idx, top_k_gates


class GraniteMoeHybridMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.input_size = args.hidden_size
        self.hidden_size = args.intermediate_size
        self.switch_mlp = SwitchGLU(
            self.input_size, self.hidden_size, args.num_local_experts
        )
        self.router = GraniteMoeHybridTopKGating(
            input_size=self.input_size,
            num_experts=args.num_local_experts,
            top_k=args.num_experts_per_tok,
        )

    def __call__(self, x: mx.array) -> mx.array:
        token_ids, gates = self.router(x)
        y = self.switch_mlp(x, token_ids)
        return (y * gates[..., None]).sum(axis=-2)


class GraniteMoeHybridSharedMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.input_linear = nn.Linear(
            args.hidden_size, args.shared_intermediate_size * 2, bias=False
        )
        self.output_linear = nn.Linear(
            args.shared_intermediate_size, args.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.input_linear(x), 2, axis=-1)
        return self.output_linear(swiglu(gate, up))


class GraniteMoeHybridMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        mlp_bias = args.mlp_bias

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class GraniteMoeHybridLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.residual_multiplier = args.residual_multiplier
        self.use_moe = args.use_moe

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        if layer_type == "mamba":
            self.mamba = GraniteMoeHybridMamba2Mixer(args)
        elif layer_type == "attention":
            self.self_attn = GraniteMoeHybridAttention(args)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        # MoE or dense MLP after attention/mamba
        if self.use_moe:
            self.shared_mlp = GraniteMoeHybridSharedMLP(args)
            self.block_sparse_moe = GraniteMoeHybridMoE(args)
        else:
            # Dense MLP mode
            self.mlp = GraniteMoeHybridMLP(args)

        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # First block: either Mamba or Attention
        residual = x
        hidden_states = self.input_layernorm(x)

        if self.layer_type == "mamba":
            hidden_states = self.mamba(hidden_states, mask=mask, cache=cache)
        else:
            hidden_states = self.self_attn(hidden_states, mask=mask, cache=cache)

        hidden_states = residual + hidden_states * self.residual_multiplier

        # Second block: MoE + shared_mlp OR dense MLP
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)

        if self.use_moe:
            moe_out = self.block_sparse_moe(normed)
            shared_out = self.shared_mlp(normed)
            mlp_out = moe_out + shared_out
        else:
            mlp_out = self.mlp(normed)

        hidden_states = residual + mlp_out * self.residual_multiplier

        return hidden_states
