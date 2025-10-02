# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import KVCache, MambaCache
from .rope_utils import initialize_rope
from .ssm import ssm_update
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    # Required fields (no defaults)
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_bias: bool
    embedding_multiplier: float
    attention_multiplier: float
    logits_scaling: float
    residual_multiplier: float
    layer_types: List[str]
    rms_norm_eps: float
    rope_theta: float

    # Optional fields (with defaults)
    # MoE parameters (optional for dense mode)
    num_local_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    shared_intermediate_size: Optional[int] = None

    # Mamba parameters (optional for non-hybrid mode)
    mamba_n_heads: Optional[int] = None
    mamba_d_head: Optional[int] = None
    mamba_proj_bias: Optional[bool] = None
    mamba_d_state: Optional[int] = None
    mamba_d_conv: Optional[int] = None
    mamba_n_groups: Optional[int] = None
    mamba_conv_bias: Optional[bool] = None

    # Dense MLP parameters (for non-MoE mode)
    mlp_bias: bool = False

    # Other optional parameters
    position_embedding_type: str = "rope"
    tie_word_embeddings: bool = True
    time_step_limit: Tuple[float, float] = (0.001, 100.0)

    # Mode flags - inferred from num_local_experts
    @property
    def use_moe(self) -> bool:
        return bool(self.num_local_experts)


class GraniteMoeHybridRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array = None) -> mx.array:
        if gate is not None:
            hidden_states = hidden_states * nn.silu(gate)
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

    def _apply_conv(
        self, conv_input: mx.array, cache: Optional[MambaCache] = None
    ) -> mx.array:
        if cache is None or cache[0] is None:
            conv_state = mx.zeros(
                (conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim),
                dtype=conv_input.dtype,
            )
        else:
            conv_state = cache[0]

        padded_input = mx.concatenate([conv_state, conv_input], axis=1)

        if cache is not None:
            cache[0] = padded_input[:, -(self.conv_kernel_size - 1) :]

        conv_output = self.conv1d(padded_input)
        return nn.silu(conv_output)

    def _ssm(
        self,
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        state: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)

        y, state = ssm_update(
            hidden_states,
            self.A_log,
            B,
            C,
            self.D,
            dt,
            self.dt_bias,
            state,
            self.time_step_limit,
            mask,
        )

        return y.reshape(batch_size, seq_len, self.intermediate_size), state

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[MambaCache] = None,
    ) -> mx.array:

        projected = self.in_proj(hidden_states)

        gate, conv_input, dt = mx.split(
            projected,
            [self.intermediate_size, self.intermediate_size + self.conv_dim],
            axis=-1,
        )

        if mask is not None:
            conv_input = mx.where(mask[..., None], conv_input, 0)
        conv_output = self._apply_conv(conv_input, cache)

        hidden_states_ssm, B, C = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )
        state = cache[1] if cache else None
        y, state = self._ssm(hidden_states_ssm, B, C, dt, state, mask)
        if cache:
            cache[1] = state
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
        top_k_idx = mx.argpartition(logits, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
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
        return self.output_linear(nn.silu(gate) * up)


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
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


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


class GraniteMoeHybridModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            GraniteMoeHybridLayer(args, layer_type) for layer_type in args.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.embedding_multiplier = args.embedding_multiplier

        # Handle hybrid vs non-hybrid mode
        self.fa_idx = (
            args.layer_types.index("attention")
            if "attention" in args.layer_types
            else None
        )
        self.ssm_idx = (
            args.layer_types.index("mamba") if "mamba" in args.layer_types else None
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        hidden_states = self.embed_tokens(inputs) * self.embedding_multiplier

        if cache is None:
            cache = [None] * len(self.layers)

        # Create masks based on what layer types exist
        attn_mask = None
        mamba_mask = None

        if self.fa_idx is not None:
            attn_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        if self.ssm_idx is not None:
            mamba_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            mask = attn_mask if layer.layer_type == "attention" else mamba_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GraniteMoeHybridModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.logits_scaling = args.logits_scaling

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache=cache)

        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)

        return out / self.logits_scaling

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.layer_type == "mamba":
                caches.append(MambaCache())
            elif layer.layer_type == "attention":
                caches.append(KVCache())
        return caches

    def sanitize(self, weights):
        # Handle conv1d weights
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)

        # Handle MoE weight transformation to SwitchGLU format (only for MoE models)
        if (
            self.args.use_moe
            and "model.layers.0.block_sparse_moe.input_linear.weight" in weights
        ):
            for l in range(self.args.num_hidden_layers):
                prefix = f"model.layers.{l}.block_sparse_moe"

                input_weight = weights.pop(f"{prefix}.input_linear.weight")
                _, expert_hidden, _ = input_weight.shape

                # Split into gate and up projections (each half of expert_hidden)
                gate_proj = input_weight[:, : expert_hidden // 2, :]
                up_proj = input_weight[:, expert_hidden // 2 :, :]
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_proj
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = up_proj

                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.output_linear.weight"
                )

        # Handle dense MLP weight transformation (for dense models)
        elif (
            not self.args.use_moe
            and "model.layers.0.shared_mlp.input_linear.weight" in weights
        ):
            for l in range(self.args.num_hidden_layers):
                prefix = f"model.layers.{l}.shared_mlp"

                # Transform shared_mlp weights to standard mlp weights
                input_weight = weights.pop(f"{prefix}.input_linear.weight")
                # Split into gate and up projections (each half)
                gate_proj, up_proj = mx.split(input_weight, 2, axis=0)
                weights[f"model.layers.{l}.mlp.gate_proj.weight"] = gate_proj
                weights[f"model.layers.{l}.mlp.up_proj.weight"] = up_proj

                weights[f"model.layers.{l}.mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.output_linear.weight"
                )

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if self.args.use_moe and path.endswith("router.layer"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
