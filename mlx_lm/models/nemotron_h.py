import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, Mamba2Cache
from .rope_utils import initialize_rope


class UnifiedCache:
    def __init__(self, attn_cache: list[KVCache], mamba_cache: Mamba2Cache):
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
    hybrid_override_pattern: str
    head_dim: Optional[int] = None
    num_heads: Optional[int] = None
    layers_block_type: Optional[List[str]] = None
    expand: Optional[int] = None

    def __post_init__(self):
        if self.expand is None:
            self.expand = self.intermediate_size // self.hidden_size

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.num_heads is None:
            self.num_heads = self.num_attention_heads

        if self.hybrid_override_pattern:
            self.layers_block_type = self._parse_hybrid_pattern(
                self.hybrid_override_pattern
            )
        else:
            self.layers_block_type = ["mamba"] * self.num_hidden_layers

    def _parse_hybrid_pattern(self, pattern: str) -> List[str]:
        layer_types = []
        for char in pattern:
            if char == "M":
                layer_types.append("mamba")
            elif char == "*":
                layer_types.append("attention")
            elif char == "-":
                layer_types.append("mlp")
        if len(layer_types) != self.num_hidden_layers:
            print(
                f"Warning: Pattern length {len(layer_types)} doesn't match num_hidden_layers {self.num_hidden_layers}"
            )
            if len(layer_types) < self.num_hidden_layers:
                layer_types.extend(
                    ["mamba"] * (self.num_hidden_layers - len(layer_types))
                )
            else:
                layer_types = layer_types[: self.num_hidden_layers]
        return layer_types


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
        self.use_conv_bias = args.use_conv_bias
        self.n_groups = args.n_groups
        self.head_dim = args.mamba_head_dim
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
        self, conv_input: mx.array, mamba_cache: Optional[Mamba2Cache] = None
    ) -> mx.array:
        batch_size, seq_len, in_ch = conv_input.shape
        if self.conv_kernel_size > 1:
            if mamba_cache is not None:
                left_ctx = mamba_cache.conv_states[self.layer_idx][:, 1:, :]
                padded_input = mx.concatenate([left_ctx, conv_input], axis=1)
            else:
                padded_input = mx.pad(
                    conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)]
                )
        else:
            padded_input = conv_input

        conv_output = self.conv1d(padded_input)

        conv_output = conv_output[:, :seq_len, :]
        if mamba_cache is not None and self.conv_kernel_size > 1:
            state_slice = padded_input[:, -self.conv_kernel_size :, :]
            mamba_cache.update_conv_state(self.layer_idx, state_slice)

        return nn.silu(conv_output)

    def _ssm(
        self,
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        mamba_cache: Optional[Mamba2Cache] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        dt = nn.softplus(dt + self.dt_bias)
        dt = mx.clip(dt, self.time_step_limit[0], self.time_step_limit[1])

        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        B_rep = mx.repeat(
            B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size),
            self.heads_per_group,
            axis=2,
        )
        C_rep = mx.repeat(
            C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size),
            self.heads_per_group,
            axis=2,
        )

        A = -mx.exp(self.A_log.astype(mx.float32))
        A = A[None, :, None, None]

        if mamba_cache is not None:
            h = mamba_cache.get_ssm_state(self.layer_idx)
        else:
            h = mx.zeros(
                (batch_size, self.num_heads, self.head_dim, self.ssm_state_size)
            )

        outputs = []
        for t in range(seq_len):
            dt_t = dt[:, t, :].reshape(batch_size, self.num_heads, 1, 1)
            dA = mx.exp(dt_t * A)
            dA = mx.broadcast_to(
                dA, (batch_size, self.num_heads, self.head_dim, self.ssm_state_size)
            )
            dB_base = dt_t * B_rep[:, t, :, None, :]
            dB = mx.broadcast_to(
                dB_base,
                (batch_size, self.num_heads, self.head_dim, self.ssm_state_size),
            )
            h = dA * h + dB * hidden_states[:, t, :, :, None]
            y_t = (
                mx.sum(C_rep[:, t, :, None, :] * h, axis=-1)
                + self.D[None, :, None] * hidden_states[:, t]
            )
            outputs.append(y_t)

        if mamba_cache is not None:
            mamba_cache.update_ssm_state(self.layer_idx, h)

        y = mx.stack(outputs, axis=1)
        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        mamba_cache: Optional[Mamba2Cache] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        projected = self.in_proj(hidden_states)

        gate = projected[..., : self.intermediate_size]
        conv_input = projected[
            ..., self.intermediate_size : self.intermediate_size + self.conv_dim
        ]
        dt = projected[..., -self.num_heads :]
        conv_output = self._apply_conv(conv_input, mamba_cache)

        hidden_states2 = conv_output[..., : self.intermediate_size]
        B = conv_output[
            ...,
            self.intermediate_size : self.intermediate_size
            + self.n_groups * self.ssm_state_size,
        ]
        C = conv_output[
            ..., self.intermediate_size + self.n_groups * self.ssm_state_size :
        ]

        y = self._ssm(hidden_states2, B, C, dt, mamba_cache)

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
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = args.max_position_embeddings
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

        self.rope = initialize_rope(
            self.head_dim,
            getattr(args, "rope_theta", 10000.0),
            False,
            getattr(args, "rope_scaling", None),
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        attn_cache: Optional[List[KVCache]] = None,
    ) -> mx.array:
        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(
            B, L, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        layer_kv = None
        layer_offset = 0
        if attn_cache is not None:
            layer_kv = attn_cache[self.layer_idx]
            if hasattr(layer_kv, "offset"):
                layer_offset = layer_kv.offset
        queries = (
            self.rope(queries, offset=layer_offset)
            if layer_offset
            else self.rope(queries)
        )
        keys = self.rope(keys, offset=layer_offset) if layer_offset else self.rope(keys)
        if layer_kv is not None:
            keys, values = layer_kv.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=layer_kv, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class NemotronHMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.up_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=args.mlp_bias
        )
        self.down_proj = nn.Linear(
            args.intermediate_size, args.hidden_size, bias=args.mlp_bias
        )
        if args.mlp_hidden_act == "relu2":
            self.activation = lambda x: nn.relu(x) ** 2
        elif args.mlp_hidden_act == "silu":
            self.activation = nn.silu
        else:
            self.activation = nn.silu

    def __call__(self, x):
        return self.down_proj(self.activation(self.up_proj(x)))


class NemotronHBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.residual_in_fp32 = args.residual_in_fp32
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.block_type = args.layers_block_type[layer_idx]

        if self.block_type == "mamba":
            self.mixer = NemotronHMamba2Mixer(args, layer_idx)
        elif self.block_type == "attention":
            self.mixer = NemotronHAttention(args, layer_idx)
        elif self.block_type == "mlp":
            self.mixer = NemotronHMLP(args)
        else:
            raise ValueError(f"Invalid block type: {self.block_type}")

    def __call__(
        self,
        x,
        attention_mask=None,
        attn_cache: Optional[List[KVCache]] = None,
        mamba_cache: Optional[Mamba2Cache] = None,
    ):
        residual = x
        hidden_states = self.norm(x)
        if self.residual_in_fp32:
            residual = residual.astype(mx.float32)

        if self.block_type == "mamba":
            hidden_states = self.mixer(
                hidden_states, mamba_cache=mamba_cache, attention_mask=attention_mask
            )
        elif self.block_type == "attention":
            hidden_states = self.mixer(
                hidden_states, mask=attention_mask, attn_cache=attn_cache
            )
        elif self.block_type == "mlp":
            hidden_states = self.mixer(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states


class NemotronHModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            NemotronHBlock(args, layer_idx=idx) for idx in range(args.num_hidden_layers)
        ]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def _create_mamba_mask_2d(self, inputs):
        B, L = inputs.shape[:2]
        return mx.ones((B, L), dtype=mx.float32)

    def __call__(
        self,
        inputs,
        causal_mask: Optional[mx.array] = None,
        mamba_mask: Optional[mx.array] = None,
        attn_cache: Optional[List[KVCache]] = None,
        mamba_cache: Optional[Mamba2Cache] = None,
    ):
        hidden_states = self.embeddings(inputs)

        if causal_mask is None:
            causal_mask = create_attention_mask(hidden_states, attn_cache)

        if mamba_mask is None:
            mamba_mask = self._create_mamba_mask_2d(hidden_states)

        for layer_idx, mixer_block in enumerate(self.layers):
            if mixer_block.block_type == "mamba":
                layer_mask = mamba_mask
            elif mixer_block.block_type == "attention":
                layer_mask = causal_mask
            elif mixer_block.block_type == "mlp":
                layer_mask = None
            else:
                raise ValueError(f"Invalid block_type: {mixer_block.block_type}")

            hidden_states = mixer_block(
                hidden_states,
                attention_mask=layer_mask,
                attn_cache=attn_cache,
                mamba_cache=mamba_cache,
            )

        return self.norm_f(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type

        self.backbone = NemotronHModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        causal_mask: Optional[mx.array] = None,
        mamba_mask: Optional[mx.array] = None,
        attn_cache: Optional[List[KVCache]] = None,
        mamba_cache: Optional[Mamba2Cache] = None,
        cache: Optional[Any] = None,
    ):
        if cache is not None and attn_cache is None and mamba_cache is None:
            if isinstance(cache, UnifiedCache):
                attn_cache = cache.attn_cache
                mamba_cache = cache.mamba_cache
            elif isinstance(cache, (tuple, list)):
                if len(cache) == 2 and isinstance(cache[0], list):
                    attn_cache = cache[0]
                    mamba_cache = cache[1]
                else:
                    attn_cache = cache
                    B = inputs.shape[0]
                    intermediate_size = (
                        self.args.mamba_num_heads * self.args.mamba_head_dim
                    )
                    conv_dim = (
                        intermediate_size
                        + 2 * self.args.n_groups * self.args.ssm_state_size
                    )
                    mamba_cache = Mamba2Cache(
                        batch_size=B, conv_dim=conv_dim, args=self.args
                    )
            elif isinstance(cache, Mamba2Cache):
                mamba_cache = cache
        out = self.backbone(
            inputs,
            causal_mask=causal_mask,
            mamba_mask=mamba_mask,
            attn_cache=attn_cache,
            mamba_cache=mamba_cache,
        )
        return self.lm_head(out)

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self, batch_size: int = 1, dtype=mx.float32):
        attn_cache = [KVCache() for _ in range(self.args.num_hidden_layers)]
        intermediate_size = self.args.mamba_num_heads * self.args.mamba_head_dim
        conv_dim = intermediate_size + 2 * self.args.n_groups * self.args.ssm_state_size
        mamba_cache = Mamba2Cache(
            batch_size=batch_size, conv_dim=conv_dim, args=self.args
        )
        return UnifiedCache(attn_cache, mamba_cache)

    def sanitize(self, weights):
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return weights
