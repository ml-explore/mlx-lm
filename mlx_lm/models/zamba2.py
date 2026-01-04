# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import CacheList, KVCache, MambaCache
from .rope_utils import initialize_rope
from .ssm import ssm_update


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "zamba2"
    vocab_size: int = 32000
    hidden_size: int = 2560
    num_hidden_layers: int = 54
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    intermediate_size: int = 10240
    hidden_act: str = "gelu"
    attention_hidden_size: Optional[int] = None  # 2 * hidden_size for concat input
    attention_head_dim: Optional[int] = None
    rms_norm_eps: float = 1e-5
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    n_mamba_heads: int = 80
    mamba_headdim: Optional[int] = None
    mamba_ngroups: int = 1
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    use_shared_attention_adapter: bool = False
    use_shared_mlp_adapter: bool = True
    # Adapter rank is used only when the corresponding adapter feature is enabled.
    adapter_rank: int = 128
    num_mem_blocks: int = 2
    hybrid_layer_ids: Optional[List[int]] = None
    layers_block_type: Optional[List[str]] = None  # Alternative to hybrid_layer_ids
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = True
    use_conv_bias: bool = True
    use_mem_rope: bool = True  # Whether to apply RoPE in attention (model-dependent; e.g. 1.2B/7B=True, 2.7B=False)

    def __post_init__(self):
        # Compute hybrid_layer_ids from layers_block_type if provided
        if self.hybrid_layer_ids is None:
            if self.layers_block_type is not None:
                self.hybrid_layer_ids = [i for i, t in enumerate(self.layers_block_type) if t == "hybrid"]
            else:
                # Default for 54-layer model
                self.hybrid_layer_ids = [6, 12, 18, 24, 30, 36, 42, 47, 51]

        # Compute derived dimensions
        if self.attention_hidden_size is None:
            self.attention_hidden_size = 2 * self.hidden_size
        if self.attention_head_dim is None:
            self.attention_head_dim = self.attention_hidden_size // self.num_attention_heads

        # Mamba dimensions
        self.mamba_d_ssm = self.mamba_expand * self.hidden_size
        if self.mamba_headdim is None:
            self.mamba_headdim = self.mamba_d_ssm // self.n_mamba_heads

        # Projection dimensions
        self.groups_time_state_size = self.mamba_ngroups * self.mamba_d_state
        self.conv_dim = self.mamba_d_ssm + 2 * self.groups_time_state_size
        self.projection_size = self.mamba_d_ssm + self.conv_dim + self.n_mamba_heads


class Zamba2RMSNormGated(nn.Module):
    """RMSNorm w/ optional SiLU gating, following Zamba2 pattern."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, norm_before_gate: bool = False):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps
        self.norm_before_gate = norm_before_gate

    def __call__(self, hidden_states: mx.array, gate: Optional[mx.array] = None) -> mx.array:
        if not self.norm_before_gate and gate is not None:
            hidden_states = hidden_states * nn.silu(gate)

        hidden_states = mx.fast.rms_norm(hidden_states, self.weight, self.eps)

        if self.norm_before_gate and gate is not None:
            hidden_states = hidden_states * nn.silu(gate)

        return hidden_states


class Zamba2MambaMixer(nn.Module):
    """Mamba mixer block using ssm_update from ssm.py."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.mamba_d_ssm = args.mamba_d_ssm
        self.ssm_state_size = args.mamba_d_state
        self.conv_kernel_size = args.mamba_d_conv
        self.n_mamba_heads = args.n_mamba_heads
        self.head_dim = args.mamba_headdim
        self.n_groups = args.mamba_ngroups
        self.groups_time_state_size = args.groups_time_state_size
        self.conv_dim = args.conv_dim
        self.projection_size = args.projection_size

        self.time_step_limit = (args.time_step_min, args.time_step_max)

        # Input projection: hidden_size -> projection_size
        self.in_proj = nn.Linear(self.hidden_size, self.projection_size, bias=False)

        # Convolution: conv_dim channels, depthwise
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=args.use_conv_bias,
            padding=0,
        )

        # SSM parameters
        self.A_log = mx.log(mx.arange(1, self.n_mamba_heads + 1, dtype=mx.float32))
        self.D = mx.ones((self.n_mamba_heads,))
        self.dt_bias = mx.ones((self.n_mamba_heads,))

        # Gated normalization
        self.norm = Zamba2RMSNormGated(self.mamba_d_ssm, eps=args.rms_norm_eps)

        # Output projection: mamba_d_ssm -> hidden_size
        self.out_proj = nn.Linear(self.mamba_d_ssm, self.hidden_size, bias=False)

    def _apply_conv(
        self, conv_input: mx.array, cache: Optional[MambaCache] = None
    ) -> mx.array:
        """Apply 1D convolution w/ state caching."""
        if cache is None or cache[0] is None:
            conv_state = mx.zeros(
                (conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim),
                dtype=conv_input.dtype,
            )
        else:
            conv_state = cache[0]

        padded_input = mx.concatenate([conv_state, conv_input], axis=1)

        if cache is not None:
            cache[0] = padded_input[:, -(self.conv_kernel_size - 1):]

        conv_output = self.conv1d(padded_input)
        return nn.silu(conv_output)

    def __call__(
        self, x: mx.array,
        cache: Optional[MambaCache] = None,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape

        projected = self.in_proj(x)

        gate, conv_input, dt = mx.split(
            projected,
            [self.mamba_d_ssm, self.mamba_d_ssm + self.conv_dim],
            axis=-1,
        )

        if mask is not None:
            conv_input = mx.where(mask[..., None], conv_input, 0)

        conv_output = self._apply_conv(conv_input, cache)

        hidden_states, B, C = mx.split(
            conv_output,
            [self.mamba_d_ssm, self.mamba_d_ssm + self.groups_time_state_size],
            axis=-1,
        )

        hidden_states = hidden_states.reshape(batch_size, seq_len, self.n_mamba_heads, self.head_dim)
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)

        # Get SSM state from cache
        state = cache[1] if cache is not None else None

        y, new_state = ssm_update(
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

        if cache is not None:
            cache[1] = new_state

        y = y.reshape(batch_size, seq_len, self.mamba_d_ssm)

        y = self.norm(y, gate)

        return self.out_proj(y)


class _Zamba2Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class Zamba2Attention(nn.Module):
    """Multi-head attention for Zamba2 shared transformer blocks."""

    def __init__(self, args: ModelArgs, num_adapters: int = 0, block_id: int = 0):
        super().__init__()
        self.attention_hidden_size = args.attention_hidden_size
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.attention_head_dim
        # Zamba2 uses (head_dim / 2) for scaling, not head_dim
        self.scale = (self.head_dim / 2) ** -0.5
        self.use_adapters = args.use_shared_attention_adapter and num_adapters > 0
        self.adapter_rank = args.adapter_rank
        self.num_mem_blocks = args.num_mem_blocks
        self.block_id = block_id
        self.use_mem_rope = args.use_mem_rope

        # Q, K, V projections from attention_hidden_size (concatenated input)
        self.q_proj = nn.Linear(self.attention_hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.attention_hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.attention_hidden_size, self.num_kv_heads * self.head_dim, bias=False)

        # Output projection to hidden_size
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Adapters for Q, K, V: one per hybrid layer position
        # HF uses nn.Identity placeholders for blocks that don't own the adapter.
        if self.use_adapters:
            self.linear_q_adapter_list = []
            self.linear_k_adapter_list = []
            self.linear_v_adapter_list = []
            for i in range(num_adapters):
                if i % self.num_mem_blocks == self.block_id:
                    self.linear_q_adapter_list.append(
                        [
                            nn.Linear(self.attention_hidden_size, args.adapter_rank, bias=False),
                            nn.Linear(args.adapter_rank, self.num_heads * self.head_dim, bias=False),
                        ]
                    )
                    self.linear_k_adapter_list.append(
                        [
                            nn.Linear(self.attention_hidden_size, args.adapter_rank, bias=False),
                            nn.Linear(args.adapter_rank, self.num_kv_heads * self.head_dim, bias=False),
                        ]
                    )
                    self.linear_v_adapter_list.append(
                        [
                            nn.Linear(self.attention_hidden_size, args.adapter_rank, bias=False),
                            nn.Linear(args.adapter_rank, self.num_kv_heads * self.head_dim, bias=False),
                        ]
                    )
                else:
                    self.linear_q_adapter_list.append(_Zamba2Identity())
                    self.linear_k_adapter_list.append(_Zamba2Identity())
                    self.linear_v_adapter_list.append(_Zamba2Identity())

        # RoPE
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            False,  # traditional
            None,  # scaling
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        adapter_idx: int = 0,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Apply adapters if available
        if self.use_adapters and adapter_idx < len(self.linear_q_adapter_list):
            q_adapter = self.linear_q_adapter_list[adapter_idx]
            k_adapter = self.linear_k_adapter_list[adapter_idx]
            v_adapter = self.linear_v_adapter_list[adapter_idx]

            if isinstance(q_adapter, list):
                queries = queries + q_adapter[1](q_adapter[0](x))
                keys = keys + k_adapter[1](k_adapter[0](x))
                values = values + v_adapter[1](v_adapter[0](x))
            else:
                queries = queries + q_adapter(x)
                keys = keys + k_adapter(x)
                values = values + v_adapter(x)

        # Reshape for multi-head attention
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE only if use_mem_rope is True (1.2B uses it, 2.7B doesn't)
        if self.use_mem_rope:
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Zamba2MLP(nn.Module):
    """MLP for the shared transformer blocks."""

    def __init__(self, args: ModelArgs, num_adapters: int = 0, block_id: int = 0):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.intermediate_size = args.intermediate_size
        self.use_adapters = args.use_shared_mlp_adapter and num_adapters > 0
        self.adapter_rank = args.adapter_rank
        self.num_mem_blocks = args.num_mem_blocks
        self.block_id = block_id
        self.hidden_act = args.hidden_act

        self.gate_up_proj = nn.Linear(args.hidden_size, 2 * args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

        if self.use_adapters:
            self.gate_up_proj_adapter_list = []
            for i in range(num_adapters):
                if i % self.num_mem_blocks == self.block_id:
                    self.gate_up_proj_adapter_list.append(
                        [
                            nn.Linear(args.hidden_size, args.adapter_rank, bias=False),
                            nn.Linear(args.adapter_rank, 2 * args.intermediate_size, bias=False),
                        ]
                    )
                else:
                    self.gate_up_proj_adapter_list.append(_Zamba2Identity())

    def __call__(self, x: mx.array, adapter_idx: int = 0) -> mx.array:
        gate_up = self.gate_up_proj(x)

        if self.use_adapters and adapter_idx < len(self.gate_up_proj_adapter_list):
            adapter = self.gate_up_proj_adapter_list[adapter_idx]
            if isinstance(adapter, list):
                gate_up = gate_up + adapter[1](adapter[0](x))
            else:
                gate_up = gate_up + adapter(x)

        gate, up = mx.split(gate_up, 2, axis=-1)

        if self.hidden_act == "gelu":
            hidden = nn.gelu(gate) * up
        elif self.hidden_act == "silu":
            hidden = nn.silu(gate) * up
        else:
            raise ValueError(f"Unsupported hidden_act={self.hidden_act!r} for zamba2")

        return self.down_proj(hidden)


class Zamba2SharedTransformerBlock(nn.Module):
    """Shared transformer block (attention + MLP) used in hybrid layers."""

    def __init__(self, args: ModelArgs, num_hybrid_layers: int, block_id: int = 0):
        super().__init__()
        self.block_id = block_id
        # Input layernorm operates on concatenated input (attention_hidden_size)
        self.input_layernorm = nn.RMSNorm(args.attention_hidden_size, eps=args.rms_norm_eps)
        self.self_attn = Zamba2Attention(args, num_adapters=num_hybrid_layers, block_id=block_id)
        # Pre-FF layernorm operates on hidden_size (after attention output)
        self.pre_ff_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.feed_forward = Zamba2MLP(args, num_adapters=num_hybrid_layers, block_id=block_id)

    def __call__(
        self,
        concat_input: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        adapter_idx: int = 0,
    ) -> mx.array:
        # Input layernorm on concatenated input
        h = self.input_layernorm(concat_input)

        # Self attention: (B, L, 5120) -> (B, L, 2560)
        h = self.self_attn(h, mask=mask, cache=cache, adapter_idx=adapter_idx)

        # Pre-FF layernorm then MLP (no residual inside the shared block)
        h = self.pre_ff_layernorm(h)
        h = self.feed_forward(h, adapter_idx=adapter_idx)

        return h


class Zamba2MambaDecoderLayer(nn.Module):
    """Standard Mamba decoder layer w/ layernorm and residual."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mamba = Zamba2MambaMixer(args)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[MambaCache] = None,
        mask: Optional[mx.array] = None,
        transformer_hidden_states: Optional[mx.array] = None,
    ) -> mx.array:
        # Residual is the ORIGINAL input, before adding transformer output
        residual = x

        # Add transformer output if provided (in hybrid layers)
        if transformer_hidden_states is not None:
            x = x + transformer_hidden_states

        h = self.input_layernorm(x)
        h = self.mamba(h, cache=cache, mask=mask)
        return residual + h


class Zamba2HybridLayer(nn.Module):
    """Hybrid layer combining shared transformer and mamba decoder."""

    def __init__(self, args: ModelArgs, hybrid_layer_idx: int):
        super().__init__()
        self.hybrid_layer_idx = hybrid_layer_idx
        self.num_mem_blocks = args.num_mem_blocks

        # Linear projection for transformer output
        self.linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        # Mamba decoder (unique per hybrid layer)
        self.mamba_decoder = Zamba2MambaDecoderLayer(args)

    def __call__(
        self,
        hidden_states: mx.array,
        original_hidden_states: mx.array,
        shared_transformers: List["Zamba2SharedTransformerBlock"],
        attn_mask: Optional[mx.array] = None,
        ssm_mask: Optional[mx.array] = None,
        mamba_cache: Optional[MambaCache] = None,
        attn_cache: Optional[KVCache] = None,
    ) -> mx.array:
        # Concatenate hidden_states w/ ORIGINAL embeddings (not itself!)
        concat_input = mx.concatenate([hidden_states, original_hidden_states], axis=-1)

        # Select the appropriate shared transformer for this hybrid layer
        block_idx = self.hybrid_layer_idx % self.num_mem_blocks
        shared_transformer = shared_transformers[block_idx]

        # Shared transformer path
        transformer_out = shared_transformer(
            concat_input,
            mask=attn_mask,
            cache=attn_cache,
            adapter_idx=self.hybrid_layer_idx,
        )

        # Linear projection of transformer output
        transformer_hidden_states = self.linear(transformer_out)

        # Mamba decoder w/ proper residual:
        # residual = hidden_states, mamba_input = hidden_states + transformer_hidden_states
        # output = residual + mamba(norm(mamba_input))
        output = self.mamba_decoder(
            hidden_states,
            cache=mamba_cache,
            mask=ssm_mask,
            transformer_hidden_states=transformer_hidden_states,
        )

        return output


class Zamba2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.hybrid_layer_ids = set(args.hybrid_layer_ids)

        # Create shared transformer blocks (num_mem_blocks of them)
        # Each block handles adapters for hybrid layers where layer_idx % num_mem_blocks == block_id
        num_hybrid_layers = len(args.hybrid_layer_ids)
        self.shared_transformers = [
            Zamba2SharedTransformerBlock(args, num_hybrid_layers, block_id=k)
            for k in range(args.num_mem_blocks)
        ]

        # Build layers
        self.layers = []
        hybrid_idx = 0
        for i in range(args.num_hidden_layers):
            if i in self.hybrid_layer_ids:
                self.layers.append(Zamba2HybridLayer(args, hybrid_idx))
                hybrid_idx += 1
            else:
                self.layers.append(Zamba2MambaDecoderLayer(args))

        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        # Keep original embeddings for concatenation in hybrid layers
        # Critical for Zamba2's hybrid architecture
        original_hidden_states = h

        if cache is None:
            cache = [None] * len(self.layers)

        # Find first caches for mask creation
        attn_cache = None
        mamba_cache = None
        for i, layer in enumerate(self.layers):
            if cache[i] is None:
                continue
            if isinstance(layer, Zamba2HybridLayer):
                if attn_cache is None:
                    attn_cache = cache[i][1]  # KVCache
                if mamba_cache is None:
                    mamba_cache = cache[i][0]  # MambaCache
            else:
                if mamba_cache is None:
                    mamba_cache = cache[i]

        attn_mask = create_attention_mask(h, attn_cache)
        ssm_mask = create_ssm_mask(h, mamba_cache)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            if isinstance(layer, Zamba2HybridLayer):
                if c is not None:
                    mamba_c, attn_c = c[0], c[1]
                else:
                    mamba_c, attn_c = None, None
                h = layer(
                    h,
                    original_hidden_states=original_hidden_states,
                    shared_transformers=self.shared_transformers,
                    attn_mask=attn_mask,
                    ssm_mask=ssm_mask,
                    mamba_cache=mamba_c,
                    attn_cache=attn_c,
                )
            else:
                h = layer(h, cache=c, mask=ssm_mask)

        return self.final_layernorm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.args = args
        self.model = Zamba2Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def make_cache(self) -> List[Any]:
        """Create cache for all layers."""
        caches = []
        hybrid_layer_ids = set(self.args.hybrid_layer_ids)
        for i in range(self.args.num_hidden_layers):
            if i in hybrid_layer_ids:
                # Hybrid layers need both Mamba and KV cache
                caches.append(CacheList(MambaCache(), KVCache()))
            else:
                # Standard Mamba layers only need Mamba cache
                caches.append(MambaCache())
        return caches

    def sanitize(self, weights: dict) -> dict:
        """Sanitize weights for loading from known Zamba2 MLX exports.

        The Zamba2 weights we encounter come in two layouts:

        1) "layers" layout (preferred):
           - "model.layers.<layer_id>.*" (mamba and hybrid layers)
           - "model.layers.<layer_id>.shared_transformer.*" (duplicated across hybrid layers)

        2) "blocks" layout:
           - "model.blocks.<block_id>.*" (shared transformer blocks)
           - "model.mamba_layers.<layer_id>.*" (mamba sublayers for every layer)
           - "model.linear_layers.<hybrid_idx>.weight" (projection for hybrid layers)

        Some repos (notably the 7B) contain both layouts side-by-side in different
        shard sets. Since `load_model()` merges all `model*.safetensors`, this
        method must pick a single consistent layout.

        Strategy:
        - If any "model.layers." keys exist, keep only the "layers" layout keys.
        - Otherwise, convert the "blocks" layout into this model's parameter names.
        """

        def _transpose_conv_weight(key: str, value: mx.array) -> mx.array:
            if "conv1d.weight" in key and value.ndim == 3 and value.shape[-1] != 1:
                return value.moveaxis(2, 1)
            return value

        def _sanitize_layers_layout(source: dict) -> dict:
            sanitized: dict[str, mx.array] = {}

            hybrid_layer_ids = self.args.hybrid_layer_ids
            hybrid_layer_id_set = set(hybrid_layer_ids)
            num_mem_blocks = self.args.num_mem_blocks

            # Track which block we've already seen weights for
            # Key: block_idx, Value: layer_id of first occurrence
            first_block_sources: dict[int, int] = {}

            # Build mapping: layer_id -> block_idx
            layer_to_block: dict[int, int] = {}
            for hybrid_idx, layer_id in enumerate(hybrid_layer_ids):
                block_idx = hybrid_idx % num_mem_blocks
                layer_to_block[layer_id] = block_idx
                first_block_sources.setdefault(block_idx, layer_id)

            for key, value in source.items():
                new_key = key
                value = _transpose_conv_weight(key, value)

                # Remap per-hybrid-layer shared_transformer weights into tied blocks.
                if ".shared_transformer." in key:
                    parts = key.split(".")
                    try:
                        layer_idx = int(parts[2])
                    except (IndexError, ValueError):
                        layer_idx = None

                    if layer_idx is not None and layer_idx in hybrid_layer_id_set:
                        block_idx = layer_to_block[layer_idx]

                        # Only take weights from the first hybrid layer that uses this block.
                        if layer_idx != first_block_sources[block_idx]:
                            continue

                        new_key = key.replace(
                            f"model.layers.{layer_idx}.shared_transformer.",
                            f"model.shared_transformers.{block_idx}.",
                        )

                sanitized[new_key] = value

            return sanitized

        def _sanitize_blocks_layout(source: dict) -> dict:
            sanitized: dict[str, mx.array] = {}

            hybrid_layer_ids = self.args.hybrid_layer_ids
            hybrid_layer_id_set = set(hybrid_layer_ids)

            for key, value in source.items():
                if key == "model.embed_tokens.weight" or key == "lm_head.weight":
                    sanitized[key] = value
                    continue

                # Mamba layers.
                if key.startswith("model.mamba_layers."):
                    parts = key.split(".")
                    try:
                        layer_idx = int(parts[2])
                    except (IndexError, ValueError):
                        continue

                    remainder = ".".join(parts[3:])
                    if layer_idx in hybrid_layer_id_set:
                        new_key = f"model.layers.{layer_idx}.mamba_decoder.{remainder}"
                    else:
                        new_key = f"model.layers.{layer_idx}.{remainder}"

                    sanitized[new_key] = _transpose_conv_weight(new_key, value)
                    continue

                # Hybrid projection linears.
                if key.startswith("model.linear_layers."):
                    parts = key.split(".")
                    try:
                        hybrid_idx = int(parts[2])
                    except (IndexError, ValueError):
                        continue

                    if hybrid_idx >= len(hybrid_layer_ids):
                        continue

                    layer_id = hybrid_layer_ids[hybrid_idx]
                    remainder = ".".join(parts[3:])
                    sanitized[f"model.layers.{layer_id}.linear.{remainder}"] = value
                    continue

                # Shared transformer blocks.
                if key.startswith("model.blocks."):
                    # model.blocks.<block_id>.<rest>
                    parts = key.split(".")
                    try:
                        block_id = int(parts[2])
                    except (IndexError, ValueError):
                        continue

                    rest = ".".join(parts[3:])

                    # Attention + norms are 1:1.
                    if (
                        rest.startswith("self_attn.") or
                        rest.startswith("input_layernorm.") or
                        rest.startswith("pre_ff_layernorm.")
                    ):
                        sanitized[f"model.shared_transformers.{block_id}.{rest}"] = value
                        continue

                    # Feed-forward projection weights.
                    if rest == "feed_forward.linear_fc1.weight":
                        sanitized[f"model.shared_transformers.{block_id}.feed_forward.gate_up_proj.weight"] = value
                        continue
                    if rest == "feed_forward.linear_fc2.weight":
                        sanitized[f"model.shared_transformers.{block_id}.feed_forward.down_proj.weight"] = value
                        continue

                    # Feed-forward LoRA weights.
                    a_marker = ".feed_forward.linear_fc1_lora_A_list."
                    b_marker = ".feed_forward.linear_fc1_lora_B_list."

                    if a_marker in key:
                        idx_str = key.split(a_marker, 1)[1].split(".", 1)[0]
                        try:
                            adapter_idx = int(idx_str)
                        except ValueError:
                            continue
                        sanitized[
                            f"model.shared_transformers.{block_id}.feed_forward.gate_up_proj_adapter_list.{adapter_idx}.0.weight"] = value
                        continue

                    if b_marker in key:
                        idx_str = key.split(b_marker, 1)[1].split(".", 1)[0]
                        try:
                            adapter_idx = int(idx_str)
                        except ValueError:
                            continue
                        sanitized[
                            f"model.shared_transformers.{block_id}.feed_forward.gate_up_proj_adapter_list.{adapter_idx}.1.weight"
                        ] = value
                        continue

            return sanitized

        has_layers_layout = any(k.startswith("model.layers.") for k in weights)

        if has_layers_layout:
            # Drop incompatible parallel exports (e.g. blocks/mamba_layers/linear_layers).
            pruned = {
                k: v
                for k, v in weights.items()
                if not (
                    k.startswith("model.blocks.")
                    or k.startswith("model.mamba_layers.")
                    or k.startswith("model.linear_layers.")
                )
            }
            sanitized = _sanitize_layers_layout(pruned)
        else:
            sanitized = _sanitize_blocks_layout(weights)

        if self.args.tie_word_embeddings:
            sanitized.pop("lm_head.weight", None)

        return sanitized

    @property
    def layers(self):
        return self.model.layers
