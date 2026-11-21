# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.nemotron_nas.config import AttentionConfig, FFNConfig, ModelArgs

__all__ = [
    "Attention",
    "LinearSubblockReplacement",
    "MLP",
    "TransformerBlock",
]

# Activation function mapping
_ACT2FN = {
    "silu": nn.silu,
    "relu": nn.relu,
    "gelu": nn.gelu,
    "gelu_new": nn.gelu_approx,
    "gelu_fast": nn.gelu_approx,
}


class LinearSubblockReplacement(nn.Module):
    """A simple linear layer used to replace Attention or MLP blocks."""

    def __init__(self, hidden_size: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        # Accepts potential extra args (like mask, cache) but ignores them
        return self.linear(x)


class Attention(nn.Module):
    """Standard GQA Attention mechanism for layers that use it."""

    def __init__(self, args: ModelArgs, attention_config: AttentionConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = n_heads // attention_config.n_heads_in_group

        self.head_dim = head_dim = args.hidden_size // n_heads
        if (self.head_dim * n_heads) != dim:
            raise ValueError(
                f"hidden_size ({dim}) must be divisible by num_attention_heads ({n_heads})"
            )

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        # Initialize RoPE based on global config
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            False,  # Llama uses traditional=False
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


def _find_multiple(n: int, k: int) -> int:
    """Finds the smallest multiple of k greater than or equal to n."""
    if n % k == 0:
        return n
    return n + k - (n % k)


def _ffn_mult_to_intermediate_size(ffn_mult: float, n_embd: int) -> int:
    """Calculates intermediate size based on multiplier, rounding up to multiple of 256."""
    intermediate_size = int(2 * ffn_mult * n_embd / 3)
    return _find_multiple(intermediate_size, 256)


class MLP(nn.Module):
    """Standard Feed-Forward Network for layers that use it."""

    def __init__(self, args: ModelArgs, ffn_config: FFNConfig):
        super().__init__()

        dim = args.hidden_size
        # Calculate intermediate dim based on layer's specific config
        hidden_dim = _ffn_mult_to_intermediate_size(ffn_config.ffn_mult, dim)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=args.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)

        self.act_fn = args.hidden_act
        if self.act_fn not in _ACT2FN:
            raise ValueError(f"Unknown activation function: {args.hidden_act}")

    def __call__(self, x) -> mx.array:
        act_fn = _ACT2FN[self.act_fn]
        return self.down_proj(act_fn(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """A single transformer block, potentially heterogeneous based on config."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        # Get the specific configuration for this layer
        block_config = args.block_configs[layer_idx]
        self.attention_config = block_config.attention
        self.ffn_config = block_config.ffn

        # Conditionally initialize Input LayerNorm (needed unless Attention is no-op)
        if not self.attention_config.no_op:
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        else:
            self.input_layernorm = None

        # Conditionally initialize Attention block
        if self.attention_config.no_op:
            self.self_attn = None
        elif self.attention_config.replace_with_linear:
            self.self_attn = LinearSubblockReplacement(
                args.hidden_size, args.attention_bias
            )
        else:
            # Standard attention for this layer
            self.self_attn = Attention(args, self.attention_config)

        # Conditionally initialize Post-Attention LayerNorm (needed unless FFN is no-op)
        if not self.ffn_config.no_op:
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
        else:
            self.post_attention_layernorm = None

        # Conditionally initialize MLP block
        if self.ffn_config.no_op:
            self.mlp = None
        elif self.ffn_config.replace_with_linear:
            self.mlp = LinearSubblockReplacement(args.hidden_size, args.mlp_bias)
        else:
            # Standard MLP for this layer
            self.mlp = MLP(args, self.ffn_config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        # Attention part (Input Norm -> Attention -> Residual)
        if self.self_attn is not None:
            residual = x
            h = self.input_layernorm(x)
            attn_out = self.self_attn(h, mask=mask, cache=cache)
            x = residual + attn_out

        # MLP part (Post-Attention Norm -> MLP -> Residual)
        if self.mlp is not None:
            residual = x
            h = self.post_attention_layernorm(x)
            mlp_out = self.mlp(h)
            x = residual + mlp_out

        return x
