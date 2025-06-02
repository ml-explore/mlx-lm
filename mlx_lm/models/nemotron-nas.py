# Copyright © 2025 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass(frozen=True)
class AttentionConfig:
    no_op: bool = False
    replace_with_linear: bool = False
    sparsify: Optional[list[str]] = None
    n_heads_in_group: Optional[int] = None  # GQA group size
    window_length: Optional[int] = None  # Not directly used here, placeholder
    num_sink_tokens: Optional[int] = None  # Not directly used here, placeholder
    use_prefill_window_in_sink_attention: bool = (
        False  # Not directly used here, placeholder
    )
    unshifted_sink: bool = False  # Not directly used here, placeholder

    def __post_init__(self):
        # Ensure consistency: If no-op or linear, other attn params are irrelevant
        if self.no_op or self.replace_with_linear:
            # Use object.__setattr__ because the dataclass is frozen
            object.__setattr__(self, "n_heads_in_group", None)
            object.__setattr__(self, "window_length", None)
            object.__setattr__(self, "num_sink_tokens", None)
        # If it's a standard attention block, n_heads_in_group must be provided
        elif not self.no_op:
            if self.n_heads_in_group is None:
                raise ValueError(
                    "n_heads_in_group must be specified for active attention blocks"
                )
            if self.n_heads_in_group <= 0:
                raise ValueError(
                    f"n_heads_in_group must be positive, got {self.n_heads_in_group}"
                )


@dataclass(frozen=True)
class FFNConfig:
    no_op: bool = False
    replace_with_linear: bool = False
    sparsify: Optional[list[str]] = None
    ffn_mult: Optional[float] = None

    def __post_init__(self):
        # Ensure consistency: If no-op or linear, ffn_mult is irrelevant
        if self.no_op or self.replace_with_linear:
            object.__setattr__(self, "ffn_mult", None)
        # If it's a standard FFN block, ffn_mult must be provided
        elif not self.no_op:
            if self.ffn_mult is None:
                raise ValueError("ffn_mult must be specified for active FFN blocks")
            # Round to prevent potential floating point inconsistencies if needed
            object.__setattr__(self, "ffn_mult", round(self.ffn_mult, 6))


@dataclass(frozen=True)
class BlockConfig:
    attention: AttentionConfig
    ffn: FFNConfig

    @classmethod
    def from_dict(cls, data: dict):
        # Helper to create BlockConfig from a dictionary (e.g., loaded from JSON)
        attn_conf = AttentionConfig(**data.get("attention", {}))
        ffn_conf = FFNConfig(**data.get("ffn", {}))
        return cls(attention=attn_conf, ffn=ffn_conf)


def _find_multiple(n: int, k: int) -> int:
    """Finds the smallest multiple of k greater than or equal to n."""
    if n % k == 0:
        return n
    return n + k - (n % k)


def _ffn_mult_to_intermediate_size(ffn_mult: float, n_embd: int) -> int:
    """Calculates intermediate size based on multiplier, rounding up to multiple of 256."""
    intermediate_size = int(2 * ffn_mult * n_embd / 3)
    return _find_multiple(intermediate_size, 256)


# Activation function mapping
_ACT2FN = {
    "silu": nn.silu,
    "relu": nn.relu,
    "gelu": nn.gelu,
    "gelu_new": nn.gelu_approx,
    "gelu_fast": nn.gelu_approx,
}


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "nemotron-nas"
    hidden_size: int = 8192
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    rms_norm_eps: float = 1e-5
    vocab_size: int = 128256
    block_configs: list = field(default_factory=list)  # List of BlockConfig or dicts
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 500000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = False

    def __post_init__(self):
        # Automatically parse block_configs if they are loaded as dicts
        if self.block_configs and isinstance(self.block_configs[0], dict):
            self.block_configs = [
                BlockConfig.from_dict(conf) for conf in self.block_configs
            ]

        if len(self.block_configs) != self.num_hidden_layers:
            raise ValueError(
                f"Number of block_configs ({len(self.block_configs)}) must match "
                f"num_hidden_layers ({self.num_hidden_layers})"
            )

        # Basic validation for RoPE scaling if provided
        if self.rope_scaling:
            if "factor" not in self.rope_scaling:
                raise ValueError("rope_scaling must contain 'factor'")
            rope_type = self.rope_scaling.get("rope_type")
            if rope_type is None:
                raise ValueError("rope_scaling must contain 'rope_type'")

        # Validate individual block configs (post_init in dataclasses already does some)
        for i, block_conf in enumerate(self.block_configs):
            attn_conf = block_conf.attention
            if not attn_conf.no_op and not attn_conf.replace_with_linear:
                if self.num_attention_heads % attn_conf.n_heads_in_group != 0:
                    raise ValueError(
                        f"Layer {i}: num_attention_heads ({self.num_attention_heads}) "
                        f"must be divisible by n_heads_in_group ({attn_conf.n_heads_in_group})"
                    )


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

        try:
            self.act_fn = _ACT2FN[args.hidden_act]
        except KeyError:
            raise ValueError(f"Unknown activation function: {args.hidden_act}")

    def __call__(self, x) -> mx.array:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LinearSubblockReplacement(nn.Module):
    """A simple linear layer used to replace Attention or MLP blocks."""

    def __init__(self, hidden_size: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        # Accepts potential extra args (like mask, cache) but ignores them
        return self.linear(x)


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


class NemotronNASModel(nn.Module):
    """The core Nemotron-NAS style transformer model."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            h = layer(h, mask, cache=cache[i])

        return self.norm(h)


class Model(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = NemotronNASModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        else:
            self.lm_head = None

    def __call__(
        self,
        inputs: mx.array,
        mask=None,
        cache=None,
    ):
        out = self.model(inputs, mask=mask, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers
