# Copyright © 2023-2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    """Configuration arguments for the LLaDA model."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 10000.0
    mask_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self):
        """Set default token IDs and key-value heads if not provided."""
        self.mask_token_id = self.mask_token_id or 126336
        self.eos_token_id = self.eos_token_id or 126081
        self.num_key_value_heads = self.num_key_value_heads or self.num_attention_heads

    @classmethod
    def from_dict(cls, params):
        """Create ModelArgs from a configuration dictionary.

        Args:
            params (dict): Configuration parameters from config.json.

        Returns:
            ModelArgs: Initialized model arguments.
        """
        mapped_params = {
            "model_type": params.get("model_type"),
            "hidden_size": params.get("d_model"),
            "num_hidden_layers": params.get("n_layers"),
            "intermediate_size": params.get("mlp_hidden_size"),
            "num_attention_heads": params.get("n_heads"),
            "rms_norm_eps": params.get("rms_norm_eps"),
            "vocab_size": params.get("vocab_size"),
            "num_key_value_heads": params.get("n_kv_heads"),
            "rope_theta": params.get("rope_theta", 10000.0),
            "mask_token_id": params.get("mask_token_id"),
            "eos_token_id": params.get("eos_token_id"),
            "attention_bias": params.get("include_qkv_bias", False),
            "mlp_bias": params.get("include_bias", False),
        }
        return cls(
            **{k: v for k, v in mapped_params.items() if k in cls.__annotations__}
        )


class LLaDABlock(nn.Module):
    """Transformer block for the LLaDA model"""

    def __init__(self, args: ModelArgs):
        """Initialize the LLaDA transformer block.

        Args:
            args (ModelArgs): Model configuration arguments.
        """
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim**-0.5
        self.args = args

        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.attn_out = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias
        )

        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ff_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        hidden_dim = args.intermediate_size
        self.ff_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.ff_out = nn.Linear(hidden_dim, dim, bias=args.mlp_bias)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        """Process input through attention and feed-forward layers.

        Args:
            x (mx.array): Input tensor [batch_size, seq_len, hidden_size].
            mask (mx.array, optional): Attention mask.
            cache (Any, optional): Key-value cache (unused in diffusion).

        Returns:
            mx.array: Output tensor [batch_size, seq_len, hidden_size].
        """
        # Attention path
        h = mx.fast.rms_norm(
            x, weight=self.attn_norm.weight, eps=self.args.rms_norm_eps
        )
        B, L, D = h.shape
        queries = self.q_proj(h).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(h).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(h).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = mx.fast.rope(
            queries,
            dims=self.head_dim,
            traditional=False,
            base=self.args.rope_theta,
            scale=1.0,
            offset=0,
        )
        keys = mx.fast.rope(
            keys,
            dims=self.head_dim,
            traditional=False,
            base=self.args.rope_theta,
            scale=1.0,
            offset=0,
        )

        attn_output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        h = x + self.attn_out(attn_output)

        # Feed-forward path
        out = mx.fast.rms_norm(
            h, weight=self.ff_norm.weight, eps=self.args.rms_norm_eps
        )
        ff = self.ff_proj(out)
        up = self.up_proj(out)
        out = h + self.ff_out(nn.silu(ff) * up)
        return out


class LLaDAModel(nn.Module):
    """Core LLaDA model for diffusion-based text generation."""

    def __init__(self, args: ModelArgs):
        """Initialize the LLaDA core model.

        Args:
            args (ModelArgs): Model configuration arguments.
        """
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [LLaDABlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if mask is None and cache is None:  # Only create mask for diffusion (no cache)
            mask = create_attention_mask(h, cache)

        for layer in self.layers:
            h = layer(h, mask, cache)
        h = self.norm(h)
        return self.lm_head(h)


class Model(nn.Module):
    """Top-level LLaDA model for diffusion-based text generation."""

    def __init__(self, args: ModelArgs):
        """Initialize the LLaDA model wrapper.

        Args:
            args (ModelArgs): Model configuration arguments.
        """
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LLaDAModel(args)
        self.mask_token_id = args.mask_token_id
        assert args.vocab_size > 0, "Vocabulary size must be positive."

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        return self.model(inputs, mask, cache)

    def sanitize(self, weights):
        """Remap PyTorch weights from transformers to MLX format."""
        sanitized_weights = {}
        for key, value in weights.items():
            new_key = key
            if "model.transformer.blocks" in key:
                new_key = key.replace("model.transformer.blocks", "model.layers")
            elif "model.transformer.wte.weight" in key:
                new_key = "model.embed_tokens.weight"
            elif "model.transformer.ln_f.weight" in key:
                new_key = "model.norm.weight"
            elif "model.transformer.ff_out.weight" in key:
                new_key = "model.lm_head.weight"
            new_key = new_key.replace("attn_norm.weight", "attn_norm.weight")
            new_key = new_key.replace("ff_norm.weight", "ff_norm.weight")
            new_key = new_key.replace("q_proj.weight", "q_proj.weight")
            new_key = new_key.replace("k_proj.weight", "k_proj.weight")
            new_key = new_key.replace("v_proj.weight", "v_proj.weight")
            new_key = new_key.replace("attn_out.weight", "attn_out.weight")
            new_key = new_key.replace("ff_proj.weight", "ff_proj.weight")
            new_key = new_key.replace("up_proj.weight", "up_proj.weight")
            new_key = new_key.replace("ff_out.weight", "ff_out.weight")
            sanitized_weights[new_key] = value
        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers
