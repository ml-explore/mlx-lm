# Copyright © 2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "exaone4"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 102400
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    
    # Exaone4-specific parameters
    reorder_qk_norm: bool = True
    sliding_window: Optional[int] = 2048
    sliding_window_pattern: str = "LLLG"

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelArgs":
        return cls(
            model_type=params.get("model_type", "exaone4"),
            hidden_size=params.get("hidden_size", 4096),
            num_hidden_layers=params.get("num_hidden_layers", 32),
            intermediate_size=params.get("intermediate_size", 11008),
            num_attention_heads=params.get("num_attention_heads", 32),
            num_key_value_heads=params.get("num_key_value_heads", 8),
            vocab_size=params.get("vocab_size", 102400),
            max_position_embeddings=params.get("max_position_embeddings", 32768),
            rms_norm_eps=params.get("rms_norm_eps", 1e-5),
            rope_theta=params.get("rope_theta", 10000.0),
            rope_scaling=params.get("rope_scaling", None),
            tie_word_embeddings=params.get("tie_word_embeddings", False),
            attention_bias=params.get("attention_bias", False),
            mlp_bias=params.get("mlp_bias", False),
            reorder_qk_norm=params.get("reorder_qk_norm", True),
            sliding_window=params.get("sliding_window", 2048),
            sliding_window_pattern=params.get("sliding_window_pattern", "LLLG"),
        )


def check_is_sliding(args: ModelArgs, layer_idx: int) -> bool:
    """Check if the layer uses sliding window attention (local attention)."""
    if args.sliding_window is None:
        return False
    
    if isinstance(args.sliding_window_pattern, str):
        # The last layer always uses global attention
        if layer_idx == args.num_hidden_layers - 1:
            return False
        
        pattern = args.sliding_window_pattern
        return pattern[layer_idx % len(pattern)] == "L"
    
    return False


class Exaone4Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim ** -0.5
        
        # Determine attention type
        self.is_sliding = check_is_sliding(args, layer_idx)
        self.sliding_window = args.sliding_window if self.is_sliding else None
        
        # Projection layers
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=args.attention_bias
        )
        
        # QK normalization (only when reorder_qk_norm is True)
        if args.reorder_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        
        # RoPE initialization (only for global attention)
        if not self.is_sliding:
            self.rope = initialize_rope(
                self.head_dim,
                args.rope_theta,
                False,  # traditional
                args.rope_scaling,
                args.max_position_embeddings,
            )
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        
        # Q, K, V projection
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        # Reshape for multi-head attention
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        
        # Apply QK normalization
        if self.args.reorder_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)
        
        # Apply RoPE (only for global attention)
        if not self.is_sliding and hasattr(self, 'rope'):
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)
        else:
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
        
        # Attention computation
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        
        # Reshape output
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Exaone4MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.intermediate_size = args.intermediate_size
        
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=args.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=args.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.mlp_bias
        )
    
    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Exaone4DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.hidden_size = args.hidden_size
        
        self.self_attn = Exaone4Attention(args, layer_idx)
        self.mlp = Exaone4MLP(args)
        
        # Normalization layers (position depends on reorder_qk_norm)
        if args.reorder_qk_norm:
            # Post-norm structure
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.post_feedforward_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
        else:
            # Pre-norm structure
            self.input_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.pre_feedforward_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        
        # Self attention
        if not self.args.reorder_qk_norm:
            x = self.input_layernorm(x)
        
        x = self.self_attn(x, mask, cache)
        
        if self.args.reorder_qk_norm:
            x = self.post_attention_layernorm(x)
        
        x = residual + x
        
        # MLP
        residual = x
        
        if not self.args.reorder_qk_norm:
            x = self.pre_feedforward_layernorm(x)
        
        x = self.mlp(x)
        
        if self.args.reorder_qk_norm:
            x = self.post_feedforward_layernorm(x)
        
        x = residual + x
        
        return x


class Exaone4Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Exaone4DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    
    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        
        if mask is None:
            mask = create_attention_mask(h, cache)
        
        if cache is None:
            cache = [None] * len(self.layers)
        
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)
        
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Exaone4Model(args)
        
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    
    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, mask, cache)
        
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        
        return out
    
    def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Remove unused weights from the model."""
        # Remove unused precomputed rotary freqs
        weights = {
            k: v for k, v in weights.items() 
            if "rotary_emb.inv_freq" not in k
        }
        
        # Remove lm_head if using tied embeddings
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        
        return weights
    
    @property
    def layers(self):
        return self.model.layers
