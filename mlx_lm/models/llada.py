# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "llada"
    # Map LLaDA config names to standard names
    d_model: int = 4096  # hidden_size
    n_layers: int = 32  # num_hidden_layers  
    mlp_hidden_size: int = 12288  # intermediate_size
    n_heads: int = 32  # num_attention_heads
    rms_norm_eps: float = 1e-05
    vocab_size: int = 126464
    n_kv_heads: Optional[int] = None  # num_key_value_heads
    max_sequence_length: int = 4096  # max_position_embeddings
    rope_theta: float = 500000.0
    rope: bool = True
    weight_tying: bool = False  # tie_word_embeddings
    mask_token_id: int = 126336
    # Additional LLaDA-specific parameters from config
    activation_type: str = "silu"
    attention_dropout: float = 0.0
    embedding_dropout: float = 0.0
    residual_dropout: float = 0.0
    include_bias: bool = False
    include_qkv_bias: bool = False
    layer_norm_type: str = "rms"
    mlp_ratio: int = 4
    block_type: str = "llama"

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

    @property
    def hidden_size(self):
        return self.d_model
    
    @property
    def num_hidden_layers(self):
        return self.n_layers
    
    @property
    def intermediate_size(self):
        return self.mlp_hidden_size
    
    @property
    def num_attention_heads(self):
        return self.n_heads
    
    @property
    def num_key_value_heads(self):
        return self.n_kv_heads
    
    @property
    def max_position_embeddings(self):
        return self.max_sequence_length
    
    @property
    def rope_traditional(self):
        return False
    
    @property
    def rope_scaling(self):
        return None
    
    @property
    def tie_word_embeddings(self):
        return self.weight_tying
    
    @property
    def attention_bias(self):
        return self.include_qkv_bias
    
    @property
    def mlp_bias(self):
        return self.include_bias
    
    @property
    def head_dim(self):
        return self.d_model // self.n_heads


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.attn_out = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
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

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

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
        return self.attn_out(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        if hasattr(args, "mlp_bias"):
            mlp_bias = args.mlp_bias
        else:
            mlp_bias = False

        self.ff_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.ff_out = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)

    def __call__(self, x) -> mx.array:
        return self.ff_out(nn.silu(self.ff_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        
        dim = args.hidden_size
        n_heads = args.num_attention_heads
        n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim or args.hidden_size // n_heads
        attention_bias = args.attention_bias

        # Attention components directly in TransformerBlock
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.attn_out = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)
        
        # MLP components directly in TransformerBlock
        hidden_dim = args.intermediate_size
        mlp_bias = args.mlp_bias
        self.ff_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.ff_out = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        
        # Layer norms
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ff_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        
        # Store args for attention computation
        self.args = args
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # Attention
        h_attn = self.attn_norm(x)
        B, L, D = h_attn.shape

        queries, keys, values = self.q_proj(h_attn), self.k_proj(h_attn), self.v_proj(h_attn)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

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
        r = self.attn_out(output)
        h = x + r
        
        # MLP
        h_ff = self.ff_norm(h)
        r = self.ff_out(nn.silu(self.ff_proj(h_ff)) * self.up_proj(h_ff))
        out = h + r
        return out


class LLaDaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
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
        self.model = LLaDaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, mask, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs
        weights = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
        
        # Only remove lm_head if tie_word_embeddings is True
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
            weights.pop("lm_head.scales", None)
            weights.pop("lm_head.biases", None)
        
        return weights

    @property
    def layers(self):
        return self.model.layers