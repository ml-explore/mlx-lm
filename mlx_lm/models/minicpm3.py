# Copyright © 2023-2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    dim_model_base: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    q_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    kv_lora_rank: int
    scale_depth: float
    scale_emb: float
    max_position_embeddings: int
    attention_bias: bool = False
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[str, float]]] = None
    tie_word_embeddings: bool = False


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.qk_rope_head_dim = self.args.qk_rope_head_dim
        self.qk_nope_head_dim = self.args.qk_nope_head_dim
        self.attention_bias = self.args.attention_bias
        self.kv_lora_rank = self.args.kv_lora_rank
        self.num_heads = self.args.num_attention_heads
        self.q_lora_rank = self.args.q_lora_rank
        self.hidden_size = self.args.hidden_size

        self.v_head_dim = self.hidden_size // self.args.num_attention_heads
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)

        self.q_a_proj = nn.Linear(
            self.hidden_size, self.q_lora_rank, bias=self.attention_bias
        )
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank)

        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=self.attention_bias,
        )

        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank)

        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=self.attention_bias,
        )

        self.rope = initialize_rope(
            args.qk_rope_head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Dict[str, mx.array]] = None
    ):
        B, L, _ = x.shape

        # Project query
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, L, self.num_heads, self.q_head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))  # [B, num_heads, L, q_head_dim]
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        # Project key and value
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim)
        k_pe = mx.transpose(k_pe, (0, 2, 1, 3))  # [B, 1, L, qk_rope_head_dim]
        
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.reshape(B, L, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        kv = mx.transpose(kv, (0, 2, 1, 3))  # [B, num_heads, L, qk_nope_head_dim + v_head_dim]
        
        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        # Apply RoPE to the query and key parts that need position embedding
        if cache is not None:
            q_pe = self.rope(q_pe, offset=cache.offset)
            k_pe = self.rope(k_pe, offset=cache.offset)
        else:
            q_pe = self.rope(q_pe)
            k_pe = self.rope(k_pe)

        # Create the full query and key tensors by combining the parts
        # Broadcast k_pe to all heads
        k_pe_broadcasted = mx.broadcast_to(k_pe, (B, self.num_heads, L, self.qk_rope_head_dim))
        
        # Use concatenate for queries
        queries = mx.concatenate([q_nope, q_pe], axis=-1)
        
        # Use concatenate for keys
        keys = mx.concatenate([k_nope, k_pe_broadcasted], axis=-1)

        # Update cache if needed
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # Perform attention
        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.softmax_scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_hidden_layers = args.num_hidden_layers

        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        self.scale_depth = args.scale_depth
        self.num_hidden_layers = args.num_hidden_layers

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r * (self.scale_depth / mx.sqrt(self.num_hidden_layers))
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r * (self.scale_depth / mx.sqrt(self.num_hidden_layers))
        return out


class MiniCPM3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.embed_tokens(inputs) * self.args.scale_emb

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniCPM3Model(args)

        if not self.args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)

        if not self.args.tie_word_embeddings:
            out = self.lm_head(out / (self.args.hidden_size / self.args.dim_model_base))
        else:
            out = out @ self.model.embed_tokens.weight.T

        return out

    def sanitize(self, weights):
        if "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        
        # Add missing scaling_factor and inv_freq for each RoPE layer
        for i in range(len(self.model.layers)):
            rope = self.model.layers[i].self_attn.rope
            
            # Check and add scaling_factor if missing
            scaling_factor_key = f"model.layers.{i}.self_attn.rope.scaling_factor"
            if scaling_factor_key not in weights:
                scale = rope.max_position_embeddings / rope.original_max_position_embeddings
                scaling_factor = mx.sqrt(
                    1 + mx.log(scale) / mx.log(rope.original_max_position_embeddings)
                )
                weights[scaling_factor_key] = scaling_factor
            
            # Check and add inv_freq if missing
            inv_freq_key = f"model.layers.{i}.self_attn.rope.inv_freq"
            if inv_freq_key not in weights:
                dims = rope.dim
                base = rope.base
                inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float16) / dims))
                weights[inv_freq_key] = inv_freq
        
        return weights

    @property
    def layers(self):
        return self.model.layers
