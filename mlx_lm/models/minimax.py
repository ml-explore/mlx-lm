# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    num_experts_per_tok: int
    num_local_experts: int
    shared_intermediate_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    vocab_size: int
    block_size: int = 256
    tie_word_embeddings: bool = False
    shared_moe_mode: str = "sigmoid"
    full_attn_alpha_factor: float = 3.5565588200778455
    full_attn_beta_factor: float = 1.0
    linear_attn_alpha_factor: float = 3.5565588200778455
    linear_attn_beta_factor: float = 1.0
    mlp_alpha_factor: float = 3.5565588200778455
    mlp_beta_factor: float = 1.0
    layer_types: List[str] = None
    head_dim: Optional[int] = None
    use_qk_norm: bool = True


class MiniMaxAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()


        self.hidden_dim = hidden_size = args.hidden_size

        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = head_dim = (
            hidden_size // args.num_attention_heads
            if args.head_dim is None
            else args.head_dim
        )
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim * self.num_attention_heads, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(head_dim * self.num_key_value_heads, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(head_dim, traditional=False, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)


        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
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



class MiniMaxSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok

        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.intermediate_size, args.num_local_experts
        )
        self.e_score_correction_bias = mx.zeros((args.num_local_experts,))

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x.astype(mx.float32))

        scores = mx.sigmoid(gates)
        orig_scores = scores
        scores = scores + self.e_score_correction_bias

        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(orig_scores, inds, axis=-1)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y


class MiniMaxDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.mlp_alpha_factor = args.mlp_alpha_factor
        self.mlp_beta_factor = args.mlp_beta_factor

        
        self.self_attn = MiniMaxAttention(args)
        self.attn_alpha_factor = args.full_attn_alpha_factor
        self.attn_beta_factor = args.full_attn_beta_factor

        self.block_sparse_moe = MiniMaxSparseMoeBlock(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = (
            self.input_layernorm(x) * self.attn_alpha_factor
            + self.self_attn(x, mask, cache) * self.attn_beta_factor
        )
        r = (
            self.block_sparse_moe(self.post_attention_layernorm(x))
            * self.mlp_alpha_factor
            + r * self.mlp_beta_factor
        )
        return r


class MiniMaxModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = [
            MiniMaxDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]

        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniMaxModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs=inputs, mask=mask, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        # Dequantize FP8 weights first before any other processing
        keys_to_remove = []
        for key in list(weights.keys()):
            if key.endswith('.weight_scale_inv'):
                # Get the corresponding weight key
                weight_key = key.replace('.weight_scale_inv', '.weight')
                if weight_key in weights:
                    scale_inv = weights[key]
                    weight = weights[weight_key]
                    
                    # Handle block-wise quantization
                    if scale_inv.ndim == 2 and weight.ndim == 2:
                        scale_h, scale_w = scale_inv.shape
                        weight_h, weight_w = weight.shape
                        
                        # Calculate block size
                        block_h = weight_h // scale_h
                        block_w = weight_w // scale_w
                        
                        # Expand scale_inv to match weight dimensions using reshape + tile
                        # Method 1: Using reshape and tile (most efficient)
                        expanded_scale = scale_inv[:, None, :, None]  # [scale_h, 1, scale_w, 1]
                        expanded_scale = mx.tile(expanded_scale, (1, block_h, 1, block_w))  # [scale_h, block_h, scale_w, block_w]
                        expanded_scale = expanded_scale.reshape(weight_h, weight_w)
                        
                        weights[weight_key] = weight * expanded_scale
                        
                    elif scale_inv.shape == weight.shape:
                        # Element-wise multiplication
                        weights[weight_key] = weight * scale_inv
                        
                    elif scale_inv.ndim == 1 and weight.ndim == 2:
                        # Per-channel scaling
                        if scale_inv.shape[0] == weight.shape[0]:
                            weights[weight_key] = weight * scale_inv[:, None]
                        elif scale_inv.shape[0] == weight.shape[1]:
                            weights[weight_key] = weight * scale_inv[None, :]
                        else:
                            raise ValueError(f"Incompatible shapes: weight {weight.shape}, scale {scale_inv.shape}")
                    else:
                        raise ValueError(f"Unsupported quantization format: weight {weight.shape}, scale {scale_inv.shape}")
                    
                    keys_to_remove.append(key)
        
        # Remove the scale_inv keys after dequantization
        for key in keys_to_remove:
            weights.pop(key)

        # Handle MoE expert weights restructuring
        if "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in weights:
            return weights
        
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
            for orig_name, new_name in mapping.items():
                if f"{prefix}.block_sparse_moe.experts.0.{orig_name}.weight" in weights:
                    to_join = [
                        weights.pop(
                            f"{prefix}.block_sparse_moe.experts.{e}.{orig_name}.weight"
                        )
                        for e in range(self.args.num_local_experts)
                    ]
                    weights[
                        f"{prefix}.block_sparse_moe.switch_mlp.{new_name}.weight"
                    ] = mx.stack(to_join)
         
        return weights

    @property
    def layers(self):
        return self.model.layers