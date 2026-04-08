# Copyright © 2025 Apple Inc.

# DFlash Draft Model - Ported from reference implementation
# Reference: https://github.com/jianc99/dflash

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .cache import KVCache
from .base import scaled_dot_product_attention
from .activations import swiglu


def rotate_half(x: mx.array) -> mx.array:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_single(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply RoPE to a single tensor (for applying to subset of k)."""
    cos = mx.expand_dims(cos, 0)
    sin = mx.expand_dims(sin, 0)

    # Concatenate cos/sin with themselves to match head_dim
    cos = mx.concatenate([cos, cos], axis=-1)
    sin = mx.concatenate([sin, sin], axis=-1)

    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> Tuple[mx.array, mx.array]:
    # cos/sin: [seq_len, head_dim/2] - duplicate for full head_dim
    # Expand dims for broadcasting: [seq_len, head_dim/2] -> [1, 1, seq_len, head_dim/2]
    cos = mx.expand_dims(cos, 0)
    cos = mx.expand_dims(cos, 0)
    sin = mx.expand_dims(sin, 0)
    sin = mx.expand_dims(sin, 0)

    # Concatenate cos/sin with themselves to match head_dim
    cos = mx.concatenate([cos, cos], axis=-1)
    sin = mx.concatenate([sin, sin], axis=-1)

    q_len = q.shape[-2]
    cos_q = cos[..., -q_len:, :]
    sin_q = sin[..., -q_len:, :]

    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


@dataclass
class ModelArgs:
    model_type: str = "qwen3"
    hidden_size: int = 2560
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 9728
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000
    rope_traditional: bool = False
    attention_bias: bool = False
    vocab_size: int = 248320
    tie_word_embeddings: bool = True
    # DFlash-specific
    block_size: int = 4
    num_target_layers: int = 32
    target_layer_ids: Optional[List[int]] = None
    mask_token_id: Optional[int] = None

    @classmethod
    def from_dict(cls, params: Dict[str, Any], weights: Dict[str, mx.array] = None):
        # Extract DFlash config
        dflash_config = params.pop("dflash_config", None)
        if dflash_config:
            params.setdefault("block_size", dflash_config.get("block_size", 16))
            params.setdefault("num_target_layers", dflash_config.get("num_target_layers", 32))
            params.setdefault("target_layer_ids", dflash_config.get("target_layer_ids"))
            params.setdefault("mask_token_id", dflash_config.get("mask_token_id"))

        # Detect actual dimensions from weights if available
        # The HuggingFace configs have wrong hidden_size - detect from actual weights
        if weights is not None and "layers.0.self_attn.q_proj.weight" in weights:
            q_proj_shape = weights["layers.0.self_attn.q_proj.weight"].shape
            # Shape is (out_dim, in_dim)
            # out_dim = num_heads * head_dim (total Q size)
            # in_dim = hidden_size (input hidden size)
            actual_hidden_size = q_proj_shape[1]
            q_out_dim = q_proj_shape[0]
            # Compute head_dim from q_out_dim and num_heads
            if "num_attention_heads" in params:
                head_dim = q_out_dim // params["num_attention_heads"]
                # Override hidden_size to match actual weights
                params["hidden_size"] = actual_hidden_size

        # Filter out unsupported parameters
        filtered_params = {k: v for k, v in params.items() if k in cls.__dataclass_fields__}
        return cls(**filtered_params)


class DFlashAttention(nn.Module):
    """DFlash dual attention layer."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size  # = 2560
        self.num_heads = args.num_attention_heads  # = 32
        self.num_key_value_heads = args.num_key_value_heads  # = 8

        # DFlash uses head_dim = 128, not hidden_size // num_heads
        # Q: 4096 = 32 * 128
        # K/V: 1024 = 8 * 128
        self.head_dim = 128
        self.q_proj_size = self.num_heads * self.head_dim  # = 4096
        self.kv_proj_size = self.num_key_value_heads * self.head_dim  # = 1024

        self.scaling = self.head_dim ** -0.5

        # Projections have different output dimensions
        self.q_proj = nn.Linear(self.hidden_size, self.q_proj_size, bias=args.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_proj_size, bias=args.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_proj_size, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.q_proj_size, self.hidden_size, bias=args.attention_bias)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """
        Args:
            hidden_states: Noise embeddings [B, L, D]
            target_hidden: Compressed context features [B, ctx_len, D]
            position_embeddings: (cos, sin) for RoPE
            mask: Attention mask
            cache: KV cache
        """
        B, L, D = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        # Q from noise only
        queries = self.q_proj(hidden_states)
        queries = queries.reshape(B, L, self.num_heads, -1)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)

        # K/V from concatenated context + noise
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        # Concatenate and reshape
        k = mx.concatenate([k_ctx, k_noise], axis=1)
        v = mx.concatenate([v_ctx, v_noise], axis=1)
        k = k.reshape(B, ctx_len + L, self.num_key_value_heads, -1)
        v = v.reshape(B, ctx_len + L, self.num_key_value_heads, -1)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply RoPE to both Q and full K (context + noise)
        # Reference applies RoPE uniformly to all of K, with Q using last q_len positions
        cos, sin = position_embeddings
        queries, k = apply_rotary_pos_emb(queries, k, cos, sin)

        # Update cache - standard KVCache append
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # Scaled dot-product attention
        output = scaled_dot_product_attention(
            queries, k, v, cache=cache, scale=self.scaling, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # Self-attention with target context
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_hidden, position_embeddings, mask, cache)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class RoPE(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # DFlash uses fixed head_dim of 128, not hidden_size // num_heads
        self.head_dim = 128

    def __call__(self, hidden_states: mx.array, position_ids: mx.array) -> Tuple[mx.array, mx.array]:
        """Compute cos/sin for RoPE.

        Args:
            hidden_states: [B, L, D]
            position_ids: [B, L] - absolute positions

        Returns:
            (cos, sin) for RoPE application - shape (seq_len, rotary_dim)
        """
        seq_len = position_ids.shape[-1]
        position_ids = position_ids.astype(mx.float32)

        # Base frequency computation - rotary_dim is head_dim
        rotary_dim = self.head_dim
        inv_freq = 1.0 / (self.args.rope_theta ** (mx.arange(0, rotary_dim, 2) / rotary_dim))

        # Compute position indices - shape (seq_len,)
        position_ids = position_ids.reshape(-1)

        # Compute rotary embeddings
        t = position_ids[:, None]  # (seq_len, 1)
        freqs = t * inv_freq[None, :]  # (seq_len, rotary_dim/2)

        cos = mx.cos(freqs)  # (seq_len, rotary_dim/2)
        sin = mx.sin(freqs)  # (seq_len, rotary_dim/2)

        return cos, sin


class Model(nn.Module):
    """DFlash draft model for MLX-LM."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # Expose key attributes
        self.block_size = args.block_size
        self.mask_token_id = args.mask_token_id
        self.target_layer_ids = args.target_layer_ids or [0]
        self.model_type = args.model_type

        # Create decoder layers
        self.layers = [
            DFlashDecoderLayer(args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]

        # Feature compression
        num_layers = len(self.target_layer_ids)
        self.fc = nn.Linear(num_layers * args.hidden_size, args.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Final normalization
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # RoPE
        self.rotary_emb = RoPE(args)

    def make_cache(self):
        return [KVCache() for _ in range(self.args.num_hidden_layers)]

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Clean up weights before loading.

        The HuggingFace DFlash models have incorrect config - using actual weight shapes.
        """
        # No reshaping needed - from_dict now detects actual dimensions from weights
        return weights

    def __call__(
        self,
        position_ids: mx.array,
        noise_embedding: mx.array,
        target_hidden: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """
        Args:
            position_ids: [B, L] - absolute positions for noise tokens
            noise_embedding: [B, L, D] - embeddings from mask tokens
            target_hidden: [B, ctx_len, num_layers * D] - raw target features
            cache: KV caches

        Returns:
            Hidden states [B, L, D]
        """
        # Compress target context features
        B, ctx_len, num_layers_times_D = target_hidden.shape
        target_hidden_flat = target_hidden.reshape(B * ctx_len, num_layers_times_D)
        compressed_target_flat = self.hidden_norm(self.fc(target_hidden_flat))
        target_hidden = compressed_target_flat.reshape(B, ctx_len, -1)

        # Compute position embeddings
        position_embeddings = self.rotary_emb(noise_embedding, position_ids)

        # Process through decoder layers
        hidden_states = noise_embedding
        for layer, c in zip(self.layers, cache or [None] * len(self.layers)):
            hidden_states = layer(hidden_states, target_hidden, position_embeddings, mask=None, cache=c)

        return self.norm(hidden_states)
