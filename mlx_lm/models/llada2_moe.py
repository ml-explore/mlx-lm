"""
LLaDA2 MoE (Large Language Diffusion with mAsking - Mixture of Experts) model.

This module contains the model architecture only. For generation, use
mlx_lm.llada2_generate.generate() or mlx_lm.llada2_generate.stream_generate().

Reference: https://huggingface.co/inclusionAI/LLaDA2.0-mini
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "llada2_moe"
    vocab_size: int = 30592
    hidden_size: int = 1024
    intermediate_size: Optional[int] = None
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 0
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 16384
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rope_traditional: bool = False
    partial_rotary_factor: float = 0.5
    use_qkv_bias: bool = False
    use_qk_norm: bool = True
    use_bias: bool = True
    tie_word_embeddings: bool = False
    # MoE configuration
    num_experts: Optional[int] = 16
    num_experts_per_tok: int = 2
    num_shared_experts: int = 0
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    moe_intermediate_size: Optional[int] = None
    first_k_dense_replace: int = 0

    def __post_init__(self):
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        # Partial rotary: only apply RoPE to a portion of head_dim
        self.rotary_dim = int(self.head_dim * args.partial_rotary_factor)

        # Combined QKV projection
        self.query_key_value = nn.Linear(
            self.hidden_size,
            (self.n_heads + 2 * self.n_kv_heads) * self.head_dim,
            bias=args.use_qkv_bias,
        )
        self.dense = nn.Linear(
            self.n_heads * self.head_dim, self.hidden_size, bias=args.use_bias
        )

        # Optional QK normalization
        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        # Initialize RoPE for the rotary portion
        self.rope = initialize_rope(
            self.rotary_dim,
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
        B, L, _ = x.shape

        # Project to Q, K, V
        qkv = self.query_key_value(x)

        # Split into Q, K, V
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        queries, keys, values = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)

        # Reshape for attention
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # Apply QK normalization if enabled
        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        # Apply partial rotary embeddings
        if self.rotary_dim > 0:
            q_rot, q_pass = (
                queries[..., : self.rotary_dim],
                queries[..., self.rotary_dim :],
            )
            k_rot, k_pass = keys[..., : self.rotary_dim], keys[..., self.rotary_dim :]

            if cache is not None:
                q_rot = self.rope(q_rot, offset=cache.offset)
                k_rot = self.rope(k_rot, offset=cache.offset)
            else:
                q_rot = self.rope(q_rot)
                k_rot = self.rope(k_rot)

            queries = mx.concatenate([q_rot, q_pass], axis=-1)
            keys = mx.concatenate([k_rot, k_pass], axis=-1)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else args.intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_experts
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor

        self.weight = mx.zeros((self.num_experts, args.hidden_size))
        self.expert_bias = mx.zeros((self.num_experts,))

    def __call__(self, x: mx.array):
        # Compute gate logits with expert bias
        gates = x @ self.weight.T + self.expert_bias
        scores = mx.softmax(gates, axis=-1, precise=True)

        # Group-limited top-k routing
        if self.n_group > 1 and self.topk_group < self.n_group:
            bsz, seq_len = x.shape[:2]
            # Reshape to group experts: [bsz, seq_len, n_group, experts_per_group]
            scores_grouped = scores.reshape(
                bsz, seq_len, self.n_group, self.num_experts // self.n_group
            )
            # Get max score per group
            group_scores = scores_grouped.max(axis=-1, keepdims=True)

            # Select top groups (zero out non-selected groups)
            k = self.n_group - self.topk_group
            group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
            scores_grouped = mx.put_along_axis(
                scores_grouped,
                mx.broadcast_to(
                    group_idx, (*group_idx.shape[:-1], scores_grouped.shape[-1])
                ),
                mx.array(0.0, scores_grouped.dtype),
                axis=-2,
            )
            scores = scores_grouped.reshape(bsz, seq_len, -1)

        # Select top-k experts
        k = self.top_k
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(scores, inds, axis=-1)

        # Apply routing scaling factor
        scores = scores * self.routed_scaling_factor

        return inds, scores


class SparseMoEBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_experts_per_tok = args.num_experts_per_tok

        # Use efficient SwitchGLU for expert computation
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
        )

        self.gate = MoEGate(args)

        # Optional shared experts
        if args.num_shared_experts and args.num_shared_experts > 0:
            shared_intermediate_size = args.moe_intermediate_size * args.num_shared_experts
            self.shared_experts = MLP(args, intermediate_size=shared_intermediate_size)
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)

        # Compute expert outputs using efficient gather-mm
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        # Add shared expert contribution
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)

        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx

        # Note: named "attention" to match HF weight naming
        self.attention = Attention(args)

        # Use MoE for layers >= first_k_dense_replace, dense MLP otherwise
        if (
            args.num_experts is not None
            and args.num_experts > 0
            and layer_idx >= args.first_k_dense_replace
        ):
            self.mlp = SparseMoEBlock(args)
            self.is_moe = True
        else:
            self.mlp = MLP(args)
            self.is_moe = False

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
        # Self attention with residual
        r = self.attention(self.input_layernorm(x), mask, cache)
        h = x + r

        # MLP/MoE with residual
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r

        return out


class LLaDA2MoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx) for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        # Use provided mask or create default causal mask
        if mask is None:
            mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LLaDA2MoeModel(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, mask=mask)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs
        weights = {
            k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k
        }

        # Remap word_embeddings to embed_tokens
        for param_type in ["weight", "scales", "biases"]:
            old_key = f"model.word_embeddings.{param_type}"
            new_key = f"model.embed_tokens.{param_type}"
            if old_key in weights:
                weights[new_key] = weights.pop(old_key)

        # Handle expert weight consolidation for MoE layers
        for l in range(self.args.num_hidden_layers):
            # Skip dense layers
            if l < self.args.first_k_dense_replace:
                continue

            prefix = f"model.layers.{l}"

            # Check if we have individual expert weights to consolidate
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                for param_type in ["weight", "scales", "biases"]:
                    expert_key = f"{prefix}.mlp.experts.0.{proj_name}.{param_type}"
                    if expert_key in weights:
                        # Stack all expert weights
                        to_join = [
                            weights.pop(
                                f"{prefix}.mlp.experts.{e}.{proj_name}.{param_type}"
                            )
                            for e in range(self.args.num_experts)
                        ]
                        weights[
                            f"{prefix}.mlp.switch_mlp.{proj_name}.{param_type}"
                        ] = mx.stack(to_join)

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        return weights

    @property
    def layers(self):
        return self.model.layers
