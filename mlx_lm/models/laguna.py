# Copyright © 2026 Apple Inc.

# Laguna M.1: MoE (256 experts, top-16, sigmoid routing) with GQA, per-head
# QK-norm, softplus attention output gating, and RoPE with YaRN scaling.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "laguna"
    vocab_size: int = 100352
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 70
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 500000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False
    # MoE
    num_experts: int = 256
    num_experts_per_tok: int = 16
    moe_intermediate_size: int = 1024
    shared_expert_intermediate_size: int = 1024
    norm_topk_prob: bool = True
    moe_routed_scaling_factor: float = 1.0
    # Layer types: "dense" or "sparse" per layer
    mlp_layer_types: Optional[List[str]] = None
    # Gating: "per-element" enables softplus output gating on attention
    gating: Optional[str] = "per-element"


class LagunaAttention(nn.Module):
    """Laguna attention: GQA + QK-norm + softplus output gating."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        # Laguna-specific: output gating projection
        self.g_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)

        # QK normalization (per-head RMSNorm before RoPE)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            dims=self.head_dim,
            base=args.rope_theta,
            traditional=False,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape to [B, n_heads, L, head_dim]
        queries = self.q_norm(
            queries.reshape(B, L, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # RoPE
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Attention
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Laguna-specific: softplus gating BEFORE o_proj
        gate = nn.softplus(self.g_proj(x))
        output = output * gate

        return self.o_proj(output)


class LagunaMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LagunaRouter(nn.Module):
    """Sigmoid-based top-k router (not softmax)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_experts
        self.norm_topk_prob = args.norm_topk_prob
        self.weight = mx.zeros((self.num_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.num_experts,))

    def __call__(self, x: mx.array):
        # Sigmoid scoring in float32 for precision
        router_logits = x @ self.weight.T
        scores = mx.sigmoid(router_logits.astype(mx.float32))

        # Apply correction bias for expert selection (not for final weights)
        corrected_scores = scores + self.e_score_correction_bias

        # Top-k selection on corrected scores
        k = self.top_k
        inds = mx.argpartition(-corrected_scores, kth=k - 1, axis=-1)[..., :k]
        # Use original scores (without bias) for weighting
        routing_weights = mx.take_along_axis(scores, inds, axis=-1)

        if self.norm_topk_prob:
            routing_weights = routing_weights / mx.sum(
                routing_weights, axis=-1, keepdims=True
            )

        return inds, routing_weights.astype(x.dtype)


class LagunaSparseMoeBlock(nn.Module):
    """Laguna MoE: sigmoid router + SwitchGLU experts + shared expert."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = LagunaRouter(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
        )
        self.shared_expert = LagunaMLP(
            args.hidden_size, args.shared_expert_intermediate_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)

        # Routed experts via SwitchGLU
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

        # Add shared expert
        y = y + self.shared_expert(x)

        return y


class LagunaDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = LagunaAttention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        # Dense or sparse MLP based on mlp_layer_types
        is_sparse = (
            args.mlp_layer_types is not None
            and layer_idx < len(args.mlp_layer_types)
            and args.mlp_layer_types[layer_idx] == "sparse"
        )
        if is_sparse:
            self.mlp = LagunaSparseMoeBlock(args)
        else:
            self.mlp = LagunaMLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LagunaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            LagunaDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
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
        self.model = LagunaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        # Dequantize FP8 compressed-tensors: uint8 + weight_scale → bfloat16
        new_weights = {}
        for k, v in weights.items():
            if k.endswith("weight_scale"):
                continue  # handled with corresponding weight
            if k.endswith(".weight") and f"{k}_scale" in weights:
                scale = weights[f"{k}_scale"]
                new_weights[k] = self._dequant_fp8_block(v, scale)
            else:
                new_weights[k] = v
        weights = new_weights

        # Remap e_score_correction_bias:
        # HF: model.layers.X.mlp.experts.e_score_correction_bias
        # MLX: model.layers.X.mlp.gate.e_score_correction_bias
        remapped = {}
        for k, v in weights.items():
            if "mlp.experts.e_score_correction_bias" in k:
                new_key = k.replace(
                    "mlp.experts.e_score_correction_bias",
                    "mlp.gate.e_score_correction_bias",
                )
                remapped[new_key] = v
            else:
                remapped[k] = v
        weights = remapped

        # Stack per-expert weights into SwitchGLU format
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                key0 = f"{prefix}.mlp.experts.0.{n}.weight"
                if key0 in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{n}.weight")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{n}.weight"] = mx.stack(
                        to_join
                    )

        # Remove unused keys
        weights = {
            k: v
            for k, v in weights.items()
            if "rotary_emb.inv_freq" not in k
        }

        return weights

    @staticmethod
    def _dequant_fp8_block(weight: mx.array, scale: mx.array) -> mx.array:
        """Dequantize FP8 block-quantized weight (uint8 + scale → bfloat16).

        Block size is 128x128 (inferred from scale shape vs weight shape).
        """
        m, n = weight.shape
        sm, sn = scale.shape
        bs_m = m // sm  # block size along rows
        bs_n = n // sn  # block size along cols

        # Reinterpret uint8 as float8 E4M3, then cast to bfloat16
        weight_bf16 = mx.from_fp8(weight, mx.bfloat16)

        # Apply block-wise scale
        weight_bf16 = weight_bf16.reshape(sm, bs_m, sn, bs_n)
        weight_bf16 = weight_bf16 * scale[:, None, :, None]
        weight_bf16 = weight_bf16.reshape(m, n)

        return weight_bf16

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # Don't quantize router gate weights — keep full precision
            if "mlp.gate.weight" in path:
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
