# Copyright © 2025 Apple Inc.

from dataclasses import dataclass

from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
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
    tie_word_embeddings: bool = False
    scoring_func: str = "sigmoid"
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
        self.scale = head_dim ** -0.5

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

        self.use_qk_norm = args.use_qk_norm if hasattr(args, 'use_qk_norm') else False
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

        scores = scores / mx.sum(scores, axis=-1, keepdims=True) + 1e-20

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y


class MiniMaxDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.self_attn = MiniMaxAttention(args)

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
        r = x + self.self_attn(self.input_layernorm(x), mask, cache)
        r = r + self.block_sparse_moe(self.post_attention_layernorm(r))
        return r


class MiniMaxModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = [
            MiniMaxDecoderLayer(args=args)
            for _ in range(args.num_hidden_layers)
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
        keys_to_remove = []
        dequantized_count = 0

        for key in list(weights.keys()):
            if not key.endswith(".weight_scale_inv"):
                continue

            weight_key = key.replace(".weight_scale_inv", ".weight")
            if weight_key not in weights:
                continue

            scale_inv = mx.array(weights[key])           # float32, [H_s, W_s]
            bf16_weight = mx.array(weights[weight_key])  # bfloat16, [H, W]

            H, W = bf16_weight.shape
            Hs, Ws = scale_inv.shape

            # Handle block-wise quantization (weight_block_size: [128, 128])

            if scale_inv.ndim == 2:
                if H % 128 != 0 or W % 128 != 0:
                    raise ValueError(f"Weight {weight_key} not divisible by 128")
                if (Hs, Ws) != (H // 128, W // 128):
                    raise ValueError(f"Scale shape mismatch: {scale_inv.shape} vs {(H//128, W//128)}")

                s = scale_inv.reshape(Hs, 1, Ws, 1)
                s = mx.tile(s, (1, 128, 1, 128))
                s = s.reshape(H, W)
            else:
                s = scale_inv  # fallback

            # Dequantize
            fp32_weight = (bf16_weight.astype(mx.float32) * s.astype(mx.float32))
            weights[weight_key] = fp32_weight
            dequantized_count += 1
            keys_to_remove.append(key)

        # Clean up scale_inv keys
        for k in keys_to_remove:
            weights.pop(k, None)

       
        # Step 2: Handle MoE expert weights restructuring
        if not any("block_sparse_moe.experts.0.w1.weight" in k for k in weights):
            return weights

        mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.block_sparse_moe"
            for src, dst in mapping.items():
                expert_weights = [
                    weights.pop(f"{prefix}.experts.{e}.{src}.weight")
                    for e in range(self.args.num_local_experts)
                ]
                weights[f"{prefix}.switch_mlp.{dst}.weight"] = mx.stack(expert_weights, axis=0)

        return weights
    @property
    def layers(self):
        return self.model.layers