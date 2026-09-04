# Copyright © 2026 Apple Inc.
# MiniMax-M3 text backbone (text-only LLM extraction of MiniMaxM3VL).
#
# M3 extends MiniMax-M2 with: Gemma-style RMSNorm (scale by 1+w, fp32),
# per-head QK-norm, partial RoPE (rotary_dim < head_dim), SwiGLU-OAI activation
# (clamped gate/up with an (up+1) term), a shared expert + routed_scaling_factor
# in the MoE, and the first few layers being dense MLPs instead of MoE.
#
# MiniMax Sparse Attention (MSA) is implemented here as full causal attention.
# For sequences up to index_topk_blocks * index_block_size (= 2048) tokens the
# MSA indexer selects *every* key block, so full attention is numerically exact;
# beyond that it is the dense (un-approximated) attention MSA approximates, so
# quality is preserved at the cost of long-context speed/memory.

from dataclasses import dataclass, field
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
    dense_intermediate_size: int
    shared_intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    num_local_experts: int
    num_experts_per_tok: int
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    vocab_size: int
    head_dim: int = 128
    max_position_embeddings: int = 1048576
    routed_scaling_factor: float = 2.0
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    scoring_func: str = "sigmoid"
    use_qk_norm: bool = True
    tie_word_embeddings: bool = False
    # Per-layer MLP dispatch: "sparse" -> MoE block, "dense" -> dense MLP.
    mlp_layer_types: Optional[List[str]] = None


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm: normalize in fp32 and scale by ``weight + 1``."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def _extra_repr(self):
        return f"{self.weight.shape[0]}, eps={self.eps}"

    def __call__(self, x):
        ot = x.dtype
        x = x.astype(mx.float32)
        x = x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)
        return (x * (1.0 + self.weight.astype(mx.float32))).astype(ot)


def swiglu_oai(x_gate, x_up, alpha: float, limit: float):
    """GPT-OSS / MiniMax-M3 clamped SwiGLU: (clamp(up)+1) * gate*sigmoid(alpha*gate)."""
    gate = mx.minimum(x_gate, limit)
    up = mx.clip(x_up, -limit, limit)
    return (up + 1.0) * (gate * mx.sigmoid(gate * alpha))


class SwiGLUOAI(nn.Module):
    """Activation callable for SwitchGLU: receives (x_up, x_gate)."""

    def __init__(self, alpha: float, limit: float):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def __call__(self, x_up, x_gate):
        return swiglu_oai(x_gate, x_up, self.alpha, self.limit)


class MiniMaxM3MLP(nn.Module):
    """Dense SwiGLU-OAI MLP (used by the first dense layers and the shared expert)."""

    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        self.alpha = args.swiglu_alpha
        self.limit = args.swiglu_limit
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(
            swiglu_oai(self.gate_proj(x), self.up_proj(x), self.alpha, self.limit)
        )


class MiniMaxM3Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * head_dim, args.hidden_size, bias=False
        )

        # M3 uses per-head Gemma QK-norm over the head dimension.
        self.q_norm = GemmaRMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(args.rotary_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_attention_heads, self.head_dim)
        keys = self.k_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)

        # Per-head QK-norm over the head dim, before transpose / RoPE.
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

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


class MiniMaxM3SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.routed_scaling_factor = args.routed_scaling_factor

        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_local_experts,))
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.intermediate_size,
            args.num_local_experts,
            activation=SwiGLUOAI(args.swiglu_alpha, args.swiglu_limit),
        )
        self.shared_experts = MiniMaxM3MLP(args, args.shared_intermediate_size)

    def __call__(self, x):
        gates = self.gate(x.astype(mx.float32))
        scores = mx.sigmoid(gates)
        orig_scores = scores
        scores = scores + self.e_score_correction_bias

        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(orig_scores, inds, axis=-1)
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        weights = (weights * self.routed_scaling_factor).astype(x.dtype)

        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2)
        return y + self.shared_experts(x)


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = MiniMaxM3Attention(args)
        self.is_sparse = (args.mlp_layer_types or ["sparse"] * args.num_hidden_layers)[
            layer_idx
        ] == "sparse"
        if self.is_sparse:
            self.block_sparse_moe = MiniMaxM3SparseMoeBlock(args)
        else:
            self.mlp = MiniMaxM3MLP(args, args.dense_intermediate_size)
        self.input_layernorm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        r = x + self.self_attn(self.input_layernorm(x), mask, cache)
        mlp = self.block_sparse_moe if self.is_sparse else self.mlp
        return r + mlp(self.post_attention_layernorm(r))


class MiniMaxM3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MiniMaxM3DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, mask=None, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniMaxM3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, mask=None, cache=None):
        out = self.model(inputs, mask, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        skip_prefixes = (
            "vision_tower",
            "multi_modal_projector",
            "patch_merge_mlp",
            "model.vision_tower",
            "model.multi_modal_projector",
        )

        def keep(k):
            if k.startswith(skip_prefixes):
                return False
            if ".self_attn.index_" in k:  # MSA lightning indexer — dropped
                return False
            if ".mtp." in k or k.startswith("mtp.") or "model.mtp" in k:
                return False
            return True

        def rename(k):
            if k.startswith("language_model.model."):
                return "model." + k[len("language_model.model.") :]
            if k.startswith("language_model.lm_head."):
                return "lm_head." + k[len("language_model.lm_head.") :]
            if k.startswith("language_model."):
                return k[len("language_model.") :]
            return k

        renamed = {}
        for k, v in weights.items():
            if not keep(k):
                continue
            renamed[rename(k)] = v
        weights = renamed

        # Stack per-expert w1/w2/w3 into SwitchGLU's batched experts.
        if (
            "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"
            not in weights
        ):
            mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
            for l in range(self.args.num_hidden_layers):
                prefix = f"model.layers.{l}.block_sparse_moe"
                if f"{prefix}.experts.0.w1.weight" not in weights:
                    continue
                for orig, new in mapping.items():
                    stacked = mx.stack(
                        [
                            weights.pop(f"{prefix}.experts.{e}.{orig}.weight")
                            for e in range(self.args.num_local_experts)
                        ]
                    )
                    weights[f"{prefix}.switch_mlp.{new}.weight"] = stacked

        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        # Keep the router correction bias in fp32.
        return lambda k: "e_score_correction_bias" not in k

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # Routers stay high-precision (small, sensitive to quantization).
            if path.endswith("block_sparse_moe.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
