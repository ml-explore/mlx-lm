# Copyright © 2026 Apple Inc.
# Cohere2 MoE (Command A+) support for MLX
# Ported by @eauchs

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "cohere2_moe"
    hidden_size: int = 4096
    head_dim: int = 128
    num_hidden_layers: int = 32
    intermediate_size: int = 4096
    num_attention_heads: int = 128
    num_key_value_heads: int = 8
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 4
    shared_expert_combination_strategy: str = "average"
    expert_selection_fn: str = "sigmoid"
    norm_topk_prob: bool = True
    rope_theta: float = 50000.0
    vocab_size: int = 262144
    layer_norm_eps: float = 1e-05
    logit_scale: float = 1.0
    attention_bias: bool = False
    sliding_window: int = 4096
    sliding_window_pattern: int = 4
    use_parallel_block: bool = True
    use_qk_norm: bool = False
    use_embedding_sharing: bool = True
    first_k_dense_replace: int = 0
    prefix_dense_intermediate_size: int = 16384
    prefix_dense_sliding_window_pattern: int = 1
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.layer_types is None:
            pattern = ["sliding_attention"] * (self.sliding_window_pattern - 1) + [
                "full_attention"
            ]
            self.layer_types = (
                pattern * (self.num_hidden_layers // len(pattern) + 1)
            )[: self.num_hidden_layers]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        # Cohere2 uses interleaved (gptj-style) RoPE
        self.rope = nn.RoPE(head_dim, traditional=True, base=args.rope_theta)

        self.use_sliding_window = args.layer_types[layer_idx] == "sliding_attention"

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Cohere2: RoPE only applied to sliding window layers
        if self.use_sliding_window:
            if cache is None:
                queries = self.rope(queries)
                keys = self.rope(keys)
            else:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        sdpa_type = mx.float32 if queries.dtype == mx.float16 else queries.dtype
        output = scaled_dot_product_attention(
            queries.astype(sdpa_type),
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        ).astype(queries.dtype)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """Dense MLP used for shared experts and any prefix dense layers."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Cohere2MoeSparseMoeBlock(nn.Module):
    """Sparse MoE block with sigmoid routing and shared experts."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.intermediate_size

        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.expert_selection_fn = args.expert_selection_fn

        # Router — no bias in BF16, bias added by NVFP4 quantization in W4A4
        self.gate = nn.Linear(dim, args.num_experts, bias=False)

        # Routed experts via SwitchGLU (no bias in BF16)
        self.switch_mlp = SwitchGLU(
            dim, intermediate_size, args.num_experts, bias=False
        )

        # Shared expert — single MLP with intermediate_size * num_shared_experts
        self.num_shared_experts = args.num_shared_experts
        self.shared_expert_combination_strategy = (
            args.shared_expert_combination_strategy
        )
        if self.num_shared_experts > 0:
            shared_intermediate = intermediate_size * self.num_shared_experts
            self.shared_experts = MLP(dim, shared_intermediate)

    def __call__(self, x: mx.array) -> mx.array:
        # Router scores
        gates = self.gate(x)

        # Cohere A+ uses sigmoid routing (not softmax)
        if self.expert_selection_fn == "sigmoid":
            gates = mx.sigmoid(gates)
        else:
            gates = mx.softmax(gates, axis=-1, precise=True)

        # Top-k expert selection
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)

        # Routed expert computation
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        # Shared expert computation
        if self.num_shared_experts > 0:
            shared_out = self.shared_experts(x)
            if self.shared_expert_combination_strategy == "average":
                y = (y + shared_out) / 2
            else:
                y = y + shared_out

        return y


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Attention(args, layer_idx)

        # MoE or dense MLP
        if layer_idx >= args.first_k_dense_replace:
            self.mlp = Cohere2MoeSparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.prefix_dense_intermediate_size)

        # Cohere2 uses LayerNorm (not RMSNorm) — no bias in BF16 checkpoint
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps, bias=False
        )

        self.use_parallel_block = args.use_parallel_block

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        h = self.input_layernorm(x)

        if self.use_parallel_block:
            # Cohere2 parallel block: attn and MLP computed from same input
            attn_h = self.self_attn(h, mask, cache)
            ff_h = self.mlp(h)
            return attn_h + ff_h + x
        else:
            # Sequential (standard transformer)
            attn_h = self.self_attn(h, mask, cache)
            h = attn_h + x
            ff_h = self.mlp(self.input_layernorm(h))
            return ff_h + h


class Cohere2MoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.window_size = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        # Final norm — no bias in BF16 checkpoint
        self.norm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps, bias=False
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        # Build masks for sliding window and full attention layers
        full_cache = None
        swa_cache = None
        for i, c in enumerate(cache):
            if self.args.layer_types[i] == "full_attention" and full_cache is None:
                full_cache = c
            elif self.args.layer_types[i] == "sliding_attention" and swa_cache is None:
                swa_cache = c

        full_mask = create_attention_mask(h, full_cache)
        swa_mask = create_attention_mask(
            h, swa_cache, window_size=self.window_size
        )

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            is_full = self.args.layer_types[i] == "full_attention"
            mask = full_mask if is_full else swa_mask
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = Cohere2MoeModel(args)
        self.args = args

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        if self.args.use_embedding_sharing:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        out = out * self.args.logit_scale
        return out

    def sanitize(self, weights):
        # Remove vision-related weights and rotary_emb
        sanitized = {}
        for k, v in weights.items():
            # Skip vision tower, projector, and rotary embedding weights
            if any(
                s in k
                for s in (
                    "vision_tower",
                    "multi_modal_projector",
                    "image_newline",
                    "rotary_emb.inv_freq",
                )
            ):
                continue

            # Strip model.language_model. prefix from vision wrapper
            clean_k = k
            if k.startswith("model.language_model."):
                clean_k = k.replace("model.language_model.", "model.")
            elif k.startswith("language_model."):
                clean_k = k.replace("language_model.", "model.")

            sanitized[clean_k] = v

        weights = sanitized

        # Handle tied embeddings
        if self.args.use_embedding_sharing:
            weights.pop("lm_head.weight", None)

        # Repack individual expert weights into stacked SwitchGLU format
        # BF16: experts.{E}.{gate,up,down}_proj.weight
        # W4A4: experts.{E}.{gate,up,down}_proj.weight_packed (+ scales)
        if "model.layers.0.mlp.switch_mlp.up_proj.weight" not in weights:
            for l in range(self.args.num_hidden_layers):
                prefix = f"model.layers.{l}.mlp"
                # Detect suffix: .weight for BF16, .weight_packed for W4A4
                for suffix in ["weight", "weight_packed"]:
                    expert_key = f"{prefix}.experts.0.gate_proj.{suffix}"
                    if expert_key not in weights:
                        continue
                    for n in ["up_proj", "down_proj", "gate_proj"]:
                        # Stack the main weight tensor
                        to_join = [
                            weights.pop(f"{prefix}.experts.{e}.{n}.{suffix}")
                            for e in range(self.args.num_experts)
                        ]
                        weights[f"{prefix}.switch_mlp.{n}.{suffix}"] = mx.stack(
                            to_join
                        )
                        # Stack associated scale/bias tensors if present (W4A4)
                        for extra in [
                            "weight_scale",
                            "weight_global_scale",
                            "input_global_scale",
                            "bias",
                        ]:
                            extra_key = f"{prefix}.experts.0.{n}.{extra}"
                            if extra_key in weights:
                                to_join_extra = [
                                    weights.pop(
                                        f"{prefix}.experts.{e}.{n}.{extra}"
                                    )
                                    for e in range(self.args.num_experts)
                                ]
                                weights[
                                    f"{prefix}.switch_mlp.{n}.{extra}"
                                ] = mx.stack(to_join_extra)

        return weights

    @property
    def quant_predicate(self):
        """Keep router gate and attention at higher precision during quantization."""

        def predicate(path, _):
            if "mlp.gate" in path:
                return {"group_size": 64, "bits": 8}
            if "self_attn" in path:
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def make_cache(self):
        caches = []
        for i in range(self.args.num_hidden_layers):
            if self.args.layer_types[i] == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                )
        return caches

    @property
    def layers(self):
        return self.model.layers
