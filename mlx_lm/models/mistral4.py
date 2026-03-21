# Copyright © 2026 Apple Inc.
#
# Mistral 4 (MoE + MLA) model support.
# Targets: Mistral Small 4 (119B total, 6B active, 128 experts, 4 active per token).
# MoE routing via switch_layers.SwitchGLU. MLA attention uses explicit kv_b_proj
# linear layer for KV decompression (distinct from DeepSeek V3's MultiLinear approach).

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .pipeline import PipelineMixin
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "mistral4"
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-5
    vocab_size: int = 131072
    head_dim: int = 128
    max_position_embeddings: int = 262144
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 1000000000.0
    rope_parameters: Optional[Dict[str, Union[float, str]]] = None
    rope_scaling: Optional[Dict] = None
    tie_word_embeddings: bool = False
    layer_types: Optional[List[str]] = None
    sliding_window: Optional[int] = None

    # MoE fields
    n_routed_experts: int = 128
    num_experts_per_tok: int = 4
    moe_intermediate_size: int = 2048
    n_shared_experts: Optional[int] = 1
    first_k_dense_replace: int = 0

    # MLA fields
    kv_lora_rank: int = 256
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 64
    qk_head_dim: int = 128
    v_head_dim: int = 128

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


class MLAAttention(nn.Module):
    """Multi-head Latent Attention (MLA) from DeepSeek V3, adapted for Mistral 4.
    Uses compressed KV projections with LoRA-style decomposition."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.v_head_dim = args.v_head_dim
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim

        self.scale = self.q_head_dim ** -0.5

        # Q projection: compressed via LoRA
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        # KV projection: compressed with MQA-style sharing
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)

        # Mistral 4 uses explicit kv_b_proj (unlike DeepSeek's MultiLinear approach)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        rope_theta = args.rope_theta
        if args.rope_parameters and "rope_theta" in args.rope_parameters:
            rope_theta = args.rope_parameters["rope_theta"]

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=rope_theta,
            traditional=True,
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

        # Compressed Q
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        # Compressed KV
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        # RoPE
        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        # Decompress KV via kv_b_proj
        kv = self.kv_b_proj(kv_latent)
        kv = kv.reshape(B, -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        kv = kv.transpose(0, 2, 1, 3)
        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        # Combine positional and non-positional key components
        keys = mx.concatenate([k_nope, mx.broadcast_to(k_pe, k_nope.shape[:-1] + k_pe.shape[-1:])], axis=-1)
        queries = mx.concatenate([q_nope, q_pe], axis=-1)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class StandardAttention(nn.Module):
    """Standard multi-head attention for dense layers."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or args.hidden_size // self.n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        rope_theta = args.rope_theta
        if args.rope_parameters and "rope_theta" in args.rope_parameters:
            rope_theta = args.rope_parameters["rope_theta"]

        self.rope = initialize_rope(
            self.head_dim,
            rope_theta,
            False,
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
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """Standard dense MLP with SwiGLU activation."""

    def __init__(self, args: ModelArgs, intermediate_size: int = None):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with shared expert and top-K routing."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok

        # Router gate
        self.gate = nn.Linear(args.hidden_size, args.n_routed_experts, bias=False)

        # Routed experts (SwitchGLU from switch_layers.py)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            bias=False,
        )

        # Shared expert(s)
        if args.n_shared_experts is not None and args.n_shared_experts > 0:
            shared_intermediate = args.moe_intermediate_size * args.n_shared_experts
            self.shared_experts = MLP(args, intermediate_size=shared_intermediate)
        else:
            self.shared_experts = None

    def __call__(self, x):
        # Route tokens to experts
        gate_logits = self.gate(x)
        k = self.num_experts_per_tok

        # Top-K expert selection with softmax scores
        inds = mx.stop_gradient(mx.argpartition(-gate_logits, kth=k - 1, axis=-1)[..., :k])
        scores = mx.softmax(gate_logits, axis=-1)
        scores = mx.take_along_axis(scores, inds, axis=-1)
        # Normalize scores
        scores = scores / scores.sum(axis=-1, keepdims=True)

        # Expert computation
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

        # Add shared expert output
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)

        return y


class TransformerBlock(nn.Module):
    """Transformer block that supports both dense and MoE+MLA configurations."""

    def __init__(self, args: ModelArgs, layer_idx: int = 0, use_sliding: bool = False):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.use_sliding = use_sliding

        # Determine if this layer uses MoE or dense MLP
        is_dense = layer_idx < args.first_k_dense_replace
        use_moe = not is_dense and args.n_routed_experts is not None and args.n_routed_experts > 0

        # Determine if this layer uses MLA or standard attention
        use_mla = args.kv_lora_rank is not None and args.kv_lora_rank > 0 and not is_dense

        # Attention
        if use_mla:
            self.self_attn = MLAAttention(args)
        else:
            self.self_attn = StandardAttention(args)

        # Feedforward
        if use_moe:
            self.mlp = MoELayer(args)
        else:
            self.mlp = MLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.use_mla = use_mla

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class LanguageModel(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(
                args=args,
                layer_idx=i,
                use_sliding=(self.layer_types[i] == "sliding_attention" if i < len(self.layer_types) else False),
            )
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        mask = create_attention_mask(h, cache[0])

        for l, c in zip(self.pipeline_layers, cache):
            h = l(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
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
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # Handle weight_scale_inv pattern
        new_weights = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                weight = weights[wk]
                new_weights[wk] = weight * scale_inv
            elif "activation_scale" in k:
                continue
            elif k not in new_weights:
                new_weights[k] = v

        return new_weights

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.use_sliding and self.model.sliding_window:
                caches.append(RotatingKVCache(max_size=self.model.sliding_window))
            else:
                caches.append(KVCache())
        return caches
