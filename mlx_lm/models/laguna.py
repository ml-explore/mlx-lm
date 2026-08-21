from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "laguna"
    vocab_size: int = 100352
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 40
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    qkv_bias: bool = False
    gating: bool = True
    tie_word_embeddings: bool = False
    sliding_window: Optional[int] = 512
    partial_rotary_factor: Optional[float] = None
    rope_parameters: Optional[Dict[str, Any]] = None
    layer_types: Optional[List[str]] = None
    num_attention_heads_per_layer: Optional[List[int]] = None
    mlp_layer_types: Optional[List[str]] = None
    # MoE
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    moe_routed_scaling_factor: float = 1.0
    moe_router_logit_softcapping: float = 0.0
    moe_router_score_func: str = "sigmoid"

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        if self.num_attention_heads_per_layer is None:
            self.num_attention_heads_per_layer = [
                self.num_attention_heads
            ] * self.num_hidden_layers


def _layer_rope(args: ModelArgs, layer_type: str):
    """Build the RoPE module for a given layer type.

    ``rope_parameters`` is a nested ``{layer_type: rope_dict}`` mapping (the
    v5 Laguna-XS schema). Each sub-dict carries its own ``partial_rotary_factor``
    so that full-attention layers (YaRN, partial rotary) and sliding-attention
    layers (default RoPE, full rotary) can differ.
    """
    rope_params = (args.rope_parameters or {}).get(layer_type) or {}
    # ``rope_theta`` is the base; the rest configure the scaling.
    base = rope_params.get("rope_theta", 10000.0)
    partial = rope_params.get(
        "partial_rotary_factor",
        args.partial_rotary_factor if args.partial_rotary_factor is not None else 1.0,
    )
    dims = int(args.head_dim * partial)
    scaling_config = {k: v for k, v in rope_params.items() if k != "rope_theta"}
    if "rope_type" not in scaling_config and "type" not in scaling_config:
        scaling_config["rope_type"] = "default"
    return initialize_rope(
        dims,
        base=base,
        traditional=False,
        scaling_config=scaling_config,
        max_position_embeddings=args.max_position_embeddings,
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads_per_layer[layer_idx]
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.gating = args.gating

        layer_type = args.layer_types[layer_idx]
        self.use_sliding = layer_type == "sliding_attention"

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.qkv_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.qkv_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.qkv_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        # QK normalization (like Qwen3), applied per head over ``head_dim``.
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        # Optional per-head attention output gating (softplus).
        if self.gating:
            self.g_proj = nn.Linear(dim, n_heads, bias=False)

        self.rope = _layer_rope(args, layer_type)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)
        values = values.reshape(B, L, self.n_kv_heads, -1)

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

        if self.gating:
            # gate: [B, L, n_heads]; broadcast across head_dim.
            gate = nn.softplus(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
            output = output.reshape(B, L, self.n_heads, self.head_dim)
            output = (output * gate[..., None]).reshape(B, L, -1)

        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_experts
        self.softcap = args.moe_router_logit_softcapping
        self.score_func = args.moe_router_score_func
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.num_experts,))

    def _routing_scores(self, logits: mx.array) -> mx.array:
        if self.score_func == "sigmoid":
            return mx.sigmoid(logits)
        if self.score_func == "sqrtsoftplus":
            return mx.sqrt(nn.softplus(logits))
        raise ValueError(f"Unknown moe_router_score_func: {self.score_func!r}")

    def __call__(self, x):
        logits = (x @ self.weight.T).astype(mx.float32)
        if self.softcap > 0.0:
            logits = mx.tanh(logits / self.softcap) * self.softcap

        scores = self._routing_scores(logits)
        scores_for_selection = scores + self.e_score_correction_bias
        inds = mx.argpartition(-scores_for_selection, kth=self.top_k - 1, axis=-1)[
            ..., : self.top_k
        ]
        weights = mx.take_along_axis(scores, inds, axis=-1)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        return inds, weights


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.routed_scaling_factor = args.moe_routed_scaling_factor
        self.gate = MoEGate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
        )
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)

    def __call__(self, x):
        shared_out = self.shared_expert(x)

        inds, weights = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(x.dtype)

        return y * self.routed_scaling_factor + shared_out


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.use_sliding = self.self_attn.use_sliding
        if args.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = MoE(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
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
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LagunaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.sliding_window = args.sliding_window
        self.layer_types = args.layer_types
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args, idx) for idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.fa_idx = self.layer_types.index("full_attention")
        self.swa_idx = None
        for e, l in enumerate(self.layers):
            if l.use_sliding:
                self.swa_idx = e
                break

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
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        swa_mask = fa_mask
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, cache=c)

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
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.model.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]
