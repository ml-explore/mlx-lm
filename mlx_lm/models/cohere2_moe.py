# Cohere2-MoE support for mlx-lm.
#
# Architecture: Cohere2MoeForCausalLM (model_type "cohere2_moe"), used by
# Cohere's North-Mini-Code-1.0 (30B total / 3B active MoE coding model).
#
# It is the Cohere2 (Command R7B) decoder — parallel attention+MLP block, a
# single LayerNorm(bias=False) per layer, 3:1 sliding:global attention with
# NoPE on the global layers, logit_scale, tied embeddings — with the dense MLP
# replaced by a sigmoid-routed MoE on all but the first `first_k_dense_replace`
# layers.
#
# Deltas from our laguna.py MoE port (same family, different vendor):
#   * parallel block + single LayerNorm (not sequential + RMSNorm)
#   * no QK-norm, no per-head attention output gating (poolside-only)
#   * no shared expert, no router e_score_correction_bias
#   * NoPE (no RoPE) on full_attention (global) layers; RoPE only on sliding
#   * full-attention layers are chosen from the explicit `layer_types` array
#     (North's phase is i % 4 == 0), never a modulus guess.
#
# Checkpoint note: the mlx-community bf16/8bit repacks were converted via
# mlx-vlm and wrap every tensor under a `language_model.` prefix, with experts
# already stacked into `switch_mlp` and the router already at `mlp.gate`. The
# only sanitize needed is the prefix strip (same gotcha as the AtomicChat
# Laguna repack). See NORTH_MINI_CODE_PLAN.md.
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 2048
    head_dim: int = 128
    num_hidden_layers: int = 49
    intermediate_size: int = 768
    prefix_dense_intermediate_size: int = 3072
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    vocab_size: int = 262144
    rope_theta: float = 50000.0
    layer_norm_eps: float = 1e-5
    logit_scale: float = 1.0
    attention_bias: bool = False
    sliding_window: int = 4096
    max_position_embeddings: int = 500000
    tie_word_embeddings: bool = True
    # MoE
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 0
    norm_topk_prob: bool = False
    first_k_dense_replace: int = 1
    expert_selection_fn: str = "sigmoid"
    layer_types: Optional[List[str]] = None
    # tolerated-but-unused config keys (kept so BaseModelArgs.from_dict is happy)
    use_parallel_block: bool = True
    use_qk_norm: bool = False

    def __post_init__(self):
        if self.layer_types is None:
            # Fall back to the North 3:1 pattern: full attention every 4th layer
            # starting at layer 0.
            self.layer_types = [
                "full_attention" if (i % 4 == 0) else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must match num_hidden_layers.")


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Cohere2MoeSparseBlock(nn.Module):
    """Sigmoid-routed MoE. No shared expert, no router correction bias.

    Router weights live at `mlp.gate.weight`; stacked experts at
    `mlp.switch_mlp.{gate,up,down}_proj.*` — matching the checkpoint layout,
    so no key remapping is needed.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.use_sigmoid = args.expert_selection_fn == "sigmoid"
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.intermediate_size, args.num_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        logits = self.gate(x).astype(mx.float32)
        scores = mx.sigmoid(logits) if self.use_sigmoid else mx.softmax(logits, axis=-1)

        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k])
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        weights = weights.astype(dtype)

        y = self.switch_mlp(x, inds)
        return mx.sum(y * weights[..., None], axis=-2)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"

        dim = args.hidden_size
        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias
        )

        # RoPE only on sliding layers; global (full_attention) layers are NoPE.
        self.rope = (
            nn.RoPE(self.head_dim, traditional=True, base=args.rope_theta)
            if self.is_sliding
            else None
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if self.rope is not None:
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


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        if layer_idx < args.first_k_dense_replace or args.num_experts == 0:
            self.mlp = MLP(args.hidden_size, args.prefix_dense_intermediate_size)
        else:
            self.mlp = Cohere2MoeSparseBlock(args)
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps, bias=False
        )
        self.attention_type = args.layer_types[layer_idx]

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # Parallel block (Cohere): attention and MLP both read the same
        # normalized hidden state and their outputs are summed into the residual.
        h = self.input_layernorm(x)
        attn_h = self.self_attn(h, mask, cache)
        ff_h = self.mlp(h)
        return x + attn_h + ff_h


class Cohere2MoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps, bias=False)
        self.fa_idx = args.layer_types.index("full_attention")
        self.swa_idx = (
            args.layer_types.index("sliding_attention")
            if "sliding_attention" in args.layer_types
            else None
        )

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            sliding_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.args.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            mask = (
                sliding_mask
                if layer.attention_type == "sliding_attention"
                else full_mask
            )
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Cohere2MoeModel(args)
        self.logit_scale = args.logit_scale
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out * self.logit_scale

    def make_cache(self):
        # 36 of 49 layers are sliding_attention (window 4096): a bounded
        # RotatingKVCache is both correct and far cheaper at long context. Only
        # the 13 full_attention (global, NoPE) layers need an unbounded KVCache.
        caches = []
        for lt in self.args.layer_types:
            if lt == "full_attention" or not self.args.sliding_window:
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches

    def sanitize(self, weights):
        # mlx-vlm repacks (mlx-community North-Mini-Code-1.0-*) wrap every
        # tensor under a `language_model.` prefix. Strip it so keys line up with
        # this module tree (model.* / lm_head.*). Experts are already stacked
        # into switch_mlp and the router is already mlp.gate — no remap needed.
        if any(k.startswith("language_model.") for k in weights):
            prefix = "language_model."
            weights = {
                (k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in weights.items()
            }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

    @property
    def quant_predicate(self):
        # Keep the router at 8-bit (quant-sensitive component), matching the
        # Laguna finding that the gate is the precision-critical weight.
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers
