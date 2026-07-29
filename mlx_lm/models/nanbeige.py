# Copyright © 2024 Apple Inc.
#
# MLX implementation of the Nanbeige (Nanbeige4.2) architecture.
#
# Nanbeige is a standard Llama-style GQA decoder (RMSNorm, SwiGLU MLP,
# rotate-half RoPE) with ONE non-standard runtime behavior that is active in
# the released Nanbeige4.2-3B checkpoint: weight-tied looping. The full stack of
# `num_hidden_layers` decoder layers is executed `num_loops` times with shared
# weights. Each loop keeps an INDEPENDENT KV cache (cache slot for a layer is
# `layer_idx + loop_idx * num_hidden_layers`), and `model.norm` is applied at
# the end of every loop (unless `skip_loop_final_norm`), so the normalized
# output of one loop is the input to the next.
#
# All of the other features in the reference `modeling_nanbeige.py`
# (n-gram embeddings, hyper-connection / mHC, depth-attention, double-loop
# split, QK-LayerNorm, attention/MLP bias, loop-loss training) are gated off in
# the released config and carry no weights in the checkpoint. They are NOT
# implemented here; the guard in `Model.__init__` raises if a config enables any
# of them so we never silently produce wrong outputs.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False

    # --- Nanbeige loop parameters (the one live non-standard feature) ---
    num_loops: int = 1
    skip_loop_final_norm: bool = False

    # --- Features present in the reference code but OFF in the released
    # checkpoint. Declared so we can detect and refuse them explicitly. ---
    loop_loss_weights: Optional[List[float]] = None
    enable_double_loop_split: bool = False
    enable_hyper_connection: bool = False
    enable_mhc: bool = False
    enable_depth_attention: bool = False
    qk_layernorm: bool = False
    emb_neighbor_num: Optional[int] = None
    insert_ngram_layer_idx: Optional[List[int]] = None
    ngram_insert_all_layers: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        self.rope = initialize_rope(
            self.head_dim,
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
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

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


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=args.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
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


class NanbeigeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.num_loops = max(1, args.num_loops)
        self.skip_loop_final_norm = args.skip_loop_final_norm
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
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

        n_layers = self.num_hidden_layers
        if cache is None:
            cache = [None] * (n_layers * self.num_loops)

        # All per-loop caches share the same offset at the start of this call,
        # so a single mask (built from the first loop's first cache) is valid
        # for every layer in every loop.
        mask = create_attention_mask(h, cache[0])

        for loop_idx in range(self.num_loops):
            base = loop_idx * n_layers
            for i, layer in enumerate(self.layers):
                h = layer(h, mask, cache=cache[base + i])
            if not self.skip_loop_final_norm:
                h = self.norm(h)

        if self.skip_loop_final_norm:
            h = self.norm(h)

        return h


_UNSUPPORTED_FLAGS = (
    ("enable_double_loop_split", True),
    ("enable_hyper_connection", True),
    ("enable_mhc", True),
    ("enable_depth_attention", True),
    ("qk_layernorm", True),
    ("ngram_insert_all_layers", True),
)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self._check_supported(args)
        self.args = args
        self.model_type = args.model_type
        self.model = NanbeigeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @staticmethod
    def _check_supported(args: ModelArgs):
        enabled = [
            name for name, on in _UNSUPPORTED_FLAGS if getattr(args, name, False) == on
        ]
        if args.emb_neighbor_num is not None:
            enabled.append("emb_neighbor_num (n-gram embeddings)")
        if args.insert_ngram_layer_idx:
            enabled.append("insert_ngram_layer_idx (n-gram layer fusion)")
        if args.loop_loss_weights:
            enabled.append("loop_loss_weights (multi-loop training objective)")
        if enabled:
            raise NotImplementedError(
                "This MLX Nanbeige port supports the released Nanbeige4.2 checkpoint "
                "(plain looped Llama). The config enables unsupported feature(s): "
                + ", ".join(enabled)
                + ". These carry extra modules/weights that are not implemented here."
            )

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
        weights = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # One independent KV cache per (loop, layer): slot = loop*num_layers + i.
        return [
            KVCache()
            for _ in range(self.model.num_hidden_layers * self.model.num_loops)
        ]
