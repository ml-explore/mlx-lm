# Copyright © 2023-2024 Apple Inc.
#
# MLX port of the Spark-X2.5 architecture (XHToken/Spark-X2.5-4B).
# Hybrid sliding-window / full attention with head-wise output gating.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "spark2_5"
    vocab_size: int = 131072
    hidden_size: int = 2560
    intermediate_size: int = 10240
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "gelu"
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    mlp_bias: bool = False
    attention_dropout: float = 0.0
    headwise_attn_output_gate: bool = True
    gate_attn_act_mode: str = "sigmoid"
    sliding_window: int = 512
    max_position_embeddings: int = 1048576
    tie_word_embeddings: bool = True
    layer_types: Optional[List[str]] = None
    rope_parameters: Optional[Dict[str, Dict[str, Union[float, int]]]] = None

    def get_rope_theta(self, layer_type: str) -> float:
        params = (self.rope_parameters or {}).get(layer_type, {})
        return params.get("rope_theta", 10000.0)

    def get_partial_rotary_factor(self, layer_type: str) -> float:
        params = (self.rope_parameters or {}).get(layer_type, {})
        return params.get("partial_rotary_factor", 1.0)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()

        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        self.q_k_v_proj = nn.Linear(
            args.hidden_size, self.q_dim + 2 * self.kv_dim, bias=args.attention_bias
        )
        self.g_proj = (
            nn.Linear(args.hidden_size, self.n_heads, bias=args.attention_bias)
            if args.headwise_attn_output_gate
            else None
        )
        self.out_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=args.attention_bias
        )

        # Rotate only the first rope_dim features of the head.
        rope_dim = int(self.head_dim * args.get_partial_rotary_factor(layer_type))
        self.rope = nn.RoPE(rope_dim, base=args.get_rope_theta(layer_type))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.q_k_v_proj(x)
        queries = qkv[..., : self.q_dim]
        keys = qkv[..., self.q_dim : self.q_dim + self.kv_dim]
        values = qkv[..., self.q_dim + self.kv_dim :]

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        if self.g_proj is not None:
            gate = mx.sigmoid(self.g_proj(x).astype(mx.float32))
            gate = gate.reshape(B, L, self.n_heads, 1).transpose(0, 2, 1, 3)
            output = output * gate.astype(output.dtype)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.gate_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=args.mlp_bias
        )
        self.up_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=args.mlp_bias
        )
        self.down_proj = nn.Linear(
            args.intermediate_size, args.hidden_size, bias=args.mlp_bias
        )

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        layer_type = (
            args.layer_types[layer_idx]
            if args.layer_types is not None
            and layer_idx < len(args.layer_types)
            else "full_attention"
        )
        self.use_sliding = layer_type == "sliding_attention"

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.self_attn = Attention(args, layer_type)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = MLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.input_layernorm(x)
        r = self.self_attn(r, mask, cache)
        h = x + r

        r = self.post_attention_layernorm(h)
        r = self.mlp(r)
        return h + r


class SparkModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_types = args.layer_types or ["full_attention"] * args.num_hidden_layers
        self.sliding_window = args.sliding_window
        self.hidden_size = args.hidden_size

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_idx=idx)
            for idx in range(self.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.fa_idx = self.layer_types.index("full_attention")
        self.swa_idx = None
        for idx, layer in enumerate(self.layers):
            if layer.use_sliding:
                self.swa_idx = idx
                break

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        swa_mask = None
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
        self.model = SparkModel(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        # The checkpoint uses "model.embedding.weight" for the (tied) embedding.
        if "model.embedding.weight" in weights:
            weights["model.embed_tokens.weight"] = weights.pop("model.embedding.weight")

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # Remove unused precomputed rotary freqs
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.model.sliding_window, keep=0)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]
