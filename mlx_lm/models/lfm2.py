# Copyright © 2023-2024 Apple Inc.
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, MambaCache
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "lfm2"
    vocab_size: int = 65536
    hidden_size: int = 1024
    intermediate_size: int = None
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = None
    norm_eps: float = 1e-05
    conv_bias: bool = False
    conv_L_cache: int = 3
    block_ff_dim: int = None
    block_multiple_of: int = 256
    block_ffn_dim_multiplier: float = 1.0
    block_auto_adjust_ff_dim: bool = True
    full_attn_idxs: Optional[list[int]] = None
    layer_types: Optional[list[str]] = None
    rope_traditional: bool = False
    rope_scaling: Optional[str] = None
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True

    def __post_init__(self):
        self.intermediate_size = self.block_ff_dim or self.intermediate_size
        if self.layer_types is None:
            full_attn_idxs = (
                self.full_attn_idxs
                if self.full_attn_idxs is not None
                else list(range(self.num_hidden_layers))
            )
            self.layer_types = [
                "full_attention" if i in full_attn_idxs else "conv"
                for i in range(self.num_hidden_layers)
            ]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        attention_bias = getattr(args, "attention_bias", False)

        self.q_layernorm = nn.RMSNorm(head_dim, eps=args.norm_eps)
        self.k_layernorm = nn.RMSNorm(head_dim, eps=args.norm_eps)

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

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

        queries = self.q_layernorm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_layernorm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, mask=mask, scale=self.scale
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class LFM2ShortConv(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer_idx: int,
    ):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.L_cache = args.conv_L_cache
        self.bias = args.conv_bias

        self.conv = nn.Conv1d(
            in_channels=args.hidden_size,
            out_channels=args.hidden_size,
            kernel_size=self.L_cache,
            groups=args.hidden_size,
            bias=self.bias,
            padding=self.L_cache - 1,
        )
        self.in_proj = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=self.bias)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ):
        seqlen = x.shape[1]
        BCx = self.in_proj(x)
        B, C, x = mx.split(BCx, 3, axis=-1)
        Bx = B * x

        if cache is not None and x.shape[1] == 1:
            conv_state = (
                cache[0]
                if cache[0] is not None
                else mx.zeros((Bx.shape[0], self.L_cache, self.args.hidden_size))
            )
            conv_state = mx.roll(conv_state, -2, -2)
            conv_state[:, -1, :] = Bx[:, 0, :]
            cache[0] = conv_state

            conv_out = mx.sum(
                conv_state.transpose(0, 2, 1) * self.conv.weight[:, :, 0],
                axis=-1,
                keepdims=True,
            )
            if self.bias:
                conv_out = conv_out + self.conv.bias
            conv_out = conv_out.reshape(Bx.shape[0], 1, -1)

        else:
            if cache is not None:
                cache[0] = Bx[:, -self.L_cache :, :]
            conv_out = self.conv(Bx)[..., :seqlen, :]

        y = C * conv_out
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        intermediate_size = args.intermediate_size
        if args.block_auto_adjust_ff_dim:
            intermediate_size = int(2 * intermediate_size / 3)
            if args.block_ffn_dim_multiplier is not None:
                intermediate_size = int(
                    args.block_ffn_dim_multiplier * intermediate_size
                )
                intermediate_size = args.block_multiple_of * (
                    (intermediate_size + args.block_multiple_of - 1)
                    // args.block_multiple_of
                )
        self.w1 = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_attention_layer = args.layer_types[layer_idx] == "full_attention"

        if self.is_attention_layer:
            self.self_attn = Attention(args)
        else:
            self.conv = LFM2ShortConv(args, layer_idx)
        self.feed_forward = MLP(args)
        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        residual = x

        if self.is_attention_layer:
            x = self.self_attn(self.operator_norm(x), mask=mask, cache=cache)
        else:
            x = self.conv(
                self.operator_norm(x),
                cache=cache,
            )

        x = x + residual

        x = x + self.feed_forward(self.ffn_norm(x))

        return x


class Lfm2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Lfm2DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]

        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if mask is None:
            first_attn_idx = self.args.layer_types.index("full_attention")
            c = [cache[first_attn_idx]] if cache is not None else None
            mask = create_attention_mask(h, c, return_array=False)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.embedding_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Lfm2Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, mask, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        sanitized_weights = {}
        for name, param in weights.items():
            if "conv.weight" in name:
                if param.shape[-1] > param.shape[1]:
                    param = param.transpose(0, 2, 1)

            sanitized_weights[name] = param
        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer_idx in range(len(self.layers)):
            if self.layers[layer_idx].is_attention_layer:
                caches.append(KVCache())
            else:
                caches.append(MambaCache())
        return caches
