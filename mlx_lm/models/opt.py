# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    vocab_size: int
    ffn_dim: int
    max_position_embeddings: int = 2048
    word_embed_proj_dim: int = None
    do_layer_norm_before: bool = True
    layer_norm_elementwise_affine: bool = True
    activation_function: str = "relu"

    def __post_init__(self):
        if self.word_embed_proj_dim is None:
            self.word_embed_proj_dim = self.hidden_size


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.ffn_dim, bias=True)
        self.fc2 = nn.Linear(args.ffn_dim, args.hidden_size, bias=True)
        self.activation_function = args.activation_function

    def __call__(self, x: mx.array) -> mx.array:
        if self.activation_function == "relu":
            return self.fc2(nn.relu(self.fc1(x)))
        else:
            return self.fc2(nn.gelu(self.fc1(x)))


class OPTDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.do_layer_norm_before = args.do_layer_norm_before
        self.self_attn = Attention(args)
        self.self_attn_layer_norm = nn.LayerNorm(args.hidden_size)
        self.mlp = MLP(args)
        self.final_layer_norm = nn.LayerNorm(args.hidden_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x

        if self.do_layer_norm_before:
            x = self.self_attn_layer_norm(x)

        x = self.self_attn(x, mask, cache)
        x = residual + x

        if not self.do_layer_norm_before:
            x = self.self_attn_layer_norm(x)

        residual = x

        if self.do_layer_norm_before:
            x = self.final_layer_norm(x)

        x = self.mlp(x)
        x = residual + x

        if not self.do_layer_norm_before:
            x = self.final_layer_norm(x)

        return x


class OPTDecoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.word_embed_proj_dim)
        # OPT position embeddings are offset by 2
        self.embed_positions = nn.Embedding(
            args.max_position_embeddings + 2, args.hidden_size
        )

        # Project if word_embed_proj_dim != hidden_size
        self.project_in = None
        self.project_out = None
        if args.word_embed_proj_dim != args.hidden_size:
            self.project_in = nn.Linear(
                args.word_embed_proj_dim, args.hidden_size, bias=False
            )
            self.project_out = nn.Linear(
                args.hidden_size, args.word_embed_proj_dim, bias=False
            )

        self.layers = [OPTDecoderLayer(args) for _ in range(args.num_hidden_layers)]

        self.do_layer_norm_before = args.do_layer_norm_before
        if self.do_layer_norm_before:
            self.final_layer_norm = nn.LayerNorm(args.hidden_size)
        else:
            self.final_layer_norm = None

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        _, L = inputs.shape

        # Token embeddings
        hidden_states = self.embed_tokens(inputs)

        # Project if needed
        if self.project_in is not None:
            hidden_states = self.project_in(hidden_states)

        # Position embeddings (offset by 2)
        if cache is None or cache[0] is None:
            offset = 0
        else:
            offset = cache[0].offset

        position_ids = mx.arange(L) + offset + 2  # OPT offset
        hidden_states = hidden_states + self.embed_positions(position_ids)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(hidden_states, cache[0])

        for layer, c in zip(self.layers, cache):
            hidden_states = layer(hidden_states, mask, cache=c)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        # Project out if needed
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        return hidden_states


class OPTModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.decoder = OPTDecoder(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        return self.decoder(inputs, cache)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = OPTModel(args)
        self.lm_head = nn.Linear(args.word_embed_proj_dim, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        new_weights = {}
        for k, v in weights.items():
            # Skip attention bias buffers
            if "self_attn.bias" in k or "self_attn.masked_bias" in k:
                continue

            new_k = k

            # HuggingFace has fc1/fc2 directly in layer, we have them in mlp module
            # model.decoder.layers.X.fc1 -> model.decoder.layers.X.mlp.fc1
            new_k = new_k.replace(".fc1.", ".mlp.fc1.")
            new_k = new_k.replace(".fc2.", ".mlp.fc2.")

            new_weights[new_k] = v

        return new_weights

    @property
    def layers(self):
        return self.model.decoder.layers
