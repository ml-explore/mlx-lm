# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "enggpt_moe"

    vocab_size: int = 131084
    hidden_size: int = 2880
    intermediate_size: int = 8640
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: Optional[int] = None

    num_experts_per_tok: int = 8
    num_experts: int = 64
    num_local_experts: Optional[int] = None
    moe_intermediate_size: int = 1080
    norm_topk_prob: bool = True

    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    attention_bias: bool = False
    tie_word_embeddings: bool = False

    decoder_sparse_step: int = 1
    mlp_only_layers: Optional[list] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        # MLX/Mixtral usa num_local_experts; EngGPT config usa num_experts.
        if self.num_local_experts is None:
            self.num_local_experts = self.num_experts

        if self.mlp_only_layers is None:
            self.mlp_only_layers = []


class MixtralAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.num_key_value_heads = args.num_key_value_heads
        self.rope_theta = args.rope_theta

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=args.attention_bias
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        keys = self.k_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)

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


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.ffn_dim = args.moe_intermediate_size
        self.num_experts = args.num_local_experts
        self.num_experts_per_tok = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(self.hidden_dim, self.ffn_dim, self.num_experts)

    def __call__(self, x: mx.array) -> mx.array:
        router_logits = self.gate(x)

        routing_weights = mx.softmax(router_logits, axis=-1, precise=True)

        k = self.num_experts_per_tok
        inds = mx.stop_gradient(
            mx.argpartition(-routing_weights, kth=k - 1, axis=-1)[..., :k]
        )

        scores = mx.take_along_axis(routing_weights, inds, axis=-1)

        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        return y

class MixtralDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size

        self.self_attn = MixtralAttention(args)

        self.mlp = MixtralSparseMoeBlock(args)
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
        out = h + r
        return out


class MixtralModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MixtralDecoderLayer(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
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
        self.model = MixtralModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        else:
            return self.lm_head(out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # Converte gli esperti EngGPT2 separati:
        #
        # model.layers.L.mlp.experts.E.gate_proj.weight
        # model.layers.L.mlp.experts.E.up_proj.weight
        # model.layers.L.mlp.experts.E.down_proj.weight
        #
        # in pesi stacked per SwitchGLU:
        #
        # model.layers.L.mlp.switch_mlp.gate_proj.weight
        # model.layers.L.mlp.switch_mlp.up_proj.weight
        # model.layers.L.mlp.switch_mlp.down_proj.weight

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.mlp"

            first_key = f"{prefix}.experts.0.gate_proj.weight"
            if first_key not in weights:
                continue

            for proj in ["gate_proj", "up_proj", "down_proj"]:
                to_join = [
                    weights.pop(f"{prefix}.experts.{e}.{proj}.weight")
                    for e in range(self.args.num_local_experts)
                ]
                weights[f"{prefix}.switch_mlp.{proj}.weight"] = mx.stack(to_join)

        return weights

    @property
    def layers(self):
        return self.model.layers
