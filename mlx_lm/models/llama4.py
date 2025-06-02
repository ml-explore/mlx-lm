# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import ChunkedKVCache, KVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class TextArgs(BaseModelArgs):
    attention_bias: bool
    attention_chunk_size: int
    head_dim: int
    hidden_act: str
    hidden_size: int
    interleave_moe_layer_step: int
    intermediate_size: int
    intermediate_size_mlp: int
    max_position_embeddings: int
    model_type: str
    num_attention_heads: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_key_value_heads: int
    num_local_experts: int
    rms_norm_eps: float
    rope_scaling: Any
    rope_theta: float
    use_qk_norm: bool
    vocab_size: int
    attn_temperature_tuning: int = 4
    floor_scale: int = 8192
    attn_scale: float = 0.1


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: Union[TextArgs, dict]
    model_type: str

    def __post_init__(self):
        self.text_config = TextArgs.from_dict(self.text_config)


class Attention(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.use_rope = int((layer_idx + 1) % 4 != 0)  # rope unused for dense layers
        self.attn_temperature_tuning = args.attn_temperature_tuning
        self.floor_scale = args.floor_scale
        self.attn_scale = args.attn_scale

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.use_qk_norm = args.use_qk_norm and self.use_rope

        if self.use_rope:
            self.rope = initialize_rope(
                head_dim,
                args.rope_theta,
                traditional=True,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
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
            offset = cache.offset
        else:
            offset = 0

        if self.use_rope:
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if self.use_qk_norm:
            queries = mx.fast.rms_norm(queries, weight=None, eps=1e-6)
            keys = mx.fast.rms_norm(keys, weight=None, eps=1e-6)

        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                mx.log(
                    mx.floor(mx.arange(offset + 1, offset + L + 1) / self.floor_scale)
                    + 1.0
                )
                * self.attn_scale
                + 1.0
            )
            attn_scales = attn_scales[:, None]
            queries = (queries * attn_scales).astype(queries.dtype)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: int = None):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = intermediate_size or args.intermediate_size

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_local_experts
        self.experts = SwitchGLU(
            args.hidden_size, args.intermediate_size, self.num_experts
        )
        self.router = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.shared_expert = MLP(args)

    def __call__(self, x) -> mx.array:
        logits = self.router(x)
        k = self.top_k
        indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(logits, indices, axis=-1)
        scores = mx.sigmoid(scores.astype(mx.float32)).astype(x.dtype)

        out = self.experts(x * scores, indices).squeeze(2)
        return out + self.shared_expert(x)


class TransformerBlock(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args, layer_idx)
        self.is_moe_layer = (layer_idx % args.interleave_moe_layer_step) == (
            args.interleave_moe_layer_step - 1
        )
        if self.is_moe_layer:
            self.feed_forward = MoE(args)
        else:
            self.feed_forward = MLP(args, args.intermediate_size_mlp)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.post_attention_layernorm(h))
        out = h + r
        return out


class LlamaModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attention_chunk_size = args.attention_chunk_size

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if cache is not None:
            for idx, c in enumerate(cache):
                if (idx + 1) % 4 != 0:
                    c.maybe_trim_front()
            start = cache[0].start_position
            offset = cache[0].offset
        else:
            start = 0
            offset = 0
        end = offset + h.shape[1]
        linds = mx.arange(start, end)
        rinds = mx.arange(offset, end)[:, None]
        block_pos = mx.abs(
            (linds // self.attention_chunk_size) - (rinds // self.attention_chunk_size)
        )
        token_pos = linds <= rinds
        chunk_mask = (block_pos == 0) & token_pos

        if mask is None:
            mask = create_attention_mask(h, cache)
        else:
            chunk_mask &= mask

        if cache is None:
            cache = [None] * len(self.layers)

        for idx, (layer, c) in enumerate(zip(self.layers, cache)):
            use_chunked_attention = (idx + 1) % 4 != 0
            if use_chunked_attention:
                local_mask = chunk_mask
            else:
                local_mask = mask
            h = layer(h, local_mask, cache=c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LlamaModel(self.args)
        self.lm_head = nn.Linear(
            self.args.hidden_size, self.args.vocab_size, bias=False
        )

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)
        return self.lm_head(out)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = LanguageModel(args.text_config)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        return self.language_model(inputs, mask, cache)

    def sanitize(self, weights):
        def to_remove(k):
            return "vision_model" in k or "multi_modal_projector" in k

        # Remove vision weights
        weights = {k: v for k, v in weights.items() if not to_remove(k)}

        # Rename expert weights for SwitchGLU
        for l in range(self.args.text_config.num_hidden_layers):
            prefix = f"language_model.model.layers.{l}.feed_forward.experts"
            if f"{prefix}.gate_up_proj" in weights:
                v = weights.pop(f"{prefix}.gate_up_proj")
                gate_k = f"{prefix}.gate_proj.weight"
                up_k = f"{prefix}.up_proj.weight"
                gate_proj, up_proj = mx.split(v, 2, axis=-1)
                weights[gate_k] = mx.swapaxes(gate_proj, 1, 2)
                weights[up_k] = mx.swapaxes(up_proj, 1, 2)
            if f"{prefix}.down_proj" in weights:
                down_proj = weights.pop(f"{prefix}.down_proj")
                weights[f"{prefix}.down_proj.weight"] = mx.swapaxes(down_proj, 1, 2)
        return weights

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        chunk_size = self.args.text_config.attention_chunk_size
        caches = []
        for i in range(len(self.layers)):
            if (i + 1) % 4 != 0:
                caches.append(ChunkedKVCache(chunk_size))
            else:
                caches.append(KVCache())
        return caches
