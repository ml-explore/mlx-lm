from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import ChunkedKVCache, KVCache
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    vocab_size: int
    intermediate_size: int
    intermediate_size_mlp: Optional[int]
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    attn_scale: float
    floor_scale: int
    use_qk_norm: bool
    attention_bias: bool = False
    head_dim: Optional[int] = None
    no_rope_layers: Optional[List[int]] = None
    layer_types: Optional[List[str]] = None
    attention_chunk_size: Optional[int] = None
    attn_temperature_tuning: Optional[bool] = False
    tie_word_embeddings: Optional[bool] = True
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    moe_layers: Optional[List[int]] = None


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.use_qk_norm = args.use_qk_norm
        self.n_heads = args.num_attention_heads
        disable_rope = False
        nrl = args.no_rope_layers
        if isinstance(nrl, list) and len(nrl) > 0:
            flags = [bool(int(v)) for v in nrl]
            if any(flags) and not all(flags):
                if layer_idx < len(flags):
                    disable_rope = flags[layer_idx]
        self.use_rope = not disable_rope
        self.floor_scale = args.floor_scale
        self.attn_scale = args.attn_scale
        self.attn_temperature_tuning = args.attn_temperature_tuning
        self.n_kv_heads = (
            args.num_key_value_heads
            if args.num_key_value_heads > 0
            else args.num_attention_heads
        )
        self.head_dim = (
            args.head_dim
            if getattr(args, "head_dim", None) is not None
            else (args.hidden_size // self.n_heads)
        )
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=args.attention_bias
        )
        self.use_qk_norm = args.use_qk_norm and self.use_rope
        self.rms_norm_eps = args.rms_norm_eps

        if self.use_rope:
            self.rope = initialize_rope(
                self.head_dim,
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

        if self.use_qk_norm:
            queries = mx.fast.rms_norm(queries, weight=None, eps=self.rms_norm_eps)
            keys = mx.fast.rms_norm(keys, weight=None, eps=self.rms_norm_eps)

        if self.use_rope:
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

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


class SwiGLUMLP(nn.Module):
    def __init__(self, dim, intermediate_size, activation=nn.silu):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DualMLP(nn.Module):
    def __init__(self, dim, intermediate_gated, intermediate_plain, activation=nn.silu):
        super().__init__()
        self.g_up = nn.Linear(dim, intermediate_gated, bias=False)
        self.g_gate = nn.Linear(dim, intermediate_gated, bias=False)
        self.g_down = nn.Linear(intermediate_gated, dim, bias=False)
        self.p_up = nn.Linear(dim, intermediate_plain, bias=False)
        self.p_down = nn.Linear(intermediate_plain, dim, bias=False)

    def __call__(self, x):
        gated_out = self.g_down(nn.silu(self.g_gate(x)) * self.g_up(x))
        plain_out = self.p_down(nn.silu(self.p_up(x)))
        return gated_out + plain_out


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.layer_idx = layer_idx

        if getattr(args, "use_dual_mlp", False):
            self.feed_forward = DualMLP(
                args.hidden_size,
                args.intermediate_size,
                args.intermediate_size_mlp or args.intermediate_size,
            )
        else:
            self.feed_forward = SwiGLUMLP(
                args.hidden_size,
                args.intermediate_size_mlp,
            )

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
        r = self.feed_forward(self.post_attention_layernorm(h))
        out = h + r
        return out


class Llama4TextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attention_chunk_size = args.attention_chunk_size

    def __call__(
        self,
        inputs: mx.array,
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

        if cache is None:
            cache = [None] * len(self.layers)

        global_mask = create_attention_mask(h, cache[3])

        for idx, (layer, c) in enumerate(zip(self.layers, cache)):
            use_chunked_attention = (idx + 1) % 4 != 0
            mask = chunk_mask if use_chunked_attention else global_mask
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Llama4TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        chunk_size = self.args.attention_chunk_size
        caches = []
        for i in range(len(self.layers)):
            if (i + 1) % 4 != 0:
                caches.append(ChunkedKVCache(chunk_size))
            else:
                caches.append(KVCache())
        return caches
