from dataclasses import dataclass
from typing import Optional, Any
import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    vocab_size: int = 65536
    hidden_size: int = 2560
    num_hidden_layers: int = 32
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_embedding: bool = True
    theta: float = 1000000.0
    max_position_embeddings: int = 128_000
    use_cache: bool = True
    norm_eps: float = 0.00001
    initializer_range: float = 0.02
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    conv_bias: bool = False
    conv_dim: int = 2560
    conv_L_cache: int = 3
    block_dim: int = 2560
    block_ff_dim: int = 12288
    block_multiple_of: int = 256
    block_ffn_dim_multiplier: float = 1.0
    block_auto_adjust_ff_dim: bool = True
    full_attn_idxs: Optional[list[int]] = None



class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.q_layernorm = nn.RMSNorm(dim, eps=args.norm_eps)
        self.k_layernorm = nn.RMSNorm(dim, eps=args.norm_eps)

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

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

        # Prepare the queries, keys and values for the attention computation
        queries = self.q_layernorm(queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3))
        keys = self.k_layernorm(keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3))
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


class LFM2ShortConv(nn.Module):
    def __init__(
        self,
        config: LFM2Config,
        dim: int,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias

        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=self.L_cache,
            groups=dim,
            bias=self.bias,
            padding=self.L_cache - 1,
        )
        self.in_proj = nn.Linear(dim, 3 * dim, bias=self.bias)
        self.out_proj = nn.Linear(dim, dim, bias=self.bias)


    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
    ):
        seqlen = x.shape[1]
        BCx = self.in_proj(x).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)

        Bx = B * x

        if cache is not None and cache.conv_cache[self.layer_idx].shape[-2] > 0:
            conv_state = cache.conv_cache[self.layer_idx]
            cache_position = mx.clip(cache.conv_position, 0, self.L_cache - 1)
            conv_state = mx.roll(conv_state, shift=-1, axis=-1)
            conv_state[:, :, cache_position] = Bx
            cache.conv_cache[self.layer_idx] = conv_state
            conv_out = mx.sum(conv_state * self.conv.weight[:, 0, :], axis=-1)
            if self.bias:
                conv_out += self.conv.bias

            conv_out = conv_out[:, :, None]
        else:
            if cache is not None:
                conv_state = mx.pad(Bx, (self.L_cache - Bx.shape[-1], 0))
                cache.conv_cache[self.layer_idx] = conv_state

            conv_out = self.conv(Bx)[..., :seqlen]

        y = C * conv_out
        y = y.transpose(-1, -2)
        y = self.out_proj(y)
        return y


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_dim: int,
        multiple_of: int,
        auto_adjust_ff_dim: bool,
        ffn_dim_multiplier: Optional[float],
    ):
        if auto_adjust_ff_dim:
            ff_dim = int(2 * ff_dim / 3)
            if ffn_dim_multiplier is not None:
                ff_dim = int(ffn_dim_multiplier * ff_dim)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w3 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))

class LFM2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args