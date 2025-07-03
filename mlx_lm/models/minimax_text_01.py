# Copyright © 2023-2024 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    attention_dropout: float
    attn_type_list: List[int]
    head_dim: int
    hidden_size: int
    intermediate_size: int
    layernorm_full_attention_alpha: float
    layernorm_full_attention_beta: float
    max_position_embeddings: int
    num_attention_heads: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_key_value_heads: int
    num_local_experts: int
    rms_norm_eps: float
    rope_theta: int
    rotary_dim: int
    router_aux_loss_coef: float
    shared_intermediate_size: int
    tie_word_embeddings: bool
    vocab_size: int
    layernorm_linear_attention_alpha: float = 1
    layernorm_linear_attention_beta: float = 1
    layernorm_mlp_alpha: float = 1
    layernorm_mlp_beta: float = 1
    mlp_bias: bool = False
    postnorm: bool = True


BLOCK: int = 256


class MiniMaxText01AttentionType0(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.head_dim = getattr(args, "head_dim", args.hidden_size // self.num_heads)
        self.hidden_size = args.hidden_size

        self.qkv_proj = nn.Linear(
            args.hidden_size, 3 * self.head_dim * self.num_heads, bias=args.mlp_bias
        )
        self.output_gate = nn.Linear(
            args.hidden_size, self.head_dim * self.num_heads, bias=args.mlp_bias
        )
        self.norm = nn.RMSNorm(self.head_dim * self.num_heads)
        self.out_proj = nn.Linear(
            self.head_dim * self.num_heads, args.hidden_size, bias=args.mlp_bias
        )

        self.offset = 0

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        slope_rate: Optional[mx.array] = None,
    ):
        b, n, d = x.shape

        # linear map
        qkv = nn.silu(self.qkv_proj(x))
        new_shape = qkv.shape[:-1] + (self.num_heads, -1)
        qkv = qkv.reshape(*new_shape)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        if cache is not None:
            keys, values = cache.update_and_fetch(k, v)

            output_parts = []
            for i in range(n):
                q_i = q[:, :, i : i + 1]

                pos = cache.offset - n + i
                if pos > 0:
                    k_past = keys[:, :, :pos]
                    v_past = values[:, :, :pos]

                    attn_weights = mx.matmul(q_i, mx.transpose(k_past, (0, 1, 3, 2)))
                    attn_output = mx.matmul(attn_weights, v_past)

                    output_parts.append(attn_output)
                else:
                    output_parts.append(
                        mx.zeros((b, self.num_heads, 1, self.head_dim), dtype=q.dtype)
                    )

            output = mx.concatenate(output_parts, axis=2)
            output = output.transpose(0, 2, 1, 3).reshape(b, n, -1)

        else:
            slope_rate = slope_rate.astype(mx.float32)

            NUM_BLOCK = (n + BLOCK - 1) // BLOCK

            array = mx.arange(BLOCK) + 1
            q_decay = mx.exp(-slope_rate * array.reshape(-1, 1))
            k_decay = mx.exp(-slope_rate * (BLOCK - array.reshape(-1, 1)))

            index = array[:, None] - array[None, :]
            s_index = slope_rate * index[None, None, :]
            s_index = mx.where(index >= 0, -s_index, float("-inf"))
            diag_decay = mx.exp(s_index)

            kv = mx.zeros(
                (b, self.num_heads, self.head_dim, self.head_dim), dtype=mx.float32
            )
            output = mx.zeros((b, self.num_heads, n, self.head_dim), dtype=q.dtype)

            for i in range(NUM_BLOCK):
                si = i * BLOCK
                ei = min(si + BLOCK, n)
                m = ei - si

                qi = q[:, :, si:ei]
                ki = k[:, :, si:ei]
                vi = v[:, :, si:ei]

                qkv_none_diag = mx.matmul(qi * q_decay[:, :m], kv).astype(mx.float32)

                qk = (
                    mx.matmul(qi, mx.transpose(ki, (0, 1, 3, 2))).astype(mx.float32)
                    * diag_decay[:, :, :m, :m]
                )
                qkv_diag = mx.matmul(qk, vi.astype(mx.float32))
                block_decay = mx.exp(-slope_rate * m)

                output_slice = qkv_none_diag + qkv_diag
                output = mx.concatenate(
                    [output[:, :, :si], output_slice, output[:, :, ei:]], axis=2
                )

                kv = block_decay * kv + mx.matmul(
                    mx.transpose(ki * k_decay[:, -m:], (0, 1, 3, 2)).astype(vi.dtype),
                    vi,
                )

            output = output.transpose(0, 2, 1, 3).reshape(b, n, -1)

        output = self.norm(output)
        output = mx.sigmoid(self.output_gate(x)) * output

        return self.out_proj(output)


class MiniMaxText01AttentionType1(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads

        self.head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.rotary_emb = initialize_rope(
            dims=args.rotary_dim or self.head_dim,
            traditional=True,
            base=args.rope_theta,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        slope_rate: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rotary_emb(queries, offset=cache.offset)
            keys = self.rotary_emb(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rotary_emb(queries)
            keys = self.rotary_emb(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MiniMaxText01SharedMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=args.mlp_bias
        )
        self.down_proj = nn.Linear(
            args.intermediate_size, args.hidden_size, bias=args.mlp_bias
        )
        self.up_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=args.mlp_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniMaxText01SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_local_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(args.hidden_size, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            input_dims=args.hidden_size,
            hidden_dims=args.intermediate_size,
            num_experts=self.num_experts,
            bias=args.mlp_bias if hasattr(args, "mlp_bias") else False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        router_logits = self.gate(x_flat)
        routing_weights = mx.softmax(router_logits, axis=1, precise=True)
        indices = mx.stop_gradient(
            mx.argpartition(-routing_weights, kth=self.top_k - 1, axis=-1)[
                ..., : self.top_k
            ]
        )
        scores = mx.take_along_axis(routing_weights, indices, axis=-1)
        y = self.switch_mlp(x_flat, indices)
        y = (y * mx.expand_dims(scores, axis=-1)).sum(axis=1)
        return y.reshape(B, L, D)


class MiniMaxText01DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, attention_type: Literal[0, 1] = 0):
        super().__init__()
        if attention_type == 0:
            self.self_attn = MiniMaxText01AttentionType0(args)
        else:
            self.self_attn = MiniMaxText01AttentionType1(args)

        self.block_sparse_moe = MiniMaxText01SparseMoeBlock(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        self.postnorm = args.postnorm
        self.layernorm_attention_alpha = args.layernorm_linear_attention_alpha
        self.layernorm_attention_beta = args.layernorm_linear_attention_beta
        self.layernorm_mlp_alpha = args.layernorm_mlp_alpha
        self.layernorm_mlp_beta = args.layernorm_mlp_beta

        shared_intermediate = args.shared_intermediate_size
        self.shared_moe = False
        if shared_intermediate > 0:
            self.shared_moe = True
            self.shared_mlp = MiniMaxText01SharedMLP(args)
            self.coefficient = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        slope_rate: Optional[mx.array] = None,
    ) -> mx.array:
        r = x
        h = self.input_layernorm(x)
        if self.postnorm:
            h = h
        attn_output = self.self_attn(x=h, mask=mask, cache=cache, slope_rate=slope_rate)
        h = (
            r * self.layernorm_attention_alpha
            + attn_output * self.layernorm_attention_beta
        )
        r = h
        h = self.post_attention_layernorm(h)
        if self.postnorm:
            r = h
        moe_h = self.block_sparse_moe(h)
        if self.shared_moe:
            output_mlp = self.shared_mlp(h)
            coef = nn.sigmoid(h @ self.coefficient)
            h = moe_h * (1 - coef) + output_mlp * coef
        else:
            h = moe_h
        return r * self.layernorm_mlp_alpha + h * self.layernorm_mlp_beta


class MiniMaxText01Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = [
            MiniMaxText01DecoderLayer(args=args, attention_type=args.attn_type_list[i])
            for i in range(args.num_hidden_layers)
        ]

        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.slopes = self._build_slope_tensor(args.num_attention_heads)

    def _build_slope_tensor(self, n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        return mx.array(get_slopes(n_attention_heads)).reshape(n_attention_heads, 1, 1)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        slope_rates = [self.slopes for _ in range(len(self.layers))]

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            sr = slope_rates[i] * (1 - i / (len(self.layers) - 1) + 1e-5)
            h = layer(h, mask=mask, cache=c, slope_rate=sr)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniMaxText01Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs=inputs, mask=mask, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if "model.slopes" not in weights:
            slopes = self.model._build_slope_tensor(self.args.num_attention_heads)
            weights["model.slopes"] = slopes

        if "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in weights:
            return weights
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
            for orig_name, new_name in mapping.items():
                if f"{prefix}.block_sparse_moe.experts.0.{orig_name}.weight" in weights:
                    to_join = [
                        weights.pop(
                            f"{prefix}.block_sparse_moe.experts.{e}.{orig_name}.weight"
                        )
                        for e in range(self.args.num_local_experts)
                    ]
                    weights[
                        f"{prefix}.block_sparse_moe.switch_mlp.{new_name}.weight"
                    ] = mx.stack(to_join)
        return weights

    @property
    def layers(self):
        return self.model.layers


# Goekdeniz-Guelmez/MiniMax01Text-Dev