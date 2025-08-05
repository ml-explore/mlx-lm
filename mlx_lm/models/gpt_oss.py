# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    num_hidden_layers: int = 36
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    vocab_size: int = 201088
    rms_norm_eps: float = 1e-05
    hidden_size: int = 2880
    intermediate_size: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    rope_theta: int = 150000
    rope_scaling: Any = None


# These operators emulate particular methods in torch that don't exist in MLX natively
def mlx_topk(a, k, axis=-1):
    """MLX equivalent of torch.topk"""
    partitioned_indices = mx.argpartition(a, kth=-k, axis=axis)
    # Extract only the top k indices (last k elements after partition)
    top_k_indices = partitioned_indices[..., -k:]
    # Get the corresponding values
    top_k_values = mx.take_along_axis(a, top_k_indices, axis=axis)
    return top_k_values, top_k_indices


@partial(mx.compile, shapeless=True)
def swiglu(x_linear, x_glu, alpha: float = 1.702, limit: float = 7.0):
    # Clamp the input values
    x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
    x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
    glu_scaled = (alpha * x_glu.astype(mx.float32)).astype(mx.bfloat16)
    negative_glu = (-glu_scaled).astype(mx.float32)
    sig = (1.0 / (1.0 + mx.exp(negative_glu))).astype(mx.bfloat16)

    out_glu = x_glu * sig
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(x, gate)


# ref. eager_attention_forward in tfm impl
def sdpa(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    S: mx.array,
    sm_scale: float,
    mask: mx.array,
):
    # Q, K, V shapes: (batch, num_heads, seqlen, head_dim)
    batch, num_kv_heads, seqlen, head_dim = K.shape
    _, num_q_heads, q_len, _ = Q.shape

    n_rep = num_q_heads // num_kv_heads
    Q = Q.reshape(batch, num_kv_heads, n_rep, q_len, head_dim)
    attn_weights = sm_scale * mx.matmul(Q, mx.expand_dims(K, axis=2).swapaxes(-1, -2))
    attn_weights = attn_weights.reshape(batch, head_dim, q_len, seqlen)

    if mask.shape[-1] != K.shape[-2]:
        mask = mask[..., -K.shape[-2] :]
    attn_weights = mx.where(mask, attn_weights, -mx.inf)

    sinks = mx.tile(S.reshape(1, -1, 1, 1), [batch, 1, q_len, 1])

    combined_logits = mx.concatenate([attn_weights, sinks], axis=-1)
    probs = mx.softmax(combined_logits, axis=-1, precise=True)
    scores = probs[..., :-1].reshape(batch, num_kv_heads, n_rep, q_len, seqlen)
    attn_output = mx.matmul(scores, mx.expand_dims(V, axis=2))
    attn_output = attn_output.reshape(batch, num_q_heads, q_len, head_dim).swapaxes(
        1, 2
    )

    return attn_output


class AttentionBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )

        self.sinks = mx.zeros((config.num_attention_heads,))

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )

        self.o_proj = nn.Linear(
            self.head_dim * config.num_attention_heads, config.hidden_size, bias=True
        )

        self.sm_scale = 1 / math.sqrt(config.head_dim)

        self.rope = initialize_rope(
            self.head_dim,
            config.rope_theta,
            traditional=False,
            scaling_config=config.rope_scaling,
        )

    def __call__(self, x: mx.array, mask: mx.array, cache=None) -> mx.array:
        input_shape = x.shape[:-1]  # (batch, seqlen)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (batch, seqlen, num_heads * head_dim) -> (batch, num_heads, seqlen, head_dim)
        q = q.reshape(*input_shape, self.num_attention_heads, self.head_dim).swapaxes(
            1, 2
        )
        k = k.reshape(*input_shape, self.num_key_value_heads, self.head_dim).swapaxes(
            1, 2
        )
        v = v.reshape(*input_shape, self.num_key_value_heads, self.head_dim).swapaxes(
            1, 2
        )

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        attn_output = sdpa(q, k, v, self.sinks, self.sm_scale, mask=mask)

        # Reshape back to original format: (batch, seqlen, hidden_size)
        attn_output = attn_output.reshape(*input_shape, -1)
        out = self.o_proj(attn_output)
        return out


class MLPBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_local_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.experts = SwitchGLU(
            input_dims=config.hidden_size,
            hidden_dims=config.intermediate_size,
            num_experts=config.num_local_experts,
            activation=SwiGLU(),
            bias=True,
        )
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.reshape(-1, self.hidden_size)

        # N.B. As elsewhere, upcast is required in linear layers
        g = self.router(x.astype(mx.float32)).astype(mx.bfloat16)
        experts, indices = mlx_topk(g, k=self.num_experts_per_tok, axis=-1)
        expert_weights = mx.softmax(experts, axis=-1, precise=True)

        # Experts block
        x = self.experts(x, indices)

        x = x * mx.expand_dims(expert_weights, axis=2)
        return x.sum(axis=1)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self_attn = AttentionBlock(config)
        self.mlp = MLPBlock(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask: mx.array, cache=None) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask, cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class GptOssMoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)

        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            x = input_embeddings
        else:
            x = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            full_mask = create_attention_mask(x, cache[1:2], return_array=True)
            sliding_window_mask = create_attention_mask(x, cache, return_array=True)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            local_mask = mask
            if mask is None and (i % 2 == 1):
                local_mask = full_mask
            elif mask is None:
                local_mask = sliding_window_mask
            if local_mask is None:
                local_mask = mx.array([True], dtype=mx.bool_)

            x = layer(x, local_mask, c)
        x = self.norm(x)
        return x


def convert_moe_packed_tensors(blocks, scales):
    assert (
        blocks.shape[:-1] == scales.shape
    ), f"{blocks.shape=} does not match {scales.shape=}"

    scales = scales.astype(mx.int32) - 127
    lut = mx.array(
        [
            +0.0,
            +0.5,
            +1.0,
            +1.5,
            +2.0,
            +3.0,
            +4.0,
            +6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=mx.bfloat16,
    )

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    idx_lo = blocks & 0x0F
    idx_hi = blocks >> 4

    out = mx.stack((lut[idx_lo], lut[idx_hi]), axis=-1).flatten(-2)
    out = (2.0**scales) * out
    out = out.reshape(*prefix_shape, G * B * 2)
    return out.astype(mx.bfloat16)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = (
            args.model_type if hasattr(args, "model_type") else "gpt_oss_moe"
        )
        self.model = GptOssMoeModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, mask: mx.array = None, cache=None):
        return self.lm_head(self.model(inputs, mask, cache))

    def sanitize(self, weights):
        if any("gate_proj.weight" in k for k in weights.keys()):
            return weights  # already sanitized

        # Needs dequantization
        if any("gate_up_proj_scales" in k for k in weights.keys()):
            new_weights = {}
            for k, v in weights.items():
                if k.endswith("_scales"):
                    pass
                elif k.endswith("_blocks"):
                    scale_k = k.replace("_blocks", "_scales")
                    scales = weights[scale_k]
                    new_v = convert_moe_packed_tensors(v, scales)
                    new_k = k.replace("_blocks", "")
                    new_weights[new_k] = new_v
                else:
                    new_weights[k] = v
            weights = new_weights

        new_weights = {}
        for k, v in weights.items():
            if "gate_up_proj" in k and "bias" not in k:
                new_weights[k.replace("gate_up_proj", "gate_proj.weight")] = v[
                    ..., ::2, :
                ]
                new_weights[k.replace("gate_up_proj", "up_proj.weight")] = v[
                    ..., 1::2, :
                ]
            elif "down_proj" in k and "bias" not in k:
                new_weights[k.replace("down_proj", "down_proj.weight")] = v
            elif "gate_up_proj_bias" in k:
                new_weights[k.replace("gate_up_proj_bias", "gate_proj.bias")] = v[
                    ..., ::2
                ]
                new_weights[k.replace("gate_up_proj_bias", "up_proj.bias")] = v[
                    ..., 1::2
                ]
            elif "down_proj_bias" in k:
                new_weights[k.replace("down_proj_bias", "down_proj.bias")] = v
            else:
                new_weights[k] = v
        return new_weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for i in range(self.args.num_hidden_layers):
            # full attn on odd indices, swa on even
            if i % 2 == 1:
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                )
        return caches
