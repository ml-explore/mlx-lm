# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import KVCache, MambaCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    attn_layer_offset: int
    attn_layer_period: int
    expert_layer_offset: int
    expert_layer_period: int
    mamba_d_conv: int
    mamba_d_state: int
    mamba_expand: int
    num_experts: int
    num_experts_per_tok: int
    rms_norm_eps: float
    max_position_embeddings: int
    vocab_size: int
    mamba_dt_rank: Union[str, int] = "auto"
    mamba_proj_bias: bool = False
    mamba_conv_bias: bool = True
    layers_block_type: Optional[List[str]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.mamba_dt_rank == "auto":
            self.mamba_dt_rank = math.ceil(self.hidden_size / 16)
        if self.layers_block_type is None:
            self.layers_block_type = [
                (
                    "attention"
                    if i % self.attn_layer_period == self.attn_layer_offset
                    else "mamba"
                )
                for i in range(self.num_hidden_layers)
            ]


class JambaMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class JambaAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


@mx.compile
def fma(a, b, c):
    return a * b + c


class JambaMambaMixer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.mamba_d_state
        self.conv_kernel_size = args.mamba_d_conv
        self.intermediate_size = args.mamba_expand * args.hidden_size
        self.time_step_rank = args.mamba_dt_rank
        self.use_conv_bias = args.mamba_conv_bias
        self.use_bias = args.mamba_proj_bias

        self.in_proj = nn.Linear(
            self.hidden_size, self.intermediate_size * 2, bias=self.use_bias
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            bias=self.use_conv_bias,
            padding=0,
        )
        self.x_proj = nn.Linear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        A = mx.repeat(
            mx.arange(1.0, self.ssm_state_size + 1.0).reshape([1, self.ssm_state_size]),
            repeats=self.intermediate_size,
            axis=0,
        )
        self.A_log = mx.log(A)
        self.D = mx.ones([self.intermediate_size])

        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=self.use_bias
        )

        self.dt_layernorm = nn.RMSNorm(self.time_step_rank, eps=args.rms_norm_eps)
        self.b_layernorm = nn.RMSNorm(self.ssm_state_size, eps=args.rms_norm_eps)
        self.c_layernorm = nn.RMSNorm(self.ssm_state_size, eps=args.rms_norm_eps)

    def ssm_step(self, x, A, state=None):
        T = x.shape[1]
        D = self.D
        deltaBC = self.x_proj(x)
        delta, B, C = mx.split(
            deltaBC,
            [self.time_step_rank, self.time_step_rank + self.ssm_state_size],
            axis=-1,
        )
        delta, B, C = self.dt_layernorm(delta), self.b_layernorm(B), self.c_layernorm(C)
        delta = nn.softplus(self.dt_proj(delta))
        new_state = mx.expand_dims(delta * x, -1) * mx.expand_dims(B, -2)
        dtA = mx.exp(mx.expand_dims(delta, -1) * A)

        # TODO, speed up prefill with chunked scan
        for t in range(T):
            if state is not None:
                new_state[:, t] = fma(state, dtA[:, t], new_state[:, t])
            state = new_state[:, t]
        y = (new_state @ mx.expand_dims(C, -1)).squeeze(-1)
        y = y + D * x
        return y, new_state[:, -1]

    def _process_sequence(self, x, conv_state, ssm_state):
        xz = self.in_proj(x)
        x, z = xz.split(indices_or_sections=2, axis=-1)
        K = self.conv_kernel_size
        if conv_state is not None:
            x_full = mx.concatenate([conv_state, x], axis=1)
        else:
            x_full = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)])
        conv_out = self.conv1d(x_full)
        conv_state = x_full[:, -(K - 1) :, :]
        x = nn.silu(conv_out)
        A = -mx.exp(self.A_log)
        y, ssm_state = self.ssm_step(x, A, ssm_state)
        z = self.out_proj(nn.silu(z) * y)
        return z, (conv_state, ssm_state)

    def __call__(self, x, cache):
        if cache is None:
            conv_state, ssm_state = None, None
        else:
            conv_state, ssm_state = cache[0], cache[1]

        output, (conv_state, ssm_state) = self._process_sequence(
            x, conv_state, ssm_state
        )

        if cache is not None:
            cache[0] = conv_state
            cache[1] = ssm_state

        return output


class JambaSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok

        self.router = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.intermediate_size, args.num_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.router(x)
        k = self.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y


class JambaDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str, layer_idx: int):
        super().__init__()
        self.is_attn = layer_type == "attention"
        if self.is_attn:
            self.self_attn = JambaAttention(args)
        else:
            self.mamba = JambaMambaMixer(args)
        if (
            args.num_experts > 1
            and (layer_idx + args.expert_layer_offset) % args.expert_layer_period == 0
        ):
            ffn_layer_class = JambaSparseMoeBlock
        else:
            ffn_layer_class = JambaMLP
        self.feed_forward = ffn_layer_class(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_ff_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_attn:
            h = self.self_attn(self.input_layernorm(x), mask, cache)
        else:
            h = self.mamba(self.input_layernorm(x), cache)
        r = x + h
        out = r + self.feed_forward(self.pre_ff_layernorm(r))
        return out


class JambaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = [
            JambaDecoderLayer(args, t, idx)
            for idx, t in enumerate(args.layers_block_type)
        ]
        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn_idx = args.layers_block_type.index("attention")
        self.ssm_idx = args.layers_block_type.index("mamba")

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        attn_mask = create_attention_mask(h, cache[self.attn_idx])
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            mask = attn_mask if layer.is_attn else ssm_mask
            h = layer(h, mask=mask, cache=c)

        return self.final_layernorm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.args = args
        self.model = JambaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def make_cache(self):
        caches = []
        for layer in self.model.layers:
            if layer.is_attn:
                caches.append(KVCache())
            else:
                caches.append(MambaCache())
        return caches

    def sanitize(self, weights):
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for l in range(self.args.num_hidden_layers):
            base = f"model.layers.{l}.feed_forward"
            if not any(key.startswith(f"{base}.experts.") for key in weights.keys()):
                continue

            for proj in ["gate_proj", "down_proj", "up_proj"]:
                for name in ["weight", "bias", "scales", "biases"]:
                    expert_tensors = [
                        weights.pop(f"{base}.experts.{e}.{proj}.{name}")
                        for e in range(len(weights))
                        if f"{base}.experts.{e}.{proj}.{name}" in weights
                    ]
                    if expert_tensors:
                        weights[f"{base}.switch_mlp.{proj}.{name}"] = mx.stack(
                            expert_tensors
                        )

        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("router"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
