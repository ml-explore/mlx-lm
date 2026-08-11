from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_kernel, gated_delta_ops
from .rope_utils import initialize_rope
from .switch_layers import QuantizedSwitchLinear, SwitchLinear


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "bailing_hybrid"
    architectures: list[str] | None = None
    vocab_size: int = 157184
    hidden_size: int = 1536
    intermediate_size: int = 4608
    moe_intermediate_size: int = 512
    moe_shared_expert_intermediate_size: int = 512
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    first_k_dense_replace: int = 1
    num_nextn_predict_layers: int = 0
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    layer_group_size: int = 4
    head_dim: int = 128
    q_lora_rank: int | None = 256
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    qk_head_dim: int = 192
    v_head_dim: int = 128
    short_conv_kernel_size: int = 4
    kda_lower_bound: float = -5.0
    kda_safe_gate: bool = True
    no_kda_lora: bool = True
    rms_norm_eps: float = 1e-6
    rope_theta: float = 6000000.0
    rope_scaling: dict[str, Any] | None = None
    max_position_embeddings: int = 131072
    rope_interleave: bool = True
    tie_word_embeddings: bool = False
    gated_attention_proj_granularity_type: str | None = None
    quantization_config: dict[str, Any] | None = None


def _is_kda_layer(
    layer_index: int, layer_group_size: int, num_hidden_layers: int
) -> bool:
    return not (
        (layer_index + 1) % layer_group_size == 0
        or layer_index >= num_hidden_layers // layer_group_size * layer_group_size
    )


def _is_mtp_weight(key: str, num_hidden_layers: int) -> bool:
    prefix = "model.layers."
    if not key.startswith(prefix):
        return False
    layer_index = key[len(prefix) :].split(".", 1)[0]
    return layer_index.isdigit() and int(layer_index) >= num_hidden_layers


def _uses_fp8(args: ModelArgs) -> bool:
    config = args.quantization_config
    return config is not None and config.get("quant_method") == "fp8"


def _normalize_kda_qk(
    q: mx.array, k: mx.array, head_dim: int
) -> tuple[mx.array, mx.array]:
    inv_scale = head_dim**-0.5
    eps = 1e-6 / head_dim
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, eps)
    k = inv_scale * mx.fast.rms_norm(k, None, eps)
    return q, k


def _linear(
    args: ModelArgs, input_dims: int, output_dims: int
) -> nn.Linear | nn.QuantizedLinear:
    if _uses_fp8(args):
        return nn.QuantizedLinear(
            input_dims,
            output_dims,
            bias=False,
            group_size=32,
            bits=8,
            mode="mxfp8",
        )
    return nn.Linear(input_dims, output_dims, bias=False)


def _switch_linear(
    args: ModelArgs, input_dims: int, output_dims: int
) -> SwitchLinear | QuantizedSwitchLinear:
    if _uses_fp8(args):
        return QuantizedSwitchLinear(
            input_dims,
            output_dims,
            args.num_experts,
            bias=False,
            group_size=32,
            bits=8,
            mode="mxfp8",
        )
    return SwitchLinear(input_dims, output_dims, args.num_experts, bias=False)


class BailingMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        self.gate_proj = _linear(args, args.hidden_size, intermediate_size)
        self.up_proj = _linear(args, args.hidden_size, intermediate_size)
        self.down_proj = _linear(args, intermediate_size, args.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class BailingSwitchGLU(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.moe_intermediate_size
        self.gate_proj = _switch_linear(args, dim, hidden_dim)
        self.up_proj = _switch_linear(args, dim, hidden_dim)
        self.down_proj = _switch_linear(args, hidden_dim, dim)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        up = self.up_proj(x, indices)
        gate = self.gate_proj(x, indices)
        x = self.down_proj(swiglu(gate, up), indices)
        return x.squeeze(-2)


def _group_expert_select(
    logits: mx.array,
    expert_bias: mx.array,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
) -> tuple[mx.array, mx.array]:
    scores = mx.sigmoid(logits.astype(mx.float32))
    routing_scores = scores + expert_bias
    routing_scores = mx.unflatten(routing_scores, axis=-1, shape=(n_group, -1))
    group_scores = mx.topk(routing_scores, 2, axis=-1).sum(axis=-1, keepdims=True)
    groups_to_mask = n_group - topk_group
    group_idx = mx.argpartition(group_scores, kth=groups_to_mask - 1, axis=-2)[
        ..., :groups_to_mask, :
    ]
    routing_scores = mx.put_along_axis(
        routing_scores,
        mx.stop_gradient(group_idx),
        mx.array(float("-inf")),
        axis=-2,
    )
    routing_scores = mx.flatten(routing_scores, -2, -1)
    indices = mx.argpartition(-routing_scores, kth=top_k - 1, axis=-1)[..., :top_k]
    selected = mx.take_along_axis(scores, indices, axis=-1)
    selected = selected / (selected.sum(axis=-1, keepdims=True) + 1e-20)
    return indices, selected * routed_scaling_factor


class BailingGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.expert_bias = mx.zeros((args.num_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        return _group_expert_select(
            logits,
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
        )


class BailingSparseMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = BailingGate(args)
        self.switch_mlp = BailingSwitchGLU(args)
        shared_size = args.moe_shared_expert_intermediate_size * args.num_shared_experts
        self.shared_experts = BailingMLP(args, shared_size)

    def __call__(self, x: mx.array) -> mx.array:
        indices, scores = self.gate(x)
        routed = self.switch_mlp(x, indices)
        routed = (routed * scores[..., None]).sum(axis=-2).astype(x.dtype)
        return routed + self.shared_experts(x)


class BailingMLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim
        self.scale = self.qk_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                args.hidden_size,
                args.num_attention_heads * args.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(args.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                args.num_attention_heads * args.qk_head_dim,
                bias=False,
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            args.kv_lora_rank + args.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            args.kv_lora_rank,
            args.num_attention_heads * (args.qk_nope_head_dim + args.v_head_dim),
            bias=False,
        )
        self.gated_attention_proj_granularity_type = (
            args.gated_attention_proj_granularity_type
        )
        if self.gated_attention_proj_granularity_type is None:
            self.g_proj = None
        elif self.gated_attention_proj_granularity_type == "head_wise":
            self.g_proj = nn.Linear(
                args.hidden_size, args.num_attention_heads, bias=False
            )
        elif self.gated_attention_proj_granularity_type == "element_wise":
            self.g_proj = nn.Linear(
                args.hidden_size,
                args.num_attention_heads * args.v_head_dim,
                bias=False,
            )
        else:
            raise ValueError(
                "Unsupported gated_attention_proj_granularity_type: "
                f"{self.gated_attention_proj_granularity_type!r}"
            )
        self.dense = _linear(
            args,
            args.num_attention_heads * args.v_head_dim,
            args.hidden_size,
        )
        if not args.rope_interleave:
            raise ValueError("Bailing V3 MLX currently requires rope_interleave=true")
        self.rope = initialize_rope(
            args.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(batch, length, self.num_heads, self.qk_head_dim).transpose(
            0, 2, 1, 3
        )
        q_nope, q_rope = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed = self.kv_a_proj_with_mqa(x)
        compressed, k_rope = mx.split(compressed, [self.kv_lora_rank], axis=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed))
        kv = kv.reshape(
            batch,
            length,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ).transpose(0, 2, 1, 3)
        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)
        k_rope = k_rope.reshape(batch, length, 1, self.qk_rope_head_dim).transpose(
            0, 2, 1, 3
        )

        offset = cache.offset if cache is not None else 0
        q_rope = self.rope(q_rope, offset=offset)
        k_rope = self.rope(k_rope, offset=offset)
        k_rope = mx.broadcast_to(
            k_rope, (batch, self.num_heads, length, self.qk_rope_head_dim)
        )
        queries = mx.concatenate([q_nope, q_rope], axis=-1)
        keys = mx.concatenate([k_nope, k_rope], axis=-1)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3)
        if self.g_proj is not None:
            gate = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
            if self.gated_attention_proj_granularity_type == "head_wise":
                output = output * gate[..., None]
            else:
                output = output * gate.reshape(output.shape)
        return self.dense(output.reshape(batch, length, -1))


class BailingKDA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        if not args.no_kda_lora:
            raise ValueError("Bailing V3 MLX currently requires no_kda_lora=true")
        if not args.kda_safe_gate:
            raise ValueError("Bailing V3 MLX currently requires kda_safe_gate=true")
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.projection_size = self.num_heads * self.head_dim
        self.conv_kernel_size = args.short_conv_kernel_size
        self.lower_bound = args.kda_lower_bound
        self.rms_norm_eps = args.rms_norm_eps

        self.q_proj = _linear(args, args.hidden_size, self.projection_size)
        self.k_proj = _linear(args, args.hidden_size, self.projection_size)
        self.v_proj = _linear(args, args.hidden_size, self.projection_size)
        self.q_conv1d = nn.Conv1d(
            self.projection_size,
            self.projection_size,
            self.conv_kernel_size,
            groups=self.projection_size,
            bias=False,
            padding=0,
        )
        self.k_conv1d = nn.Conv1d(
            self.projection_size,
            self.projection_size,
            self.conv_kernel_size,
            groups=self.projection_size,
            bias=False,
            padding=0,
        )
        self.v_conv1d = nn.Conv1d(
            self.projection_size,
            self.projection_size,
            self.conv_kernel_size,
            groups=self.projection_size,
            bias=False,
            padding=0,
        )
        self.A_log = mx.zeros((self.num_heads,), dtype=mx.float32)
        self.f_proj = _linear(args, args.hidden_size, self.projection_size)
        self.dt_bias = mx.zeros((self.projection_size,), dtype=mx.float32)
        self.b_proj = nn.Linear(args.hidden_size, self.num_heads, bias=False)
        self.g_proj = _linear(args, args.hidden_size, self.projection_size)
        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = _linear(args, self.projection_size, args.hidden_size)

    def _causal_conv(
        self,
        x: mx.array,
        conv: nn.Conv1d,
        cache: ArraysCache | None,
        cache_index: int,
        mask: mx.array | None,
    ) -> mx.array:
        batch, length, dim = x.shape
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)
        previous = None if cache is None else cache[cache_index]
        if previous is None:
            previous = mx.zeros((batch, self.conv_kernel_size - 1, dim), dtype=x.dtype)
        conv_input = mx.concatenate([previous, x], axis=1)
        if cache is not None:
            keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, length)
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                cache[cache_index] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[cache_index] = mx.contiguous(conv_input[:, -keep:, :])
        return nn.silu(conv(conv_input))

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        q = self._causal_conv(self.q_proj(x), self.q_conv1d, cache, 0, mask)
        k = self._causal_conv(self.k_proj(x), self.k_conv1d, cache, 1, mask)
        v = self._causal_conv(self.v_proj(x), self.v_conv1d, cache, 2, mask)
        q = q.reshape(batch, length, self.num_heads, self.head_dim)
        k = k.reshape(batch, length, self.num_heads, self.head_dim)
        v = v.reshape(batch, length, self.num_heads, self.head_dim)

        q, k = _normalize_kda_qk(q, k, self.head_dim)

        raw_gate = self.f_proj(x).reshape(batch, length, self.num_heads, self.head_dim)
        dt_bias = self.dt_bias.reshape(self.num_heads, self.head_dim)
        log_decay = self.lower_bound * mx.sigmoid(
            mx.exp(self.A_log.astype(mx.float32))[None, None, :, None]
            * (raw_gate.astype(mx.float32) + dt_bias)
        )
        decay = mx.exp(log_decay)
        beta = mx.sigmoid(self.b_proj(x).astype(mx.float32))
        state = None if cache is None else cache[3]
        if state is None:
            state = mx.zeros(
                (
                    batch,
                    self.num_heads,
                    self.head_dim,
                    self.head_dim,
                ),
                dtype=mx.float32,
            )
        if mx.default_device() == mx.gpu and mx.metal.is_available():
            output, state = gated_delta_kernel(q, k, v, decay, beta, state, mask)
        else:
            output, state = gated_delta_ops(q, k, v, decay, beta, state, mask)
        if cache is not None:
            cache[3] = state
            cache.advance(length)

        output_gate = self.g_proj(x).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        output = self.o_norm(output)
        output = output * mx.sigmoid(output_gate.astype(mx.float32)).astype(
            output.dtype
        )
        return self.o_proj(output.reshape(batch, length, -1))


class BailingDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_index: int):
        super().__init__()
        self.is_linear = _is_kda_layer(
            layer_index, args.layer_group_size, args.num_hidden_layers
        )
        self.attention = BailingKDA(args) if self.is_linear else BailingMLA(args)
        if layer_index >= args.first_k_dense_replace:
            self.mlp = BailingSparseMoE(args)
        else:
            self.mlp = BailingMLP(args, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        hidden = x + self.attention(self.input_layernorm(x), mask, cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class BailingModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            BailingDecoderLayer(args, layer_index)
            for layer_index in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.linear_index = 0
        self.attention_index = args.layer_group_size - 1

    def __call__(self, inputs: mx.array, cache: Any | None = None) -> mx.array:
        hidden = self.word_embeddings(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        attention_mask = create_attention_mask(hidden, cache[self.attention_index])
        linear_mask = create_ssm_mask(hidden, cache[self.linear_index])
        for layer, layer_cache in zip(self.layers, cache):
            mask = linear_mask if layer.is_linear else attention_mask
            hidden = layer(hidden, mask=mask, cache=layer_cache)
        return self.norm(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self._validate_quantization_config(args.quantization_config)
        self.args = args
        self.model_type = args.model_type
        self.model = BailingModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def __call__(self, inputs: mx.array, cache: Any | None = None) -> mx.array:
        hidden = self.model(inputs, cache)
        return self.lm_head(hidden)

    def make_cache(self):
        return [
            ArraysCache(size=4) if layer.is_linear else KVCache()
            for layer in self.layers
        ]

    @staticmethod
    def _validate_quantization_config(config: dict[str, Any] | None) -> None:
        if config is None:
            return
        expected = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        }
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Bailing V3 MLX requires an FP8 E4M3 checkpoint with UE8M0 "
                "scales and weight_block_size=[128, 128]"
            )

    def sanitize(self, weights):
        weights = {
            key: value
            for key, value in weights.items()
            if not _is_mtp_weight(key, self.args.num_hidden_layers)
        }
        scale_keys = [key for key in weights if key.endswith("weight_scale_inv")]
        uses_fp8 = _uses_fp8(self.args)
        if not uses_fp8 and scale_keys:
            raise ValueError(
                "Unquantized Bailing V3 checkpoints must not contain "
                "weight_scale_inv tensors"
            )
        for scale_key in scale_keys:
            weight_key = scale_key.removesuffix("_scale_inv")
            weight = weights.pop(weight_key)
            scale = weights.pop(scale_key)
            output_dims, input_dims = weight.shape
            weights[weight_key] = weight.view(mx.uint32)
            expanded_scale = mx.repeat(scale.view(mx.uint8), 128, axis=0)
            expanded_scale = mx.repeat(expanded_scale, 4, axis=1)
            scale_name = weight_key.removesuffix("weight") + "scales"
            weights[scale_name] = expanded_scale[:output_dims, : input_dims // 32]

        for layer_index in range(
            self.args.first_k_dense_replace, self.args.num_hidden_layers
        ):
            prefix = f"model.layers.{layer_index}.mlp"
            for projection in ("gate_proj", "up_proj", "down_proj"):
                suffixes = ("weight", "scales") if uses_fp8 else ("weight",)
                for suffix in suffixes:
                    expert_weights = [
                        weights.pop(f"{prefix}.experts.{expert}.{projection}.{suffix}")
                        for expert in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.switch_mlp.{projection}.{suffix}"] = mx.stack(
                        expert_weights
                    )

        for key, value in list(weights.items()):
            if "_conv1d.weight" in key and value.shape[-1] != 1:
                weights[key] = value.moveaxis(2, 1)
        return weights
