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
from .gated_delta import gated_delta_update
from .mla import MultiLinear
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "bailing_hybrid"
    vocab_size: int = 157184
    hidden_size: int = 1536
    intermediate_size: int = 4608
    moe_intermediate_size: int = 512
    moe_shared_expert_intermediate_size: int = 512
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    first_k_dense_replace: int = 1
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    layer_group_size: int = 4
    head_dim: int = 128
    q_lora_rank: int | None = 256
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
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


def _normalize_kda_qk(
    q: mx.array, k: mx.array, head_dim: int
) -> tuple[mx.array, mx.array]:
    inv_scale = head_dim**-0.5
    eps = 1e-6 / head_dim
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, eps)
    k = inv_scale * mx.fast.rms_norm(k, None, eps)
    return q, k


class BailingMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


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
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
        )
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
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.scale = self.qk_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                args.hidden_size,
                args.num_attention_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(args.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                args.num_attention_heads * self.qk_head_dim,
                bias=False,
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            args.kv_lora_rank + args.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            args.qk_nope_head_dim, args.kv_lora_rank, args.num_attention_heads
        )
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, args.v_head_dim, args.num_attention_heads
        )
        self.g_proj = nn.Linear(args.hidden_size, args.num_attention_heads, bias=False)
        self.dense = nn.Linear(
            args.num_attention_heads * args.v_head_dim,
            args.hidden_size,
            bias=False,
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
        kv_latent = self.kv_a_layernorm(compressed)
        k_rope = k_rope.reshape(batch, length, 1, self.qk_rope_head_dim).transpose(
            0, 2, 1, 3
        )

        offset = cache.offset if cache is not None else 0
        q_rope = self.rope(q_rope, offset=offset)
        k_rope = self.rope(k_rope, offset=offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)
        if cache is not None:
            kv_latent, k_rope = cache.update_and_fetch(kv_latent, k_rope)

        pe_scores = (q_rope * self.scale) @ k_rope.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if length == 1:
            q_nope = self.embed_q(q_nope)
            keys = values = kv_latent
        else:
            keys = self.embed_q(kv_latent, transpose=False)
            values = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=pe_scores,
        )
        if length == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3)
        gate = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
        output = output * gate[..., None]
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

        self.q_proj = nn.Linear(args.hidden_size, self.projection_size, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.projection_size, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.projection_size, bias=False)
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
        self.f_proj = nn.Linear(args.hidden_size, self.projection_size, bias=False)
        self.dt_bias = mx.zeros((self.projection_size,), dtype=mx.float32)
        self.b_proj = nn.Linear(args.hidden_size, self.num_heads, bias=False)
        self.g_proj = nn.Linear(args.hidden_size, self.projection_size, bias=False)
        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_size, args.hidden_size, bias=False)

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
        state = None if cache is None else cache[3]
        output, state = gated_delta_update(
            q,
            k,
            v,
            raw_gate,
            self.b_proj(x).astype(mx.float32),
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=state,
            mask=mask,
            use_kernel=not self.training,
            lower_bound=self.lower_bound,
        )
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
        self.linear_index = next(
            (index for index, layer in enumerate(self.layers) if layer.is_linear), None
        )
        self.attention_index = next(
            (index for index, layer in enumerate(self.layers) if not layer.is_linear),
            None,
        )

    def __call__(self, inputs: mx.array, cache: Any | None = None) -> mx.array:
        hidden = self.word_embeddings(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        attention_mask = (
            create_attention_mask(
                hidden, cache[self.attention_index], return_array=True
            )
            if self.attention_index is not None
            else None
        )
        linear_mask = (
            create_ssm_mask(hidden, cache[self.linear_index])
            if self.linear_index is not None
            else None
        )
        for layer, layer_cache in zip(self.layers, cache):
            mask = linear_mask if layer.is_linear else attention_mask
            hidden = layer(hidden, mask=mask, cache=layer_cache)
        return self.norm(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = BailingModel(args)
        self.lm_head = (
            None
            if args.tie_word_embeddings
            else nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        )

    @property
    def layers(self):
        return self.model.layers

    def __call__(self, inputs: mx.array, cache: Any | None = None) -> mx.array:
        hidden = self.model(inputs, cache)
        if self.lm_head is None:
            return self.model.word_embeddings.as_linear(hidden)
        return self.lm_head(hidden)

    def make_cache(self):
        return [
            ArraysCache(size=4) if layer.is_linear else KVCache()
            for layer in self.layers
        ]

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path in ("model.word_embeddings", "lm_head"):
                return False
            if path.endswith(
                (
                    "attention.q_a_proj",
                    "attention.q_b_proj",
                    "attention.kv_a_proj_with_mqa",
                    "attention.embed_q",
                    "attention.unembed_out",
                    "attention.b_proj",
                )
            ):
                return False
            if path.endswith(("attention.q_proj", "attention.g_proj")):
                layer_index = int(path.split(".")[2])
                return self.layers[layer_index].is_linear
            return True

        return predicate

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        weights = {
            key: value
            for key, value in weights.items()
            if not _is_mtp_weight(key, self.args.num_hidden_layers)
        }
        for layer_index in range(self.args.num_hidden_layers):
            layer_prefix = f"model.layers.{layer_index}"
            if layer_index >= self.args.first_k_dense_replace:
                mlp_prefix = f"{layer_prefix}.mlp"
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    expert_prefix = f"{mlp_prefix}.experts.0.{projection}"
                    if f"{expert_prefix}.weight" not in weights:
                        continue
                    expert_weights = [
                        weights.pop(
                            f"{mlp_prefix}.experts.{expert}.{projection}.weight"
                        )
                        for expert in range(self.args.num_experts)
                    ]
                    weights[f"{mlp_prefix}.switch_mlp.{projection}.weight"] = mx.stack(
                        expert_weights
                    )

            attention_prefix = f"{layer_prefix}.attention"
            kv_b_key = f"{attention_prefix}.kv_b_proj.weight"
            if kv_b_key not in weights:
                continue
            num_heads = self.args.num_attention_heads
            head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim
            if f"{attention_prefix}.kv_b_proj.scales" in weights:
                raise ValueError(
                    "Quantized Bailing V3 kv_b_proj weights are not supported"
                )
            value = weights.pop(kv_b_key)
            value = value.reshape(num_heads, head_dim, -1)
            wk = mx.contiguous(
                value[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
            )
            wv = mx.contiguous(value[:, self.args.qk_nope_head_dim :, :])
            weights[f"{attention_prefix}.embed_q.weight"] = wk
            weights[f"{attention_prefix}.unembed_out.weight"] = wv

        for key, value in list(weights.items()):
            if "_conv1d.weight" in key and value.shape[-1] != 1:
                weights[key] = value.moveaxis(2, 1)
        return weights
