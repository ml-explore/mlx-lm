# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "laguna"
    vocab_size: int = 100352
    hidden_size: int = 3072
    intermediate_size: int = 12288
    num_hidden_layers: int = 48
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    attention_bias: bool = False
    qkv_bias: bool = False
    attention_dropout: float = 0.0
    gating: bool | str = "per-head"
    sliding_window: Optional[int] = 512
    rope_parameters: Optional[Dict[str, Any]] = None
    layer_types: Optional[List[str]] = None
    num_attention_heads_per_layer: Optional[List[int]] = None
    num_experts: int = 256
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 1024
    shared_expert_intermediate_size: int = 1024
    norm_topk_prob: bool = True
    moe_routed_scaling_factor: float = 2.5
    moe_router_logit_softcapping: float = 0.0
    moe_apply_router_weight_on_input: bool = False
    decoder_sparse_step: int = 1
    mlp_only_layers: Optional[List[int]] = None
    mlp_layer_types: Optional[List[str]] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per hidden layer")
        invalid_types = set(self.layer_types) - {
            "full_attention",
            "sliding_attention",
        }
        if invalid_types:
            raise ValueError(f"Unsupported attention types: {sorted(invalid_types)}")
        if "sliding_attention" in self.layer_types and (
            self.sliding_window is None or self.sliding_window <= 0
        ):
            raise ValueError("sliding_window must be positive for sliding attention")

        if self.num_attention_heads_per_layer is None:
            self.num_attention_heads_per_layer = [
                self.num_attention_heads
            ] * self.num_hidden_layers
        if len(self.num_attention_heads_per_layer) != self.num_hidden_layers:
            raise ValueError(
                "num_attention_heads_per_layer must contain one entry per hidden layer"
            )
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if any(
            h <= 0 or h % self.num_key_value_heads
            for h in self.num_attention_heads_per_layer
        ):
            raise ValueError(
                "Each attention head count must be positive and divisible by "
                "num_key_value_heads"
            )

        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        if self.decoder_sparse_step <= 0:
            raise ValueError("decoder_sparse_step must be positive")
        if self.moe_apply_router_weight_on_input:
            raise ValueError("moe_apply_router_weight_on_input is not supported")

        if self.mlp_only_layers is None:
            self.mlp_only_layers = [0]
        if any(i < 0 or i >= self.num_hidden_layers for i in self.mlp_only_layers):
            raise ValueError("mlp_only_layers contains an invalid layer index")
        if self.mlp_layer_types is not None:
            if len(self.mlp_layer_types) != self.num_hidden_layers:
                raise ValueError(
                    "mlp_layer_types must contain one entry per hidden layer"
                )
            invalid_mlp_types = set(self.mlp_layer_types) - {"dense", "sparse"}
            if invalid_mlp_types:
                raise ValueError(
                    f"Unsupported MLP layer types: {sorted(invalid_mlp_types)}"
                )

        if self.gating not in (False, True, "per-head", "per_head", "per-element"):
            raise ValueError(f"Unsupported attention gating mode: {self.gating}")
        if self.qkv_bias or self.attention_bias:
            raise ValueError("Laguna checkpoints use bias-free attention projections")
        if self.hidden_act != "silu":
            raise ValueError("Laguna currently supports only the silu activation")

        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 500000.0,
                    "partial_rotary_factor": 1.0,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                },
            }
        if "full_attention" in self.rope_parameters:
            for layer_type in set(self.layer_types):
                if layer_type not in self.rope_parameters:
                    raise ValueError(
                        f"rope_parameters is missing nested {layer_type!r} settings"
                    )
            rope_configs = [
                self.rope_parameters[layer_type] for layer_type in set(self.layer_types)
            ]
        else:
            rope_configs = [self.rope_parameters]
        for rope_config in rope_configs:
            rotary_dims = int(
                self.head_dim * rope_config.get("partial_rotary_factor", 1.0)
            )
            if rotary_dims <= 0 or rotary_dims > self.head_dim or rotary_dims % 2:
                raise ValueError(
                    "Rotary dimensions must be a positive, even value no larger "
                    "than head_dim"
                )


class LagunaMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        intermediate_size = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def router_select(
    logits: mx.array,
    correction_bias: mx.array,
    top_k: int,
    normalize: bool,
    output_dtype,
):
    """Select experts with corrected sigmoid scores but return unbiased weights."""
    scores = mx.sigmoid(logits.astype(mx.float32))
    selection_scores = scores + correction_bias.astype(mx.float32)
    indices = mx.argpartition(-selection_scores, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if normalize:
        weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    return indices, weights.astype(output_dtype)


class LagunaTopKRouter(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.softcap = float(args.moe_router_logit_softcapping or 0.0)
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        logits = self.gate(x).astype(mx.float32)
        if self.softcap > 0:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        return router_select(
            logits,
            self.e_score_correction_bias,
            self.top_k,
            self.norm_topk_prob,
            x.dtype,
        )


class LagunaSparseMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.routed_scaling_factor = args.moe_routed_scaling_factor
        self.gate = LagunaTopKRouter(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=False,
        )
        self.share_expert = LagunaMLP(
            args, intermediate_size=args.shared_expert_intermediate_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        indices, weights = self.gate(x)
        routed = self.switch_mlp(x, indices)
        routed = (routed * weights[..., None]).sum(axis=-2).astype(x.dtype)
        return routed * self.routed_scaling_factor + self.share_expert(x)


def apply_attention_gate(
    output: mx.array,
    gate_logits: mx.array,
    num_heads: int,
    head_dim: int,
    per_head: bool,
) -> mx.array:
    """Apply Laguna's float32 softplus gate before the output projection."""
    gate = nn.softplus(gate_logits.astype(mx.float32)).astype(output.dtype)
    if per_head:
        output = output.reshape(*output.shape[:-1], num_heads, head_dim)
        output = output * gate[..., None]
        return output.reshape(*output.shape[:-2], num_heads * head_dim)
    return output * gate


class LagunaAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.num_heads = args.num_attention_heads_per_layer[layer_idx]
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        gating = args.gating
        self.use_gate = bool(gating)
        self.gate_per_head = gating in ("per-head", "per_head")
        if self.use_gate:
            gate_dims = (
                self.num_heads if self.gate_per_head else self.num_heads * self.head_dim
            )
            self.g_proj = nn.Linear(args.hidden_size, gate_dims, bias=False)

        layer_type = args.layer_types[layer_idx]
        if "full_attention" in args.rope_parameters:
            rope_params = args.rope_parameters[layer_type]
        else:
            rope_params = args.rope_parameters
        rotary_dims = int(self.head_dim * rope_params.get("partial_rotary_factor", 1.0))
        if rotary_dims <= 0 or rotary_dims > self.head_dim or rotary_dims % 2:
            raise ValueError(
                "Rotary dimensions must be a positive, even value no larger than head_dim"
            )
        self.rope = initialize_rope(
            dims=rotary_dims,
            base=rope_params.get("rope_theta", 10000.0),
            traditional=False,
            scaling_config=rope_params,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        queries = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim)
        keys = self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
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
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        if self.use_gate:
            output = apply_attention_gate(
                output,
                self.g_proj(x),
                self.num_heads,
                self.head_dim,
                self.gate_per_head,
            )
        return self.o_proj(output)


class LagunaDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = LagunaAttention(args, layer_idx)
        self.is_sliding = self.self_attn.is_sliding

        if args.mlp_layer_types is not None:
            is_sparse = args.mlp_layer_types[layer_idx] == "sparse"
        else:
            is_sparse = (
                layer_idx not in args.mlp_only_layers
                and (layer_idx + 1) % args.decoder_sparse_step == 0
            )
        self.mlp = LagunaSparseMoE(args) if is_sparse else LagunaMLP(args)
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
        h = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class LagunaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            LagunaDecoderLayer(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._full_idx = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_sliding), None
        )
        self._sliding_idx = next(
            (i for i, layer in enumerate(self.layers) if layer.is_sliding), None
        )

    def __call__(self, inputs: mx.array, cache: Optional[List[Any]] = None) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * self.num_layers
        if len(cache) != self.num_layers:
            raise ValueError("cache must contain one entry per hidden layer")

        full_mask = (
            create_attention_mask(h, cache[self._full_idx])
            if self._full_idx is not None
            else None
        )
        sliding_mask = (
            create_attention_mask(
                h,
                cache[self._sliding_idx],
                window_size=self.args.sliding_window,
            )
            if self._sliding_idx is not None
            else None
        )
        for layer, layer_cache in zip(self.layers, cache):
            mask = sliding_mask if layer.is_sliding else full_mask
            h = layer(h, mask=mask, cache=layer_cache)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LagunaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Optional[List[Any]] = None) -> mx.array:
        output = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(output)
        return self.lm_head(output)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                if layer.is_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    def sanitize(self, weights):
        weights = dict(weights)

        def move_key(source, destination):
            if source not in weights:
                return
            if destination in weights:
                raise ValueError(
                    "Ambiguous Laguna checkpoint keys: both "
                    f"{source!r} and {destination!r} are present"
                )
            weights[destination] = weights.pop(source)

        for layer_idx, layer in enumerate(self.layers):
            if not isinstance(layer.mlp, LagunaSparseMoE):
                continue
            prefix = f"model.layers.{layer_idx}.mlp"

            correction_key = f"{prefix}.experts.e_score_correction_bias"
            move_key(correction_key, f"{prefix}.gate.e_score_correction_bias")
            move_key(f"{prefix}.gate.weight", f"{prefix}.gate.gate.weight")

            expert_prefix = f"{prefix}.experts."
            projections = ("gate_proj", "up_proj", "down_proj")
            suffixes = ("weight", "scales", "biases")
            expert_groups = {}
            unsupported_expert_keys = []
            for key in weights:
                if not key.startswith(expert_prefix):
                    continue
                parts = key[len(expert_prefix) :].split(".")
                if not parts[0].isdigit():
                    continue
                if (
                    len(parts) != 3
                    or parts[1] not in projections
                    or parts[2] not in suffixes
                ):
                    unsupported_expert_keys.append(key)
                    continue
                expert_groups.setdefault((parts[1], parts[2]), {})[int(parts[0])] = key

            if unsupported_expert_keys:
                examples = ", ".join(repr(k) for k in unsupported_expert_keys[:3])
                raise ValueError(
                    f"Unsupported Laguna expert keys for layer {layer_idx}; "
                    f"examples: {examples}"
                )

            if expert_groups:
                expected_ids = set(range(self.args.num_experts))
                problems = []
                groups_to_stack = []
                for projection in projections:
                    group = (projection, "weight")
                    ids = set(expert_groups.get(group, {}))
                    if ids != expected_ids:
                        missing = sorted(expected_ids - ids)
                        unexpected = sorted(ids - expected_ids)
                        missing_examples = [
                            f"{expert_prefix}{expert}.{projection}.weight"
                            for expert in missing[:3]
                        ]
                        problems.append(
                            f"{projection}.weight: {len(ids)}/"
                            f"{self.args.num_experts}; missing examples "
                            f"{missing_examples}; unexpected ids {unexpected[:3]}"
                        )
                    else:
                        groups_to_stack.append(group)

                for group, experts in expert_groups.items():
                    if group[1] == "weight":
                        continue
                    ids = set(experts)
                    if ids != expected_ids:
                        missing = sorted(expected_ids - ids)
                        unexpected = sorted(ids - expected_ids)
                        projection, suffix = group
                        missing_examples = [
                            f"{expert_prefix}{expert}.{projection}.{suffix}"
                            for expert in missing[:3]
                        ]
                        problems.append(
                            f"{projection}.{suffix}: {len(ids)}/"
                            f"{self.args.num_experts}; missing examples "
                            f"{missing_examples}; unexpected ids {unexpected[:3]}"
                        )
                    else:
                        groups_to_stack.append(group)

                if problems:
                    observed = sum(len(experts) for experts in expert_groups.values())
                    expected_weights = self.args.num_experts * len(projections)
                    raise ValueError(
                        f"Incomplete Laguna expert tensors for layer {layer_idx}: "
                        f"found {observed} individual tensors; expected at least "
                        f"{expected_weights} weight tensors. " + "; ".join(problems)
                    )

                for projection, suffix in groups_to_stack:
                    destination = f"{prefix}.switch_mlp.{projection}.{suffix}"
                    if destination in weights:
                        raise ValueError(
                            "Ambiguous Laguna expert tensors: both individual "
                            f"expert keys and fused key {destination!r} are present"
                        )
                    experts = expert_groups[(projection, suffix)]
                    values = [
                        weights.pop(experts[expert])
                        for expert in range(self.args.num_experts)
                    ]
                    weights[destination] = mx.stack(values, axis=0)

        remaining_expert_keys = []
        expert_marker = ".mlp.experts."
        for key in weights:
            if expert_marker not in key:
                continue
            tail = key.split(expert_marker, 1)[1]
            if tail.split(".", 1)[0].isdigit():
                remaining_expert_keys.append(key)
        if remaining_expert_keys:
            examples = ", ".join(repr(k) for k in remaining_expert_keys[:3])
            raise ValueError(
                "Unconsumed Laguna individual expert keys remain; the layer may "
                f"not be configured as sparse. Examples: {examples}"
            )

        for source in [k for k in weights if ".mlp.shared_expert." in k]:
            move_key(
                source,
                source.replace(".mlp.shared_expert.", ".mlp.share_expert."),
            )
        return weights

    @property
    def cast_predicate(self):
        def predicate(key):
            return "e_score_correction_bias" not in key

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            lowered = path.lower()
            if "norm" in lowered or "bias" in lowered:
                return False
            # The sparse router module is ``mlp.gate`` and its Linear is
            # ``mlp.gate.gate``.  Do not confuse that subtree with the dense
            # MLP's independently quantizable ``mlp.gate_proj``.
            dotted = f".{lowered}"
            if ".mlp.gate." in dotted or dotted.endswith(".mlp.gate"):
                return False
            return True

        return predicate
