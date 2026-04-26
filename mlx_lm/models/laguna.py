from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    qkv_bias: bool = False
    attention_bias: bool = False
    gating: Union[bool, str] = True
    tie_word_embeddings: bool = False
    rope_theta: float = 500000.0
    rope_parameters: Optional[Dict[str, Union[float, str]]] = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    partial_rotary_factor: Optional[float] = None
    rope_style: str = "rotate-half"
    sliding_window: Optional[int] = None
    layer_types: Optional[List[str]] = None
    num_attention_heads_per_layer: Optional[List[int]] = None
    swa_rope_parameters: Optional[Dict[str, Union[float, str]]] = None
    swa_attention_sink_enabled: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=lambda: [0])
    moe_routed_scaling_factor: float = 1.0
    moe_apply_router_weight_on_input: bool = False
    moe_router_logit_softcapping: float = 0.0
    moe_router_use_sigmoid: bool = True

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must match num_hidden_layers.")

        if self.num_attention_heads_per_layer is None:
            self.num_attention_heads_per_layer = [
                self.num_attention_heads
            ] * self.num_hidden_layers
        if len(self.num_attention_heads_per_layer) != self.num_hidden_layers:
            raise ValueError(
                "num_attention_heads_per_layer must match num_hidden_layers."
            )
        if any(
            h % self.num_key_value_heads for h in self.num_attention_heads_per_layer
        ):
            raise ValueError(
                "Every query-head count must be divisible by num_key_value_heads."
            )

        if self.rope_parameters is None:
            self.rope_parameters = (
                dict(self.rope_scaling)
                if self.rope_scaling is not None
                else {"rope_type": "default", "rope_theta": self.rope_theta}
            )
        else:
            self.rope_parameters = dict(self.rope_parameters)

        if self.swa_rope_parameters is not None:
            self.swa_rope_parameters = dict(self.swa_rope_parameters)

        if self.partial_rotary_factor is not None:
            self.rope_parameters.setdefault(
                "partial_rotary_factor", self.partial_rotary_factor
            )
            if self.swa_rope_parameters is not None:
                self.swa_rope_parameters.setdefault(
                    "partial_rotary_factor", self.partial_rotary_factor
                )


def _rope_base(args: ModelArgs, rope_config: Dict[str, Union[float, str]]) -> float:
    return float(rope_config.get("rope_theta", args.rope_theta))


def _rope_dims(args: ModelArgs, rope_config: Dict[str, Union[float, str]]) -> int:
    partial = float(rope_config.get("partial_rotary_factor", 1.0))
    return int(args.head_dim * partial)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class LagunaTopKRouter(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.use_sigmoid = args.moe_router_use_sigmoid
        self.router_logit_softcapping = args.moe_router_logit_softcapping
        self.proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        dtype = x.dtype
        logits = self.proj(x).astype(mx.float32)
        if self.router_logit_softcapping > 0.0:
            c = self.router_logit_softcapping
            logits = mx.tanh(logits / c) * c

        scores = mx.sigmoid(logits) if self.use_sigmoid else mx.softmax(logits, axis=-1)
        corrected_scores = scores + self.e_score_correction_bias.astype(scores.dtype)

        k = self.top_k
        inds = mx.stop_gradient(
            mx.argpartition(-corrected_scores, kth=k - 1, axis=-1)[..., :k]
        )
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        return inds, weights.astype(dtype)


class LagunaSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        if args.moe_apply_router_weight_on_input:
            raise NotImplementedError(
                "moe_apply_router_weight_on_input=True is not supported."
            )
        self.routed_scaling_factor = args.moe_routed_scaling_factor
        self.gate = LagunaTopKRouter(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = mx.sum(y * scores[..., None], axis=-2)
        if self.routed_scaling_factor != 1.0:
            y = y * self.routed_scaling_factor
        return y + self.shared_expert(x)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        self.n_heads = args.num_attention_heads_per_layer[layer_idx]
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.gate_per_head = args.gating == "per-head"
        self.gating = bool(args.gating)
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = args.sliding_window if self.is_sliding else None

        dim = args.hidden_size
        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=args.qkv_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.qkv_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.qkv_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias
        )

        if self.gating:
            gate_dim = (
                self.n_heads
                if self.gate_per_head
                else self.n_heads * self.head_dim
            )
            self.g_proj = nn.Linear(dim, gate_dim, bias=False)

        if self.is_sliding and args.swa_attention_sink_enabled:
            self.sink = mx.zeros((self.n_heads,))
        else:
            self.sink = None

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        rope_config = (
            args.swa_rope_parameters
            if self.is_sliding and args.swa_rope_parameters is not None
            else args.rope_parameters
        )
        self.rope = initialize_rope(
            _rope_dims(args, rope_config),
            base=_rope_base(args, rope_config),
            traditional=False,
            scaling_config=rope_config,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(
            queries.reshape(B, L, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
            sinks=self.sink,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        if self.gating:
            gate = nn.softplus(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
            if self.gate_per_head:
                shape = output.shape
                output = (
                    output.reshape(B, L, self.n_heads, self.head_dim)
                    * gate[..., None]
                ).reshape(shape)
            else:
                output = output * gate

        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        if (layer_idx not in args.mlp_only_layers) and (
            args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
        ):
            self.mlp = LagunaSparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.attention_type = args.layer_types[layer_idx]

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LagunaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx) for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx = args.layer_types.index("full_attention")
        self.swa_idx = (
            args.layer_types.index("sliding_attention")
            if "sliding_attention" in args.layer_types
            else None
        )

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

        full_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            sliding_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.args.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            mask = (
                sliding_mask
                if layer.attention_type == "sliding_attention"
                else full_mask
            )
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LagunaModel(args)
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
        return self.lm_head(out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        weights = self._unpack_compressed_tensors(weights)
        weights = self._remap_router_weights(weights)
        weights = self._stack_experts(weights)
        return {
            k: v
            for k, v in weights.items()
            if "rotary_emb.inv_freq" not in k
            and not k.endswith(".self_attn.k_scale")
            and not k.endswith(".self_attn.v_scale")
        }

    def _unpack_compressed_tensors(self, weights):
        if not any(k.endswith(".weight_shape") for k in weights):
            return weights

        new_weights = {}
        for k, v in weights.items():
            if k.endswith(".weight_shape"):
                base = k[: -len("weight_shape")]
                if (
                    f"{base}weight_packed" in weights
                    and f"{base}weight_scale" in weights
                ):
                    scales = weights[f"{base}weight_scale"]
                    new_weights[f"{base}weight"] = weights[
                        f"{base}weight_packed"
                    ].view(mx.uint32)
                    new_weights[f"{base}scales"] = scales
                    new_weights[f"{base}biases"] = (-8 * scales).astype(scales.dtype)
            elif k.endswith(".weight_packed") or k.endswith(".weight_scale"):
                base = k.rsplit(".", 1)[0] + "."
                if f"{base}weight_shape" in weights:
                    continue
                new_weights[k] = v
            else:
                new_weights[k] = v
        return new_weights

    def _remap_router_weights(self, weights):
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp"
            gate_weight = f"{prefix}.gate.weight"
            if gate_weight in weights:
                weights[f"{prefix}.gate.proj.weight"] = weights.pop(gate_weight)

            legacy_bias = f"{prefix}.experts.e_score_correction_bias"
            if legacy_bias in weights:
                weights[f"{prefix}.gate.e_score_correction_bias"] = weights.pop(
                    legacy_bias
                )
        return weights

    def _stack_experts(self, weights):
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp"
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                for suffix in ["weight", "scales", "biases"]:
                    first_key = f"{prefix}.experts.0.{proj}.{suffix}"
                    if first_key not in weights:
                        continue
                    weights[f"{prefix}.switch_mlp.{proj}.{suffix}"] = mx.stack(
                        [
                            weights.pop(f"{prefix}.experts.{e}.{proj}.{suffix}")
                            for e in range(self.args.num_experts)
                        ]
                    )
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate.proj"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k
        return predicate

    @property
    def layers(self):
        return self.model.layers
