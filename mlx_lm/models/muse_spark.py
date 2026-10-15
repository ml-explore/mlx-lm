# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .pipeline import PipelineMixin
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU

_RESIDUAL_GATE_EPS = 1e-3
_ROUTER_NORM_EPS = 1e-15
_FP32_PARAMS = ("mlp.gate.", "residual_gate.")


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "muse_spark"
    hidden_size: int = 8192
    num_hidden_layers: int = 62
    intermediate_size: int = 16384
    num_attention_heads: int = 128
    num_key_value_heads: int = 16
    head_dim: int = 64
    vocab_size: int = 202048
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    sliding_window: int = 2048
    qk_scale_factor: float = 5.4730064863875
    output_multiplier: float = 0.17677669529663687
    final_logit_softcapping: float = 20.0
    max_position_embeddings: int = 1048576
    rope_theta: float = 500000.0
    rope_parameters: Optional[Dict[str, Any]] = None
    layer_types: Optional[List[str]] = None
    layer_rope_theta: Optional[List[float]] = None
    mlp_layer_types: Optional[List[str]] = None
    moe_hidden_size: int = 4096
    moe_intermediate_size: int = 4096
    num_local_experts: int = 256
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 1.0
    residual_gate_temperature: float = 0.3
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params):
        # HF checkpoints nest the language model under "text_config".
        if "text_config" in params:
            merged = dict(params)
            merged.update(params["text_config"])
            merged["model_type"] = params.get("model_type", cls.model_type)
            params = merged
        return super().from_dict(params)

    def __post_init__(self):
        if self.rope_parameters and "rope_theta" in self.rope_parameters:
            self.rope_theta = self.rope_parameters["rope_theta"]
        if self.layer_types is None:
            self.layer_types = [
                (
                    "full_attention"
                    if (self.num_hidden_layers - 1 - i) % 4 == 0
                    else "sliding_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        if self.layer_rope_theta is None:
            self.layer_rope_theta = [
                0 if t == "full_attention" else self.rope_theta
                for t in self.layer_types
            ]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["sparse"] * self.num_hidden_layers


# Scale in FP32 and cast once, like the reference. mx.fast.rms_norm, as used by
# muse_glimmer, casts before it scales and changes BF16 results.
@mx.compile
def _scaled_rms_norm(x, weight, eps):
    x32 = x.astype(mx.float32)
    x32 = x32 * mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    return (x32 * weight.astype(mx.float32)).astype(x.dtype)


@mx.compile
def _input_scaled_rms_norm(x, weight, eps):
    # The reference rounds the scaled input to the activation dtype before the norm.
    x = (x.astype(mx.float32) * weight.astype(mx.float32)).astype(x.dtype)
    return mx.fast.rms_norm(x, None, eps)


# alpha = sqrt(1 - beta^2) without the cancellation, so alpha^2 + beta^2 = 1.
@mx.compile
def _residual_gate(residual, hidden, gate, temperature, eps):
    w = gate.astype(mx.float32) / temperature
    beta = mx.sigmoid(w)
    alpha = mx.sqrt(mx.maximum(mx.sigmoid(-w) * (1.0 + beta), eps))
    return (alpha * residual + beta * hidden).astype(residual.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return _scaled_rms_norm(x, self.weight, self.eps)


class ScalelessRMSNorm(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, None, self.eps)


class CenteredRMSNorm(nn.Module):
    # norm(x) * (1 + weight)
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x):
        return _scaled_rms_norm(x, 1.0 + self.weight, self.eps)


class InputScaledRMSNorm(nn.Module):
    # norm(x * (1 + weight))
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x):
        return _input_scaled_rms_norm(x, 1.0 + self.weight, self.eps)


class ResidualGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = mx.zeros((args.hidden_size,), dtype=mx.float32)
        self.temperature = args.residual_gate_temperature

    def __call__(self, residual, hidden):
        return _residual_gate(
            residual, hidden, self.gate, self.temperature, _RESIDUAL_GATE_EPS
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.qk_scale_factor = args.qk_scale_factor
        self.eps = args.rms_norm_eps
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"

        bias = args.attention_bias
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)
        self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)

        theta = args.layer_rope_theta[layer_idx]
        self.rope = (
            initialize_rope(
                self.head_dim,
                base=float(theta),
                traditional=False,
                scaling_config=args.rope_parameters,
                max_position_embeddings=args.max_position_embeddings,
            )
            if theta
            else None
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1)

        queries = mx.fast.rms_norm(queries, None, self.eps)
        # The reference multiplies in FP32; a BF16 scalar would round the factor.
        queries = (queries.astype(mx.float32) * self.qk_scale_factor).astype(
            values.dtype
        )
        keys = mx.fast.rms_norm(keys, None, self.eps)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = mx.fast.rms_norm(output, None, self.eps)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = output * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim, hidden = args.hidden_size, args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Router(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.output_multiplier = args.output_multiplier
        self.routed_scaling_factor = args.routed_scaling_factor
        self.weight = mx.zeros(
            (args.num_local_experts, args.hidden_size), dtype=mx.float32
        )
        self.e_score_correction_bias = mx.zeros(
            (args.num_local_experts,), dtype=mx.float32
        )

    def __call__(self, x):
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits * self.output_multiplier)
        # The bias only steers the expert choice; the weights use raw scores.
        choice = scores + self.e_score_correction_bias.astype(mx.float32)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k])
        weights = mx.take_along_axis(scores, inds, axis=-1)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + _ROUTER_NORM_EPS)
        return inds, weights * self.routed_scaling_factor


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.moe_hidden_size
        self.gate = Router(args)
        self.pre_expert_proj = nn.Linear(args.hidden_size, dim, bias=False)
        self.pre_expert_norm = CenteredRMSNorm(dim, args.rms_norm_eps)
        self.switch_mlp = SwitchGLU(
            dim, args.moe_intermediate_size, args.num_local_experts
        )
        self.post_expert_norm = InputScaledRMSNorm(dim, args.post_norm_eps)
        self.post_expert_proj = nn.Linear(dim, args.hidden_size, bias=False)
        self.sharding_group = None

    def __call__(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        inds, weights = self.gate(x)
        h = self.pre_expert_norm(self.pre_expert_proj(x))
        y = self.switch_mlp(h, inds)
        if self.sharding_group is not None:
            # The expert norm needs the full output of every expert.
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        y = self.post_expert_norm(y)
        y = (y.astype(mx.float32) * weights[..., None]).sum(axis=-2)
        return self.post_expert_proj(y.astype(h.dtype))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = (
            MoE(args) if args.mlp_layer_types[layer_idx] == "sparse" else MLP(args)
        )
        self.input_layernorm = CenteredRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.pre_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.rms_norm_eps
        )
        self.post_feedforward_layernorm = InputScaledRMSNorm(
            args.hidden_size, args.post_norm_eps
        )
        self.post_attention_residual_gate = ResidualGate(args)
        self.post_feedforward_residual_gate = ResidualGate(args)
        self.post_norm_eps = args.post_norm_eps

    def __call__(self, x, mask=None, cache=None):
        h = self.self_attn(self.input_layernorm(x), mask, cache)
        # post_attention_layernorm has no weight.
        h = mx.fast.rms_norm(h, None, self.post_norm_eps)
        x = self.post_attention_residual_gate(x, h)

        h = self.mlp(self.pre_feedforward_layernorm(x))
        h = self.post_feedforward_layernorm(h)
        return self.post_feedforward_residual_gate(x, h)


class MuseSparkModel(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        # The embeddings get a scaleless RMSNorm, not a sqrt(dim) scale.
        self.embed_norm = ScalelessRMSNorm(args.rms_norm_eps)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def embed(self, inputs):
        return self.embed_norm(self.embed_tokens(inputs))

    def _masks(self, h, layers, cache):
        made = {}
        for layer, c in zip(layers, cache):
            sliding = layer.self_attn.is_sliding
            if sliding not in made:
                window = self.args.sliding_window if sliding else None
                made[sliding] = create_attention_mask(h, c, window_size=window)
            if len(made) == 2:
                break
        return made

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embed(inputs)

        layers = self.pipeline_layers
        if cache is None:
            cache = [None] * len(layers)
        masks = self._masks(h, layers, cache)

        rank, size = self.pipeline_rank, self.pipeline_size
        if rank < size - 1:
            h = mx.distributed.recv_like(h, rank + 1)

        for layer, c in zip(layers, cache):
            h = layer(h, masks[layer.self_attn.is_sliding], c)

        if rank != 0:
            h = mx.distributed.send(h, (rank - 1) % size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)
        if size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MuseSparkModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        cap = self.args.final_logit_softcapping
        logits = out.astype(mx.float32) * (self.args.output_multiplier / cap)
        return (mx.tanh(logits) * cap).astype(out.dtype)

    def sanitize(self, weights):
        out = {}
        for key, value in weights.items():
            # Drop the vision and audio towers of HF and mlx-vlm checkpoints.
            if key.startswith(
                ("model.vision_", "model.audio_tower.", "vision_", "audio_tower.")
            ):
                continue
            if key.startswith("model.language_model."):
                key = "model." + key[len("model.language_model.") :]
            elif key.startswith("language_model."):
                key = key[len("language_model.") :]
            if key.endswith(".mlp.experts.gate_up_proj"):
                # HF stores the experts as (experts, 2 * hidden, dim), gate first.
                prefix = key[: -len("experts.gate_up_proj")] + "switch_mlp."
                gate, up = mx.split(value, 2, axis=1)
                out[prefix + "gate_proj.weight"] = gate
                out[prefix + "up_proj.weight"] = up
            elif key.endswith(".mlp.experts.down_proj"):
                prefix = key[: -len("experts.down_proj")] + "switch_mlp."
                out[prefix + "down_proj.weight"] = value
            else:
                key = key.replace(
                    ".mlp.experts.post_expert_norm.", ".mlp.post_expert_norm."
                )
                out[key] = value
        if self.args.tie_word_embeddings:
            out.pop("lm_head.weight", None)
        return out

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        for layer in self.model.layers:
            attn = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "gate_proj"):
                setattr(
                    attn,
                    name,
                    shard_linear(getattr(attn, name), "all-to-sharded", group=group),
                )
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.n_heads //= N
            attn.n_kv_heads //= N

            mlp = layer.mlp
            if isinstance(mlp, MLP):
                mlp.gate_proj = shard_linear(
                    mlp.gate_proj, "all-to-sharded", group=group
                )
                mlp.up_proj = shard_linear(mlp.up_proj, "all-to-sharded", group=group)
                mlp.down_proj = shard_linear(
                    mlp.down_proj, "sharded-to-all", group=group
                )
            else:
                # Experts shard their intermediate dimension. The router and the
                # pre/post expert projections are replicated.
                mlp.sharding_group = group
                switch = mlp.switch_mlp
                shard_inplace(switch.gate_proj, "all-to-sharded", group=group)
                shard_inplace(switch.up_proj, "all-to-sharded", group=group)
                shard_inplace(switch.down_proj, "sharded-to-all", group=group)

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    @property
    def cast_predicate(self):
        # Keep the router and the residual gates in FP32.
        def predicate(key):
            return not any(part in key for part in _FP32_PARAMS)

        return predicate

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                if layer.self_attn.is_sliding
                else KVCache()
            )
            for layer in self.layers
        ]
