# Copyright © 2025 Apple Inc.

# SenseNova-Loop-9B — MLX port (qwen3_5 backbone + Loop visits + mHC routing)
# Ported from mlx-lm qwen3_next.py. Text-only; visual/mtp/loop_gate dropped.
#
# Architecture (official config):
#   32 physical layers, layer_types = [lin,lin,lin,full]*8  (full at idx 3,7,11,...,31)
#   Loop:  plan = [0..3] + [4..27]*2 + [28..31]  = 56 execution slots
#   mHC:   layers 4..27 use MHCRouter (FP32) to route over `streams` residual streams
#   GDN:   SEPARATE projections (in_proj_qkv / z / a / b), block split [key_dim,key_dim,value_dim]
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import KVCache

# mlx-lm renamed MambaCache -> ArraysCache(size=2) upstream; support both.
try:
    from .cache import ArraysCache

    def _ssm_cache():
        return ArraysCache(size=2)

except ImportError:  # older mlx-lm
    from .cache import MambaCache

    def _ssm_cache():
        return MambaCache()


from .gated_delta import gated_delta_update
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    partial_rotary_factor: float
    max_position_embeddings: int
    rms_norm_eps: float
    vocab_size: int
    tie_word_embeddings: bool
    attention_bias: bool
    attn_output_gate: bool
    full_attention_interval: int
    layer_types: List[str]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    rope_parameters: Dict[str, Any]
    loop_settings: Dict[str, Any]
    mhc: Dict[str, Any]

    @classmethod
    def from_dict(cls, params):
        d = dict(params)
        tc = d.get("text_config") or {}
        merged = {**tc, **d}  # top-level (loop_settings/mhc/model_type) wins
        return super().from_dict(merged)


def build_plan(loop_settings, num_layers):
    start = loop_settings["start"]
    end = loop_settings["end"]
    count = loop_settings["counts"][0]
    return (
        tuple(range(start))
        + tuple(range(start, end)) * count
        + tuple(range(end, num_layers))
    )


# --------------------------------------------------------------------------- #
# RMSNorm (gated) — matches HF Qwen3_5RMSNormGated: x*rsqrt*weight*silu(gate)
# --------------------------------------------------------------------------- #
class SenseNovaRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(
        self, hidden_states: mx.array, gate: Optional[mx.array] = None
    ) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            x = x * nn.silu(gate)
        return x


# --------------------------------------------------------------------------- #
# GatedDeltaNet with SEPARATE projections (SenseNova / HF qwen3_5 layout)
# --------------------------------------------------------------------------- #
class SenseNovaGatedDeltaNet(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_k_heads = args.linear_num_key_heads
        self.num_v_heads = args.linear_num_value_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel = args.linear_conv_kernel_dim
        self.eps = args.rms_norm_eps

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel,
            groups=self.conv_dim,
            padding=0,
        )
        self.in_proj_qkv = nn.Linear(
            args.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(args.hidden_size, self.value_dim, bias=False)
        self.in_proj_a = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_b = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.A_log = mx.zeros(self.num_v_heads)
        self.dt_bias = mx.zeros(self.num_v_heads)
        self.norm = SenseNovaRMSNormGated(self.head_v_dim, eps=self.eps)
        self.out_proj = nn.Linear(self.value_dim, args.hidden_size, bias=False)

    def __call__(self, inputs, mask=None, cache=None):
        B, S, _ = inputs.shape
        mixed_qkv = self.in_proj_qkv(inputs)  # (B,S,conv_dim)
        z = self.in_proj_z(inputs)  # (B,S,value_dim)
        b = self.in_proj_b(inputs)  # (B,S,Hv)
        a = self.in_proj_a(inputs)  # (B,S,Hv)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel - 1, self.conv_dim), dtype=inputs.dtype
            )

        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)  # (B, K-1+S, C)
        if cache is not None:
            cache[0] = conv_input[:, -(self.conv_kernel - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))  # (B,S,conv_dim)

        # block split [key_dim, key_dim, value_dim]
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)

        state = cache[1] if cache is not None else None
        inv_scale = self.head_k_dim**-0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = state

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


# --------------------------------------------------------------------------- #
# Full attention (gated q, q/k head-norm, partial rotary + YaRN)
# --------------------------------------------------------------------------- #
class SenseNovaAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        rope = args.rope_parameters
        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            base=rope["rope_theta"],
            traditional=False,
            scaling_config=rope,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)
        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
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
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))


# --------------------------------------------------------------------------- #
# MLP (no MoE)
# --------------------------------------------------------------------------- #
class SenseNovaMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# mHC (multi-head compressor) routing — pure tensor ops, FP32
# --------------------------------------------------------------------------- #
class MHCRouter(nn.Module):
    def __init__(self, dim: int, settings: Dict[str, Any]):
        super().__init__()
        self.streams = settings["streams"]
        self.iterations = settings["iterations"]
        self.fixed_identity = settings["fixed_identity"]
        self.pre_scale = settings.get("pre_scale", 1.0)
        s = self.streams
        out = 2 * s + (0 if self.fixed_identity else s * s)
        self.weight = mx.zeros((out, s * dim), dtype=mx.float32)
        self.bias = mx.zeros((out,), dtype=mx.float32)

    def _sinkhorn(self, logits):
        logits = logits.astype(mx.float32)
        for _ in range(self.iterations):
            logits = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            logits = logits - mx.logsumexp(logits, axis=-2, keepdims=True)
        return logits.exp()

    def __call__(self, residual):
        s = self.streams
        flat = residual.flatten(-2).astype(mx.float32)
        normalized = flat * mx.rsqrt(flat.square().mean(axis=-1, keepdims=True) + 1e-6)
        logits = mx.matmul(normalized, self.weight.T) + self.bias  # (..., out)
        pre = mx.sigmoid(logits[..., :s])
        post = 2.0 * mx.sigmoid(logits[..., s : 2 * s])
        if self.fixed_identity:
            eye = mx.eye(s, dtype=mx.float32)
            mix = mx.broadcast_to(eye, (*flat.shape[:-1], s, s))
        else:
            mix = self._sinkhorn(logits[..., 2 * s :].reshape(*flat.shape[:-1], s, s))
        branch = mx.matmul(
            mx.expand_dims(pre.astype(residual.dtype), -2), residual
        ).squeeze(-2)
        if self.pre_scale != 1.0:
            branch = branch * self.pre_scale
        return branch, post, mix

    @staticmethod
    def merge(residual, branch, post, mix):
        mixed = mx.matmul(mix.astype(residual.dtype), residual)
        update = mx.expand_dims(post.astype(branch.dtype), -1) * mx.expand_dims(
            branch, -2
        )
        return mixed + update


def _enter(hidden, physical, settings):
    s = settings["streams"]
    if physical == settings["start"]:
        return mx.array(
            mx.broadcast_to(
                mx.expand_dims(hidden, -2), (*hidden.shape[:-1], s, hidden.shape[-1])
            )
        )
    return hidden.reshape(*hidden.shape[:-1], s, -1)


def _leave(residual, physical, settings):
    if physical == settings["end"] - 1:
        return residual.mean(-2)
    return residual.flatten(-2)


# --------------------------------------------------------------------------- #
# Decoder layers
# --------------------------------------------------------------------------- #
class SenseNovaDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        if self.is_linear:
            self.linear_attn = SenseNovaGatedDeltaNet(args)
        else:
            self.self_attn = SenseNovaAttention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = SenseNovaMLP(args.hidden_size, args.intermediate_size)

    def __call__(self, x, mask=None, cache=None):
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class SenseNovaMHCDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int, mhc_settings: Dict[str, Any]):
        super().__init__()
        self._mhc_physical = layer_idx
        self._mhc_settings = mhc_settings
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        if self.is_linear:
            self.linear_attn = SenseNovaGatedDeltaNet(args)
        else:
            self.self_attn = SenseNovaAttention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = SenseNovaMLP(args.hidden_size, args.intermediate_size)
        self.mhc_attn = MHCRouter(args.hidden_size, mhc_settings)
        self.mhc_ffn = MHCRouter(args.hidden_size, mhc_settings)

    def __call__(self, x, mask=None, cache=None):
        physical = self._mhc_physical
        s = self._mhc_settings
        residual = _enter(x, physical, s)
        branch, post, mix = self.mhc_attn(residual)
        branch = self.input_layernorm(branch)
        if self.is_linear:
            branch = self.linear_attn(branch, mask, cache)
        else:
            branch = self.self_attn(branch, mask, cache)
        residual = MHCRouter.merge(residual, branch, post, mix)
        branch, post, mix = self.mhc_ffn(residual)
        branch = self.mlp(self.post_attention_layernorm(branch))
        residual = MHCRouter.merge(residual, branch, post, mix)
        return _leave(residual, physical, s)


# --------------------------------------------------------------------------- #
# Text model (loop scheduler)
# --------------------------------------------------------------------------- #
class SenseNovaLoopTextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            (
                SenseNovaMHCDecoderLayer(args, i, args.mhc)
                if args.mhc["start"] <= i < args.mhc["end"]
                else SenseNovaDecoderLayer(args, i)
            )
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.plan = build_plan(args.loop_settings, args.num_hidden_layers)
        # slots in plan order for mask generation
        self._first_full = next(
            i
            for i, p in enumerate(self.plan)
            if args.layer_types[p] == "full_attention"
        )
        self._first_lin = next(
            i
            for i, p in enumerate(self.plan)
            if args.layer_types[p] == "linear_attention"
        )

    def __call__(self, inputs, cache=None):
        hidden = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.plan)
        fa_mask = create_attention_mask(hidden, cache[self._first_full])
        ssm_mask = create_ssm_mask(hidden, cache[self._first_lin])
        for exec_idx, physical in enumerate(self.plan):
            c = cache[exec_idx]
            mask = fa_mask if self.layers[physical].is_linear is False else ssm_mask
            hidden = self.layers[physical](hidden, mask=mask, cache=c)
        return self.norm(hidden)


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #
class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = SenseNovaLoopTextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            KVCache() if self.args.layer_types[p] == "full_attention" else _ssm_cache()
            for p in self.model.plan
        ]

    def sanitize(self, weights):
        # Idempotency guard: load_model() calls sanitize() on EVERY load,
        # including reload of an already-converted MLX dir. The raw HF
        # checkpoint is fingerprinted by the 'model.language_model.' prefix;
        # a saved MLX dir has no such keys. Only transform raw-HF weights.
        # (Without this, the +1.0 norm offset is applied twice on reload ->
        #  norms ~2x -> garbled generation; verified mean 1.033 vs 2.033.)
        if not any(k.startswith("model.language_model.") for k in weights):
            return weights
        out = {}
        for k, v in weights.items():
            if (
                k.startswith("model.visual.")
                or k.startswith("mtp.")
                or "loop_gate" in k
            ):
                continue
            if k == "lm_head.weight":
                out[k] = v
                continue
            if k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model.") :]
            out[k] = v

        norm_suffix = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k in list(out.keys()):
            v = out[k]
            if "conv1d.weight" in k and v.shape[-1] != 1:
                out[k] = v.moveaxis(2, 1)
            if any(k.endswith(sfx) for sfx in norm_suffix) and v.ndim == 1:
                out[k] = v + 1.0
        return out

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # mHC router (fp32 Sinkhorn) + GDN decay/bias stay full precision
            if "mhc_attn" in path or "mhc_ffn" in path:
                return False
            if path.endswith(".A_log") or path.endswith(".dt_bias"):
                return False
            # All norm weights (1-D, multiplicative) stay full precision
            if "norm.weight" in path:
                return False
            return True

        return predicate

    def cast_predicate(self, path):
        """convert's --dtype cast step: return False to keep a tensor at its
        native (F32) precision. mHC router + GDN A_log/dt_bias must stay F32 —
        bf16 mHC breaks the 20-iter Sinkhorn (verified: 6-bit output went
        garbage when --dtype bfloat16 downcast them)."""
        if "mhc_attn" in path or "mhc_ffn" in path:
            return False
        if path.endswith(".A_log") or path.endswith(".dt_bias"):
            return False
        return True
