# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_kernel, gated_delta_ops
from .mla import MultiLinear
from .switch_layers import SwitchGLU

# rda_config keys this model reads; any other key is an error.
_RDA_CONFIG_KEYS = frozenset(
    {
        "num_heads",
        "head_dim",
        "conv_kernel_size",
        "full_rank_gate",
        "gate_lower_bound",
        "rda_layers",
        "mla_layers",
    }
)


# Adapted from mlx-lm (MIT License).
# SiTU activation: beta * tanh(gate / beta) * sigmoid(gate) times linear_beta * tanh(x / linear_beta), computed in float32
# and rounded once to the input dtype, as one compiled kernel.
@partial(mx.compile, shapeless=True)
def _situ(x, gate, beta, linear_beta):
    dtype = x.dtype
    gate = gate.astype(mx.float32)
    x = x.astype(mx.float32)
    a = beta * mx.tanh(gate / beta) * mx.sigmoid(gate)
    if linear_beta is not None:
        x = linear_beta * mx.tanh(x / linear_beta)
    return (a * x).astype(dtype)


class SiTU(nn.Module):
    def __init__(self, beta: float = 1.0, linear_beta: Optional[float] = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return _situ(x, gate, self.beta, self.linear_beta)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "raon1_mini"
    hidden_size: int = 2560
    num_hidden_layers: int = 25
    num_attention_heads: int = 40
    head_dim: int = 128
    vocab_size: int = 259840
    intermediate_size: int = 12032
    rms_norm_eps: float = 1e-5
    mla_latent_norm_eps: float = 1e-6
    raon_mhc_selector_eps: float = 1e-5
    raon_mhc_streams: int = 4
    raon_mhc_post_factor: float = 2.0
    hidden_act: str = "situ"
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0
    num_experts: Optional[int] = 128
    num_experts_per_token: int = 6
    moe_intermediate_size: int = 1088
    num_shared_experts: int = 1
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    use_grouped_topk: bool = False
    routed_expert_hidden_size: Optional[int] = None
    latent_moe_use_norm: bool = False
    q_lora_rank: int = 576
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True
    rda_config: Optional[dict] = None
    rda_decay_rank: int = 128
    num_nextn_predict_layers: int = 1
    attn_res_block_size: Optional[int] = None
    tie_word_embeddings: bool = False
    raon_fp32_export_head: bool = False
    raon_global_scaled_linears: Optional[list] = None

    def __post_init__(self):
        if self.hidden_act != "situ" or self.moe_router_activation_func != "sigmoid":
            raise ValueError("RAON requires SiTU activation and sigmoid routing")
        if (
            self.use_grouped_topk
            or self.routed_expert_hidden_size is not None
            or self.latent_moe_use_norm
        ):
            raise ValueError("Grouped routing and latent MoE are not supported")
        if not self.mla_use_nope or self.attn_res_block_size is not None:
            raise ValueError("RAON requires NoPE MLA and mHC residual streams")
        if self.raon_mhc_streams < 2 or self.moe_layer_freq < 1:
            raise ValueError("Invalid stream count or MoE layer frequency")
        if self.num_experts and not 0 < self.num_experts_per_token <= self.num_experts:
            raise ValueError("Invalid number of active experts")
        if self.rda_config is None:
            raise ValueError("config must set rda_config")
        unknown = set(self.rda_config) - _RDA_CONFIG_KEYS
        missing = {"rda_layers", "mla_layers"} - set(self.rda_config)
        if unknown or missing:
            raise ValueError(
                f"rda_config keys: unknown {sorted(unknown)}, missing {sorted(missing)}"
            )
        if not self.rda_config.get("full_rank_gate", True):
            raise ValueError("Only full-rank RDA gates are supported")
        linear = self.rda_config["rda_layers"]
        full = self.rda_config["mla_layers"]
        if sorted(linear + full) != list(range(1, self.num_hidden_layers + 1)):
            raise ValueError("Attention layer lists must partition all layers")


class StreamMix(nn.Module):
    def __init__(self, streams, hidden):
        super().__init__()
        size = (streams - 1) ** 2
        self.phi_z = mx.zeros((size, streams * hidden))
        self.alpha_z = mx.array(1.0)
        self.b_z = mx.zeros((size,))
        self.eta = mx.array(0.0)


class MHC(nn.Module):
    def __init__(self, args):
        super().__init__()
        n, hidden = args.raon_mhc_streams, args.hidden_size
        self.num_streams = n
        self.eps = args.raon_mhc_selector_eps
        self.post_factor = args.raon_mhc_post_factor
        self.phi_pre = mx.zeros((n, n * hidden))
        self.phi_post = mx.zeros((n, n * hidden))
        self.alpha_pre = mx.array(1.0)
        self.alpha_post = mx.array(1.0)
        self.b_pre = mx.zeros((n,))
        self.b_post = mx.zeros((n,))
        self.res = StreamMix(n, hidden)
        self._helmert = mx.array(
            [
                [
                    (1.0 if i < k else -float(k) if i == k else 0.0)
                    / math.sqrt(k * (k + 1))
                    for k in range(1, n)
                ]
                for i in range(n)
            ]
        )
        self._eye = mx.eye(n)
        self._chart = (self._eye + mx.ones((n, n)) / n) * 0.5

    def route(self, stream):
        n = self.num_streams
        flat = stream.reshape(-1, n * stream.shape[-1])
        weights = mx.concatenate([self.phi_pre, self.phi_post, self.res.phi_z])
        inv = mx.rsqrt(
            mx.square(flat.astype(mx.float32)).mean(-1, keepdims=True) + self.eps
        )
        z = (flat @ weights.astype(flat.dtype).T).astype(mx.float32) * inv
        pre = mx.sigmoid(
            self.alpha_pre.astype(mx.float32) * z[:, :n] + self.b_pre.astype(mx.float32)
        )
        post = self.post_factor * mx.sigmoid(
            self.alpha_post.astype(mx.float32) * z[:, n : 2 * n]
            + self.b_post.astype(mx.float32)
        )
        chart = self.res.alpha_z.astype(mx.float32) * z[
            :, 2 * n :
        ] + self.res.b_z.astype(mx.float32)
        e = self._helmert @ chart.reshape(-1, n - 1, n - 1) @ self._helmert.T
        rho = mx.maximum((-e / self._chart).max(axis=(-2, -1)), 0)
        phi = self._chart + e / (1 + rho[:, None, None])
        gamma = mx.sigmoid(self.res.eta.astype(mx.float32))
        return pre, post, (1 - gamma) * self._eye + gamma * phi

    @staticmethod
    def aggregate(stream, pre):
        return (stream.astype(mx.float32) * pre[..., None]).sum(-2).astype(stream.dtype)

    @staticmethod
    def mix(stream, branch, post, residual):
        return (
            residual @ stream.astype(mx.float32)
            + post[..., None] * branch.astype(mx.float32)[:, None]
        ).astype(stream.dtype)


def _bound_cache_graph(cache, names, every=64):
    # Raon never reads the batch cache offsets during decode, so their lazy
    # update chains grow every step. Evaluate them from time to time.
    count = cache.__dict__.get("_raon_steps", 0) + 1
    cache.__dict__["_raon_steps"] = count
    if count % every == 0:
        arrays = [getattr(cache, name, None) for name in names]
        mx.eval(*[a for a in arrays if isinstance(a, mx.array)])


class ShortConv(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.weight = mx.zeros((channels, kernel_size, 1))
        self.kernel_size = kernel_size

    def __call__(self, x, state, mask, lengths):
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)
        keep = self.kernel_size - 1
        if state is None:
            state = mx.zeros((x.shape[0], keep, x.shape[-1]), dtype=x.dtype)
        signal = mx.concatenate([state, x], axis=1)
        out = mx.conv1d(
            signal.astype(mx.float32),
            self.weight.astype(mx.float32),
            groups=x.shape[-1],
        )
        out = nn.silu(out).astype(x.dtype)
        if lengths is not None:
            ends = mx.clip(lengths, 0, x.shape[1])
            indices = (ends[:, None] + mx.arange(keep))[..., None]
            state = mx.take_along_axis(signal, indices, axis=1)
        else:
            state = mx.contiguous(signal[:, -keep:, :]) if keep else signal[:, :0, :]
        return out, state


class RDA(nn.Module):
    def __init__(self, args):
        super().__init__()
        cfg = args.rda_config
        self.num_heads = cfg.get("num_heads", args.num_attention_heads)
        self.head_dim = cfg.get("head_dim", args.head_dim)
        self.proj_size = self.num_heads * self.head_dim
        self.lower_bound = cfg.get("gate_lower_bound", -5.0)
        self.eps = args.rms_norm_eps
        hidden, size = args.hidden_size, self.proj_size
        for name in ["q_proj", "k_proj", "v_proj", "g_proj"]:
            setattr(self, name, nn.Linear(hidden, size, bias=False))
        for name in ["q_conv1d", "k_conv1d", "v_conv1d"]:
            setattr(self, name, ShortConv(size, cfg.get("conv_kernel_size", 4)))
        self.f_a_proj = nn.Linear(hidden, args.rda_decay_rank, bias=False)
        self.f_b_proj = nn.Linear(args.rda_decay_rank, size, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.rda_w_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.A_log = mx.zeros((self.num_heads,))
        self.dt_bias = mx.zeros((size,))
        self.o_norm = nn.RMSNorm(self.head_dim, eps=self.eps)
        self.o_proj = nn.Linear(size, hidden, bias=False)

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        dtype = x.dtype
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)
        states = [None] * 4 if cache is None else cache.cache
        lengths = None if cache is None else cache.lengths
        q, qs = self.q_conv1d(self.q_proj(x), states[0], mask, lengths)
        k, ks = self.k_conv1d(self.k_proj(x), states[1], mask, lengths)
        v, vs = self.v_conv1d(self.v_proj(x), states[2], mask, lengths)
        shape = (batch, length, self.num_heads, self.head_dim)
        q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)

        def normalize(value):
            value = value.astype(mx.float32)
            return (
                value
                / mx.maximum(mx.sqrt(mx.square(value).sum(-1, keepdims=True)), 1e-6)
            ).astype(dtype)

        q, k = normalize(q) * self.head_dim**-0.5, normalize(k)
        # RDA can enlarge v beyond FP16 range before beta compensates it.
        update_dtype = mx.float32 if dtype == mx.float16 else dtype
        if dtype == mx.float16:
            q, k = q.astype(update_dtype), k.astype(update_dtype)
        beta_raw = self.b_proj(x).astype(mx.float32)
        delta = self.rda_w_proj(x).astype(mx.float32)
        ratio = mx.exp(-nn.softplus(-beta_raw - delta) + nn.softplus(-beta_raw))
        v = (v.astype(mx.float32) * ratio[..., None]).astype(update_dtype)
        decay = self.f_b_proj(self.f_a_proj(x)).astype(mx.float32).reshape(shape)
        decay = (
            decay
            + self.dt_bias.astype(mx.float32).reshape(self.num_heads, self.head_dim)
        ) * mx.exp(self.A_log.astype(mx.float32))[:, None]
        log_decay = (
            -nn.softplus(decay)
            if self.lower_bound is None
            else self.lower_bound * mx.sigmoid(decay)
        )
        beta = mx.sigmoid(beta_raw).astype(update_dtype)
        state = states[3]
        if state is None:
            state = mx.zeros(
                (batch, self.num_heads, self.head_dim, self.head_dim), dtype=mx.float32
            )
        use_kernel = (
            not self.training
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and self.head_dim >= 32
            and self.head_dim % 32 == 0
        )
        update = gated_delta_kernel if use_kernel else gated_delta_ops
        out, state = update(q, k, v, mx.exp(log_decay), beta, state, mask)
        if cache is not None:
            cache.cache = [qs, ks, vs, state]
            cache.advance(length)
            _bound_cache_graph(cache, ("lengths", "left_padding"))
        normed = mx.fast.rms_norm(
            out.astype(mx.float32), self.o_norm.weight.astype(mx.float32), self.eps
        )
        gate = mx.sigmoid(self.g_proj(x).astype(mx.float32).reshape(shape))
        return self.o_proj(
            (normed * gate).astype(dtype).reshape(batch, length, self.proj_size)
        )


# Prompt processing: the rope scores are an additive mask [heads, rows, keys]. For long prompts this is the largest
# buffer, so queries are taken in blocks and each block is evaluated before the next mask is built.
_PREFILL_QUERY_BLOCK = 128


def _blocked_prefill_attention(self, q_nope, q_extra, key, value, extra, mask):
    outs = []
    length = q_nope.shape[2]
    # Attention with 8 or fewer query rows uses another kernel, so a short last block joins the one before.
    starts = list(range(0, length, _PREFILL_QUERY_BLOCK))
    if len(starts) > 1 and length - starts[-1] <= 8:
        starts.pop()
    for i, s0 in enumerate(starts):
        s1 = starts[i + 1] if i + 1 < len(starts) else length
        pe = (q_extra[:, :, s0:s1] * self.scale) @ extra.swapaxes(-1, -2)
        if mask is not None:
            pe = mx.where(mask[..., s0:s1, :], pe, mx.finfo(pe.dtype).min)
        outs.append(
            mx.fast.scaled_dot_product_attention(
                q_nope[:, :, s0:s1], key, value, scale=self.scale, mask=pe
            )
        )
        mx.eval(outs[-1])
    return mx.concatenate(outs, axis=2)


class MLA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.nope = args.qk_nope_head_dim
        self.extra = args.qk_rope_head_dim
        self.value_dim = args.v_head_dim
        self.latent_dim = args.kv_lora_rank
        self.scale = (self.nope + self.extra) ** -0.5
        self.use_gate = args.mla_use_output_gate
        hidden = args.hidden_size
        self.q_a_proj = nn.Linear(hidden, args.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.mla_latent_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, self.num_heads * (self.nope + self.extra), bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, self.latent_dim + self.extra, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.latent_dim, eps=args.mla_latent_norm_eps)
        self.embed_q = MultiLinear(self.nope, self.latent_dim, self.num_heads)
        self.unembed_out = MultiLinear(self.latent_dim, self.value_dim, self.num_heads)
        self.o_proj = nn.Linear(self.num_heads * self.value_dim, hidden, bias=False)
        if self.use_gate:
            self.g_proj = nn.Linear(hidden, self.num_heads * self.value_dim, bias=False)

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(batch, length, self.num_heads, self.nope + self.extra).transpose(
            0, 2, 1, 3
        )
        q_nope, q_extra = mx.split(q, [self.nope], axis=-1)
        latent, extra = mx.split(self.kv_a_proj_with_mqa(x), [self.latent_dim], axis=-1)
        latent = self.kv_a_layernorm(latent)[:, None]
        extra = extra[:, None]
        if cache is not None:
            latent, extra = cache.update_and_fetch(latent, extra)
            _bound_cache_graph(cache, ("offset", "left_padding"))
        quantized = hasattr(cache, "bits")
        if quantized:
            pe_scores = mx.quantized_matmul(
                q_extra * self.scale,
                *extra,
                transpose=True,
                group_size=cache.group_size,
                bits=cache.bits,
            )
        else:
            pe_scores = (q_extra * self.scale) @ extra.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(mask, pe_scores, mx.finfo(pe_scores.dtype).min)
        attn_cache = cache
        if length == 1:
            q_nope = self.embed_q(q_nope)
            key = value = latent
        else:
            if quantized:
                latent = mx.dequantize(
                    *latent, group_size=cache.group_size, bits=cache.bits
                )
                attn_cache = None
            key = self.embed_q(latent, transpose=False)
            value = self.unembed_out(latent)
        if quantized and length == 1:
            # The extra NoPE scores have one bias per query head.
            scores = mx.quantized_matmul(
                q_nope * self.scale,
                *key,
                transpose=True,
                group_size=cache.group_size,
                bits=cache.bits,
            )
            scores = mx.softmax(scores + pe_scores, axis=-1, precise=True)
            out = mx.quantized_matmul(
                scores,
                *value,
                transpose=False,
                group_size=cache.group_size,
                bits=cache.bits,
            )
        elif not quantized and length > _PREFILL_QUERY_BLOCK:
            out = _blocked_prefill_attention(
                self, q_nope, q_extra, key, value, extra, mask
            )
        else:
            out = scaled_dot_product_attention(
                q_nope, key, value, cache=attn_cache, scale=self.scale, mask=pe_scores
            )
        if length == 1:
            out = self.unembed_out(out)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        if self.use_gate:
            out = out * mx.sigmoid(self.g_proj(x))
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, args, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)
        self.activation = SiTU(
            args.activation_situ_beta, args.activation_situ_linear_beta
        )

    def __call__(self, x):
        return self.down_proj(self.activation(self.up_proj(x), self.gate_proj(x)))


class Router(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.num_experts,))
        self.top_k = args.num_experts_per_token
        self.renormalize = args.moe_renormalize
        self.scale = args.routed_scaling_factor

    def __call__(self, x):
        scores = mx.sigmoid(x.astype(mx.float32) @ self.weight.astype(mx.float32).T)
        ranked = scores + self.e_score_correction_bias.astype(mx.float32)
        indices = mx.stop_gradient(mx.argsort(-ranked, axis=-1)[..., : self.top_k])
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if self.renormalize:
            weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)
        return indices, weights * self.scale


class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gate = Router(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            activation=SiTU(
                args.activation_situ_beta, args.activation_situ_linear_beta
            ),
        )
        self.shared_experts = (
            MLP(args, args.moe_intermediate_size * args.num_shared_experts)
            if args.num_shared_experts
            else None
        )

    def __call__(self, x):
        indices, weights = self.gate(x)
        out = self.switch_mlp(x, indices)
        out = (out.astype(mx.float32) * weights[..., None]).sum(-2).astype(x.dtype)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class GlobalScaledLinear(nn.Module):
    """An nvfp4 linear with one float32 scale for the full tensor."""

    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.weight = mx.zeros((output_dims, input_dims // 8), mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // 16), mx.uint8)
        self.global_scale = mx.array(1.0, mx.float32)
        self.freeze()

    def __call__(self, x):
        y = mx.quantized_matmul(
            x,
            self["weight"],
            self["scales"],
            transpose=True,
            group_size=16,
            bits=4,
            mode="nvfp4",
        )
        # quantized_matmul has no global scale, so apply it to the output.
        return (y.astype(mx.float32) * self["global_scale"]).astype(y.dtype)


class DecoderLayer(nn.Module):
    def __init__(self, args, index):
        super().__init__()
        self.is_linear = index + 1 in args.rda_config["rda_layers"]
        self.is_moe = (
            bool(args.num_experts)
            and index >= args.first_k_dense_replace
            and index % args.moe_layer_freq == 0
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.self_attn = RDA(args) if self.is_linear else MLA(args)
        self.self_attn_raon_mhc = MHC(args)
        self.mlp_raon_mhc = MHC(args)
        if self.is_moe:
            self.moe = MoE(args)
        else:
            self.mlp = MLP(args, args.intermediate_size)

    def __call__(self, stream, mask=None, cache=None):
        batch, length, streams, hidden = stream.shape
        rows = stream.reshape(-1, streams, hidden)
        pre, post, residual = self.self_attn_raon_mhc.route(stream)
        x = self.input_layernorm(MHC.aggregate(rows, pre)).reshape(
            batch, length, hidden
        )
        branch = self.self_attn(x, mask, cache).reshape(-1, hidden)
        rows = MHC.mix(rows, branch, post, residual)
        pre, post, residual = self.mlp_raon_mhc.route(rows.reshape(stream.shape))
        x = self.post_attention_layernorm(MHC.aggregate(rows, pre)).reshape(
            batch, length, hidden
        )
        branch = (self.moe(x) if self.is_moe else self.mlp(x)).reshape(-1, hidden)
        return MHC.mix(rows, branch, post, residual).reshape(stream.shape)


class RaonModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.streams = args.raon_mhc_streams
        self._linear_index = next(
            (i for i, layer in enumerate(self.layers) if layer.is_linear), None
        )
        self._full_index = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_linear), None
        )

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        cache = [None] * len(self.layers) if cache is None else cache
        if len(cache) != len(self.layers):
            raise ValueError("Cache must have one entry per layer")
        linear_mask = (
            create_ssm_mask(h, cache[self._linear_index])
            if self._linear_index is not None
            else None
        )
        full_mask = (
            create_attention_mask(h, cache[self._full_index], return_array=True)
            if self._full_index is not None
            else None
        )
        stream = mx.repeat(h[:, :, None], self.streams, axis=2)
        for layer, layer_cache in zip(self.layers, cache):
            stream = layer(
                stream, linear_mask if layer.is_linear else full_mask, layer_cache
            )
        return self.norm(stream.astype(mx.float32).mean(axis=2).astype(h.dtype))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = RaonModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        for path in args.raon_global_scaled_linears or []:
            *parents, name = path.split(".")
            parent = self
            for part in parents:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            linear = getattr(parent, name)
            if not isinstance(linear, nn.Linear):
                raise ValueError(f"{path} is not a linear layer")
            output_dims, input_dims = linear.weight.shape
            setattr(parent, name, GlobalScaledLinear(input_dims, output_dims))

    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        return (
            self.model.embed_tokens.as_linear(out)
            if self.args.tie_word_embeddings
            else self.lm_head(out)
        )

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            ArraysCache(4) if layer.is_linear else KVCache() for layer in self.layers
        ]

    def sanitize(self, weights):
        weights = dict(weights)
        args = self.args
        for k, v in weights.items():
            if (
                k.endswith("self_attn.f_a_proj.weight")
                and v.shape[0] != args.rda_decay_rank
            ):
                raise ValueError(
                    f"{k} has {v.shape[0]} rows but rda_decay_rank is {args.rda_decay_rank}"
                )
        for i in range(
            args.num_hidden_layers,
            args.num_hidden_layers + args.num_nextn_predict_layers,
        ):
            prefix = f"model.layers.{i}."
            weights = {
                key: value
                for key, value in weights.items()
                if not key.startswith(prefix)
            }
        for i, layer in enumerate(self.layers):
            prefix = f"model.layers.{i}."
            if layer.is_linear:
                merged = prefix + "self_attn.qkv_conv1d.weight"
                if merged in weights:
                    parts = mx.split(weights.pop(merged), 3, axis=0)
                    for name, part in zip(["q_conv1d", "k_conv1d", "v_conv1d"], parts):
                        weights[prefix + f"self_attn.{name}.weight"] = part
                for name in ["q_conv1d", "k_conv1d", "v_conv1d"]:
                    key = prefix + f"self_attn.{name}.weight"
                    if key in weights and weights[key].ndim == 2:
                        weights[key] = weights[key][..., None]
            else:
                key = prefix + "self_attn.kv_b_proj.weight"
                if key in weights:
                    w = weights.pop(key).reshape(
                        args.num_attention_heads,
                        args.qk_nope_head_dim + args.v_head_dim,
                        args.kv_lora_rank,
                    )
                    weights[prefix + "self_attn.embed_q.weight"] = w[
                        :, : args.qk_nope_head_dim
                    ].swapaxes(-1, -2)
                    weights[prefix + "self_attn.unembed_out.weight"] = w[
                        :, args.qk_nope_head_dim :
                    ]
            if layer.is_moe:
                moe = prefix + "moe."
                if moe + "experts.0.gate_proj.weight" in weights:
                    for name in ["gate_proj", "up_proj", "down_proj"]:
                        src = [
                            moe + f"experts.{e}.{name}" for e in range(args.num_experts)
                        ]
                        dst = moe + f"switch_mlp.{name}"
                        weights[dst + ".weight"] = mx.stack(
                            [weights.pop(k + ".weight") for k in src]
                        )
                        # The original checkpoint stores mxfp8 experts: e4m3 bytes and
                        # one e8m0 scale per 32 inputs, the same bytes as MLX mxfp8.
                        if src[0] + ".weight_scale_inv" in weights:
                            weights[dst + ".scales"] = mx.stack(
                                [weights.pop(k + ".weight_scale_inv") for k in src]
                            )
                            weights[dst + ".weight"] = weights[dst + ".weight"].view(
                                mx.uint32
                            )
        if args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def cast_predicate(self):
        return lambda path: not (
            path.endswith(
                (
                    "A_log",
                    "dt_bias",
                    "e_score_correction_bias",
                    "res.eta",
                    "global_scale",
                )
            )
            or (self.args.raon_fp32_export_head and path == "lm_head.weight")
        )

    @property
    def quant_predicate(self):
        return lambda path, module: not (
            path.endswith("moe.gate")
            or isinstance(module, GlobalScaledLinear)
            or (self.args.raon_fp32_export_head and path == "lm_head")
        )
