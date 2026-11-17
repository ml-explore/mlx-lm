# Copyright © 2026 Apple Inc.

from functools import partial
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.layers.attention import MultiLinear, scaled_dot_product_attention
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.layers.recurrent import gated_delta_update, normalize_qk
from mlx_lm.models.cache import KVCache
from mlx_lm.models.kimi_k3.config import TextArgs
from mlx_lm.models.kimi_k3.kernels import attn_res_combine
from mlx_lm.models.kimi_linear.layers import ShortConv1d

__all__ = [
    "KimiK3DecoderLayer",
    "KimiK3DeltaAttention",
    "KimiK3MLAAttention",
    "KimiK3MLP",
    "KimiK3SparseMoE",
    "ResidualBlocks",
    "SiTU",
]


@partial(mx.compile, shapeless=True)
def _situ(x, gate, beta, linear_beta):
    dtype = x.dtype
    gate = gate.astype(mx.float32)
    x = x.astype(mx.float32)
    a = beta * mx.tanh(gate / beta) * mx.sigmoid(gate)
    if linear_beta is not None:
        x = linear_beta * mx.tanh(x / linear_beta)
    return (a * x).astype(dtype)


class KimiK3MLP(nn.Module):
    def __init__(self, args: TextArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        dim = args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.beta = args.activation_situ_beta or 1.0
        self.linear_beta = args.activation_situ_linear_beta

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _situ(self.up_proj(x), self.gate_proj(x), self.beta, self.linear_beta)
        )


class ResidualBlocks:
    def __init__(self, eps: float):
        self.eps = eps
        self.raw = None
        self.inv_rms = None

    def append(self, x: mx.array):
        xf = x.astype(mx.float32)
        n = mx.rsqrt((xf * xf).mean(axis=-1) + self.eps)[None]
        r = x[None]
        if self.raw is None:
            self.raw = r
            self.inv_rms = n
        else:
            self.raw = mx.concatenate([self.raw, r])
            self.inv_rms = mx.concatenate([self.inv_rms, n])


class KimiK3DeltaAttention(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        cfg = args.linear_attn_config

        self.layer_idx = layer_idx
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.conv_kernel = cfg["short_conv_kernel_size"]
        self.projection_dim = self.num_heads * self.head_dim
        self.scale = float(self.head_dim) ** -0.5
        self.lower_bound = cfg.get("gate_lower_bound", None)
        self.use_full_rank_gate = cfg.get("use_full_rank_gate", False)

        hidden = args.hidden_size
        self.qkv_proj = nn.Linear(hidden, 3 * self.projection_dim, bias=False)
        self.qkv_conv = ShortConv1d(3 * self.projection_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)

        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        else:
            self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)

        self.A_log = mx.log(
            mx.random.uniform(low=1.0, high=16.0, shape=(self.num_heads,))
        )
        self.dt_bias = mx.zeros((self.projection_dim,))

        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)
        self._step = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        dtype = x.dtype
        P = self.projection_dim

        if cache is not None:
            conv_state, ssm_state = cache
            lengths = cache.lengths
        else:
            conv_state = None
            ssm_state = None
            lengths = None

        if (
            T == 1
            and not self.training
            and mask is None
            and lengths is None
            and cache is not None
            and mx.metal.is_available()
        ):
            if conv_state is None:
                conv_state = mx.zeros((B, self.conv_kernel - 1, 3 * P), dtype=dtype)
            if ssm_state is None:
                ssm_state = mx.zeros(
                    (B, self.num_heads, self.head_dim, self.head_dim),
                    dtype=mx.float32,
                )
            if self._step is None:
                self._step = mx.compile(self._decode_core)
            y, conv_state, ssm_state = self._step(x, conv_state, ssm_state)
            cache[0] = conv_state
            cache[1] = ssm_state
            cache.advance(1)
            return y

        if conv_state is None:
            conv_state = mx.zeros((B, self.conv_kernel - 1, 3 * P), dtype=dtype)

        qkv, conv_state = self.qkv_conv(self.qkv_proj(x), conv_state, mask, lengths)

        if cache is not None:
            cache[0] = conv_state

        q = qkv[..., :P].reshape(B, T, self.num_heads, self.head_dim)
        k = qkv[..., P : 2 * P].reshape(B, T, self.num_heads, self.head_dim)
        v = qkv[..., 2 * P :].reshape(B, T, self.num_heads, self.head_dim)

        q, k = normalize_qk(q, k, inv_scale=self.scale, eps=1e-6)

        a_logits = self.f_b_proj(self.f_a_proj(x)).reshape(
            B, T, self.num_heads, self.head_dim
        )
        b_logits = self.b_proj(x).reshape(B, T, self.num_heads)

        out, ssm_state = gated_delta_update(
            q,
            k,
            v,
            a_logits,
            b_logits,
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=ssm_state,
            mask=mask,
            use_kernel=not self.training,
            lower_bound=self.lower_bound,
        )

        if cache is not None:
            cache[1] = ssm_state
            cache.advance(T)

        if self.use_full_rank_gate:
            gate = self.g_proj(x)
        else:
            gate = self.g_b_proj(self.g_a_proj(x))
        gate = gate.reshape(B, T, self.num_heads, self.head_dim)
        out = (
            self.o_norm(out.reshape(B, T, self.num_heads, self.head_dim))
            * mx.sigmoid(gate)
        ).reshape(B, T, -1)
        return self.o_proj(out)

    def _decode_core(self, x, conv_state, ssm_state):
        B = x.shape[0]
        P = self.projection_dim
        qkv, conv_state = self.qkv_conv(self.qkv_proj(x), conv_state, None, None)

        q = qkv[..., :P].reshape(B, 1, self.num_heads, self.head_dim)
        k = qkv[..., P : 2 * P].reshape(B, 1, self.num_heads, self.head_dim)
        v = qkv[..., 2 * P :].reshape(B, 1, self.num_heads, self.head_dim)

        q, k = normalize_qk(q, k, inv_scale=self.scale, eps=1e-6)

        a_logits = self.f_b_proj(self.f_a_proj(x)).reshape(
            B, 1, self.num_heads, self.head_dim
        )
        b_logits = self.b_proj(x).reshape(B, 1, self.num_heads)

        out, ssm_state = gated_delta_update(
            q,
            k,
            v,
            a_logits,
            b_logits,
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=ssm_state,
            mask=None,
            use_kernel=True,
            lower_bound=self.lower_bound,
        )

        if self.use_full_rank_gate:
            gate = self.g_proj(x)
        else:
            gate = self.g_b_proj(self.g_a_proj(x))
        gate = gate.reshape(B, 1, self.num_heads, self.head_dim)
        out = (
            self.o_norm(out.reshape(B, 1, self.num_heads, self.head_dim))
            * mx.sigmoid(gate)
        ).reshape(B, 1, -1)
        return self.o_proj(out), conv_state, ssm_state


class KimiK3MLAAttention(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        if not args.mla_use_nope:
            raise ValueError("Only NoPE MLA is supported (mla_use_nope=True)")
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = self.q_head_dim**-0.5
        self.use_gate = args.mla_use_output_gate

        hidden = args.hidden_size
        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(hidden, self.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(
                hidden, self.num_heads * self.q_head_dim, bias=False
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)
        if self.use_gate:
            self.g_proj = nn.Linear(
                hidden, self.num_heads * self.v_head_dim, bias=False
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if self.q_lora_rank is not None:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )

        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.use_gate:
            output = output * mx.sigmoid(self.g_proj(x))
        return self.o_proj(output)


class SiTU(nn.Module):
    def __init__(self, beta: float = 1.0, linear_beta: Optional[float] = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return _situ(x, gate, self.beta, self.linear_beta)


@mx.compile
def _group_expert_select(
    gates: mx.array,
    bias: Optional[mx.array],
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
) -> Tuple[mx.array, mx.array]:
    in_type = gates.dtype
    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    if bias is not None:
        scores = scores + bias.astype(scores.dtype)

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores,
            mx.stop_gradient(group_idx),
            mx.array(0.0, dtype=scores.dtype),
            axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    inds = mx.stop_gradient(inds)
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)

    if top_k > 1 and renormalize:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator

    return inds, (scores * routed_scaling_factor).astype(in_type)


class KimiK3SparseMoE(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        hidden = args.hidden_size
        experts = args.num_experts
        self.latent_size = args.routed_expert_hidden_size

        expert_dim = self.latent_size or hidden
        self.gate = nn.Linear(hidden, experts, bias=False)
        self.switch_mlp = SwitchGLU(
            expert_dim,
            args.moe_intermediate_size,
            experts,
            activation=SiTU(
                args.activation_situ_beta or 1.0, args.activation_situ_linear_beta
            ),
        )
        self.e_score_correction_bias = mx.zeros((experts,), dtype=mx.float32)

        if self.latent_size is not None:
            self.routed_expert_down_proj = nn.Linear(
                hidden, self.latent_size, bias=False
            )
            self.routed_expert_up_proj = nn.Linear(self.latent_size, hidden, bias=False)
            if args.latent_moe_use_norm:
                self.routed_expert_norm = nn.RMSNorm(
                    self.latent_size, eps=args.rms_norm_eps
                )
            else:
                self.routed_expert_norm = None
        else:
            self.routed_expert_norm = None

        if args.num_shared_experts:
            shared_hidden = args.moe_intermediate_size * args.num_shared_experts
            self.shared_experts = KimiK3MLP(args, intermediate_size=shared_hidden)
        else:
            self.shared_experts = None

        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        scores = self.gate(x)
        inds, weights = _group_expert_select(
            scores,
            self.e_score_correction_bias,
            self.args.num_experts_per_token,
            self.args.num_expert_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.moe_renormalize,
        )
        y = self.routed_expert_down_proj(x) if self.latent_size is not None else x
        y = self.switch_mlp(y, inds)
        y = (y * weights[..., None]).sum(axis=-2)
        shared = self.shared_experts(x) if self.shared_experts is not None else None
        if self.sharding_group is not None:
            if shared is not None:
                split = y.shape[-1]
                combined = mx.distributed.all_sum(
                    mx.concatenate([y, shared], axis=-1), group=self.sharding_group
                )
                y, shared = mx.split(combined, [split], axis=-1)
            else:
                y = mx.distributed.all_sum(y, group=self.sharding_group)
        if self.routed_expert_norm is not None:
            y = self.routed_expert_norm(y)
        if self.latent_size is not None:
            y = self.routed_expert_up_proj(y)
        if shared is not None:
            y = y + shared
        return y


def _attn_res_mix(
    blocks: ResidualBlocks,
    partial_sum: mx.array,
    w_eff: mx.array,
    eps: float,
    use_kernel: bool = True,
) -> mx.array:
    if blocks.raw is None:
        return partial_sum
    return attn_res_combine(
        blocks.raw, blocks.inv_rms, partial_sum, w_eff, eps, use_kernel=use_kernel
    )


class KimiK3DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.eps = args.rms_norm_eps
        kda_layers = args.linear_attn_config["kda_layers"]
        self.is_linear = (layer_idx + 1) in kda_layers

        if self.is_linear:
            self.self_attn = KimiK3DeltaAttention(args, layer_idx)
        else:
            self.self_attn = KimiK3MLAAttention(args)

        if (
            (args.num_experts or 0) > 0
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        ):
            self.mlp = KimiK3SparseMoE(args)
        else:
            self.mlp = KimiK3MLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        self.use_attn_res = args.attn_res_block_size is not None
        if self.use_attn_res:
            self.is_block_start = layer_idx % args.attn_res_block_size == 0
            self.self_attention_res_proj = nn.Linear(args.hidden_size, 1, bias=False)
            self.self_attention_res_norm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.mlp_res_proj = nn.Linear(args.hidden_size, 1, bias=False)
            self.mlp_res_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self._attn_res_w_eff = None
            self._mlp_res_w_eff = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        blocks: Optional[ResidualBlocks] = None,
    ) -> Tuple[mx.array, Optional[ResidualBlocks]]:
        if not self.use_attn_res:
            h = x + self.self_attn(self.input_layernorm(x), mask, cache)
            return h + self.mlp(self.post_attention_layernorm(h)), blocks

        if self.training or self._attn_res_w_eff is None:
            self._attn_res_w_eff = self.self_attention_res_norm.weight.astype(
                mx.float32
            ) * self.self_attention_res_proj.weight.reshape(-1)
            self._mlp_res_w_eff = self.mlp_res_norm.weight.astype(
                mx.float32
            ) * self.mlp_res_proj.weight.reshape(-1)

        partial_sum = x
        h = _attn_res_mix(
            blocks, partial_sum, self._attn_res_w_eff, self.eps, not self.training
        )
        if self.is_block_start:
            blocks.append(partial_sum)
            partial_sum = None

        y = self.self_attn(self.input_layernorm(h), mask, cache)
        partial_sum = y if partial_sum is None else partial_sum + y

        h = _attn_res_mix(
            blocks, partial_sum, self._mlp_res_w_eff, self.eps, not self.training
        )
        partial_sum = partial_sum + self.mlp(self.post_attention_layernorm(h))
        return partial_sum, blocks
