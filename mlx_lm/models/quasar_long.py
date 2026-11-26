# Copyright © 2025 Apple Inc.

# Quasar-Long (silx-ai/Quasar-Preview): a hybrid linear-attention MoE model.
#
# Every layer runs standard GQA softmax attention. Layers in
# ``hybrid_attention_layers`` additionally run ONE linear-attention branch
# (quasar / raven / gla, assigned by ``hybrid_layerwise_cycle``) whose output is
# gated and added to the GQA output:
#
#     out = gqa(x) + sigmoid(replace_alpha_raw) * adapter(channel_gain * global_gain * branch(x))
#
# Branch math (faithful ports of the reference fla / goombalab-raven ops):
#   * gla    -> fla.ops.simple_gla   (scalar-gated linear attention, ALiBi slope)
#   * raven  -> fla.ops.gsa          (gated slot attention, Mamba2 decay + top-k router)
#   * quasar -> fla.ops.quasar       (gated delta-rule; re-derived from the fused kernel)

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import ArraysCache, CacheList, KVCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "quasar_long"
    vocab_size: int = 157184
    hidden_size: int = 2048
    intermediate_size: int = 5120
    moe_intermediate_size: int = 512
    num_hidden_layers: int = 20
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    num_experts: int = 256
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    first_k_dense_replace: int = 1
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.5
    use_qk_norm: bool = True
    use_qkv_bias: bool = False
    use_bias: bool = False
    use_nope: bool = True
    long_context_mode: str = "rope_short_nope_long"
    nope_after_position: int = 512
    max_position_embeddings: int = 5000000
    tie_word_embeddings: bool = False
    hybrid_attention_layers: List[int] = field(default_factory=list)
    hybrid_branch_layout: str = "layerwise"
    hybrid_layerwise_cycle: List[str] = field(default_factory=list)
    hybrid_quasar_enabled: bool = True
    hybrid_raven_enabled: bool = True
    hybrid_gla_enabled: bool = True
    hybrid_raven_slots: int = 64
    hybrid_raven_topk: int = 32
    hybrid_raven_decay_type: str = "Mamba2"
    hybrid_output_adapter_rank: int = 16
    hybrid_output_adapter_alpha: Optional[float] = None

    def branch_for_layer(self, layer_idx: int) -> Optional[str]:
        """Return 'quasar' | 'raven' | 'gla' | None for a given layer."""
        if layer_idx not in self.hybrid_attention_layers:
            return None
        enabled = {
            "quasar": self.hybrid_quasar_enabled,
            "raven": self.hybrid_raven_enabled,
            "gla": self.hybrid_gla_enabled,
        }
        cycle = [b.strip().lower() for b in self.hybrid_layerwise_cycle if enabled.get(b.strip().lower())]
        if not cycle:
            cycle = [name for name, on in enabled.items() if on] or ["quasar"]
        order = sorted(self.hybrid_attention_layers)
        pos = order.index(layer_idx)
        return cycle[pos % len(cycle)]


# ---------------------------------------------------------------------------
# norms / rope helpers
# ---------------------------------------------------------------------------
def _rms_norm(x: mx.array, weight: Optional[mx.array], eps: float) -> mx.array:
    return mx.fast.rms_norm(x.astype(mx.float32), weight, eps).astype(x.dtype)


class GroupRMSNorm(nn.Module):
    """RMSNorm over ``hidden // group_size``-wide groups (gla g_norm)."""

    def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.group_size = group_size
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        *lead, h = x.shape
        g = self.group_size
        y = x.reshape(*lead, g, h // g).astype(mx.float32)
        y = y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + self.eps)
        y = y.reshape(*lead, h)
        return (self.weight * y.astype(x.dtype)).astype(x.dtype)


def quasar_rope(
    x: mx.array, offset: int, rope_dim: int, base: float, nope_after: int, use_nope: bool
) -> mx.array:
    """Partial NeoX-style RoPE on x[..., :rope_dim], identity for pos >= nope_after.

    x layout: (B, H, L, D). Rotates the first ``rope_dim`` channels.
    """
    B, H, L, D = x.shape
    half = rope_dim // 2
    pos = (offset + mx.arange(L)).astype(mx.float32)
    inv_freq = base ** (-(mx.arange(0, rope_dim, 2).astype(mx.float32)) / rope_dim)
    freqs = pos[:, None] * inv_freq[None, :]            # (L, half)
    emb = mx.concatenate([freqs, freqs], axis=-1)       # (L, rope_dim)
    cos = mx.cos(emb)
    sin = mx.sin(emb)
    if use_nope:
        keep = (pos < nope_after)[:, None]
        cos = mx.where(keep, cos, mx.ones_like(cos))
        sin = mx.where(keep, sin, mx.zeros_like(sin))
    cos = cos[None, None, :, :].astype(x.dtype)
    sin = sin[None, None, :, :].astype(x.dtype)

    xr, xp = x[..., :rope_dim], x[..., rope_dim:]
    x1, x2 = xr[..., :half], xr[..., half:]
    rot = mx.concatenate([-x2, x1], axis=-1)
    xr = xr * cos + rot * sin
    return mx.concatenate([xr, xp], axis=-1)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    """Sigmoid group-limited top-k router (DeepSeek-V3 style)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.routed_scaling_factor
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.expert_bias = mx.zeros((args.num_experts,))

    def __call__(self, x):
        scores = mx.sigmoid((x @ self.weight.T).astype(mx.float32))
        orig = scores
        scores = scores + self.expert_bias
        ng, tg = self.n_group, self.topk_group
        scores_g = mx.unflatten(scores, axis=-1, shape=(ng, -1))
        group_scores = mx.topk(scores_g, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = ng - tg
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores_g = mx.put_along_axis(scores_g, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2)
        scores = mx.flatten(scores_g, -2, -1)

        k = self.top_k
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(orig, inds, axis=-1)
        if self.top_k > 1 and self.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        return inds, weights.astype(x.dtype)


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = MoEGate(args)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.num_experts)
        self.shared_experts = MLP(
            args.hidden_size, args.moe_intermediate_size * args.num_shared_experts
        )

    def __call__(self, x):
        inds, weights = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(x.dtype)
        return y + self.shared_experts(x)


# ---------------------------------------------------------------------------
# linear-attention branches (recurrent scans)
# ---------------------------------------------------------------------------
def _gla_recurrence(q, k, v, g, scale, state=None):
    """Faithful port of fla simple_gla naive recurrence.

    q,k,v: (B, L, H, D) ; g: (B, L, H) log-gate. Returns (o (B,L,H,V), final_state).
    """
    B, L, H, K = q.shape
    V = v.shape[-1]
    q = q.transpose(0, 2, 1, 3) * scale
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    g = g.transpose(0, 2, 1)                                   # (B,H,L)
    S = mx.zeros((B, H, K, V), dtype=mx.float32) if state is None else state
    outs = []
    for i in range(L):
        gate = mx.exp(g[:, :, i])[:, :, None, None]
        kv = k[:, :, i, :, None] * v[:, :, i, None, :]
        S = S * gate + kv
        outs.append((q[:, :, i, :, None] * S).sum(axis=-2))
    o = mx.stack(outs, axis=2).transpose(0, 2, 1, 3)          # (B,L,H,V)
    return o, S


def _gsa_recurrence(q, k, v, s, g, scale, state=None):
    """Faithful port of fla gsa naive recurrence (g provided, NG=1).

    q,k: (B,L,H,P) ; v: (B,L,H,P) ; s,g: (B,L,H,M). Returns (o (B,L,H,P), (hk,hv)).
    """
    B, L, H, P = q.shape
    M = s.shape[-1]
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    s = s.transpose(0, 2, 1, 3)
    g = g.transpose(0, 2, 1, 3)
    hk = mx.zeros((B, H, P, M), dtype=mx.float32) if state is None else state[0]
    ok = []
    for i in range(L):
        gi = mx.exp(g[:, :, i])
        hk = hk * gi[:, :, None, :] + k[:, :, i, :, None] * s[:, :, i, None, :]
        ok.append(((q[:, :, i] * scale)[:, :, :, None] * hk).sum(axis=-2))
    ok = mx.stack(ok, axis=2)                                  # (B,H,L,M)
    qv = mx.softmax(ok, axis=-1)
    hv = mx.zeros((B, H, M, P), dtype=mx.float32) if state is None else state[1]
    ov = []
    for i in range(L):
        gi = mx.exp(g[:, :, i])
        hv = hv * gi[:, :, :, None] + s[:, :, i, :, None] * v[:, :, i, None, :]
        ov.append((qv[:, :, i, :, None] * hv).sum(axis=-2))
    o = mx.stack(ov, axis=2).transpose(0, 2, 1, 3)            # (B,L,H,P)
    return o, (hk, hv)


def _build_slopes(n_heads: int) -> mx.array:
    def slopes_pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        sl = slopes_pow2(n_heads)
    else:
        cp = 2 ** math.floor(math.log2(n_heads))
        sl = slopes_pow2(cp)
        extra = slopes_pow2(2 * cp)[0::2][: n_heads - cp]
        sl = sl + extra
    return mx.array(sl, dtype=mx.float32)


class GLABranch(nn.Module):
    """simple_gla: scalar-gated linear attention (state K x V per head)."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.n_heads = args.num_attention_heads
        self.n_kv = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.groups = self.n_heads // self.n_kv
        self.rope_dim = int(self.head_dim * args.partial_rotary_factor)

        self.query_key_value = nn.Linear(
            args.hidden_size, (self.n_heads + 2 * self.n_kv) * self.head_dim, bias=args.use_qkv_bias
        )
        if args.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.dense = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=args.use_bias)
        self.g_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.g_norm = GroupRMSNorm(self.n_heads * self.head_dim, self.n_heads, eps=args.rms_norm_eps)
        slope = -_build_slopes(self.n_heads)
        if args.num_hidden_layers > 1 and layer_idx is not None:
            slope = slope * (1 - max(layer_idx - 1, 0) / (args.num_hidden_layers - 1) + 1e-5)
        self.slope = slope

    def __call__(self, x, offset=0, state=None):
        B, L, _ = x.shape
        qkv = self.query_key_value(x).reshape(B, L, self.n_heads + 2 * self.n_kv, self.head_dim)
        q = qkv[:, :, : self.n_heads]
        k = qkv[:, :, self.n_heads : self.n_heads + self.n_kv]
        v = qkv[:, :, self.n_heads + self.n_kv :]
        if self.args.use_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)
        # RoPE (unsqueeze over heads); operate in (B,H,L,D)
        q = quasar_rope(q.transpose(0, 2, 1, 3), offset, self.rope_dim, self.args.rope_theta,
                        self.args.nope_after_position, self.args.use_nope).transpose(0, 2, 1, 3)
        k = quasar_rope(k.transpose(0, 2, 1, 3), offset, self.rope_dim, self.args.rope_theta,
                        self.args.nope_after_position, self.args.use_nope).transpose(0, 2, 1, 3)
        if self.groups > 1:
            k = mx.repeat(k, self.groups, axis=2)
            v = mx.repeat(v, self.groups, axis=2)

        scale = self.head_dim**-0.5
        decay = self.slope.astype(mx.float32)                 # (H,) negative
        g = mx.broadcast_to(decay[None, None, :], (B, L, self.n_heads))  # (B,L,H) log-gate
        o, S = _gla_recurrence(q.astype(mx.float32), k.astype(mx.float32),
                               v.astype(mx.float32), g, scale, state)  # o: (B,L,H,V)
        o = o.reshape(B, L, -1).astype(x.dtype)
        o = self.g_norm(o)
        o = o * mx.sigmoid(self.g_proj(x))
        return self.dense(o), S


class RavenBranch(nn.Module):
    """Gated slot attention (GSA) with Mamba2 decay and top-k slot router."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.n_heads = args.num_attention_heads
        self.n_kv = args.num_key_value_heads
        self.groups = self.n_heads // self.n_kv
        self.head_dim = args.head_dim          # head_k_dim == head_v_dim == 128
        self.num_slots = args.hybrid_raven_slots
        self.topk = args.hybrid_raven_topk
        key_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv * self.head_dim

        self.q_proj = nn.Linear(args.hidden_size, key_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, kv_dim, bias=False)
        self.r_proj = nn.Linear(args.hidden_size, self.n_heads * self.num_slots, bias=False)
        self.a_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.A_log = mx.zeros((self.n_heads,))
        self.dt_bias = mx.zeros((self.n_heads,))
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.g_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(key_dim, args.hidden_size, bias=False)

    def __call__(self, x, offset=0, state=None):
        B, L, _ = x.shape
        H, P, M = self.n_heads, self.head_dim, self.num_slots
        q = self.q_proj(x).reshape(B, L, H, P)
        k = self.k_proj(x).reshape(B, L, self.n_kv, P)
        v = self.v_proj(x).reshape(B, L, self.n_kv, P)
        router = self.r_proj(x).reshape(B, L, H, M)

        # Mamba2 decay: f = -exp(A_log) * softplus(a_proj + dt_bias), per-head scalar
        a = self.a_proj(x).astype(mx.float32) + self.dt_bias
        f_head = (-mx.exp(self.A_log.astype(mx.float32)) * nn.softplus(a))[..., None]  # (B,L,H,1)

        # swish feature map on q,k then qk-norm
        q = nn.silu(q)
        k = nn.silu(k)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = nn.silu(v)

        # sigmoid top-k slot router
        scores = mx.sigmoid(router.astype(mx.float32))                 # (B,L,H,M)
        idx = mx.argpartition(-scores, kth=self.topk - 1, axis=-1)[..., : self.topk]
        w = mx.take_along_axis(scores, idx, axis=-1)
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-9)
        s_multihot = mx.put_along_axis(mx.zeros_like(scores), idx, w, axis=-1)

        f = f_head * s_multihot          # (B,L,H,M)  decay (<=0) on selected slots
        s = 1.0 - mx.exp(f)              # (B,L,H,M)  slot scores

        # expand kv heads 4 -> 16
        if self.groups > 1:
            k = mx.repeat(k, self.groups, axis=2)
            v = mx.repeat(v, self.groups, axis=2)

        o, new_state = _gsa_recurrence(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
            s.astype(mx.float32), f.astype(mx.float32), scale=1.0, state=state,
        )                                                           # o: (B,L,H,P)
        o = o.reshape(B, L, -1).astype(x.dtype)
        o = _rms_norm(nn.silu(o), self.g_norm.weight, self.args.rms_norm_eps)
        return self.o_proj(o), new_state


class QuasarBranch(nn.Module):
    """Gated delta-rule linear attention (fla.ops.quasar, re-derived).

    State S is (head_dim_k x head_dim_v) per head. Per step:
        beta_t   = sigmoid(b_proj)                         (per head)
        psi_t    = -exp(A_log) * softplus(f_proj + dt_bias) (per key channel)
        ||k||^2  = sum_d k_t^2  (clamped to >= 0.05)
        alpha_t  = (1 - exp(-beta * ||k||^2)) / ||k||^2
        S        = (exp(psi) over key axis) * S
        S        = S + alpha * k_t^T (v_t - k_t @ S)
        o_t      = q_t @ S
    q,k are l2-normalized; q is scaled by head_dim^-0.5.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        d = self.n_heads * self.head_dim
        self.q_proj = nn.Linear(args.hidden_size, d, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, d, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, d, bias=False)
        self.b_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.f_proj = nn.Linear(args.hidden_size, d, bias=False)
        # g_proj is a 2-layer Sequential -> stored as an indexed list (g_proj.0 / g_proj.1)
        self.g_proj = [
            nn.Linear(args.hidden_size, self.head_dim, bias=False),
            nn.Linear(self.head_dim, d, bias=True),
        ]
        self.o_proj = nn.Linear(d, args.hidden_size, bias=False)
        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.A_log = mx.zeros((self.n_heads,))
        self.dt_bias = mx.zeros((d,))

    def __call__(self, x, offset=0, state=None):
        B, L, _ = x.shape
        H, P = self.n_heads, self.head_dim
        # branch-local rms-normalize of the input (no learnable weight)
        xf = x.astype(mx.float32)
        xn = (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + 1e-6)).astype(x.dtype)

        q = nn.silu(self.q_proj(xn)).reshape(B, L, H, P)
        k = nn.silu(self.k_proj(xn)).reshape(B, L, H, P)
        v = nn.silu(self.v_proj(xn)).reshape(B, L, H, P)
        beta = mx.sigmoid(self.b_proj(xn).astype(mx.float32))            # (B,L,H)
        fg = self.f_proj(xn).reshape(B, L, H, P).astype(mx.float32)
        fg = fg + self.dt_bias.reshape(H, P)[None, None]
        psi = -mx.exp(self.A_log.astype(mx.float32))[None, None, :, None] * nn.softplus(fg)  # (B,L,H,P)

        # l2-norm q,k then scale q
        qf = q.astype(mx.float32)
        kf = k.astype(mx.float32)
        qf = qf * mx.rsqrt((qf * qf).sum(-1, keepdims=True) + 1e-12)
        kf = kf * mx.rsqrt((kf * kf).sum(-1, keepdims=True) + 1e-12)
        qf = qf * (P**-0.5)
        vf = v.astype(mx.float32)

        # (B,H,L,P)
        qf = qf.transpose(0, 2, 1, 3)
        kf = kf.transpose(0, 2, 1, 3)
        vf = vf.transpose(0, 2, 1, 3)
        psi = psi.transpose(0, 2, 1, 3)
        beta = beta.transpose(0, 2, 1)                                   # (B,H,L)

        if state is None:
            S = mx.zeros((B, H, P, P), dtype=mx.float32)                 # (key x value)
        else:
            S = state
        outs = []
        for i in range(L):
            k_i = kf[:, :, i, :]                                         # (B,H,P)
            v_i = vf[:, :, i, :]
            q_i = qf[:, :, i, :]
            knorm2 = mx.maximum((k_i * k_i).sum(-1, keepdims=True), 0.05)  # (B,H,1)
            alpha = (1.0 - mx.exp(-beta[:, :, i, None] * knorm2)) / (knorm2 + 1e-8)
            decay = mx.exp(psi[:, :, i, :])                              # (B,H,P) over key axis
            S = S * decay[:, :, :, None]
            v_pred = (k_i[:, :, :, None] * S).sum(axis=-2)               # (B,H,V)
            delta = v_i - v_pred
            S = S + alpha[:, :, :, None] * (k_i[:, :, :, None] * delta[:, :, None, :])
            o = (q_i[:, :, :, None] * S).sum(axis=-2)                    # (B,H,V)
            outs.append(o)
        o = mx.stack(outs, axis=2)                                      # (B,H,L,P)
        o = o.transpose(0, 2, 1, 3).astype(x.dtype)                     # (B,L,H,P)

        # gated rms-norm output: rms(o)*weight * sigmoid(g_proj(x))
        gate = self.g_proj[1](self.g_proj[0](x)).reshape(B, L, H, P)
        y = _rms_norm(o, self.o_norm.weight, self.args.rms_norm_eps)
        y = y * mx.sigmoid(gate)
        y = y.reshape(B, L, -1)
        return self.o_proj(y), S


# ---------------------------------------------------------------------------
# hybrid attention wrapper
# ---------------------------------------------------------------------------
_BRANCH_STATE_SIZE = {"gla": 1, "raven": 2, "quasar": 1}


class HybridAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        # Standard GQA params live DIRECTLY on this module (matches checkpoint:
        # `attention.query_key_value`, `attention.dense`, ...).
        self.n_heads = args.num_attention_heads
        self.n_kv = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.rope_dim = int(self.head_dim * args.partial_rotary_factor)
        self.query_key_value = nn.Linear(
            args.hidden_size, (self.n_heads + 2 * self.n_kv) * self.head_dim, bias=args.use_qkv_bias
        )
        self.dense = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=args.use_bias)
        if args.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.branch_name = args.branch_for_layer(layer_idx)
        if self.branch_name is None:
            return
        rank = args.hybrid_output_adapter_rank
        alpha = args.hybrid_output_adapter_alpha
        self.adapter_scale = (alpha if alpha is not None else max(rank, 1)) / max(rank, 1)
        self.replace_alpha_raw = mx.zeros((1,))
        self.branch_mix_logits = mx.zeros((3,))
        self.branch_output_gain = mx.ones((1,))
        self.branch_global_output_gain = mx.ones((1,))
        self.branch_output_channel_gain = mx.ones((args.hidden_size,))
        self.branch_local_window_mix_logit = mx.zeros((1,))
        self.branch_output_adapter_down = nn.Linear(args.hidden_size, rank, bias=False)
        self.branch_output_adapter_up = nn.Linear(rank, args.hidden_size, bias=False)
        if self.branch_name == "gla":
            self.gla_attention = GLABranch(args, layer_idx)
        elif self.branch_name == "raven":
            self.raven_attention = RavenBranch(args, layer_idx)
        elif self.branch_name == "quasar":
            self.quasar_attention = QuasarBranch(args, layer_idx)

    def _gqa(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        qkv = self.query_key_value(x).reshape(B, L, self.n_heads + 2 * self.n_kv, self.head_dim)
        q = qkv[:, :, : self.n_heads]
        k = qkv[:, :, self.n_heads : self.n_heads + self.n_kv]
        v = qkv[:, :, self.n_heads + self.n_kv :]
        if self.args.use_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        rope = lambda t: quasar_rope(t, offset, self.rope_dim, self.args.rope_theta,
                                     self.args.nope_after_position, self.args.use_nope)
        q, k = rope(q), rope(k)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        return self.dense(out.transpose(0, 2, 1, 3).reshape(B, L, -1))

    def __call__(self, x, mask=None, cache=None):
        if self.branch_name is None:
            return self._gqa(x, mask=mask, cache=cache)

        attn_cache = cache[0] if cache is not None else None
        gqa = self._gqa(x, mask=mask, cache=attn_cache)

        offset = attn_cache.offset - x.shape[1] if attn_cache is not None else 0
        bcache = cache[1] if cache is not None else None
        n_state = _BRANCH_STATE_SIZE[self.branch_name]
        if bcache is None or bcache[0] is None:
            st = None
        elif n_state == 1:
            st = bcache[0]
        else:
            st = tuple(bcache[j] for j in range(n_state))

        if self.branch_name == "gla":
            branch_out, new_state = self.gla_attention(x, offset, st)
            new_state = [new_state]
        elif self.branch_name == "raven":
            branch_out, new_state = self.raven_attention(x, offset, st)
            new_state = list(new_state)
        else:
            branch_out, new_state = self.quasar_attention(x, offset, st)
            new_state = [new_state]

        if bcache is not None:
            for j, s in enumerate(new_state):
                bcache[j] = s

        # global_gain * branch (output_gain cancels), then channel gain, then LoRA adapter
        lin = self.branch_global_output_gain * branch_out
        lin = lin * self.branch_output_channel_gain.reshape(1, 1, -1)
        lin = lin + self.adapter_scale * self.branch_output_adapter_up(
            self.branch_output_adapter_down(lin)
        )
        alpha = mx.sigmoid(self.replace_alpha_raw)
        return gqa + alpha * lin.astype(gqa.dtype)


# ---------------------------------------------------------------------------
# decoder / model
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attention = HybridAttention(args, layer_idx)
        if layer_idx >= args.first_k_dense_replace:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.attention(self.input_layernorm(x), mask=mask, cache=cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class QuasarLongModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None):
        h = self.word_embeddings(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        # the standard-attention KV cache lives at cache[i][0] for hybrid layers
        first = cache[0][0] if isinstance(cache[0], CacheList) else cache[0]
        mask = create_attention_mask(h, first)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = QuasarLongModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for i in range(self.args.num_hidden_layers):
            branch = self.args.branch_for_layer(i)
            if branch is None:
                caches.append(KVCache())
            else:
                caches.append(CacheList(KVCache(), ArraysCache(size=_BRANCH_STATE_SIZE[branch])))
        return caches

    def sanitize(self, weights):
        new = {}
        E = self.args.num_experts
        inter = self.args.moe_intermediate_size
        for k, v in weights.items():
            if k.endswith("rotary_emb.inv_freq"):
                continue
            if k.endswith("mlp.experts_w12"):
                # (E, hidden, 2*inter) -> gate/up each (E, inter, hidden)
                gate = v[:, :, :inter].transpose(0, 2, 1)
                up = v[:, :, inter:].transpose(0, 2, 1)
                p = k[: -len("experts_w12")]
                new[p + "switch_mlp.gate_proj.weight"] = gate
                new[p + "switch_mlp.up_proj.weight"] = up
                continue
            if k.endswith("mlp.experts_w3"):
                # (E, inter, hidden) -> down (E, hidden, inter)
                p = k[: -len("experts_w3")]
                new[p + "switch_mlp.down_proj.weight"] = v.transpose(0, 2, 1)
                continue
            new[k] = v
        return new

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("gate") or "adapter" in path or path.endswith("a_proj") or path.endswith("b_proj"):
                return False
            return True

        return predicate
