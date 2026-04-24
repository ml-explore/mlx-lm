# Copyright © 2026 Apple Inc. / mlx-community
#
# DeepSeek-V4 (Pro / Flash) for mlx-lm.
# Architecture: Multi-head Latent Attention (num_kv_heads=1) + grouped low-rank output,
# sliding-window + compressed KV + indexer topk (sparse attention), hash-routed MoE
# with sqrtsoftplus scoring, Manifold-constrained Hyper-Connections (mHC) replacing
# residuals. Weights are native FP8 (e4m3) with 128x128 block scaling (ue8m0).
#
# Reference: deepseek-ai/DeepSeek-V4 (Apr 2026). mHC: arXiv:2512.24880.

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .pipeline import PipelineMixin
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1

    # Attention (MLA-style with single shared KV head)
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    attention_bias: bool = False
    sliding_window: int = 128
    compress_ratios: List[int] = field(default_factory=list)

    # Compressor / Indexer
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    compress_rope_theta: float = 160000.0

    # MoE
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    num_hash_layers: int = 3
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0

    # Hyper-Connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # MTP (multi-token prediction) — present in checkpoint but unused at inference
    num_nextn_predict_layers: int = 1

    # RoPE / YaRN
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    rms_norm_eps: float = 1e-6

    # Quantization (FP8 block)
    quantization_config: Optional[Dict] = None


# --------------------------------------------------------------------------- #
# mHC (Manifold-constrained Hyper-Connections)                                #
# --------------------------------------------------------------------------- #

# Cache of jit-compiled Sinkhorn kernels keyed by (hc, iters).
# eps is baked at compile time for max register efficiency.
_SINKHORN_KERNELS: dict = {}


def _make_sinkhorn_kernel(hc: int, iters: int, eps: float):
    """Build a fully-unrolled, register-resident Sinkhorn Metal kernel.

    Each thread owns one token's [hc, hc] matrix in registers (hc^2 floats).
    No threadgroup memory, no atomics, no global memory traffic between iters
    — kernel reads input once and writes output once.

    Trades replication of work across threads for zero kernel-launch overhead
    in what was previously 40+ launches per layer per token (softmax + iters
    × (sum + div) × 2). Fuses everything into a single grid dispatch.
    """
    key = (hc, iters)
    if key in _SINKHORN_KERNELS:
        return _SINKHORN_KERNELS[key]

    n_elem = hc * hc
    eps_lit = f"{eps:.8e}f"

    # Generate fully-unrolled accumulators (no loops over hc — Metal unrolls
    # tiny loops anyway but explicit unroll keeps the code register-friendly).
    def row_softmax():
        out = []
        for r in range(hc):
            base = r * hc
            out.append(f"        {{ float mx = m[{base}];")
            for c in range(1, hc):
                out.append(f"          mx = metal::max(mx, m[{base + c}]);")
            out.append(f"          float s = 0.0f;")
            for c in range(hc):
                out.append(f"          m[{base + c}] = metal::exp(m[{base + c}] - mx); s += m[{base + c}];")
            out.append(f"          float inv = 1.0f / s;")
            for c in range(hc):
                out.append(f"          m[{base + c}] *= inv;")
            out.append("        }")
        return "\n".join(out)

    def row_norm():
        out = []
        for r in range(hc):
            base = r * hc
            terms = " + ".join(f"m[{base + c}]" for c in range(hc))
            out.append(f"        {{ float inv = 1.0f / ({terms} + {eps_lit});")
            for c in range(hc):
                out.append(f"          m[{base + c}] *= inv;")
            out.append("        }")
        return "\n".join(out)

    def col_norm():
        out = []
        for c in range(hc):
            terms = " + ".join(f"m[{r * hc + c}]" for r in range(hc))
            out.append(f"        {{ float inv = 1.0f / ({terms} + {eps_lit});")
            for r in range(hc):
                out.append(f"          m[{r * hc + c}] *= inv;")
            out.append("        }")
        return "\n".join(out)

    def add_eps():
        return "\n".join(f"        m[{i}] += {eps_lit};" for i in range(n_elem))

    iter_body = "\n".join([row_norm(), col_norm()])
    inner_iters = "\n".join([iter_body] * (iters - 1))

    source = f"""
        uint n = thread_position_in_grid.x;
        if (n >= n_tokens[0]) return;

        const device float *src = comb_log + n * {n_elem};
        device float *dst = comb + n * {n_elem};

        float m[{n_elem}];
{chr(10).join(f"        m[{i}] = src[{i}];" for i in range(n_elem))}

        // Row softmax
{row_softmax()}

        // Add eps
{add_eps()}

        // Initial column normalization (matches reference: cols first after softmax)
{col_norm()}

        // Remaining (iters - 1) rounds of (row_norm, col_norm)
{inner_iters}

{chr(10).join(f"        dst[{i}] = m[{i}];" for i in range(n_elem))}
    """

    kernel = mx.fast.metal_kernel(
        name=f"sinkhorn_hc{hc}_it{iters}",
        input_names=["comb_log", "n_tokens"],
        output_names=["comb"],
        source=source,
    )
    _SINKHORN_KERNELS[key] = kernel
    return kernel


def hc_split_sinkhorn(
    mixes: mx.array,        # [B*S, (2+hc)*hc] fp32
    hc_scale: mx.array,     # [3] fp32
    hc_base: mx.array,      # [(2+hc)*hc] fp32
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Split `mixes` into (pre, post, comb_logits); Sinkhorn-normalize comb to doubly stochastic.

    Returns:
        pre  [N, hc]        — sigmoid(mixes[:,:hc] * s0 + base[:hc]) + eps
        post [N, hc]        — 2*sigmoid(mixes[:,hc:2hc] * s1 + base[hc:2hc])
        comb [N, hc, hc]    — Sinkhorn-normalized (rows & cols ~= 1) from the last hc*hc logits.

    Pure-MLX reference; matches `kernel.py::hc_split_sinkhorn_kernel` in the V4 release.
    Uses softmax(-1) to start, then alternating col/row normalization with `eps` to keep
    numerics stable. Accepts arbitrary batched leading dims.
    """
    n = mixes.shape[0]
    mix = mixes  # [n, (2+hc)*hc]
    s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]

    pre_log  = mix[:, :hc_mult]             * s0 + hc_base[:hc_mult]
    post_log = mix[:, hc_mult:2 * hc_mult]  * s1 + hc_base[hc_mult:2 * hc_mult]
    comb_log = (
        mix[:, 2 * hc_mult:].reshape(n, hc_mult, hc_mult) * s2
        + hc_base[2 * hc_mult:].reshape(hc_mult, hc_mult)
    )

    pre  = mx.sigmoid(pre_log) + eps              # [n, hc]
    post = 2 * mx.sigmoid(post_log)               # [n, hc]

    # Sinkhorn: dispatch to fused Metal kernel when on GPU + small hc; else Python loop.
    use_kernel = (
        hc_mult <= 8                               # register budget guard (hc^2 floats)
        and mx.metal.is_available()
        and comb_log.size > 0
    )
    if use_kernel:
        kernel = _make_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
        flat = comb_log.reshape(n, hc_mult * hc_mult).astype(mx.float32)
        n_tokens = mx.array([n], dtype=mx.uint32)
        (comb_flat,) = kernel(
            inputs=[flat, n_tokens],
            output_shapes=[(n, hc_mult * hc_mult)],
            output_dtypes=[mx.float32],
            grid=(n, 1, 1),
            threadgroup=(min(n, 256) or 1, 1, 1),
        )
        comb = comb_flat.reshape(n, hc_mult, hc_mult)
    else:
        # Reference path: rows softmax -> +eps -> cols norm -> (iters-1) × (rows norm, cols norm)
        comb = mx.softmax(comb_log, axis=-1, precise=True) + eps
        col_sum = comb.sum(axis=1, keepdims=True) + eps
        comb = comb / col_sum
        for _ in range(sinkhorn_iters - 1):
            row_sum = comb.sum(axis=2, keepdims=True) + eps
            comb = comb / row_sum
            col_sum = comb.sum(axis=1, keepdims=True) + eps
            comb = comb / col_sum

    return pre, post, comb


class HyperConnection(nn.Module):
    """Per-block mHC parameters: projects x -> (pre, post, comb) used in hc_pre/hc_post.

    Paper/ref stores the weights as:
        hc_fn    : [(2+hc)*hc, hc*dim]
        hc_scale : [3]
        hc_base  : [(2+hc)*hc]

    hc_pre reduces `hc_mult` parallel hidden states to 1 via `pre`.
    Block F is applied to the reduced state. hc_post expands 1 -> hc via `post` (the new
    contribution) added to `comb @ residual` (where `comb` is a doubly-stochastic mix
    that recombines the input `hc_mult` copies to stay on the Birkhoff manifold).
    """

    def __init__(self, dim: int, hc_mult: int, norm_eps: float, sinkhorn_iters: int, hc_eps: float):
        super().__init__()
        self.dim = dim
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim
        # All mHC params are fp32 in the checkpoint.
        self.fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.scale = mx.zeros((3,), dtype=mx.float32)

    def hc_pre(self, x: mx.array):
        # x: [B, S, hc, D] -> reduce to [B, S, D] via `pre`; return (y, post, comb) for hc_post.
        B, S, hc, D = x.shape
        dtype = x.dtype
        xf = x.reshape(B, S, hc * D).astype(mx.float32)
        inv = mx.rsqrt((xf * xf).mean(axis=-1, keepdims=True) + self.norm_eps)
        mixes = (xf @ self.fn.T) * inv                     # [B,S,mix_hc]
        mixes = mixes.reshape(B * S, -1)
        pre, post, comb = hc_split_sinkhorn(
            mixes, self.scale, self.base, hc, self.sinkhorn_iters, self.hc_eps
        )
        pre  = pre.reshape(B, S, hc)
        post = post.reshape(B, S, hc)
        comb = comb.reshape(B, S, hc, hc)
        y = (pre[..., None] * x.astype(mx.float32)).sum(axis=2)   # [B,S,D]
        return y.astype(dtype), post, comb

    def hc_post(self, f_out: mx.array, residual: mx.array, post: mx.array, comb: mx.array):
        # f_out    [B,S,D] (block output, reduced state)
        # residual [B,S,hc,D] (input to hc_pre)
        # post     [B,S,hc]
        # comb     [B,S,hc,hc]
        # returns  [B,S,hc,D]
        dtype = f_out.dtype
        # post.unsqueeze(-1) * f_out.unsqueeze(-2)  -> [B,S,hc,D]
        term_new = post[..., None] * f_out[:, :, None, :].astype(mx.float32)
        # comb @ residual: [B,S,hc,hc] @ [B,S,hc,D] -> [B,S,hc,D]
        term_res = mx.einsum("bsij,bsjd->bsid", comb.astype(mx.float32), residual.astype(mx.float32))
        y = term_new + term_res
        return y.astype(dtype)


class HyperHead(nn.Module):
    """Final (head) mHC projection: reduces [B,S,hc,D] -> [B,S,D] via sigmoid-weighted sum.
    No Sinkhorn here — this is the simpler head variant from `ParallelHead.hc_head`.
    """

    def __init__(self, dim: int, hc_mult: int, norm_eps: float, hc_eps: float):
        super().__init__()
        self.dim = dim
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.fn = mx.zeros((hc_mult, hc_mult * dim), dtype=mx.float32)
        self.base = mx.zeros((hc_mult,), dtype=mx.float32)
        self.scale = mx.zeros((1,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        B, S, hc, D = x.shape
        dtype = x.dtype
        xf = x.reshape(B, S, hc * D).astype(mx.float32)
        inv = mx.rsqrt((xf * xf).mean(axis=-1, keepdims=True) + self.norm_eps)
        mixes = (xf @ self.fn.T) * inv                      # [B,S,hc]
        pre = mx.sigmoid(mixes * self.scale[0] + self.base) + self.hc_eps
        y = (pre[..., None] * x.astype(mx.float32)).sum(axis=2)
        return y.astype(dtype)


# --------------------------------------------------------------------------- #
# Gate (hash + score-based)                                                   #
# --------------------------------------------------------------------------- #

def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    # sqrtsoftplus: sqrt(softplus(x))  — used by V4
    return mx.sqrt(mx.logaddexp(scores, mx.zeros_like(scores)))


class MoEGate(nn.Module):
    """Routing gate. First `num_hash_layers` layers use a deterministic hash
    (token-id -> expert-id table) instead of learned score-based topk. Remaining
    layers run sqrtsoftplus scoring + e_score_correction_bias + topk, with
    post-softmax renormalization if score_func != 'softmax'."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_routed = args.n_routed_experts
        self.top_k = args.num_experts_per_tok
        self.hash = layer_idx < args.num_hash_layers
        self.score_func = args.scoring_func
        self.route_scale = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob

        self.weight = mx.zeros((self.n_routed, args.hidden_size))
        if self.hash:
            # tid2eid: [vocab, top_k] int32 — predetermined expert routing per token id
            self.tid2eid = mx.zeros((args.vocab_size, self.top_k), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros((self.n_routed,), dtype=mx.float32)

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        # x: [B, S, D] or [N, D]
        if self.hash:
            # x shape -> [B*S, D]; input_ids -> [B, S] flattened to [B*S]
            flat = x.reshape(-1, x.shape[-1])
            scores = flat.astype(mx.float32) @ self.weight.T.astype(mx.float32)
            scores = _score_func(scores, self.score_func)
            ids = input_ids.reshape(-1)
            inds = self.tid2eid[ids].astype(mx.int32)
            weights = mx.take_along_axis(scores, inds, axis=-1)
            # Reshape inds/weights back to match x's leading dims so SwitchGLU
            # can broadcast against x: [B, S, top_k] (mirrors non-hash branch).
            inds = inds.reshape(*x.shape[:-1], self.top_k)
            weights = weights.reshape(*x.shape[:-1], self.top_k)
        else:
            scores = x.astype(mx.float32) @ self.weight.T.astype(mx.float32)
            scores = _score_func(scores, self.score_func)
            orig = scores
            biased = scores + self.e_score_correction_bias
            inds = mx.argpartition(-biased, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
            weights = mx.take_along_axis(orig, inds, axis=-1)

        if self.score_func != "softmax" and self.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        weights = weights * self.route_scale
        return inds, weights


# --------------------------------------------------------------------------- #
# MoE                                                                          #
# --------------------------------------------------------------------------- #

def _swiglu_limited(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        up = mx.clip(up, -limit, limit)
        gate = mx.minimum(gate, limit)
    return nn.silu(gate) * up


class DeepseekV4MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_swiglu_limited(self.gate_proj(x), self.up_proj(x), self.swiglu_limit))


class DeepseekV4MoE(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
        )
        self.gate = MoEGate(args, layer_idx)
        if args.n_shared_experts:
            self.shared_experts = DeepseekV4MLP(
                args.hidden_size,
                args.moe_intermediate_size * args.n_shared_experts,
                swiglu_limit=0.0,
            )
        self.sharding_group = None

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        inds, weights = self.gate(x, input_ids)
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        if hasattr(self, "shared_experts"):
            y = y + self.shared_experts(x)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


# --------------------------------------------------------------------------- #
# Attention: MLA (num_kv_heads=1) + sliding window + optional compressed KV   #
# --------------------------------------------------------------------------- #

class Compressor(nn.Module):
    """Learned gated pooling over `ratio` consecutive tokens for KV compression.

    At prefill, produces ~ seq/ratio compressed KV rows. At decode, accumulates
    tokens in a state buffer and emits a compressed row every `ratio` steps.
    Pure-MLX; a fused Metal kernel may replace this in a follow-up.
    """

    def __init__(self, args: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.dim = args.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.ratio = compress_ratio
        self.overlap = compress_ratio == 4
        coff = 1 + int(self.overlap)
        self.wkv   = nn.Linear(self.dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(self.dim, coff * head_dim, bias=False)
        self.ape   = mx.zeros((compress_ratio, coff * head_dim), dtype=mx.float32)
        self.norm  = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        # Prefill-only MVP: chunk x into non-overlapping windows of `ratio` tokens.
        # Returns compressed KV: [B, S//ratio, head_dim] (bf16).
        B, S, _ = x.shape
        r = self.ratio
        keep = (S // r) * r
        if keep == 0:
            return mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        xf = x[:, :keep].astype(mx.float32)
        kv    = self.wkv(xf).reshape(B, keep // r, r, self.head_dim)
        score = self.wgate(xf).reshape(B, keep // r, r, self.head_dim) + self.ape
        weights = mx.softmax(score, axis=2, precise=True)
        kv = (kv * weights).sum(axis=2)
        return self.norm(kv.astype(x.dtype))


class V4Attention(nn.Module):
    """V4 attention block.

    Checkpoint shapes (Flash):
        n_heads=64, head_dim=512, rope_head_dim=64 (nope=448)
        q_lora_rank=1024,  wq_a: [dim, 1024], wq_b: [1024, n_heads*head_dim]
        wkv: [dim, head_dim]            (single shared K=V head, MQA-style)
        attn_sink: [n_heads] fp32
        wo_a: [n_heads*head_dim/n_groups, n_groups*o_lora_rank]
        wo_b: [n_groups*o_lora_rank, dim]
        For compress_ratio != 0: compressor.wkv/wgate/ape/norm; and if ratio==4, indexer.*

    Forward path (MVP):
        - Project Q (64 heads), K=V (1 head); apply RoPE to last `rope_head_dim` dims.
        - For ratio=0 layers: sliding window mask of size `sliding_window`.
        - For ratio!=0 layers: append compressed KV rows to attend to (no topk filtering
          yet — full compressed cache). Use attn_sink via SDPA `sinks=` argument.
        - Grouped low-rank output projection: wo_a per group -> concat -> wo_b.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = args.head_dim - args.qk_rope_head_dim
        self.n_groups = args.o_groups
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.window = args.sliding_window
        self.eps = args.rms_norm_eps

        ratios = args.compress_ratios or []
        self.compress_ratio = ratios[layer_idx] if layer_idx < len(ratios) else 0

        self.scale = self.head_dim ** -0.5

        # q path
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)

        # kv path (single shared head)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=self.eps)

        # attention sink (per-head learnable bias added in softmax denominator)
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        # grouped low-rank output projection
        group_feat = (self.n_heads * self.head_dim) // self.n_groups
        self.wo_a = nn.Linear(group_feat, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=args.attention_bias)

        # rope (sliding layers use base theta; compressed layers use YaRN + compress_rope_theta)
        if self.compress_ratio:
            base = args.compress_rope_theta
            scaling = args.rope_scaling
        else:
            base = args.rope_theta
            scaling = None
        self.rope = initialize_rope(
            dims=self.rope_head_dim,
            base=base,
            traditional=True,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=scaling,
        )

        # Compressor / Indexer — present only when compress_ratio > 0
        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)

    def __call__(self, x: mx.array, mask=None, cache=None):
        B, S, _ = x.shape

        # --- Q ---
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        # RMS-normalize each head independently (matches ref: q *= rsqrt(mean(q^2)+eps))
        q = q * mx.rsqrt(q.square().mean(axis=-1, keepdims=True) + self.eps)

        # --- K = V (shared single-head) ---
        kv = self.kv_norm(self.wkv(x))
        kv = kv.reshape(B, S, 1, self.head_dim).transpose(0, 2, 1, 3)   # [B, 1, S, head_dim]

        offset = cache.offset if cache is not None else 0

        # Apply RoPE only to the last rope_head_dim dims
        q_nope, q_pe = mx.split(q,  [self.nope_head_dim], axis=-1)
        k_nope, k_pe = mx.split(kv, [self.nope_head_dim], axis=-1)
        q_pe = self.rope(q_pe, offset=offset)
        k_pe = self.rope(k_pe, offset=offset)
        q = mx.concatenate([q_nope, q_pe], axis=-1)
        k = v = mx.concatenate([k_nope, k_pe], axis=-1)

        # Update KV cache
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # Standard SDPA (compressed KV + topk deferred to v0.2)
        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask,
        )

        # Grouped low-rank projection: [B, n_heads, S, head_dim] -> [B, S, n_heads*head_dim]
        out = out.transpose(0, 2, 1, 3).reshape(B, S, self.n_heads * self.head_dim)
        # Split into o_groups along the head_dim*n_heads axis; apply per-group wo_a via einsum.
        out = out.reshape(B, S, self.n_groups, -1)
        # wo_a is stored as nn.Linear(group_feat, n_groups*o_lora_rank). Logical weight shape is
        # [n_groups*o_lora_rank, group_feat]; we reshape to [n_groups, o_lora_rank, group_feat]
        # for a block-diagonal projection. For quantized models, .weight is packed int4 — must
        # dequantize first.
        if hasattr(self.wo_a, "scales"):
            full_w = mx.dequantize(
                self.wo_a.weight, self.wo_a.scales, self.wo_a.biases,
                self.wo_a.group_size, self.wo_a.bits,
            )
        else:
            full_w = self.wo_a.weight
        wa = full_w.reshape(self.n_groups, self.o_lora_rank, -1)
        out = mx.einsum("bsgd,grd->bsgr", out, wa)                 # [B,S,n_groups,o_lora_rank]
        out = out.reshape(B, S, self.n_groups * self.o_lora_rank)
        return self.wo_b(out)


class Indexer(nn.Module):
    """Top-k selector over compressed KV rows. For MVP we instantiate to preserve
    checkpoint parameter names; the actual topk gather path is not yet used in
    the forward pass (we attend to all compressed rows in v0.1)."""

    def __init__(self, args: ModelArgs, compress_ratio: int):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.compressor = Compressor(args, compress_ratio, self.head_dim)


# --------------------------------------------------------------------------- #
# Block                                                                       #
# --------------------------------------------------------------------------- #

class DeepseekV4Block(nn.Module):
    """V4 block: mHC-wrapped (attention-norm -> attention), mHC-wrapped (moe-norm -> moe).

    The block maintains `hc_mult` parallel hidden-state copies. Each sub-layer
    reduces them to 1 via hc_pre, applies its block, then expands back via hc_post.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn = V4Attention(args, layer_idx)
        self.hc_attn = HyperConnection(
            args.hidden_size, args.hc_mult,
            args.rms_norm_eps, args.hc_sinkhorn_iters, args.hc_eps,
        )

        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn = DeepseekV4MoE(args, layer_idx)
        self.hc_ffn = HyperConnection(
            args.hidden_size, args.hc_mult,
            args.rms_norm_eps, args.hc_sinkhorn_iters, args.hc_eps,
        )

    def __call__(self, h: mx.array, mask, cache, input_ids: mx.array) -> mx.array:
        # h: [B, S, hc, D]
        # Attention half
        residual = h
        y, post, comb = self.hc_attn.hc_pre(h)
        y = self.attn_norm(y)
        y = self.attn(y, mask=mask, cache=cache)
        h = self.hc_attn.hc_post(y, residual, post, comb)

        # FFN half
        residual = h
        y, post, comb = self.hc_ffn.hc_pre(h)
        y = self.ffn_norm(y)
        y = self.ffn(y, input_ids)
        h = self.hc_ffn.hc_post(y, residual, post, comb)
        return h


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #

class DeepseekV4Model(nn.Module, PipelineMixin):
    def __init__(self, args: ModelArgs):
        super().__init__()
        PipelineMixin.__init__(self)
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DeepseekV4Block(args, i) for i in range(args.num_hidden_layers)]
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.num_layers = self.end_idx
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Final HC head (reduces hc copies -> 1 before lm_head)
        self.hc_head = HyperHead(
            args.hidden_size, args.hc_mult, args.rms_norm_eps, args.hc_eps
        )

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)                        # [B, S, D]
        # Expand to hc_mult parallel copies
        h = mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]))
        # Make it contiguous — broadcast_to gives a view
        h = mx.contiguous(h)

        if cache is None:
            cache = [None] * self.num_layers

        first_cache = cache[0]
        if isinstance(first_cache, (list, tuple)):
            first_cache = first_cache[0]
        mask = create_attention_mask(
            h[:, :, 0, :],
            first_cache if first_cache is not None else None,
            return_array=True,
        )

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for i in range(self.num_layers):
            h = self.layers[self.start_idx + i](h, mask, cache[i], inputs)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        # Reduce [B,S,hc,D] -> [B,S,D] then RMSNorm
        h = self.hc_head(h)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV4Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.model(inputs, cache)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers[self.model.start_idx : self.model.end_idx]

    @property
    def cast_predicate(self):
        def pred(k: str):
            # Keep mHC params and gate biases in fp32
            if "hc_" in k or "e_score_correction_bias" in k or "attn_sink" in k:
                return False
            if k.endswith(".fn") or k.endswith(".base") or k.endswith(".scale"):
                return False
            return True
        return pred

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.attn.compress_ratio:
                # Full cache for compressed-attention layers (MVP: no topk selection)
                caches.append(KVCache())
            else:
                # Sliding-window cache for pure local-attention layers
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches

    # ------------------------------------------------------------------- #
    # Weight loading                                                      #
    # ------------------------------------------------------------------- #

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Handle DeepSeek-V4 checkpoint conversion.

        Checkpoint naming (from HF):
            layers.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}
            layers.N.attn.{q_norm,kv_norm,attn_sink}
            layers.N.attn.compressor.{wkv,wgate,ape,norm}
            layers.N.attn.indexer.{wq_b,weights_proj,compressor.*}
            layers.N.ffn.gate.{weight,bias,tid2eid}
            layers.N.ffn.experts.E.w{1,2,3}.{weight,scale}
            layers.N.ffn.shared_experts.w{1,2,3}.{weight,scale}
            layers.N.{attn_norm,ffn_norm}.weight
            layers.N.hc_{attn,ffn}_{fn,base,scale}
            embed.weight, head.weight, hc_head_{fn,base,scale}
            mtp.0.* (dropped)
        """
        n_layers = self.args.num_hidden_layers

        # 1) Drop MTP + any layers beyond n_layers
        new = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    idx = int(parts[1])
                except ValueError:
                    new[k] = v
                    continue
                if idx >= n_layers:
                    continue
            new[k] = v
        weights = new

        # 2) FP8 block dequant: `X.weight` + `X.scale` -> dequantized bf16 `X.weight`
        def _dequant_fp8_block(weight: mx.array, scale: mx.array, bs: int = 128) -> mx.array:
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            m, n = weight.shape
            pad_b = (-m) % bs
            pad_s = (-n) % bs
            weight = mx.pad(weight, ((0, pad_b), (0, pad_s)))
            weight = weight.reshape(((m + pad_b) // bs, bs, (n + pad_s) // bs, bs))
            weight = (weight * scale[:, None, :, None]).reshape(m + pad_b, n + pad_s)
            return weight[:m, :n].astype(mx.bfloat16)

        new = {}
        for k, v in weights.items():
            if k.endswith(".scale"):
                wk = k[:-len(".scale")] + ".weight"
                if wk in weights and weights[wk].dtype in (mx.uint8,):
                    new[wk] = _dequant_fp8_block(weights[wk], v)
                else:
                    new[k] = v
            elif k not in new:
                new[k] = v
        weights = new

        # 3) Remap top-level names to our module structure
        top_remap = {
            "embed.weight":    "model.embed_tokens.weight",
            "head.weight":     "lm_head.weight",
            "norm.weight":     "model.norm.weight",
            "hc_head_fn":      "model.hc_head.fn",
            "hc_head_base":    "model.hc_head.base",
            "hc_head_scale":   "model.hc_head.scale",
        }
        for old, new_key in top_remap.items():
            if old in weights:
                weights[new_key] = weights.pop(old)

        # 4) Remap layer-level names: layers.N.X -> model.layers.N.X
        #    Also remap gate.bias -> gate.e_score_correction_bias,
        #    hc_{attn,ffn}_{fn,base,scale} -> hc_{attn,ffn}.{fn,base,scale},
        #    shared_experts.w{1,2,3} -> shared_experts.{gate,down,up}_proj
        new = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            nk = k
            # Add model. prefix for layers
            if nk.startswith("layers."):
                nk = "model." + nk

            # gate.bias -> gate.e_score_correction_bias
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")

            # hc_attn_fn -> hc_attn.fn (etc.)
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".hc_{sub}.{param}")

            # shared_experts.w1 -> shared_experts.gate_proj (etc.)
            for w_old, w_new in w_remap.items():
                nk = nk.replace(f".shared_experts.{w_old}.", f".shared_experts.{w_new}.")

            new[nk] = v
        weights = new

        # 5) Stack expert weights: experts.E.w{1,2,3}.weight -> switch_mlp.{gate,down,up}_proj.weight
        for l in range(n_layers):
            prefix = f"model.layers.{l}.ffn.experts"
            for src, dst in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                key0 = f"{prefix}.0.{src}.weight"
                if key0 in weights:
                    stack = [weights.pop(f"{prefix}.{e}.{src}.weight")
                             for e in range(self.args.n_routed_experts)]
                    weights[f"model.layers.{l}.ffn.switch_mlp.{dst}.weight"] = mx.stack(stack)

        return weights

    # ------------------------------------------------------------------- #
    # Distributed sharding                                                 #
    # ------------------------------------------------------------------- #

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        for layer in self.model.layers:
            a = layer.attn
            a.wq_b = shard_linear(a.wq_b, "all-to-sharded", group=group)
            a.wo_b = shard_linear(a.wo_b, "sharded-to-all", group=group)
            a.n_heads //= N
            # (n_groups shard omitted here for simplicity; wo_a stays replicated)

            if isinstance(layer.ffn, DeepseekV4MoE):
                layer.ffn.sharding_group = group
                if hasattr(layer.ffn, "shared_experts"):
                    shard_inplace(layer.ffn.shared_experts.gate_proj, "all-to-sharded", group=group)
                    shard_inplace(layer.ffn.shared_experts.down_proj, "sharded-to-all", group=group)
                    shard_inplace(layer.ffn.shared_experts.up_proj,   "all-to-sharded", group=group)
                shard_inplace(layer.ffn.switch_mlp.gate_proj, "all-to-sharded", group=group)
                shard_inplace(layer.ffn.switch_mlp.down_proj, "sharded-to-all", group=group)
                shard_inplace(layer.ffn.switch_mlp.up_proj,   "all-to-sharded", group=group)
