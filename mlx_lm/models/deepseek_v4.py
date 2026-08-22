import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoConfig, PreTrainedConfig

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchLinear


class DeepseekV4HFConfig(PreTrainedConfig):
    model_type = "deepseek_v4"

    def __init__(self, rope_scaling=None, **kwargs):
        self.rope_scaling = rope_scaling
        super().__init__(**kwargs)


AutoConfig.register("deepseek_v4", DeepseekV4HFConfig, exist_ok=True)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_hash_layers: int = 3
    num_nextn_predict_layers: int = 1
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    o_groups: int = 8
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2048
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    norm_topk_prob: bool = True
    sliding_window: int = 128
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    max_position_embeddings: int = 1048576
    attention_bias: bool = False


def _apply_inverse_rope(x: mx.array, rope_fn, offset: int) -> mx.array:
    sh = x.shape
    rd = sh[-1]
    pairs = x.reshape(*sh[:-1], rd // 2, 2)
    flip = mx.array([1.0, -1.0], dtype=x.dtype)
    y_conj = rope_fn((pairs * flip).reshape(sh), offset)
    return (y_conj.reshape(*sh[:-1], rd // 2, 2) * flip).reshape(sh)


def _hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, n_iters, eps):
    hc = hc_mult
    pre = mx.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * mx.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb_logits = (
        mixes[..., 2 * hc :].reshape(*mixes.shape[:-1], hc, hc) * hc_scale[2]
        + hc_base[2 * hc :].reshape(hc, hc)
    )
    comb = mx.softmax(comb_logits, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(n_iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, n_iters, eps, norm_eps):
    B, L, H, D = x.shape
    xf = x.reshape(B, L, H * D).astype(mx.float32)
    rsqrt = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + norm_eps)
    mixes = (xf @ hc_fn.T) * rsqrt
    pre, post, comb = _hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, n_iters, eps)
    combined = mx.sum(pre[..., None] * x.astype(mx.float32), axis=2)
    return combined.astype(x.dtype), post, comb


def _hc_post(x, residual, post, comb):
    term_new = post[..., None] * x[:, :, None, :].astype(mx.float32)
    term_res = comb.astype(mx.float32) @ residual.astype(mx.float32)
    return (term_new + term_res).astype(x.dtype)


class Compressor(nn.Module):
    def __init__(self, args: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = compress_ratio == 4
        coff = 2 if self.overlap else 1
        self.out_dim = coff * head_dim
        self.wkv = nn.Linear(args.hidden_size, self.out_dim, bias=False)
        self.wgate = nn.Linear(args.hidden_size, self.out_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, self.out_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        B, S, _ = x.shape
        r = self.compress_ratio
        keep = (S // r) * r
        if keep == 0:
            return mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        xc = x[:, :keep].astype(mx.float32)
        kv = self.wkv(xc).reshape(B, keep // r, r, -1)
        score = self.wgate(xc).reshape(B, keep // r, r, -1) + self.ape
        if self.overlap:
            d = self.head_dim
            kv_ov = mx.zeros((B, keep // r, 2 * r, d), dtype=kv.dtype)
            kv_ov[:, :, r:] = kv[:, :, :, d:]
            kv_ov[:, 1:, :r] = kv[:, :-1, :, :d]
            kv = kv_ov
            score_ov = mx.full((B, keep // r, 2 * r, d), float("-inf"), dtype=score.dtype)
            score_ov[:, :, r:] = score[:, :, :, d:]
            score_ov[:, 1:, :r] = score[:, :-1, :, :d]
            score = score_ov
        weights = mx.softmax(score, axis=2, precise=True)
        kv = (kv * weights).sum(axis=2)
        return self.norm(kv.astype(x.dtype))


class Indexer(nn.Module):
    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.wq_b = nn.Linear(args.q_lora_rank, args.index_n_heads * args.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(args.hidden_size, args.index_n_heads, bias=False)
        self.compressor = Compressor(args, compress_ratio, args.index_head_dim)


class Attention(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rd = args.qk_rope_head_dim
        self.nope_dim = args.head_dim - args.qk_rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.scale = args.head_dim ** -0.5
        self.eps = args.rms_norm_eps

        compress_ratio = args.compress_ratios[layer_id] if args.compress_ratios else 0
        self.compress_ratio = compress_ratio

        self.wq_a = nn.Linear(args.hidden_size, args.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(args.q_lora_rank, eps=self.eps)
        self.wq_b = nn.Linear(args.q_lora_rank, args.num_attention_heads * args.head_dim, bias=False)
        self.wkv = nn.Linear(args.hidden_size, args.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(args.head_dim, eps=self.eps)
        D_group = args.num_attention_heads * args.head_dim // args.o_groups
        self.wo_a = [nn.Linear(D_group, args.o_lora_rank, bias=False) for _ in range(args.o_groups)]
        self.wo_b = nn.Linear(args.o_groups * args.o_lora_rank, args.hidden_size, bias=False)
        self.attn_sink = mx.zeros((args.num_attention_heads,))

        if compress_ratio:
            self.compressor = Compressor(args, compress_ratio, args.head_dim)
            if compress_ratio == 4:
                self.indexer = Indexer(args, compress_ratio)

        if compress_ratio and args.rope_scaling is not None:
            yarn_cfg = dict(args.rope_scaling)
            rope_base = args.compress_rope_theta
        else:
            yarn_cfg = None
            rope_base = args.rope_theta

        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=rope_base,
            traditional=True,
            scaling_config=yarn_cfg,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache=None, x_full=None):
        if x_full is None:
            x_full = x
        B, L, _ = x.shape

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = q * mx.rsqrt(mx.mean(q * q, axis=-1, keepdims=True) + self.eps)

        kv = self.kv_norm(self.wkv(x))

        offset = cache.offset if cache is not None else 0

        q_nope = q[..., : self.nope_dim]
        q_pe = self.rope(q[..., self.nope_dim :], offset)
        q = mx.concatenate([q_nope, q_pe], axis=-1)

        kv_nope = kv[..., : self.nope_dim]
        kv_pe = self.rope(kv[..., self.nope_dim :].reshape(B, 1, L, self.rd), offset).squeeze(1)
        kv = mx.concatenate([kv_nope, kv_pe], axis=-1)

        compressed = None
        if self.compress_ratio and L >= self.compress_ratio:
            compressed = self.compressor(x_full)
            if compressed.shape[1] == 0:
                compressed = None

        if cache is not None:
            kv, _ = cache.update_and_fetch(kv[:, None, :, :], kv[:, None, :, :])
            kv = kv.squeeze(1)

        if compressed is not None:
            kv = mx.concatenate([compressed, kv], axis=1)
            n_comp = compressed.shape[1]
            if mask is not None and not isinstance(mask, str):
                pad_shape = list(mask.shape)
                pad_shape[-1] = n_comp
                # Compressed positions are always visible: True for bool masks, 0 for additive.
                fill = True if mask.dtype == mx.bool_ else 0
                pad_mask = mx.full(pad_shape, fill, dtype=mask.dtype)
                mask = mx.concatenate([pad_mask, mask], axis=-1)

        k = kv[:, None, :, :]
        v = kv[:, None, :, :]

        o = scaled_dot_product_attention(
            q, k, v, cache, scale=self.scale, mask=mask, sinks=self.attn_sink.astype(q.dtype)
        )
        o_nope = o[..., : self.nope_dim]
        o_pe = _apply_inverse_rope(o[..., self.nope_dim :], self.rope, offset)
        o = mx.concatenate([o_nope, o_pe], axis=-1)
        o = o.transpose(0, 2, 1, 3)

        o = o.reshape(B, L, self.n_groups, -1)
        o = mx.concatenate(
            [self.wo_a[g](o[:, :, g, :]) for g in range(self.n_groups)], axis=-1
        )
        return self.wo_b(o)


class Gate(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.topk = args.num_experts_per_tok
        self.scoring_func = args.scoring_func
        self.route_scale = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.is_hash = layer_id < args.num_hash_layers
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        if self.is_hash:
            self.tid2eid = mx.zeros((args.vocab_size, args.num_experts_per_tok), dtype=mx.int32)
        else:
            self.bias = mx.zeros((args.n_routed_experts,))

    def __call__(self, x, input_ids=None):
        scores = x.astype(mx.float32) @ self.weight.T
        if self.scoring_func == "softmax":
            scores = mx.softmax(scores, axis=-1)
        elif self.scoring_func == "sigmoid":
            scores = mx.sigmoid(scores)
        else:  # sqrtsoftplus
            scores = mx.sqrt(nn.softplus(scores))

        original_scores = scores
        if not self.is_hash:
            scores = scores + self.bias

        if self.is_hash and input_ids is not None:
            indices = self.tid2eid[input_ids.reshape(-1)]
        else:
            indices = mx.stop_gradient(
                mx.argpartition(-scores, kth=self.topk, axis=-1)[..., : self.topk]
            )

        weights = mx.take_along_axis(original_scores, indices, axis=-1)
        if self.scoring_func != "softmax" and self.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-9)
        weights = (weights * self.route_scale).astype(x.dtype)
        return weights, indices


class Expert(nn.Module):
    def __init__(self, d_in, d_out, swiglu_limit=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_out, bias=False)
        self.w2 = nn.Linear(d_out, d_in, bias=False)
        self.w3 = nn.Linear(d_in, d_out, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x):
        gate = self.w1(x)
        up = self.w3(x)
        if self.swiglu_limit > 0:
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
            gate = mx.minimum(gate, self.swiglu_limit)
        return self.w2(nn.silu(gate) * up)


class Experts(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        n = args.n_routed_experts
        d, e = args.hidden_size, args.moe_intermediate_size
        self.w1 = SwitchLinear(d, e, n, bias=False)
        self.w2 = SwitchLinear(e, d, n, bias=False)
        self.w3 = SwitchLinear(d, e, n, bias=False)
        self.swiglu_limit = args.swiglu_limit

    def __call__(self, x, indices, weights):
        x = mx.expand_dims(x, (-2, -3))
        gate = self.w1(x, indices)
        up = self.w3(x, indices)
        if self.swiglu_limit > 0:
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
            gate = mx.minimum(gate, self.swiglu_limit)
        hidden = nn.silu(gate) * up
        out = self.w2(hidden, indices)
        return (out.squeeze(-2) * weights[..., None]).sum(axis=1)


class MoE(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.gate = Gate(layer_id, args)
        self.experts = Experts(args)
        # Shared experts have NO swiglu_limit (ref Expert(dim, inter_dim) called without kwarg).
        self.shared_experts = Expert(args.hidden_size, args.moe_intermediate_size, swiglu_limit=0.0)

    def __call__(self, x, input_ids=None):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        weights, indices = self.gate(x_flat, input_ids)
        routed = self.experts(x_flat, indices, weights).reshape(B, L, D)
        shared = self.shared_experts(x)
        return (routed + shared).astype(x.dtype)


class DeepseekV4Block(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        hc = args.hc_mult
        hc_dim = hc * args.hidden_size
        mix_hc = (2 + hc) * hc
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_attn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_attn_scale = mx.zeros((3,), dtype=mx.float32)
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_ffn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_ffn_scale = mx.zeros((3,), dtype=mx.float32)
        self._args = args

    def __call__(self, x, mask=None, cache=None, input_ids=None):
        a = self._args

        residual = x
        y, post, comb = _hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            a.hc_mult, a.hc_sinkhorn_iters, a.hc_eps, a.rms_norm_eps,
        )
        attn_out = self.attn(self.attn_norm(y), mask=mask, cache=cache, x_full=y)
        x = _hc_post(attn_out, residual, post, comb)

        residual = x
        y, post, comb = _hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            a.hc_mult, a.hc_sinkhorn_iters, a.hc_eps, a.rms_norm_eps,
        )
        ffn_out = self.ffn(self.ffn_norm(y), input_ids)
        x = _hc_post(ffn_out, residual, post, comb)
        return x


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DeepseekV4Block(i, args) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        hc = args.hc_mult
        hc_dim = hc * args.hidden_size
        self.hc_head_fn = mx.zeros((hc, hc_dim), dtype=mx.float32)
        self.hc_head_base = mx.zeros((hc,), dtype=mx.float32)
        self.hc_head_scale = mx.zeros((1,), dtype=mx.float32)

    def __call__(self, inputs, cache=None):
        h = self.embed(inputs)  # [B, L, D]
        h = mx.repeat(mx.expand_dims(h, 2), self.args.hc_mult, axis=2)  # [B, L, hc, D]

        mask = create_attention_mask(h[:, :, 0, :], cache)
        for i, layer in enumerate(self.layers):
            h = layer(h, mask, cache[i] if cache is not None else None, inputs)

        # Reduce hc copies via sigmoid-weighted sum (HyperHead)
        B, L, hc, D = h.shape
        hf = h.reshape(B, L, hc * D).astype(mx.float32)
        rsqrt = mx.rsqrt(mx.mean(hf * hf, axis=-1, keepdims=True) + self.args.hc_eps)
        mixes = (hf @ self.hc_head_fn.T) * rsqrt
        pre = mx.sigmoid(mixes * self.hc_head_scale[0] + self.hc_head_base) + self.args.hc_eps
        h_out = mx.sum(pre[..., None] * h.astype(mx.float32), axis=2).astype(h.dtype)

        return self.head(self.norm(h_out))

    def sanitize(self, weights: dict) -> dict:
        # Drop MTP layers (training-only).
        weights = {k: v for k, v in weights.items() if not k.startswith("mtp.")}

        # Stack per-expert weights (raw HF) -> SwitchLinear-format batched [n_exp, ...].
        n_exp = self.args.n_routed_experts
        for l in range(self.args.num_hidden_layers):
            pfx = f"layers.{l}.ffn.experts"
            for proj in ("w1", "w2", "w3"):
                if f"{pfx}.{proj}.weight" in weights:
                    continue  # already-stacked community quant
                if f"{pfx}.0.{proj}.weight" not in weights:
                    continue
                for kind in ("weight", "scales", "biases"):
                    keys = [f"{pfx}.{e}.{proj}.{kind}" for e in range(n_exp)]
                    if all(k in weights for k in keys):
                        weights[f"{pfx}.{proj}.{kind}"] = mx.stack(
                            [weights.pop(k) for k in keys]
                        )
        return weights

    @property
    def layers_list(self):
        return self.layers

    @property
    def cast_predicate(self):
        excluded = {
            "hc_attn_fn", "hc_attn_base", "hc_attn_scale",
            "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
            "hc_head_fn", "hc_head_base", "hc_head_scale",
            "attn_sink",
        }

        def predicate(k):
            return not any(e in k for e in excluded)

        return predicate
