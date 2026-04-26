# Copyright © 2026 Apple Inc.

"""Manifold-constrained Hyper-Connections (mHC) — shared building blocks.

DeepSeek-V4 introduced multi-HyperConnection (mHC) as a residual replacement:
expand the hidden state into `hc_mult` parallel copies, mix them via a
doubly-stochastic matrix on the Birkhoff polytope, apply the block, and
recombine. This module hosts the `nn.Module` layers; the Sinkhorn projection
that produces the doubly-stochastic mixing matrix lives in
`mlx_lm.models.sinkhorn`.

Two layers:
  - HyperConnection — per-block mHC: hc_pre reduces hc_mult -> 1; block F runs;
    hc_post expands 1 -> hc_mult via (post * f_out + comb @ residual).
  - HyperHead      — final-layer head variant: sigmoid-weighted reduction
    hc_mult -> 1 with no Sinkhorn (simpler than HyperConnection).

References:
  - mHC: arXiv:2512.24880 (DeepSeek, Dec 2025)
  - HC base: arXiv:2409.19606 (Sep 2024)
"""

import mlx.core as mx
import mlx.nn as nn

from .sinkhorn import hc_split_sinkhorn


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
        self._fn_t = None  # lazy transpose cache (avoids 86 .T calls/token)

    def hc_pre(self, x: mx.array):
        B, S, hc, D = x.shape
        dtype = x.dtype
        xf = x.reshape(B, S, hc * D).astype(mx.float32)
        xf_norm = mx.fast.rms_norm(xf, weight=None, eps=self.norm_eps)
        if self._fn_t is None:
            self._fn_t = self.fn.T
        mixes = (xf_norm @ self._fn_t).reshape(B * S, -1)
        pre, post, comb = hc_split_sinkhorn(
            mixes, self.scale, self.base, hc, self.sinkhorn_iters, self.hc_eps
        )
        pre  = pre.reshape(B, S, hc)
        post = post.reshape(B, S, hc)
        comb = comb.reshape(B, S, hc, hc)
        y = (pre[..., None] * x.astype(mx.float32)).sum(axis=2)
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
        term_res = comb.astype(mx.float32) @ residual.astype(mx.float32)
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
        self._fn_t = None  # lazy transpose cache

    def __call__(self, x: mx.array):
        B, S, hc, D = x.shape
        dtype = x.dtype
        xf = x.reshape(B, S, hc * D).astype(mx.float32)
        inv = mx.rsqrt((xf * xf).mean(axis=-1, keepdims=True) + self.norm_eps)
        if self._fn_t is None:
            self._fn_t = self.fn.T
        mixes = (xf @ self._fn_t) * inv                     # [B,S,hc]
        pre = mx.sigmoid(mixes * self.scale[0] + self.base) + self.hc_eps
        y = (pre[..., None] * x.astype(mx.float32)).sum(axis=2)
        return y.astype(dtype)
