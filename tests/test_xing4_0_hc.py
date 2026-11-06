# Copyright © 2026 Apple Inc.

"""Numerical check of the MLX hyper-connection against the reference torch code.

The two torch classes below are copied verbatim from modeling_xing4_0.py in
XingChen-AGI/Xing4.0-29B-A4B (only the nn.Parameter init is filled in).
"""

import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mlx_lm.models.xing4_0 import HyperConnection, ModelArgs


class RefUnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        ).to(x.dtype)


class RefHyperConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = RefUnweightedRMSNorm(eps=config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.hc_fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(mix))
        self.hc_scale = nn.Parameter(torch.empty(3))
        self.mhc_h_res_clamp_max = config.mhc_h_res_clamp_max
        self.mhc_h_res_clamp_min = config.mhc_h_res_clamp_min

    def forward(self, hidden_streams):
        ori_dtype = hidden_streams.dtype
        hc = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre_w, post_w, comb_w = (
            F.linear(flat.to(ori_dtype), self.hc_fn.to(ori_dtype))
            .float()
            .split([hc, hc, hc * hc], dim=-1)
        )
        pre_b, post_b, comb_b = self.hc_base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.hc_scale.unbind(0)
        pre = torch.sigmoid(pre_w * pre_scale + pre_b)
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(
            *comb_w.shape[:-1], hc, hc
        ) * comb_scale + comb_b.view(hc, hc)
        comb_logits = torch.clamp(
            comb_logits, min=self.mhc_h_res_clamp_min, max=self.mhc_h_res_clamp_max
        )
        comb_max = comb_logits.amax(dim=-1, keepdim=True)
        comb = torch.exp(comb_logits - comb_max)
        for _ in range(self.hc_sinkhorn_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        collapsed = (pre.unsqueeze(-1).to(ori_dtype) * hidden_streams).sum(dim=2)
        return post.to(ori_dtype), comb.to(ori_dtype), collapsed.to(ori_dtype)


class TestXing40HyperConnection(unittest.TestCase):
    def _run(self, dtype):
        torch.manual_seed(0)
        args = ModelArgs(
            model_type="xing4_0",
            hidden_size=64,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            rms_norm_eps=1e-6,
        )
        ref = RefHyperConnection(args)
        with torch.no_grad():
            ref.hc_fn.normal_(0, 0.05)
            ref.hc_base.normal_(0, 0.5)
            ref.hc_scale.copy_(torch.tensor([0.7, 1.3, 2.1]))

        x = torch.randn(2, 5, args.hc_mult, args.hidden_size)
        with torch.no_grad():
            ref_post, ref_comb, ref_collapsed = ref(x.to(dtype))

        hc = HyperConnection(args)
        hc.load_weights(
            [
                ("hc_fn", mx.array(ref.hc_fn.detach().numpy())),
                ("hc_base", mx.array(ref.hc_base.detach().numpy())),
                ("hc_scale", mx.array(ref.hc_scale.detach().numpy())),
            ]
        )
        mx_dtype = mx.bfloat16 if dtype is torch.bfloat16 else mx.float32
        # MLX float32 matmuls are lower precision on the GPU, so the exact
        # comparison runs on the CPU.
        stream = mx.cpu if dtype is torch.float32 else mx.default_device()
        with mx.stream(stream):
            post, comb, collapsed = hc(mx.array(x.numpy()).astype(mx_dtype))
            mx.eval(post, comb, collapsed)

        for name, got, want in (
            ("post", post, ref_post),
            ("comb", comb, ref_comb),
            ("collapsed", collapsed, ref_collapsed),
        ):
            got = np.array(got.astype(mx.float32))
            want = want.float().numpy()
            err = np.abs(got - want).max()
            scale = max(np.abs(want).max(), 1e-6)
            self.assertLess(
                err / scale,
                2e-2 if dtype is torch.bfloat16 else 1e-5,
                f"{name}: max rel err {err / scale:.3e}",
            )

    def test_float32(self):
        self._run(torch.float32)

    def test_bfloat16(self):
        self._run(torch.bfloat16)

    def test_comb_is_doubly_stochastic(self):
        args = ModelArgs(model_type="xing4_0", hidden_size=64, hc_mult=4)
        hc = HyperConnection(args)
        hc.hc_fn = mx.random.normal(hc.hc_fn.shape) * 0.05
        hc.hc_base = mx.random.normal(hc.hc_base.shape)
        _, comb, _ = hc(mx.random.normal((2, 5, args.hc_mult, args.hidden_size)))
        comb = np.array(comb.astype(mx.float32))
        self.assertLess(np.abs(comb.sum(-1) - 1).max(), 1e-3)
        self.assertLess(np.abs(comb.sum(-2) - 1).max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
