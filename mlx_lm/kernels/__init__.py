# Copyright © 2026 Apple Inc.

"""Fused Metal kernels, one kernel per file."""

from .depth_attention import depth_attention
from .expert_routing import expert_routing
from .norm_add_norm import norm_add_norm, pre_norm
from .post_expert_norm_combine import post_expert_norm_combine
from .qk_norm import qk_norm
from .qk_norm_rope import qk_norm_rope
from .relative_attention import relative_attention
from .rms_norm_sigmoid_gate import rms_norm_sigmoid_gate
from .sparse_mla import sparse_mla_attention

__all__ = [
    "depth_attention",
    "expert_routing",
    "norm_add_norm",
    "post_expert_norm_combine",
    "pre_norm",
    "qk_norm",
    "qk_norm_rope",
    "relative_attention",
    "rms_norm_sigmoid_gate",
    "sparse_mla_attention",
]
