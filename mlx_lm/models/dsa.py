# Copyright © 2026 Apple Inc.
# Sparse prefill for DeepSeek sparse attention (DSA) models (deepseek_v32, glm_moe_dsa):
# attention over each query's top-k keys from the indexer.

"""Moved to ``mlx_lm.layers.dsa``; this module re-exports it."""

from mlx_lm.layers.dsa import (
    SPARSE_QUERY_BLOCK,
    blocked_topk,
    fused_sparse_attention,
    min_context,
    sparse_attention,
)

__all__ = [
    "SPARSE_QUERY_BLOCK",
    "blocked_topk",
    "fused_sparse_attention",
    "min_context",
    "sparse_attention",
]
