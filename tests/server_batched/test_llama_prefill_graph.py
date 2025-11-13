# ABOUTME: Validates helper utilities inside the LlamaPrefillGraph module.
# ABOUTME: Ensures head-dim inference works when attention modules lack metadata.

from pathlib import Path

from .util import ensure_mlx_stub

ensure_mlx_stub()

import mlx.core as mx
import pytest

from mlx_lm import load
from mlx_lm.server_batched.graph_decode.llama_prefill import (
    LlamaPrefillGraph,
    _infer_attn_head_dim,
)


class _DummyProj:
    def __init__(self, rows, cols):
        self.weight = mx.zeros((rows, cols), dtype=mx.float16)


class _DummyAttention:
    def __init__(
        self,
        *,
        n_heads,
        n_kv_heads,
        head_dim=None,
        hidden_size=None,
        proj_rows=None,
        proj_cols=None
    ):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        if head_dim is not None:
            self.head_dim = head_dim
        if hidden_size is not None:
            self.hidden_size = hidden_size
        default_rows = (
            hidden_size
            if hidden_size is not None
            else (n_heads * 16 if n_heads > 0 else 0)
        )
        rows = proj_rows if proj_rows is not None else default_rows
        cols = proj_cols if proj_cols is not None else rows
        self.q_proj = _DummyProj(rows, cols)
        kv_rows = n_kv_heads * max(
            1, head_dim or (rows // n_heads if n_heads > 0 else 1)
        )
        self.k_proj = _DummyProj(kv_rows, cols)


def test_infer_head_dim_uses_attr_when_present():
    attn = _DummyAttention(n_heads=2, n_kv_heads=2, head_dim=64, hidden_size=512)
    assert _infer_attn_head_dim(attn) == 64


def test_infer_head_dim_falls_back_to_hidden_size():
    attn = _DummyAttention(n_heads=4, n_kv_heads=2, hidden_size=256)
    assert _infer_attn_head_dim(attn) == 64


def test_infer_head_dim_uses_projection_shape_when_needed():
    attn = _DummyAttention(n_heads=2, n_kv_heads=2, proj_rows=512)
    assert _infer_attn_head_dim(attn) == 256


def test_infer_head_dim_raises_when_metadata_missing():
    attn = _DummyAttention(n_heads=0, n_kv_heads=0, proj_rows=0, proj_cols=0)
    with pytest.raises(ValueError):
        _infer_attn_head_dim(attn)


@pytest.mark.skipif(
    getattr(mx, "__file__", None) is None, reason="requires MLX runtime"
)
def test_prefill_graph_profile_mode_emits_metrics(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    model_dir = repo_root.parent / "local-models" / "toy-llama"
    if not model_dir.exists():
        pytest.skip("toy-llama assets missing")
    monkeypatch.setenv("MLXLM_PREFILL_PROFILE", "1")
    model, _ = load(str(model_dir))
    graph = LlamaPrefillGraph(model)
    chunk_len = 4
    fn = graph.get_compiled(
        batch_size=1,
        chunk_len=chunk_len,
        block_tables_shape=(1, 1),
        k_cache_shape=(
            len(model.layers),
            model.layers[0].self_attn.n_kv_heads,
            1,
            chunk_len,
            graph.head_dim,
        ),
        v_cache_shape=(
            len(model.layers),
            model.layers[0].self_attn.n_kv_heads,
            1,
            chunk_len,
            graph.head_dim,
        ),
        kv_map_shape=None,
        dtype=mx.float16,
        pending_flag=0,
    )
    tokens = mx.zeros((1, chunk_len), dtype=mx.int32)
    base_lens = mx.zeros((1,), dtype=mx.int32)
    block_tables = mx.zeros((1, 1), dtype=mx.int32)
    n_layers = len(model.layers)
    n_kv_heads = model.layers[0].self_attn.n_kv_heads
    head_dim = graph.head_dim
    cache_dtype = model.layers[0].self_attn.q_proj.weight.dtype
    k_cache = mx.zeros(
        (n_layers, n_kv_heads, 1, chunk_len, head_dim), dtype=cache_dtype
    )
    v_cache = mx.zeros_like(k_cache)
    overlay_shape = (n_layers, chunk_len, 1, n_kv_heads, head_dim)
    pending_k = mx.zeros(overlay_shape, dtype=cache_dtype)
    pending_v = mx.zeros_like(pending_k)
    fn(tokens, base_lens, block_tables, k_cache, v_cache, pending_k, pending_v)
    metrics = graph.consume_metrics()
    assert metrics["array_prefill_attn_s"] >= 0
    assert len(metrics["array_prefill_layer_attn_s"]) == n_layers


if __name__ == "__main__":
    pytest.main([__file__])
