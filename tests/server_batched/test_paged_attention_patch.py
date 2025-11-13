# ABOUTME: Tests paged attention patch helpers for prefill tiling controls.
# ABOUTME: Ensures environment knob controls paged decode reuse for prefill.

from .util import ensure_mlx_stub

ensure_mlx_stub()

import os
import unittest
from unittest import mock

import mlx.core as mx
import mlx.nn.paged_kv as pkv
import numpy as np

from mlx_lm.server_batched import paged_attention_patch as patch
from mlx_lm.server_batched.paged_context import current_layer, wrap_attention_layers


class PagedAttentionPatchTests(unittest.TestCase):
    def test_prefill_tiling_respects_env_override(self):
        queries = mx.array(np.zeros((1, 1, 4, 2), dtype=np.float32))
        k_cache = mx.zeros((1, 1, 1, 2), dtype=mx.float32)
        v_cache = mx.zeros_like(k_cache)
        block_tables = mx.zeros((1, 4), dtype=mx.int32)
        context_lens = mx.ones((1,), dtype=mx.int32)
        calls = []

        def fake_paged_attention(q_slice, *_args, **_kwargs):
            calls.append(int(q_slice.shape[2]))
            return mx.zeros_like(q_slice)

        with mock.patch.object(
            mx.fast, "paged_attention", side_effect=fake_paged_attention
        ):
            with mock.patch.dict(os.environ, {"MLXLM_PREFILL_AS_DECODE_TILE": "2"}):
                patch._paged_attention_call(
                    queries,
                    k_cache,
                    v_cache,
                    block_tables,
                    context_lens,
                    kv_head_mapping=None,
                    scale=1.0,
                    quant_kwargs={},
                )

        self.assertEqual(calls, [2, 2])

    def test_prefill_uses_native_kernel_when_base_lens_available(self):
        queries = mx.array(np.ones((1, 1, 3, 2), dtype=np.float32))
        k_cache = mx.zeros((1, 1, 1, 2), dtype=mx.float32)
        v_cache = mx.zeros_like(k_cache)
        block_tables = mx.zeros((1, 4), dtype=mx.int32)
        context_lens = mx.full((1,), 3, dtype=mx.int32)
        base_lens = mx.array([1], dtype=mx.int32)
        called = {}

        def fake_prefill(*args, **kwargs):
            called["args"] = args
            called["kwargs"] = kwargs
            return mx.zeros_like(args[0])

        with mock.patch.object(
            mx.fast, "paged_attention", side_effect=AssertionError("should not decode")
        ):
            with mock.patch.object(
                mx.fast, "paged_prefill", side_effect=fake_prefill
            ) as mock_prefill:
                patch._paged_attention_call(
                    queries,
                    k_cache,
                    v_cache,
                    block_tables,
                    context_lens,
                    kv_head_mapping=None,
                    scale=1.0,
                    quant_kwargs={},
                    prefill_base_lens=base_lens,
                )
        mock_prefill.assert_called_once()
        self.assertIs(called["args"][4], base_lens)
        self.assertIs(called["args"][5], context_lens)

    def test_decode_rewraps_detached_queries_for_native_call(self):
        queries = mx.array(np.ones((1, 2, 1, 4), dtype=np.float32))
        queries_ref = queries
        k_cache = mx.zeros((2, 8, 1, 4), dtype=mx.float32)
        v_cache = mx.zeros_like(k_cache)
        block_tables = mx.zeros((1, 8), dtype=mx.int32)
        context_lens = mx.ones((1,), dtype=mx.int32)
        contiguous_called = []

        def fake_contiguous(arr):
            if arr is queries_ref:
                raise AssertionError("decode queries must be copied before contiguous")
            contiguous_called.append(arr)
            return arr

        def fake_decode(q_slice, *_args, **_kwargs):
            return mx.zeros_like(q_slice)

        with mock.patch.object(patch.mx, "contiguous", side_effect=fake_contiguous):
            with mock.patch.object(mx.fast, "paged_attention", side_effect=fake_decode):
                patch._paged_attention_call(
                    queries,
                    k_cache,
                    v_cache,
                    block_tables,
                    context_lens,
                    kv_head_mapping=None,
                    scale=1.0,
                    quant_kwargs={},
                )

        self.assertEqual(len(contiguous_called), 1)

    def test_call_returns_none_when_kernel_missing(self):
        queries = mx.array(np.ones((1, 1, 1, 2), dtype=np.float32))
        k_cache = mx.zeros((1, 1, 1, 2), dtype=mx.float32)
        v_cache = mx.zeros_like(k_cache)
        block_tables = mx.zeros((1, 1), dtype=mx.int32)
        context_lens = mx.ones((1,), dtype=mx.int32)
        with mock.patch.object(mx.fast, "paged_attention", new=None, create=True):
            result = patch._paged_attention_call(
                queries,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                kv_head_mapping=None,
                scale=1.0,
                quant_kwargs={},
            )
        self.assertIsNone(result)

    def test_prefill_falls_back_to_reference_when_native_missing(self):
        queries = mx.array(np.ones((1, 1, 2, 2), dtype=np.float32))
        k_cache = mx.zeros((1, 1, 2, 2), dtype=mx.float32)
        v_cache = mx.zeros_like(k_cache)
        block_tables = mx.zeros((1, 2), dtype=mx.int32)
        context_lens = mx.full((1,), 2, dtype=mx.int32)
        base_lens = mx.array([1], dtype=mx.int32)
        called = {}

        def fake_reference(*args, **kwargs):
            called["args"] = args
            called["kwargs"] = kwargs
            return mx.zeros_like(args[0])

        with mock.patch.object(mx.fast, "paged_prefill", new=None, create=True):
            with mock.patch.object(
                patch, "_mx_paged_prefill_reference", side_effect=fake_reference
            ):
                patch._paged_attention_call(
                    queries,
                    k_cache,
                    v_cache,
                    block_tables,
                    context_lens,
                    kv_head_mapping=None,
                    scale=1.0,
                    quant_kwargs={},
                    prefill_base_lens=base_lens,
                )
        self.assertIs(called["args"][0], queries)
        self.assertIs(called["args"][4], context_lens)
        self.assertIs(called["kwargs"]["base_lens"], base_lens)

    def test_decode_returns_none_when_kernel_raises(self):
        queries = mx.array(np.ones((1, 1, 1, 2), dtype=np.float32))
        k_cache = mx.zeros((1, 1, 1, 2), dtype=mx.float32)
        v_cache = mx.zeros_like(k_cache)
        block_tables = mx.zeros((1, 1), dtype=mx.int32)
        context_lens = mx.ones((1,), dtype=mx.int32)
        quant_kwargs = {"kv_quant_mode": "int4_v"}
        patch._DISABLED_QUANT_MODES.clear()

        call_counter = {"count": 0}

        def fake_decode(*_args, **_kwargs):
            call_counter["count"] += 1
            raise RuntimeError("kernel rejected configuration")

        with mock.patch.object(mx.fast, "paged_attention", side_effect=fake_decode):
            result = patch._paged_attention_call(
                queries,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                kv_head_mapping=None,
                scale=1.0,
                quant_kwargs=quant_kwargs,
            )
            self.assertIsNone(result)
            # second call should skip invoking the kernel entirely
            result2 = patch._paged_attention_call(
                queries,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                kv_head_mapping=None,
                scale=1.0,
                quant_kwargs=quant_kwargs,
            )
            self.assertIsNone(result2)
            self.assertEqual(call_counter["count"], 1)


class _DummyAttention:
    def __init__(self):
        self.last_layer = None

    def __call__(self, *args, **kwargs):
        self.last_layer = current_layer()
        return None


class _DummyLayer:
    def __init__(self):
        self.self_attn = _DummyAttention()


class _DummyModel:
    def __init__(self, num_layers: int):
        self.layers = [_DummyLayer() for _ in range(num_layers)]


class PagedContextWrappingTests(unittest.TestCase):
    def test_wrap_attention_layers_sets_layer_scope(self):
        model = _DummyModel(3)
        wrap_attention_layers(model)
        for idx, layer in enumerate(model.layers):
            layer.self_attn("query")
            self.assertEqual(layer.self_attn.last_layer, idx)


if __name__ == "__main__":
    unittest.main()
