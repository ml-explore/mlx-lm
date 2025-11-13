# ABOUTME: Validates paged-attention initialization inside ModelRunner.
# ABOUTME: Ensures KVBlockManager is created only when geometry is inferrable.

from .util import ensure_mlx_stub

ensure_mlx_stub()

import types
import unittest
from unittest import mock

import mlx.core as mx

from mlx_lm.server_batched import engine as engine_module
from mlx_lm.server_batched.engine import ModelRunner


class _PagedTokenizer:
    eos_token_id = 0
    eos_token_ids = [0]

    @property
    def detokenizer(self):
        detok = types.SimpleNamespace()
        detok.add_token = lambda *_: None
        detok.finalize = lambda: None
        detok.last_segment = ""
        return detok


class _StubAttention:
    def __init__(self):
        self.n_heads = 2
        self.n_kv_heads = 1
        self.head_dim = 4
        self.q_proj = types.SimpleNamespace(weight=mx.zeros((1,), dtype=mx.float16))


class _StubLayer:
    def __init__(self):
        self.self_attn = _StubAttention()


class _PagedModel:
    def __init__(self):
        self.args = types.SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            hidden_size=8,
        )
        self.layers = [_StubLayer() for _ in range(2)]
        self.embed_tokens = types.SimpleNamespace(
            weight=mx.zeros((1,), dtype=mx.float16)
        )

    def __call__(self, tokens, cache=None):
        arr = mx.array(tokens)
        batch = int(arr.shape[0]) if arr.ndim >= 1 else 1
        return mx.zeros((batch, 1, 1), dtype=mx.float32)


class _NoLayerModel:
    def __call__(self, tokens, cache=None):
        arr = mx.array(tokens)
        batch = int(arr.shape[0]) if arr.ndim >= 1 else 1
        return mx.zeros((batch, 1, 1), dtype=mx.float32)


class ModelRunnerPagedInitTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _PagedTokenizer()

    @mock.patch.object(engine_module, "KVBlockManager")
    def test_initializes_kv_manager_when_backend_is_paged(self, mock_kv):
        mock_kv.return_value = types.SimpleNamespace()
        model = _PagedModel()
        with mock.patch.object(
            mx.fast, "_paged_attention_prewarm", autospec=True
        ) as mock_prewarm:
            runner = ModelRunner(
                model=model,
                tokenizer=self.tokenizer,
                max_num_seqs=2,
                prefill_chunk=4,
                force_legacy_generator=True,
                attn_backend="paged",
                kv_block_size=32,
                kv_pool_blocks=128,
                paged_vec_width=2,
                paged_threads_per_head=64,
            )
        self.assertTrue(runner.paged_backend_enabled)
        self.assertIsNotNone(runner.kv_manager)
        mock_kv.assert_called_once()
        kwargs = mock_kv.call_args.kwargs
        self.assertEqual(kwargs["block_size"], 32)
        self.assertEqual(kwargs["max_blocks"], 128)
        self.assertIn("kv_quantization", kwargs)
        self.assertIsNone(kwargs["kv_quantization"])
        mock_prewarm.assert_called_once()

    @mock.patch.object(engine_module, "KVBlockManager")
    def test_falls_back_to_dense_when_geometry_missing(self, mock_kv):
        model = _NoLayerModel()
        runner = ModelRunner(
            model=model,
            tokenizer=self.tokenizer,
            max_num_seqs=2,
            prefill_chunk=4,
            force_legacy_generator=True,
            attn_backend="paged",
        )
        self.assertFalse(runner.paged_backend_enabled)
        self.assertIsNone(runner.kv_manager)
        mock_kv.assert_not_called()

    @mock.patch.object(engine_module, "KVBlockManager")
    def test_initializes_quant_spec_when_requested(self, mock_kv):
        if engine_module.QuantSpec is None:
            self.skipTest("QuantSpec unavailable")
        mock_kv.return_value = types.SimpleNamespace()
        model = _PagedModel()
        runner = ModelRunner(
            model=model,
            tokenizer=self.tokenizer,
            max_num_seqs=2,
            prefill_chunk=4,
            force_legacy_generator=True,
            attn_backend="paged",
            kv_quant_mode="int4_v",
            kv_quant_group_size=32,
        )
        self.assertTrue(runner.paged_backend_enabled)
        kwargs = mock_kv.call_args.kwargs
        self.assertIsNotNone(kwargs.get("kv_quantization"))

    @mock.patch.object(engine_module, "KVBlockManager")
    def test_forces_float16_dtype_for_non_float_models(self, mock_kv):
        mock_kv.return_value = types.SimpleNamespace()
        model = _PagedModel()
        model.embed_tokens = types.SimpleNamespace(weight=mx.zeros((1,), dtype=mx.int8))
        runner = ModelRunner(
            model=model,
            tokenizer=self.tokenizer,
            max_num_seqs=2,
            prefill_chunk=4,
            force_legacy_generator=True,
            attn_backend="paged",
        )
        self.assertTrue(runner.paged_backend_enabled)
        kwargs = mock_kv.call_args.kwargs
        self.assertEqual(kwargs["dtype"], mx.float16)


if __name__ == "__main__":
    unittest.main()
