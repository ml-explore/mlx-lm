# ABOUTME: Ensures ModelRunner propagates array prefill metrics during decode.
# ABOUTME: Verifies decode stats include telemetry for scheduler consumers.

from .util import ensure_mlx_stub

ensure_mlx_stub()

import types
import unittest
from unittest import mock

import mlx.core as mx

from mlx_lm.server_batched.engine import ModelRunner


class _StubTokenizer:
    eos_token_id = 0
    eos_token_ids = [0]

    @property
    def detokenizer(self):
        detok = types.SimpleNamespace()
        detok.add_token = lambda *_: None
        detok.finalize = lambda: None
        detok.last_segment = ""
        return detok


class _StubModel:
    def __init__(self):
        self.layers = []

    def __call__(self, tokens, cache=None):
        arr = mx.array(tokens)
        batch = int(arr.shape[0]) if arr.ndim >= 1 else 1
        return mx.zeros((batch, 1, 1), dtype=mx.float32)

    def make_cache(self):
        return [types.SimpleNamespace(state=(None, None), offset=0)]


class _TelemetrySlotGenerator:
    def __init__(self):
        self.paged_cache = None
        self._last_decode_profile = {"batch_size": 1}

    def decode_chunk(
        self,
        contexts,
        *,
        max_steps,
        decode_state,
        safe_bump,
        emit_callback,
    ):
        return {
            "iterations": 1,
            "model_s": 0.0,
            "kernel_s": 0.0,
            "total_s": 0.0,
        }

    def prefill_slice_stats(self):
        return {
            "array_prefill_chunk_ms_total": 12.5,
            "array_prefill_chunk_count": 1.0,
            "array_prefill_first_chunk_ms": 12.5,
        }

    def compile_stats(self):
        return {}


class ModelRunnerTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.model = _StubModel()
        self.tokenizer = _StubTokenizer()

    def test_decode_includes_prefill_array_metrics(self):
        runner = ModelRunner(
            model=self.model,
            tokenizer=self.tokenizer,
            max_num_seqs=2,
            prefill_chunk=4,
        )
        runner.slot_generator = _TelemetrySlotGenerator()
        ctx = types.SimpleNamespace()
        ctx.state = types.SimpleNamespace(finished=False)
        stats = runner.decode([ctx])
        self.assertIn("array_prefill_chunk_ms_total", stats)
        self.assertEqual(stats["array_prefill_chunk_ms_total"], 12.5)
        self.assertIn("array_prefill_chunk_count", stats)
        self.assertEqual(stats["array_prefill_chunk_count"], 1.0)

    @mock.patch("mlx_lm.server_batched.engine.mx.get_cache_memory", return_value=512)
    @mock.patch("mlx_lm.server_batched.engine.mx.get_peak_memory", return_value=2048)
    @mock.patch("mlx_lm.server_batched.engine.mx.get_active_memory", return_value=1024)
    def test_collect_step_stats_includes_memory_metrics(
        self, mock_active, mock_peak, mock_cache
    ):
        runner = ModelRunner(
            model=self.model,
            tokenizer=self.tokenizer,
            max_num_seqs=2,
            prefill_chunk=4,
        )
        runner.kv_pool_blocks = 1024
        runner.kv_pool_meta = {"device_limit_bytes": 4096.0}
        stats = runner.collect_step_stats()
        self.assertIn("memory_active_bytes", stats)
        self.assertEqual(stats["memory_active_bytes"], 1024.0)
        self.assertIn("memory_peak_bytes", stats)
        self.assertEqual(stats["memory_peak_bytes"], 2048.0)
        self.assertIn("memory_cache_bytes", stats)
        self.assertEqual(stats["memory_cache_bytes"], 512.0)
        self.assertIn("memory_limit_bytes", stats)
        self.assertEqual(stats["memory_limit_bytes"], 4096.0)
        self.assertIn("memory_utilization", stats)
        self.assertAlmostEqual(stats["memory_utilization"], 0.25)

    def test_memory_pressure_warning_emitted_once_until_cleared(self):
        runner = ModelRunner(
            model=self.model,
            tokenizer=self.tokenizer,
            max_num_seqs=2,
            prefill_chunk=4,
        )
        runner.kv_pool_blocks = 1024
        runner.kv_pool_meta = {"device_limit_bytes": 1000.0}
        active_sequence = [900, 900, 500, 900]

        def _next_active():
            return active_sequence.pop(0)

        with mock.patch(
            "mlx_lm.server_batched.engine.mx.get_active_memory",
            side_effect=_next_active,
        ) as mock_active, mock.patch(
            "mlx_lm.server_batched.engine.mx.get_peak_memory", return_value=0
        ), mock.patch(
            "mlx_lm.server_batched.engine.mx.get_cache_memory", return_value=0
        ), mock.patch(
            "mlx_lm.server_batched.engine.logging.warning"
        ) as mock_warning:
            runner.collect_step_stats()
            mock_warning.assert_called_once()
            mock_warning.reset_mock()

            # Still above threshold; warning should not repeat.
            runner.collect_step_stats()
            mock_warning.assert_not_called()

            # Drop below reset threshold.
            runner.collect_step_stats()
            mock_warning.assert_not_called()

            # Exceed again -> warning re-emits.
            runner.collect_step_stats()
            mock_warning.assert_called_once()
        self.assertEqual(mock_active.call_count, 4)


if __name__ == "__main__":
    unittest.main()
