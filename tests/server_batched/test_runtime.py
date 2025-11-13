# ABOUTME: Tests continuous batching runtime wrapper around scheduler.
# ABOUTME: Ensures requests enqueue and trigger scheduler steps.

from .util import ensure_mlx_stub

ensure_mlx_stub()

import time
import types
import unittest
from unittest import mock

import numpy as np

from mlx_lm.generate import GenerationResponse
from mlx_lm.server_batched import runtime as runtime_module
from mlx_lm.server_batched.runtime import ContinuousBatchingRuntime, create_runtime
from mlx_lm.server_batched.state import SequenceContext, SequenceState


class DummyTokenizer:
    eos_token_id = 0

    @property
    def detokenizer(self):
        detok = types.SimpleNamespace()
        detok.add_token = lambda *_: None
        detok.finalize = lambda: None
        detok.last_segment = ""
        return detok


class DummyModel:
    def __call__(self, tokens, cache=None):
        arr = np.array(tokens)
        if arr.ndim == 2:
            batch = arr.shape[0]
            return np.zeros((batch, 1, 1), dtype=np.float32)
        return np.zeros((1, 1, 1), dtype=np.float32)

    @property
    def layers(self):
        return []


class FakeRunner:
    def __init__(self):
        self.prefill_requests = []
        self.decode_calls = 0
        self._prefill_count = 0
        self._prefill_tokens = 0
        self._prefix_hits = 0.0
        self._prefix_lookups = 0.0

    def begin_step(self):
        self._prefill_count = 0
        self._prefill_tokens = 0

    def build_context(
        self,
        request_id,
        prompt,
        *,
        max_new_tokens,
        sampler_settings,
        stopping_settings,
        logit_bias,
        repetition_penalty,
        repetition_context_size,
    ):
        state = SequenceState(
            request_id=request_id, prompt_len=len(prompt), max_new_tokens=max_new_tokens
        )
        ctx = SequenceContext(
            state=state,
            prompt_tokens=list(prompt),
            sampler_settings=sampler_settings,
            stopping_settings=stopping_settings,
            tokenizer=DummyTokenizer(),
        )
        ctx.history_tokens = list(prompt)
        return ctx

    def prefill_context(self, ctx, tokens):
        self.prefill_requests.append((ctx.state.request_id, tokens))
        ctx.state.prompt_pos += tokens
        self._prefill_count += 1
        self._prefill_tokens += tokens
        return tokens

    def decode(self, contexts):
        self.decode_calls += 1
        self._prefix_hits += len(contexts)
        self._prefix_lookups += max(len(contexts), 1)
        for ctx in contexts:
            ctx.state.generated_tokens += 1
            ctx.state.finished = True
            ctx.enqueue_event(
                GenerationResponse(
                    text="hello",
                    token=1,
                    logprobs=None,
                    from_draft=False,
                    prompt_tokens=ctx.state.prompt_len,
                    prompt_tps=0.0,
                    generation_tokens=ctx.state.generated_tokens,
                    generation_tps=0.0,
                    peak_memory=0.0,
                    finish_reason="stop",
                )
            )
        return {
            "decode_iterations": 1 if contexts else 0,
            "decode_tokens": len(contexts),
            "decode_duration_s": 0.0,
            "prefill_calls": self._prefill_count,
            "prefill_tokens": self._prefill_tokens,
            "prefill_duration_s": 0.0,
            "prefix_hits": self._prefix_hits,
            "prefix_lookups": self._prefix_lookups,
            "prefix_hit_rate": (
                (self._prefix_hits / self._prefix_lookups)
                if self._prefix_lookups
                else 0.0
            ),
            "prefix_tokens_reused": self._prefix_hits,
        }

    def collect_step_stats(self):
        return {
            "prefill_calls": self._prefill_count,
            "prefill_tokens": self._prefill_tokens,
            "prefill_duration_s": 0.0,
            "decode_iterations": 0,
            "decode_tokens": 0,
            "decode_duration_s": 0.0,
            "prefix_hits": self._prefix_hits,
            "prefix_lookups": self._prefix_lookups,
            "prefix_hit_rate": (
                (self._prefix_hits / self._prefix_lookups)
                if self._prefix_lookups
                else 0.0
            ),
            "prefix_tokens_reused": self._prefix_hits,
        }

    def debug_state(self):
        return {
            "generator_active": 0,
            "uid_count": 0,
            "prefill_calls": self._prefill_count,
            "prefill_tokens": self._prefill_tokens,
            "decode_iterations": 0,
            "decode_tokens": 0,
        }


class ErrorRunner(FakeRunner):
    """Runner that raises during prefill to simulate worker failure."""

    def prefill_context(self, ctx, tokens):
        raise RuntimeError("boom")


class ContinuousBatchingRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runner = FakeRunner()
        self.runtime = ContinuousBatchingRuntime(
            runner=self.runner,
            max_num_seqs=2,
            max_tokens_per_step=4,
            prefill_chunk=4,
        )

    def tearDown(self):
        self.runtime.shutdown()

    def test_submit_request_returns_generator(self):
        request_id, generator = self.runtime.submit_request(
            prompt_tokens=[1, 2, 3],
            max_new_tokens=1,
            sampler_settings={},
            stopping_settings={},
            logit_bias=None,
            repetition_penalty=None,
            repetition_context_size=None,
        )
        self.assertTrue(request_id)
        events = list(generator)
        self.assertEqual(len(events), 1)
        self.assertEqual(self.runner.prefill_requests[0][0], request_id)

    def test_scheduler_wakes_on_submit(self):
        request_id, generator = self.runtime.submit_request(
            prompt_tokens=[1],
            max_new_tokens=1,
            sampler_settings={},
            stopping_settings={},
            logit_bias=None,
            repetition_penalty=None,
            repetition_context_size=None,
        )
        # Await worker to process
        time_limit = time.time() + 1.0
        while self.runner.decode_calls == 0 and time.time() < time_limit:
            time.sleep(0.01)
        self.assertGreaterEqual(self.runner.decode_calls, 1)
        list(generator)

    def test_metrics_history_includes_prefix_stats(self):
        _, generator = self.runtime.submit_request(
            prompt_tokens=[4, 5, 6],
            max_new_tokens=1,
            sampler_settings={},
            stopping_settings={},
            logit_bias=None,
            repetition_penalty=None,
            repetition_context_size=None,
        )
        list(generator)
        history = self.runtime.metrics_history()
        self.assertTrue(any("prefix_hits" in entry for entry in history))
        last = history[-1]
        self.assertIn("prefix_hit_rate", last)
        self.assertIn("prefix_tokens_reused", last)

    def test_stream_raises_when_worker_fails(self):
        runtime = ContinuousBatchingRuntime(
            runner=ErrorRunner(),
            max_num_seqs=1,
            max_tokens_per_step=4,
            prefill_chunk=4,
        )
        try:
            _, generator = runtime.submit_request(
                prompt_tokens=[1, 2],
                max_new_tokens=1,
                sampler_settings={},
                stopping_settings={},
                logit_bias=None,
                repetition_penalty=None,
                repetition_context_size=None,
            )
            with self.assertRaisesRegex(RuntimeError, "continuous batching worker"):
                next(generator)
        finally:
            runtime.shutdown()


class RuntimeFactoryTests(unittest.TestCase):
    def test_force_legacy_generator_creates_legacy_runner(self):
        config = {
            "enabled": True,
            "max_num_seqs": 1,
            "max_tokens_per_step": 4,
            "prefill_chunk": 2,
            "force_legacy_generator": True,
        }
        runtime = create_runtime(config, model=DummyModel(), tokenizer=DummyTokenizer())
        self.assertIsNotNone(runtime)
        self.assertTrue(runtime.runner.use_legacy_generator)
        runtime.shutdown()

    def test_slot_generator_is_default(self):
        config = {
            "enabled": True,
            "max_num_seqs": 1,
            "max_tokens_per_step": 4,
            "prefill_chunk": 2,
        }
        runtime = create_runtime(config, model=DummyModel(), tokenizer=DummyTokenizer())
        self.assertIsNotNone(runtime)
        self.assertFalse(runtime.runner.use_legacy_generator)
        runtime.shutdown()

    def test_runtime_config_records_backend_selection(self):
        config = {
            "enabled": True,
            "max_num_seqs": 1,
            "max_tokens_per_step": 4,
            "prefill_chunk": 2,
            "attn_backend": "auto",
            "kv_block_size": 16,
            "kv_pool_blocks": 128,
            "paged_vec_width": "auto",
            "paged_threads_per_head": "auto",
            "decode_unroll_safe": False,
        }
        runtime = create_runtime(config, model=DummyModel(), tokenizer=DummyTokenizer())
        self.assertIn(config["selected_attn_backend"], ("dense", "paged"))
        self.assertIn("paged_backend_available", config)
        self.assertIn("paged_backend_enabled", config)
        self.assertEqual(config["kv_block_size"], 16)
        self.assertEqual(config["kv_pool_blocks"], 128)
        self.assertEqual(config["kv_quant_mode"], "none")
        self.assertEqual(config["kv_quant_group_size"], 64)
        self.assertIn("decode_unroll_safe", config)
        self.assertFalse(config["decode_unroll_safe"])
        runtime.shutdown()

    def test_invalid_attn_backend_raises(self):
        config = {
            "enabled": True,
            "max_num_seqs": 1,
            "max_tokens_per_step": 4,
            "prefill_chunk": 2,
            "attn_backend": "unsupported",
        }
        with self.assertRaises(ValueError):
            create_runtime(config, model=DummyModel(), tokenizer=DummyTokenizer())

    def test_invalid_kv_quant_mode_raises(self):
        config = {
            "enabled": True,
            "max_num_seqs": 1,
            "max_tokens_per_step": 4,
            "prefill_chunk": 2,
            "kv_quant_mode": "bad",
        }
        with self.assertRaises(ValueError):
            create_runtime(config, model=DummyModel(), tokenizer=DummyTokenizer())

    def test_metal_profiling_request_calls_backend(self):
        config = {
            "enabled": True,
            "max_num_seqs": 1,
            "max_tokens_per_step": 4,
            "prefill_chunk": 2,
            "metal_profiling": True,
        }
        fake_metal = types.SimpleNamespace(
            called=False,
            command_buffer_profiling_supported=lambda: True,
            set_command_buffer_profiling=lambda enabled: setattr(
                fake_metal, "called", enabled
            ),
        )
        original = getattr(runtime_module, "_METAL_PROFILING_ACTIVE")
        try:
            runtime_module._METAL_PROFILING_ACTIVE = False
            with mock.patch.object(runtime_module.mx, "metal", fake_metal):
                runtime = create_runtime(
                    config, model=DummyModel(), tokenizer=DummyTokenizer()
                )
            self.assertTrue(fake_metal.called)
            runtime.shutdown()
        finally:
            runtime_module._METAL_PROFILING_ACTIVE = original


class RuntimeConfigNormalizationTests(unittest.TestCase):
    def test_kv_pool_blocks_auto_sets_none(self):
        config = {
            "enabled": True,
            "max_num_seqs": 1,
            "max_tokens_per_step": 4,
            "prefill_chunk": 2,
            "kv_pool_blocks": "auto",
        }
        runtime_module._normalize_runtime_config(config)
        self.assertIsNone(config["kv_pool_blocks"])

    def test_env_kill_switch_forces_dense_backend(self):
        config = {
            "enabled": True,
            "max_num_seqs": 1,
            "max_tokens_per_step": 4,
            "prefill_chunk": 2,
            "attn_backend": "paged",
            "kv_block_size": 16,
            "kv_pool_blocks": 64,
            "paged_vec_width": None,
            "paged_threads_per_head": None,
            "kv_quant_mode": "none",
            "kv_quant_group_size": 64,
        }
        with mock.patch.object(
            runtime_module, "_has_paged_attention_support", return_value=True
        ):
            with mock.patch.dict("os.environ", {"MLXLM_PAGED_DISABLE": "1"}):
                runtime_module._normalize_runtime_config(config)
        self.assertEqual(config["selected_attn_backend"], "dense")
        self.assertFalse(config["paged_backend_enabled"])
        self.assertTrue(config.get("paged_backend_env_disabled"))

    def test_metal_profiling_defaults_false(self):
        config = {
            "enabled": True,
            "max_num_seqs": 1,
            "max_tokens_per_step": 4,
            "prefill_chunk": 2,
        }
        runtime_module._normalize_runtime_config(config)
        self.assertFalse(config["metal_profiling"])

    def test_env_enables_metal_profiling(self):
        config = {
            "enabled": True,
            "max_num_seqs": 1,
            "max_tokens_per_step": 4,
            "prefill_chunk": 2,
        }
        with mock.patch.dict("os.environ", {"MLXLM_METAL_PROFILING": "1"}):
            runtime_module._normalize_runtime_config(config)
        self.assertTrue(config["metal_profiling"])

    def test_decode_unroll_safe_defaults_true(self):
        config = {
            "enabled": True,
            "max_num_seqs": 1,
            "max_tokens_per_step": 4,
            "prefill_chunk": 2,
        }
        runtime_module._normalize_runtime_config(config)
        self.assertTrue(config["decode_unroll_safe"])

    def test_decode_unroll_safe_respects_config(self):
        config = {
            "enabled": True,
            "max_num_seqs": 1,
            "max_tokens_per_step": 4,
            "prefill_chunk": 2,
            "decode_unroll_safe": False,
        }
        runtime_module._normalize_runtime_config(config)
        self.assertFalse(config["decode_unroll_safe"])


if __name__ == "__main__":
    unittest.main()
