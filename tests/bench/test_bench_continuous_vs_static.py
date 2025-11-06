# ABOUTME: Exercises helper utilities used by the benchmarking harness.
# ABOUTME: Ensures sampling parity and steady-state reporting stay correct.

import threading

import pytest
import types

from bench.bench_continuous_vs_static import (
    TokenControls,
    compute_steady_state_tps,
    normalise_request_count,
    _override_tokenizer_stops,
    phase_summaries,
    should_schedule_open_loop,
    estimate_safe_max_num_seqs,
    select_token_controls,
)
from mlx_lm.server_batched.runtime import ContinuousBatchingRuntime


class DummyTokenizer:
    eos_token_id = 42
    eos_token_ids = [42, 99]


@pytest.fixture()
def dummy_tokenizer():
    return DummyTokenizer()


def test_select_token_controls_fixed_mode(dummy_tokenizer):
    controls = select_token_controls(
        mode="fixed",
        tokenizer=dummy_tokenizer,
        explicit_stops=None,
    )
    assert controls.static_stop_tokens == []
    assert controls.runtime_stop_tokens == set()
    assert controls.use_eos_stop is False
    assert controls.sampler_settings["temp"] == 0.0
    assert controls.sampler_settings["xtc_special_tokens"] == []


def test_select_token_controls_eos_mode_with_explicit_stops(dummy_tokenizer):
    controls = select_token_controls(
        mode="eos",
        tokenizer=dummy_tokenizer,
        explicit_stops={128001, 128008},
    )
    expected = [128001, 128008]
    assert sorted(controls.static_stop_tokens) == expected
    assert controls.runtime_stop_tokens == set(expected)
    assert controls.use_eos_stop is True
    assert controls.sampler_settings["temp"] == 0.0
    assert controls.sampler_settings["xtc_special_tokens"] == expected


def test_compute_steady_state_tps_filters_on_active_sequences():
    history = [
        {"active_sequences": 2, "decode_tokens": 32, "decode_duration_s": 0.08},
        {"active_sequences": 8, "decode_tokens": 32, "decode_duration_s": 0.06},
        {"active_sequences": 9, "decode_tokens": 32, "decode_duration_s": 0.05},
        {"active_sequences": 9, "decode_tokens": 32, "decode_duration_s": 0.0},
    ]
    tps = compute_steady_state_tps(history, steady_fraction=0.75, max_num_seqs=10)
    assert pytest.approx(tps, rel=1e-4) == 586.6666666666667


def test_normalise_request_count_defaults_to_concurrency():
    assert normalise_request_count(None, 12) == 12
    assert normalise_request_count(24, 12) == 24


def test_metrics_history_returns_copy():
    runtime = object.__new__(ContinuousBatchingRuntime)
    runtime._metrics_history = [{"decode_tokens": 1}]
    runtime._metrics_lock = threading.Lock()
    result = runtime.metrics_history()
    assert result == runtime._metrics_history
    result.append({"decode_tokens": 2})
    assert runtime._metrics_history == [{"decode_tokens": 1}]


def test_override_tokenizer_stops_temporarily_sets_ids(dummy_tokenizer):
    dummy = dummy_tokenizer
    dummy.eos_token_id = 1
    dummy.eos_token_ids = [1, 2]
    with _override_tokenizer_stops(dummy, [10, 11]):
        assert dummy.eos_token_id == 10
        assert dummy.eos_token_ids == [10, 11]
    assert dummy.eos_token_id == 1
    assert dummy.eos_token_ids == [1, 2]


def test_phase_summaries_segments_history():
    history = [
        {"active_sequences": 2, "decode_duration_s": 0.05, "decode_tokens": 16, "decode_batch_size": 2},
        {"active_sequences": 8, "decode_duration_s": 0.06, "decode_tokens": 32, "decode_batch_size": 8},
        {"active_sequences": 9, "decode_duration_s": 0.05, "decode_tokens": 32, "decode_batch_size": 9},
        {"active_sequences": 3, "decode_duration_s": 0.30, "decode_tokens": 6, "decode_batch_size": 3},
    ]
    phases = phase_summaries(history, steady_fraction=0.6, max_num_seqs=10, spike_threshold_s=0.2)
    assert phases["ramp"]["ticks"] == 1
    assert phases["steady"]["ticks"] == 2
    assert phases["tail"]["ticks"] == 1
    assert phases["spikes"]["count"] == 1


def test_should_schedule_open_loop_conditions():
    assert should_schedule_open_loop(
        backlog=0,
        free_slots=5,
        active_slots=3,
        max_num_seqs=8,
        target_tokens=None,
        total_tokens_emitted=0,
    )
    assert not should_schedule_open_loop(
        backlog=7,
        free_slots=0,
        active_slots=8,
        max_num_seqs=8,
        target_tokens=None,
        total_tokens_emitted=0,
    )
    assert not should_schedule_open_loop(
        backlog=8,
        free_slots=2,
        active_slots=7,
        max_num_seqs=8,
        target_tokens=None,
        total_tokens_emitted=0,
    )
    assert not should_schedule_open_loop(
        backlog=3,
        free_slots=2,
        active_slots=7,
        max_num_seqs=8,
        target_tokens=1000,
        total_tokens_emitted=1000,
    )


def _make_dummy_model(n_layers=4, n_kv_heads=4, head_dim=16, dtype_size=2):
    dtype = types.SimpleNamespace(size=dtype_size)

    class DummyProj:
        def __init__(self, dtype):
            self.weight = types.SimpleNamespace(dtype=dtype)

    class DummyAttn:
        def __init__(self):
            self.n_kv_heads = n_kv_heads
            self.head_dim = head_dim
            self.k_proj = DummyProj(dtype)
            self.v_proj = DummyProj(dtype)

    class DummyLayer:
        def __init__(self):
            self.self_attn = DummyAttn()

    class Wrapper:
        def __init__(self):
            self.layers = [DummyLayer() for _ in range(n_layers)]

    class Model:
        def __init__(self):
            self.model = Wrapper()

    return Model()


def test_estimate_safe_max_num_seqs_caps_when_exceeding_limit():
    dummy = _make_dummy_model(n_layers=4, n_kv_heads=4, head_dim=16, dtype_size=2)
    safe, meta = estimate_safe_max_num_seqs(dummy, requested=64, max_tokens=128, limit_bytes=128 * 1024)
    assert safe < 64
    assert meta is not None
    assert meta["limit_bytes"] == 128 * 1024


def test_estimate_safe_max_num_seqs_preserves_request_when_within_limit():
    dummy = _make_dummy_model(n_layers=2, n_kv_heads=2, head_dim=8, dtype_size=2)
    safe, meta = estimate_safe_max_num_seqs(dummy, requested=16, max_tokens=32, limit_bytes=512 * 1024 * 1024)
    assert safe == 16
    assert meta is None
