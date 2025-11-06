# ABOUTME: Exercises helper utilities used by the benchmarking harness.
# ABOUTME: Ensures sampling parity and steady-state reporting stay correct.

import threading

import pytest

from bench.bench_continuous_vs_static import (
    TokenControls,
    compute_steady_state_tps,
    normalise_request_count,
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
