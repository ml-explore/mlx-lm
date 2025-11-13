# ABOUTME: Exercises helper utilities used by the benchmarking harness.
# ABOUTME: Ensures sampling parity and steady-state reporting stay correct.

import sys
import threading
import types
from pathlib import Path

import pytest

import tests  # noqa: F401  # ensures ensure_mlx_stub() runs before bench imports

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from bench.bench_continuous_vs_static import (
    Result,
    TokenControls,
    _override_tokenizer_stops,
    closed_arrival_schedule,
    collect_prefill_profile_metrics,
    compute_steady_state_tps,
    estimate_safe_max_num_seqs,
    extract_apc_stats,
    extract_compile_stats,
    normalise_request_count,
    phase_summaries,
    run_static_runtime,
    select_token_controls,
    should_schedule_open_loop,
    should_use_runtime_static,
    summarize,
)

try:
    from mlx_lm.server_batched.runtime import ContinuousBatchingRuntime
except Exception:  # pragma: no cover - fallback for environments without mlx_lm deps

    class ContinuousBatchingRuntime:  # type: ignore[empty-body]
        pass


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
        {
            "active_sequences": 2,
            "decode_duration_s": 0.05,
            "decode_tokens": 16,
            "decode_batch_size": 2,
        },
        {
            "active_sequences": 8,
            "decode_duration_s": 0.06,
            "decode_tokens": 32,
            "decode_batch_size": 8,
        },
        {
            "active_sequences": 9,
            "decode_duration_s": 0.05,
            "decode_tokens": 32,
            "decode_batch_size": 9,
        },
        {
            "active_sequences": 3,
            "decode_duration_s": 0.30,
            "decode_tokens": 6,
            "decode_batch_size": 3,
        },
    ]
    phases = phase_summaries(
        history, steady_fraction=0.6, max_num_seqs=10, spike_threshold_s=0.2
    )
    assert phases["ramp"]["ticks"] == 1
    assert phases["steady"]["ticks"] == 2
    assert phases["tail"]["ticks"] == 1
    assert phases["spikes"]["count"] == 1


def test_extract_compile_stats_returns_latest_snapshot():
    history = [
        {"decode_tokens": 8},
        {
            "compile_cache_hits": 5.0,
            "compile_cache_misses": 1.0,
            "array_phase1_compile_hits": 3.0,
            "array_phase1_compile_misses": 1.0,
            "array_phase1_duration_s": 0.25,
        },
    ]
    stats = extract_compile_stats(history)
    assert stats["compile_cache_hits"] == 5.0
    assert stats["compile_cache_misses"] == 1.0
    assert stats["array_phase1_compile_hits"] == 3.0
    assert stats["array_phase1_compile_misses"] == 1.0
    assert stats["array_phase1_duration_s"] == 0.25


def test_extract_compile_stats_handles_empty_history():
    assert extract_compile_stats([]) == {}


def test_should_schedule_open_loop_conditions():
    assert should_schedule_open_loop(
        backlog=0,
        target_tokens=None,
        total_tokens_emitted=0,
    )
    assert not should_schedule_open_loop(
        backlog=7,
        target_tokens=None,
        total_tokens_emitted=0,
        backlog_limit=5,
    )
    assert not should_schedule_open_loop(
        backlog=3,
        target_tokens=1000,
        total_tokens_emitted=1000,
    )
    assert should_schedule_open_loop(
        backlog=3,
        target_tokens=1000,
        total_tokens_emitted=512,
        backlog_limit=10,
    )


def test_closed_arrival_schedule_supports_burst_mode():
    arrivals = closed_arrival_schedule(count=4, mode="burst", lmbda=1.0)
    assert arrivals == [0.0, 0.0, 0.0, 0.0]


def test_closed_arrival_schedule_defaults_to_poisson():
    arrivals = closed_arrival_schedule(count=3, mode="poisson", lmbda=2.0)
    assert len(arrivals) == 3
    assert arrivals[0] >= 0.0
    assert arrivals[1] > arrivals[0]
    assert arrivals[2] > arrivals[1]


def test_summarize_reports_ttft_window_stats():
    base_ns = 1_000_000
    results = [
        Result(
            submit_ns=0,
            first_token_ns=1_000_000,
            finish_ns=base_ns * 2,
            gen_tokens=32,
        ),
        Result(
            submit_ns=0,
            first_token_ns=2_000_000,
            finish_ns=base_ns * 3,
            gen_tokens=32,
        ),
        Result(
            submit_ns=0,
            first_token_ns=20_000_000,
            finish_ns=base_ns * 30,
            gen_tokens=32,
        ),
        Result(
            submit_ns=0,
            first_token_ns=25_000_000,
            finish_ns=base_ns * 40,
            gen_tokens=32,
        ),
    ]
    summary = summarize("unit", results, ttft_window=2)
    assert summary["ttft_median_ms"] > 10.0  # backlog pushes median up
    assert summary["ttft_window_count"] == 2
    assert summary["ttft_window_median_ms"] == pytest.approx(1.5, rel=1e-3)


def test_run_static_runtime_delegates_with_closed_loop(monkeypatch):
    from bench import bench_continuous_vs_static as bench_mod

    captured = {}

    def fake_run_continuous(*args, **kwargs):
        captured["kwargs"] = kwargs
        return (["ok"], [{"decode_tokens": 1}])

    monkeypatch.setattr(bench_mod, "run_continuous", fake_run_continuous)
    results = run_static_runtime(
        model="dummy-model",
        tokenizer="dummy-tokenizer",
        prompts=["a", "b"],
        max_tokens=32,
        lmbda=1.0,
        max_num_seqs=4,
        prefill_chunk=64,
        prefill_ramp_chunk=32,
        max_tokens_per_step=256,
        decode_unroll=2,
        stop_tokens_override=None,
        use_eos_stop=False,
        sampler_settings={"temp": 0.0},
        attn_backend="paged",
        kv_block_size=16,
        kv_pool_blocks=None,
        paged_vec_width=None,
        paged_threads_per_head=None,
        kv_quant_mode="none",
        kv_quant_group_size=64,
        decode_engine="paged-arrays",
        prefill_hybrid_threshold=0,
        prefill_ramp_budget_ms=None,
    )
    assert results == ["ok"]
    assert captured["kwargs"]["open_loop"] is False
    assert captured["kwargs"]["arrival_mode"] == "burst"
    assert captured["kwargs"]["stop_tokens_override"] is None


def test_should_use_runtime_static_modes():
    assert should_use_runtime_static(decode_engine="paged-arrays", static_mode="auto")
    assert should_use_runtime_static(decode_engine="dense", static_mode="runtime")
    assert not should_use_runtime_static(decode_engine="dense", static_mode="batch")
    assert not should_use_runtime_static(decode_engine="dense", static_mode="auto")


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
    safe, meta = estimate_safe_max_num_seqs(
        dummy, requested=64, max_tokens=128, limit_bytes=128 * 1024
    )
    assert safe < 64
    assert meta is not None
    assert meta["limit_bytes"] == 128 * 1024


def test_estimate_safe_max_num_seqs_preserves_request_when_within_limit():
    dummy = _make_dummy_model(n_layers=2, n_kv_heads=2, head_dim=8, dtype_size=2)
    safe, meta = estimate_safe_max_num_seqs(
        dummy, requested=16, max_tokens=32, limit_bytes=512 * 1024 * 1024
    )
    assert safe == 16
    assert meta is None


def test_extract_apc_stats_scans_backwards_for_prefix_metrics():
    history = [
        {
            "prefix_hits": 3.0,
            "prefix_lookups": 6.0,
            "prefix_hit_rate": 0.5,
            "prefix_tokens_reused": 192.0,
        },
        {
            "decode_tokens": 4,
            "decode_duration_s": 0.01,
        },
    ]
    stats = extract_apc_stats(history)
    assert stats["prefix_hits"] == 3.0
    assert stats["prefix_tokens_reused"] == 192.0


def _make_stub_mx():
    class _FakeArray(dict):
        pass

    class _StubMX:
        int32 = "int32"
        float16 = "float16"

        @staticmethod
        def _ensure_tuple(shape):
            if isinstance(shape, tuple):
                return shape
            if isinstance(shape, int):
                return (shape,)
            return tuple(shape)

        def zeros(self, shape, dtype=None):
            return _FakeArray(shape=self._ensure_tuple(shape), dtype=dtype)

        def zeros_like(self, other):
            return _FakeArray(shape=other["shape"], dtype=other["dtype"])

    return _StubMX()


def test_collect_prefill_profile_metrics_embeds_metrics_when_graph_available(
    monkeypatch,
):
    from bench import bench_continuous_vs_static as bench_mod

    instances = []

    class DummyGraph:
        def __init__(self, model):
            self.head_dim = 8
            self.called = False
            self._metrics = {
                "array_prefill_attn_s": 1.23,
                "array_prefill_mlp_s": 0.01,
                "array_prefill_overlay_s": 0.02,
                "array_prefill_layer_attn_s": [0.5],
                "array_prefill_layer_mlp_s": [0.01],
                "array_prefill_layer_overlay_s": [0.02],
            }
            instances.append(self)

        def get_compiled(self, **_):
            def _fn(*args):
                self.called = True

            return _fn

        def consume_metrics(self):
            return dict(self._metrics)

    def _make_model():
        attn = types.SimpleNamespace(
            n_kv_heads=2,
            head_dim=8,
            q_proj=types.SimpleNamespace(weight=types.SimpleNamespace(dtype="float16")),
        )
        layer = types.SimpleNamespace(self_attn=attn)
        return types.SimpleNamespace(layers=[layer, layer])

    model = _make_model()
    monkeypatch.setattr(bench_mod, "LlamaPrefillGraph", DummyGraph)
    monkeypatch.setattr(bench_mod, "mx", _make_stub_mx())
    profile = collect_prefill_profile_metrics(model, chunk_len=4, ramp_chunk=2)
    assert profile["chunk_len"] == 4
    assert profile["prefill_ramp_chunk"] == 2
    assert profile["array_prefill_attn_s"] == 1.23
    assert profile["array_prefill_mlp_s"] == 0.01
    assert instances[-1].called is True


def test_collect_prefill_profile_metrics_returns_none_on_graph_failure(monkeypatch):
    from bench import bench_continuous_vs_static as bench_mod

    def _boom(_model):
        raise RuntimeError("boom")

    attn = types.SimpleNamespace(
        n_kv_heads=2,
        head_dim=8,
        q_proj=types.SimpleNamespace(weight=types.SimpleNamespace(dtype="float16")),
    )
    layer = types.SimpleNamespace(self_attn=attn)
    model = types.SimpleNamespace(layers=[layer])

    monkeypatch.setattr(bench_mod, "LlamaPrefillGraph", _boom)
    profile = collect_prefill_profile_metrics(model, chunk_len=4, ramp_chunk=2)
    assert profile is None
