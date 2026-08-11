# Copyright © 2026 Apple Inc.
"""Synthetic streaming coverage for the native Qwen MTP generator."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import mlx.core as mx
import pytest

from mlx_lm.generate import (
    GenerationForwardPhase,
    NativeMTPSamplingConfig,
    mtp_generate_step,
)
from mlx_lm.models.cache import ArraysCache, KVCache, NativeMTPRequestCache

_ACTIVE_FORWARD = ContextVar("native_mtp_test_forward", default=None)


@dataclass(frozen=True)
class _Capability:
    supported: bool = True
    reason: str = "supported"


class _NativeMTPModel:
    """Tiny cache-writing target/MTP pair with a known dense next-token rule."""

    mtp_capability = _Capability()
    vocab_size = 7

    def __init__(
        self,
        *,
        wrong_draft=False,
        fail_mtp_offsets=(),
        acknowledge_positions=True,
        tamper_ack=False,
        reuse_ack=False,
    ):
        self.wrong_draft = wrong_draft
        self.fail_mtp_offsets = frozenset(fail_mtp_offsets)
        self.acknowledge_positions = acknowledge_positions
        self.tamper_ack = tamper_ack
        self.reuse_ack = reuse_ack
        self.layers = [
            type("_Linear", (), {"is_linear": True})(),
            type("_Attention", (), {"is_linear": False})(),
        ]
        self.mtp = type("_MTP", (), {"layers": [object()]})()
        self.requests = []
        self.target_offset = 0
        self.target_offsets = []
        self.mtp_offsets = []
        self.mtp_attempt_offsets = []

    def _acknowledge_forward_positions(self):
        forward = _ACTIVE_FORWARD.get()
        if forward is None or forward.logical_position_ack is None:
            return
        if not self.acknowledge_positions:
            return
        positions = forward.logical_positions
        if self.tamper_ack:
            positions = positions[:-1] + (positions[-1] + 1,)
        forward.logical_position_ack.acknowledge(positions)
        if self.reuse_ack:
            forward.logical_position_ack.acknowledge(positions)

    def make_cache(self):
        recurrent = ArraysCache(2)
        recurrent[0] = mx.array([[0.0]])
        recurrent[1] = mx.array([[0.0]])
        return [recurrent, KVCache()]

    def make_mtp_cache(self):
        return [KVCache()]

    def make_mtp_request_cache(self):
        request = NativeMTPRequestCache.create(self)
        self.requests.append(request)
        return request

    @staticmethod
    def _write_attention(entry, count):
        values = mx.zeros((1, 1, count, 32))
        entry.update_and_fetch(values, values)

    def _logits(self, inputs, *, offset=1):
        ids = (inputs.astype(mx.int32) + offset) % self.vocab_size
        logits = mx.full((1, inputs.shape[1], self.vocab_size), -20.0)
        for index in range(inputs.shape[1]):
            logits[:, index, ids[0, index]] = 20.0
        return logits

    def __call__(self, inputs, *, cache, return_hidden=False):
        self._acknowledge_forward_positions()
        count = inputs.shape[1]
        cache[0][0] = cache[0][0] + count
        cache[0][1] = cache[0][1] + count
        cache[0].advance(count)
        self._write_attention(cache[1], count)
        self.target_offset = cache[1].offset
        self.target_offsets.append(self.target_offset)
        logits = self._logits(inputs)
        hidden = inputs.astype(mx.float32)[..., None]
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(self, hidden, next_token_ids, cache):
        self._acknowledge_forward_positions()
        self.mtp_attempt_offsets.append(cache[0].offset)
        if cache[0].offset in self.fail_mtp_offsets:
            raise RuntimeError("synthetic unexpected next draft")
        self._write_attention(cache[0], next_token_ids.shape[1])
        self.mtp_offsets.append(cache[0].offset)
        offset = 2 if self.wrong_draft or cache[0].offset != self.target_offset else 1
        return self._logits(next_token_ids, offset=offset)


def _run(model, *, max_tokens=4, prompt=(0, 1), **kwargs):
    telemetry = {}
    output = list(
        mtp_generate_step(
            mx.array(prompt, dtype=mx.uint32),
            model,
            max_tokens=max_tokens,
            telemetry=telemetry,
            **kwargs,
        )
    )
    return output, telemetry


def test_greedy_mtp_matches_dense_token_rule_and_uses_target_logprobs_for_drafts():
    output, telemetry = _run(_NativeMTPModel(), max_tokens=4)
    tokens = [token for token, _, _ in output]
    from_draft = [draft for _, _, draft in output]

    assert tokens == [2, 3, 4, 5]
    assert from_draft == [False, True, False, True]
    assert telemetry == {
        "mtp_drafts": 2,
        "mtp_accepted": 2,
        "mtp_bypass_reason": None,
    }
    for token, logprobs, from_draft in output:
        if from_draft:
            assert mx.argmax(logprobs).item() == token


def test_rejected_draft_rolls_back_and_replays_the_recurrent_prefix():
    model = _NativeMTPModel(wrong_draft=True)
    output, telemetry = _run(model, max_tokens=3)

    assert [token for token, _, _ in output] == [2, 3, 4]
    assert not any(from_draft for _, _, from_draft in output)
    assert telemetry["mtp_drafts"] == 2
    assert telemetry["mtp_accepted"] == 0
    request = model.requests[-1]
    assert request.closed
    # Prompt tokens plus the two replayed prior outputs are physically retained.
    assert request.backbone[1].offset == 4
    assert request.mtp[0].offset == 3
    assert mx.array_equal(request.backbone[0][0], mx.array([[4.0]])).item()


def test_forward_context_labels_target_prefill_verify_decode_and_mtp_draft():
    model = _NativeMTPModel(wrong_draft=True)
    phases = []

    @contextmanager
    def context(forward):
        phases.append(forward.phase)
        yield

    _run(model, max_tokens=3, model_forward_context=context)
    assert GenerationForwardPhase.PREFILL in phases
    assert GenerationForwardPhase.MTP_DRAFT in phases
    assert GenerationForwardPhase.VERIFY in phases
    assert GenerationForwardPhase.DECODE in phases


class _PositionContext:
    def __init__(self):
        self.events = []
        self.active = 0

    @contextmanager
    def context(self, forward):
        self.events.append(forward)
        token = _ACTIVE_FORWARD.set(forward)
        self.active += 1
        try:
            yield
        finally:
            self.active -= 1
            _ACTIVE_FORWARD.reset(token)


def _event_contract(events):
    return [(event.phase, event.logical_positions) for event in events]


def test_sparse_positions_cover_exact_native_phase_order_and_shared_cursor():
    model = _NativeMTPModel()
    context = _PositionContext()

    output, _ = _run(
        model,
        max_tokens=4,
        prompt_logical_positions=(0, 3),
        model_forward_context=context.context,
    )

    assert [token for token, _, _ in output] == [2, 3, 4, 5]
    assert _event_contract(context.events) == [
        (GenerationForwardPhase.PREFILL, (0,)),
        (GenerationForwardPhase.MTP_DRAFT, (0,)),
        (GenerationForwardPhase.PREFILL, (3,)),
        (GenerationForwardPhase.MTP_DRAFT, (3,)),
        (GenerationForwardPhase.VERIFY, (4, 5)),
        (GenerationForwardPhase.MTP_DRAFT, (4, 5)),
        (GenerationForwardPhase.VERIFY, (6, 7)),
    ]
    request = model.requests[-1]
    assert request.backbone[1].offset == 6
    assert request.mtp[0].offset == 4
    assert request.state.next_logical_position == 8
    assert context.active == 0


def test_sparse_rejection_replays_exact_positions_without_cursor_rewind():
    model = _NativeMTPModel(wrong_draft=True)
    context = _PositionContext()

    output, _ = _run(
        model,
        max_tokens=3,
        prompt_logical_positions=(0, 3),
        model_forward_context=context.context,
    )

    assert [token for token, _, _ in output] == [2, 3, 4]
    assert _event_contract(context.events) == [
        (GenerationForwardPhase.PREFILL, (0,)),
        (GenerationForwardPhase.MTP_DRAFT, (0,)),
        (GenerationForwardPhase.PREFILL, (3,)),
        (GenerationForwardPhase.MTP_DRAFT, (3,)),
        (GenerationForwardPhase.VERIFY, (4, 5)),
        (GenerationForwardPhase.DECODE, (4,)),
        (GenerationForwardPhase.MTP_DRAFT, (4,)),
        (GenerationForwardPhase.VERIFY, (5, 6)),
        (GenerationForwardPhase.DECODE, (5,)),
    ]
    request = model.requests[-1]
    assert request.backbone[1].offset == 4
    assert request.mtp[0].offset == 3
    assert request.state.next_logical_position == 7
    assert context.active == 0


def test_sparse_cursor_advances_before_terminal_yield_and_cancel_cleanup():
    model = _NativeMTPModel()
    context = _PositionContext()
    iterator = mtp_generate_step(
        mx.array([0, 1], dtype=mx.uint32),
        model,
        max_tokens=4,
        prompt_logical_positions=(0, 3),
        model_forward_context=context.context,
    )

    assert next(iterator)[0] == 2
    request = model.requests[-1]
    assert request.state.next_logical_position == 5
    assert context.active == 0
    iterator.close()
    assert request.closed
    assert request.state.next_logical_position == 5
    assert context.active == 0

    terminal_model = _NativeMTPModel()
    terminal_context = _PositionContext()
    terminal, _ = _run(
        terminal_model,
        max_tokens=1,
        prompt_logical_positions=(0, 3),
        model_forward_context=terminal_context.context,
    )
    assert [token for token, _, _ in terminal] == [2]
    assert terminal_model.requests[-1].closed
    assert terminal_model.requests[-1].state.next_logical_position == 5
    assert terminal_context.active == 0


@pytest.mark.parametrize("wrong_draft", (False, True))
def test_sparse_position_context_is_inactive_across_every_yield(wrong_draft):
    model = _NativeMTPModel(wrong_draft=wrong_draft)
    context = _PositionContext()
    iterator = mtp_generate_step(
        mx.array([0, 1], dtype=mx.uint32),
        model,
        max_tokens=4,
        prompt_logical_positions=(0, 3),
        model_forward_context=context.context,
    )

    observed = []
    for item in iterator:
        observed.append(item[0])
        assert context.active == 0
        assert _ACTIVE_FORWARD.get() is None
    assert observed == [2, 3, 4, 5]
    assert context.active == 0


@pytest.mark.parametrize(
    ("model", "message"),
    (
        (
            _NativeMTPModel(acknowledge_positions=False),
            "generation_logical_positions_not_acknowledged",
        ),
        (
            _NativeMTPModel(tamper_ack=True),
            "generation_logical_position_ack_mismatch",
        ),
        (
            _NativeMTPModel(reuse_ack=True),
            "generation_logical_position_ack_reused",
        ),
    ),
)
def test_sparse_position_ack_missing_tampered_or_reused_fails_closed(model, message):
    context = _PositionContext()
    with pytest.raises(RuntimeError, match=message):
        _run(
            model,
            max_tokens=1,
            prompt_logical_positions=(0, 3),
            model_forward_context=context.context,
        )
    assert model.requests[-1].closed
    assert context.active == 0


def test_sparse_position_ack_rejects_context_entry_pre_ack():
    model = _NativeMTPModel()

    @contextmanager
    def pre_ack_context(forward):
        forward.logical_position_ack.acknowledge(forward.logical_positions)
        yield

    with pytest.raises(
        RuntimeError, match="generation_logical_position_ack_outside_forward"
    ):
        _run(
            model,
            max_tokens=1,
            prompt_logical_positions=(0, 3),
            model_forward_context=pre_ack_context,
        )
    assert model.requests[-1].closed


def test_dense_native_path_has_no_logical_metadata_or_cursor():
    model = _NativeMTPModel(wrong_draft=True)
    events = []

    @contextmanager
    def context(forward):
        events.append(forward)
        yield

    output, _ = _run(model, max_tokens=3, model_forward_context=context)
    assert [token for token, _, _ in output] == [2, 3, 4]
    assert events
    assert all(event.logical_positions is None for event in events)
    assert all(event.logical_position_ack is None for event in events)
    assert model.requests[-1].state.next_logical_position is None


@pytest.mark.parametrize(
    ("positions", "error", "message"),
    (
        ((0,), ValueError, "must match prompt"),
        ((0, 0), ValueError, "strictly increasing"),
        ((0, -1), ValueError, "non-negative integers"),
        ((0, True), ValueError, "non-negative integers"),
    ),
)
def test_sparse_prompt_positions_fail_before_cache_construction(
    positions, error, message
):
    model = _NativeMTPModel()
    context = _PositionContext()
    with pytest.raises(error, match=message):
        _run(
            model,
            max_tokens=1,
            prompt_logical_positions=positions,
            model_forward_context=context.context,
        )
    assert model.requests == []


def test_sparse_prompt_positions_reject_device_sequence_before_iteration():
    model = _NativeMTPModel()
    context = _PositionContext()
    with pytest.raises(TypeError, match="host sequence"):
        _run(
            model,
            max_tokens=1,
            prompt_logical_positions=mx.array([0, 3]),
            model_forward_context=context.context,
        )
    assert model.requests == []


@pytest.mark.parametrize(
    "sampling",
    (
        NativeMTPSamplingConfig(temperature=0.7, seed=17),
        NativeMTPSamplingConfig(temperature=0.7, top_p=0.9, seed=17),
        NativeMTPSamplingConfig(temperature=0.7, top_k=3, seed=17),
        NativeMTPSamplingConfig(
            temperature=0.7, min_p=0.01, min_tokens_to_keep=2, seed=17
        ),
    ),
)
def test_stochastic_sampling_configs_are_seeded_and_request_local(sampling):
    first, first_telemetry = _run(
        _NativeMTPModel(), max_tokens=4, sampling_config=sampling
    )
    second, second_telemetry = _run(
        _NativeMTPModel(), max_tokens=4, sampling_config=sampling
    )
    assert [token for token, _, _ in first] == [token for token, _, _ in second]
    assert first_telemetry == second_telemetry


def test_opaque_sampling_and_non_replay_safe_processors_fail_before_cache_mutation():
    model = _NativeMTPModel()
    with pytest.raises(ValueError, match="native_mtp_opaque_sampler_unsupported"):
        _run(
            model,
            max_tokens=1,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
        )
    assert model.requests == []

    stochastic_model = _NativeMTPModel()
    with pytest.raises(ValueError, match="native_mtp_opaque_sampler_unsupported"):
        _run(
            stochastic_model,
            max_tokens=1,
            sampling_config=NativeMTPSamplingConfig(temperature=1, seed=4),
            sampler=lambda logits: mx.argmax(logits, axis=-1),
        )
    assert stochastic_model.requests == []

    processor_model = _NativeMTPModel()
    with pytest.raises(ValueError, match="native_mtp_non_replay_safe_logits_processor"):
        _run(processor_model, logits_processors=[lambda _tokens, logits: logits])
    assert processor_model.requests == []


def test_explicitly_deterministic_sampler_preserves_greedy_equivalence():
    def deterministic_sampler(logprobs):
        return mx.argmax(logprobs, axis=-1)

    deterministic_sampler.native_mtp_deterministic = True
    output, _ = _run(_NativeMTPModel(), max_tokens=4, sampler=deterministic_sampler)
    assert [token for token, _, _ in output] == [2, 3, 4, 5]


def test_replay_safe_processor_observes_dense_histories_and_bonus_only_after_accept():
    histories = []

    def processor(tokens, logits):
        histories.append(tokens.tolist())
        return logits

    processor.native_mtp_replay_safe = True
    _run(_NativeMTPModel(), max_tokens=3, logits_processors=[processor])
    assert histories == [
        [1],
        [1, 2],
        [1, 2],
        [1, 2, 3],
    ]

    rejected_histories = []

    def rejected_processor(tokens, logits):
        rejected_histories.append(tokens.tolist())
        return logits

    rejected_processor.native_mtp_replay_safe = True
    output, _ = _run(
        _NativeMTPModel(wrong_draft=True),
        max_tokens=2,
        logits_processors=[rejected_processor],
    )
    assert [token for token, _, _ in output] == [2, 3]
    assert [1, 2, 4] not in rejected_histories
    assert mx.argmax(output[-1][1]).item() == 3


def test_prompt_and_accept_reject_paths_keep_target_and_mtp_offsets_aligned():
    accepted = _NativeMTPModel()
    _run(accepted, max_tokens=4)
    accepted_request = accepted.requests[-1]
    assert accepted.target_offsets[:2] == [1, 2]
    assert accepted.mtp_offsets[:2] == [1, 2]
    assert accepted_request.backbone[1].offset == 6
    assert accepted_request.mtp[0].offset == 4

    rejected = _NativeMTPModel(wrong_draft=True)
    _run(rejected, max_tokens=3)
    rejected_request = rejected.requests[-1]
    assert rejected.target_offsets[:2] == [1, 2]
    assert rejected.mtp_offsets[:2] == [1, 2]
    assert rejected_request.backbone[1].offset == 4
    assert rejected_request.mtp[0].offset == 3


def test_shifted_prompt_prefill_is_cache_sensitive_and_exactly_aligned():
    model = _NativeMTPModel()
    output, _ = _run(model, prompt=(0, 1, 2, 3), max_tokens=1, prefill_step_size=2)
    assert [token for token, _, _ in output] == [4]
    assert model.target_offsets[:2] == [2, 3]
    assert model.mtp_offsets[:2] == [2, 3]
    request = model.requests[-1]
    assert request.backbone[1].offset == 4
    assert request.mtp[0].offset == 3


@pytest.mark.parametrize(
    ("model", "max_tokens", "eos_token_ids", "expected", "drafts", "accepted"),
    (
        (_NativeMTPModel(fail_mtp_offsets={1}), 1, (), [2], 0, 0),
        (_NativeMTPModel(fail_mtp_offsets={1}), 4, (2,), [2], 0, 0),
        (_NativeMTPModel(fail_mtp_offsets={2}), 2, (), [2, 3], 1, 1),
        (_NativeMTPModel(fail_mtp_offsets={2}), 3, (), [2, 3, 4], 1, 1),
        (
            _NativeMTPModel(wrong_draft=True, fail_mtp_offsets={2}),
            2,
            (),
            [2, 3],
            1,
            0,
        ),
    ),
)
def test_terminal_yields_do_not_start_unobservable_speculative_work(
    model, max_tokens, eos_token_ids, expected, drafts, accepted
):
    output, telemetry = _run(model, max_tokens=max_tokens, eos_token_ids=eos_token_ids)
    assert [token for token, _, _ in output] == expected
    assert telemetry["mtp_drafts"] == drafts
    assert telemetry["mtp_accepted"] == accepted


@pytest.mark.parametrize(
    "sampling",
    (
        NativeMTPSamplingConfig(temperature=0.1, seed=9),
        NativeMTPSamplingConfig(temperature=0.7, top_p=0.5, seed=9),
        NativeMTPSamplingConfig(temperature=0.7, top_k=1, seed=9),
    ),
)
def test_filtered_sampling_returns_dense_normalized_target_logprobs(sampling):
    model = _NativeMTPModel()
    output, _ = _run(model, max_tokens=3, sampling_config=sampling)

    expected_logits = model._logits(mx.array([[1]], dtype=mx.uint32))[:, -1, :]
    expected = (
        expected_logits - mx.logsumexp(expected_logits, axis=-1, keepdims=True)
    ).squeeze(0)
    assert mx.allclose(output[0][1], expected).item()
    for _, reported_logprobs, _ in output:
        assert mx.all(mx.isfinite(reported_logprobs)).item()
        assert mx.allclose(mx.sum(mx.exp(reported_logprobs)), mx.array(1.0)).item()


def test_rejection_reports_target_logits_not_filtered_residual_distribution():
    output, _ = _run(
        _NativeMTPModel(wrong_draft=True),
        max_tokens=2,
        sampling_config=NativeMTPSamplingConfig(
            temperature=0.7, top_p=0.5, top_k=1, seed=13
        ),
    )
    assert [token for token, _, _ in output] == [2, 3]
    rejection_logprobs = output[-1][1]
    assert mx.argmax(rejection_logprobs).item() == 3
    assert mx.all(mx.isfinite(rejection_logprobs)).item()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"temperature": True}, "temperature must be a finite number"),
        ({"temperature": float("nan")}, "temperature must be a finite number"),
        ({"temperature": float("inf")}, "temperature must be a finite number"),
        ({"top_p": True}, "top_p must be a finite number"),
        ({"top_p": float("nan")}, "top_p must be a finite number"),
        ({"min_p": float("inf")}, "min_p must be a finite number"),
        ({"top_k": True}, "top_k must be an integer"),
        ({"top_k": 1.5}, "top_k must be an integer"),
        ({"min_tokens_to_keep": False}, "min_tokens_to_keep must be an integer"),
        ({"seed": True}, "seed must be an integer or None"),
        ({"seed": -1}, "seed must be non-negative"),
    ),
)
def test_native_sampling_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        NativeMTPSamplingConfig(**kwargs)


def test_top_k_is_validated_against_vocab_before_cache_mutation():
    model = _NativeMTPModel()
    with pytest.raises(ValueError, match="top_k must be smaller than vocabulary size"):
        _run(
            model,
            max_tokens=1,
            sampling_config=NativeMTPSamplingConfig(temperature=1, top_k=7),
        )
    assert model.requests == []


def test_prefix_capability_and_generator_close_fail_closed_or_cleanup():
    with pytest.raises(ValueError, match="native_mtp_prefix_reuse_unsupported"):
        list(
            mtp_generate_step(
                mx.array([0], dtype=mx.uint32),
                _NativeMTPModel(),
                prompt_cache=[],
            )
        )

    unsupported = _NativeMTPModel()
    unsupported.mtp_capability = _Capability(False, "native_mtp_weights_not_loaded")
    with pytest.raises(RuntimeError, match="native_mtp_weights_not_loaded"):
        next(mtp_generate_step(mx.array([0], dtype=mx.uint32), unsupported))

    model = _NativeMTPModel()
    iterator = mtp_generate_step(mx.array([0, 1], dtype=mx.uint32), model)
    next(iterator)
    iterator.close()
    assert model.requests[-1].closed


def test_zero_length_finishes_without_forward_or_uninitialized_state():
    model = _NativeMTPModel()
    output, telemetry = _run(model, max_tokens=0)
    assert output == []
    assert telemetry["mtp_drafts"] == 0
    assert model.requests[-1].closed
