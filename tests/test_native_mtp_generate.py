# Copyright © 2026 Apple Inc.
"""Synthetic streaming coverage for the native Qwen MTP generator."""

import importlib
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

import mlx.core as mx
import pytest

from mlx_lm.generate import (
    GenerationForward,
    GenerationForwardPhase,
    GenerationForwardPositionReceipt,
    NativeMTPSamplingConfig,
    NativeMTPSparseBootstrap,
    attested_target_forward,
    mtp_generate_step,
    stream_generate,
)
from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    NativeMTPRequestCache,
    QuantizedKVCache,
)

generate_module = importlib.import_module("mlx_lm.generate")

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
        self.mtp_input_ids = []
        self.last_mtp_cache = None

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
        self.last_mtp_cache = [KVCache()]
        return self.last_mtp_cache

    def make_mtp_request_cache(self):
        request = NativeMTPRequestCache.create(self)
        self.requests.append(request)
        return request

    @staticmethod
    def _write_attention(entry, positions):
        values = mx.array(positions, dtype=mx.float32).reshape(1, 1, -1, 1)
        values = mx.broadcast_to(values, (1, 1, len(positions), 32))
        entry.update_and_fetch(values, values)

    @staticmethod
    def _logical_positions(count, physical_start):
        forward = _ACTIVE_FORWARD.get()
        if forward is not None and forward.logical_positions is not None:
            return forward.logical_positions
        return tuple(range(physical_start, physical_start + count))

    def _logits(self, inputs, *, offset=1):
        ids = (inputs.astype(mx.int32) + offset) % self.vocab_size
        logits = mx.full((1, inputs.shape[1], self.vocab_size), -20.0)
        for index in range(inputs.shape[1]):
            logits[:, index, ids[0, index]] = 20.0
        return logits

    def __call__(self, inputs, *, cache, return_hidden=False):
        self._acknowledge_forward_positions()
        count = inputs.shape[1]
        positions = self._logical_positions(count, cache[1].offset)
        cache[0][0] = cache[0][0] + count
        cache[0][1] = cache[0][1] + count
        cache[0].advance(count)
        self._write_attention(cache[1], positions)
        self.target_offset = cache[1].offset
        self.target_offsets.append(self.target_offset)
        logits = self._logits(inputs)
        hidden = mx.stack(
            [
                inputs.astype(mx.float32),
                mx.array(positions, dtype=mx.float32)[None],
            ],
            axis=-1,
        )
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(self, hidden, next_token_ids, cache):
        self._acknowledge_forward_positions()
        self.mtp_input_ids.extend(next_token_ids.reshape(-1).tolist())
        self.mtp_attempt_offsets.append(cache[0].offset)
        if cache[0].offset in self.fail_mtp_offsets:
            raise RuntimeError("synthetic unexpected next draft")
        positions = self._logical_positions(next_token_ids.shape[1], cache[0].offset)
        self._write_attention(cache[0], positions)
        self.mtp_offsets.append(cache[0].offset)
        hidden_positions = tuple(int(position) for position in hidden[0, :, 1].tolist())
        offset = (
            2
            if self.wrong_draft
            or cache[0].offset != self.target_offset
            or hidden_positions != positions
            else 1
        )
        return self._logits(next_token_ids, offset=offset)


class _NativeMTPMoEModel(_NativeMTPModel):
    """Tiny two-expert target/MTP double with token-routed expert logits."""

    architecture = "qwen3_5_moe"
    num_experts = 2

    def _logits(self, inputs, *, offset=1):
        expert = inputs.astype(mx.int32) % self.num_experts
        expert_zero = inputs.astype(mx.int32) + offset
        expert_one = inputs.astype(mx.int32) - (self.vocab_size - offset)
        ids = mx.where(expert == 0, expert_zero, expert_one) % self.vocab_size
        logits = mx.full((1, inputs.shape[1], self.vocab_size), -20.0)
        for index in range(inputs.shape[1]):
            logits[:, index, ids[0, index]] = 20.0
        return logits


class _NativeMTPQuantizedTargetModel(_NativeMTPModel):
    """Target double with a known active packed uint32 authority payload."""

    def make_cache(self):
        recurrent = ArraysCache(2)
        recurrent[0] = mx.array([[0.0]])
        recurrent[1] = mx.array([[0.0]])
        return [recurrent, QuantizedKVCache(group_size=32, bits=8)]

    def __call__(self, inputs, *, cache, return_hidden=False):
        result = super().__call__(inputs, cache=cache, return_hidden=return_hidden)
        cache[1].keys[0][0, 0, 0, 0] = mx.array(16777216, dtype=mx.uint32)
        return result


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


class _StreamDetokenizer:
    def __init__(self):
        self.last_segment = ""
        self.finalized = False

    def add_token(self, token):
        self.last_segment = str(token)

    def finalize(self):
        self.finalized = True


class _StreamTokenizer:
    def __init__(self):
        self.detokenizer = _StreamDetokenizer()
        self.eos_token_ids = frozenset()


def _make_sparse_bootstrap(
    model,
    context,
    *,
    positions,
    selected_token_ids,
    immediate_successor_token_ids,
    chunk_sizes,
):
    target_cache = model.make_cache()
    receipts = []
    start = 0
    for chunk_size in chunk_sizes:
        end = start + chunk_size
        (_, _), receipt = attested_target_forward(
            model,
            selected_token_ids[start:end],
            target_cache,
            phase=GenerationForwardPhase.PREFILL,
            logical_positions=positions[start:end],
            immediate_successor_token_ids=immediate_successor_token_ids[
                start : min(end, len(immediate_successor_token_ids))
            ],
            model_forward_context=context.context,
        )
        receipts.append(receipt)
        start = end
    assert start == len(positions)
    return NativeMTPSparseBootstrap(
        receipts=tuple(receipts),
        selected_logical_positions=positions,
        selected_token_ids=selected_token_ids,
        immediate_successor_token_ids=immediate_successor_token_ids,
        target_cache=target_cache,
        next_logical_position=positions[-1] + 1,
    )


def test_stream_generate_dispatches_attested_sparse_bootstrap_without_prompt_array(
    monkeypatch,
):
    model = _NativeMTPModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0, 3, 7),
        selected_token_ids=(0, 3, 5),
        immediate_successor_token_ids=(1, 4),
        chunk_sizes=(1, 2),
    )
    captured = {}
    logprobs = mx.zeros((model.vocab_size,))

    def fake_mtp_generate_step(prompt, actual_model, **kwargs):
        captured["prompt"] = prompt
        captured["model"] = actual_model
        captured.update(kwargs)
        kwargs["telemetry"].update(mtp_drafts=4, mtp_accepted=3)
        yield 6, logprobs, True

    monkeypatch.setattr(generate_module, "TokenizerWrapper", _StreamTokenizer)
    monkeypatch.setattr(generate_module, "mtp_generate_step", fake_mtp_generate_step)
    tokenizer = _StreamTokenizer()

    responses = list(
        stream_generate(
            model,
            tokenizer,
            None,
            mtp=True,
            max_tokens=2,
            sparse_bootstrap=bootstrap,
            model_forward_context=context.context,
        )
    )

    assert captured["prompt"] is None
    assert captured["model"] is model
    assert captured["sparse_bootstrap"] is bootstrap
    assert captured["model_forward_context"] == context.context
    assert [response.prompt_tokens for response in responses] == [8, 8]
    assert [response.prompt_tps for response in responses] == [0.0, 0.0]
    assert [response.mtp_drafts for response in responses] == [4, 4]
    assert [response.mtp_accepted for response in responses] == [3, 3]
    assert [response.text for response in responses] == ["6", "6"]
    assert tokenizer.detokenizer.finalized
    assert responses[-1].finish_reason == "length"


def test_stream_generate_dense_native_mtp_omits_sparse_bootstrap_keyword(monkeypatch):
    captured = {}
    logprobs = mx.zeros((7,))

    def fake_mtp_generate_step(prompt, actual_model, **kwargs):
        captured["prompt"] = prompt
        captured["model"] = actual_model
        captured.update(kwargs)
        yield 2, logprobs, False

    model = _NativeMTPModel()
    monkeypatch.setattr(generate_module, "TokenizerWrapper", _StreamTokenizer)
    monkeypatch.setattr(generate_module, "mtp_generate_step", fake_mtp_generate_step)
    responses = list(
        stream_generate(
            model,
            _StreamTokenizer(),
            [0, 1],
            mtp=True,
            max_tokens=2,
        )
    )

    assert captured["prompt"].tolist() == [0, 1]
    assert captured["model"] is model
    assert "sparse_bootstrap" not in captured
    assert [response.prompt_tokens for response in responses] == [2, 2]


@pytest.mark.parametrize(
    ("prompt", "mtp", "draft_model", "bootstrap", "kwargs", "message"),
    (
        (None, False, None, "valid", {}, "native_mtp_sparse_bootstrap_requires_mtp"),
        (
            None,
            True,
            object(),
            "valid",
            {},
            "native_mtp_sparse_bootstrap_external_draft_unsupported",
        ),
        (None, True, None, None, {}, "stream_generate requires a prompt"),
        (
            (0,),
            True,
            None,
            "valid",
            {},
            "native_mtp_sparse_bootstrap_owns_selected_tokens",
        ),
        (
            None,
            True,
            None,
            "valid",
            {"prompt_logical_positions": (0,)},
            "native_mtp_sparse_bootstrap_owns_logical_positions",
        ),
        (
            None,
            True,
            None,
            "valid",
            {"prompt_cache": []},
            "native_mtp_prefix_reuse_unsupported",
        ),
        (None, True, None, object(), {}, "native_mtp_sparse_bootstrap_invalid"),
    ),
)
def test_stream_generate_rejects_invalid_sparse_none_combinations_before_prompt_mutation(
    monkeypatch, prompt, mtp, draft_model, bootstrap, kwargs, message
):
    model = _NativeMTPModel()
    context = _PositionContext()
    valid_bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0,),
        selected_token_ids=(0,),
        immediate_successor_token_ids=(),
        chunk_sizes=(1,),
    )
    if bootstrap == "valid":
        bootstrap = valid_bootstrap

    def prompt_conversion_forbidden(*_args, **_kwargs):
        raise AssertionError("prompt conversion must not run before sparse validation")

    monkeypatch.setattr(generate_module.mx, "array", prompt_conversion_forbidden)
    with pytest.raises((TypeError, ValueError), match=message):
        list(
            stream_generate(
                model,
                _StreamTokenizer(),
                prompt,
                mtp=mtp,
                draft_model=draft_model,
                sparse_bootstrap=bootstrap,
                model_forward_context=context.context,
                **kwargs,
            )
        )


@pytest.mark.parametrize(
    ("model_capability", "context", "message"),
    (
        (None, None, "native_mtp_sparse_bootstrap_requires_position_context"),
        (
            _Capability(False, "native_mtp_weights_not_loaded"),
            "context",
            "native_mtp_weights_not_loaded",
        ),
    ),
)
def test_stream_generate_sparse_rejects_context_and_capability_before_mutation(
    monkeypatch, model_capability, context, message
):
    model = _NativeMTPModel()
    position_context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        position_context,
        positions=(0,),
        selected_token_ids=(0,),
        immediate_successor_token_ids=(),
        chunk_sizes=(1,),
    )
    if model_capability is not None:
        model.mtp_capability = model_capability
    if context == "context":
        context = position_context.context

    def prompt_conversion_forbidden(*_args, **_kwargs):
        raise AssertionError("prompt conversion must not run before sparse validation")

    monkeypatch.setattr(generate_module.mx, "array", prompt_conversion_forbidden)
    with pytest.raises((RuntimeError, ValueError), match=message):
        list(
            stream_generate(
                model,
                _StreamTokenizer(),
                None,
                mtp=True,
                sparse_bootstrap=bootstrap,
                model_forward_context=context,
            )
        )
    assert model.requests == []


def test_stream_generate_sparse_bootstrap_is_one_shot_and_closes_on_failure(
    monkeypatch,
):
    monkeypatch.setattr(generate_module, "TokenizerWrapper", _StreamTokenizer)
    adopted = []
    original_adopt = NativeMTPRequestCache.adopt_sparse_target.__func__

    def capture_adopt(cls, *args, **kwargs):
        request = original_adopt(cls, *args, **kwargs)
        adopted.append(request)
        return request

    monkeypatch.setattr(
        NativeMTPRequestCache, "adopt_sparse_target", classmethod(capture_adopt)
    )

    model = _NativeMTPModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0, 3),
        selected_token_ids=(0, 3),
        immediate_successor_token_ids=(1,),
        chunk_sizes=(2,),
    )
    iterator = stream_generate(
        model,
        _StreamTokenizer(),
        None,
        mtp=True,
        max_tokens=4,
        sparse_bootstrap=bootstrap,
        model_forward_context=context.context,
    )
    assert next(iterator).token == 4
    request = adopted[-1]
    iterator.close()
    assert request.closed
    assert context.active == 0

    with pytest.raises(RuntimeError, match="native_mtp_sparse_bootstrap_already_claimed"):
        list(
            stream_generate(
                model,
                _StreamTokenizer(),
                None,
                mtp=True,
                sparse_bootstrap=bootstrap,
                model_forward_context=context.context,
            )
        )

    failing_model = _NativeMTPModel(fail_mtp_offsets={0})
    failing_context = _PositionContext()
    failing_bootstrap = _make_sparse_bootstrap(
        failing_model,
        failing_context,
        positions=(0, 3),
        selected_token_ids=(0, 3),
        immediate_successor_token_ids=(1,),
        chunk_sizes=(2,),
    )
    with pytest.raises(RuntimeError, match="synthetic unexpected next draft"):
        list(
            stream_generate(
                failing_model,
                _StreamTokenizer(),
                None,
                mtp=True,
                sparse_bootstrap=failing_bootstrap,
                model_forward_context=failing_context.context,
            )
        )
    assert adopted[-1].closed
    assert failing_context.active == 0


def test_stream_generate_sparse_zero_tokens_finalizes_with_terminal_telemetry(
    monkeypatch,
):
    monkeypatch.setattr(generate_module, "TokenizerWrapper", _StreamTokenizer)
    adopted = []
    original_adopt = NativeMTPRequestCache.adopt_sparse_target.__func__

    def capture_adopt(cls, *args, **kwargs):
        request = original_adopt(cls, *args, **kwargs)
        adopted.append(request)
        return request

    monkeypatch.setattr(
        NativeMTPRequestCache, "adopt_sparse_target", classmethod(capture_adopt)
    )
    model = _NativeMTPModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0, 3),
        selected_token_ids=(0, 3),
        immediate_successor_token_ids=(1,),
        chunk_sizes=(2,),
    )
    tokenizer = _StreamTokenizer()
    responses = list(
        stream_generate(
            model,
            tokenizer,
            None,
            mtp=True,
            max_tokens=0,
            sparse_bootstrap=bootstrap,
            model_forward_context=context.context,
        )
    )

    assert len(responses) == 1
    terminal = responses[0]
    assert tokenizer.detokenizer.finalized
    assert terminal.text == ""
    assert terminal.generation_tokens == 0
    assert terminal.prompt_tokens == 4
    assert terminal.prompt_tps == 0.0
    assert terminal.finish_reason == "length"
    assert terminal.mtp_drafts == 0
    assert terminal.mtp_accepted == 0
    assert terminal.mtp_bypass_reason is None
    assert adopted[-1].closed
    assert context.active == 0


def test_stream_generate_ordinary_prompt_path_is_unchanged(monkeypatch):
    captured = {}
    logprobs = mx.zeros((7,))

    def fake_generate_step(prompt, model, **kwargs):
        captured["prompt"] = prompt
        captured["model"] = model
        captured.update(kwargs)
        yield 2, logprobs

    model = _NativeMTPModel()
    monkeypatch.setattr(generate_module, "TokenizerWrapper", _StreamTokenizer)
    monkeypatch.setattr(generate_module, "generate_step", fake_generate_step)
    responses = list(
        stream_generate(
            model,
            _StreamTokenizer(),
            [0, 1],
            max_tokens=2,
        )
    )

    assert isinstance(captured["prompt"], mx.array)
    assert captured["prompt"].tolist() == [0, 1]
    assert captured["model"] is model
    assert "sparse_bootstrap" not in captured
    assert [response.prompt_tokens for response in responses] == [2, 2]
    assert responses[-1].mtp_bypass_reason is None


def _direct_positioned_target(model, cache, context, token_ids, positions):
    tokens = mx.array(token_ids, dtype=mx.uint32)
    forward = GenerationForward(
        model=model,
        input_tokens=tokens[None],
        cache=cache,
        phase=GenerationForwardPhase.VERIFY,
        logical_positions=positions,
    )
    with context.context(forward):
        result = model(tokens[None], cache=cache, return_hidden=True)
    mx.eval(result)
    return result


def _direct_positioned_mtp(model, cache, context, hidden, token_ids, positions):
    tokens = mx.array(token_ids, dtype=mx.uint32)
    forward = GenerationForward(
        model=model,
        input_tokens=tokens[None],
        cache=cache,
        phase=GenerationForwardPhase.MTP_DRAFT,
        logical_positions=positions,
    )
    with context.context(forward):
        logits = model.mtp_forward(hidden, tokens[None], cache)
    mx.eval(logits)
    return logits


def _per_token_native_oracle(
    model,
    *,
    positions,
    selected_token_ids,
    immediate_successor_token_ids,
    rejected,
):
    context = _PositionContext()
    target_cache = model.make_cache()
    evidence = []
    for index, (position, token_id) in enumerate(zip(positions, selected_token_ids)):
        successors = (
            (immediate_successor_token_ids[index],)
            if index < len(immediate_successor_token_ids)
            else ()
        )
        result, _ = attested_target_forward(
            model,
            (token_id,),
            target_cache,
            phase=GenerationForwardPhase.PREFILL,
            logical_positions=(position,),
            immediate_successor_token_ids=successors,
            model_forward_context=context.context,
        )
        evidence.append(result)

    mtp_cache = model.make_mtp_cache()
    for index, successor in enumerate(immediate_successor_token_ids):
        _direct_positioned_mtp(
            model,
            mtp_cache,
            context,
            evidence[index][1],
            (successor,),
            (positions[index],),
        )
    final_logits, final_hidden = evidence[-1]
    first_logprobs = final_logits[:, -1, :].squeeze(0)
    first_logprobs = first_logprobs - mx.logsumexp(first_logprobs)
    current = mx.argmax(first_logprobs).item()
    draft_logits = _direct_positioned_mtp(
        model,
        mtp_cache,
        context,
        final_hidden,
        (current,),
        (positions[-1],),
    )
    draft = mx.argmax(draft_logits[:, -1, :]).item()

    verify_positions = (positions[-1] + 1, positions[-1] + 2)
    if rejected:
        disposable = model.make_cache()
        for position, token_id in zip(positions, selected_token_ids):
            _direct_positioned_target(
                model, disposable, context, (token_id,), (position,)
            )
        verify_logits, _ = _direct_positioned_target(
            model, disposable, context, (current, draft), verify_positions
        )
        replacement = mx.argmax(verify_logits[:, 0, :]).item()
        _direct_positioned_target(
            model, target_cache, context, (current,), (verify_positions[0],)
        )
        second = replacement
    else:
        verify_logits, _ = _direct_positioned_target(
            model, target_cache, context, (current, draft), verify_positions
        )
        second = draft
    second_logprobs = verify_logits[:, 0, :].squeeze(0)
    second_logprobs = second_logprobs - mx.logsumexp(second_logprobs)
    return (current, second), (first_logprobs, second_logprobs), target_cache, mtp_cache


def _assert_kv_cache_equal(left, right):
    assert left.offset == right.offset
    active = slice(0, left.offset)
    assert mx.array_equal(left.keys[..., active, :], right.keys[..., active, :]).item()
    assert mx.array_equal(
        left.values[..., active, :], right.values[..., active, :]
    ).item()


def _assert_arrays_cache_equal(left, right):
    for left_value, right_value in zip(left.cache, right.cache):
        assert mx.array_equal(left_value, right_value).item()


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


@pytest.mark.parametrize("model_type", (_NativeMTPModel, _NativeMTPMoEModel))
def test_attested_keep_one_bootstrap_matches_dense_prefill(model_type, monkeypatch):
    dense_model = model_type()
    dense_output, dense_telemetry = _run(
        dense_model, prompt=(0, 1, 2, 3), max_tokens=4, prefill_step_size=2
    )
    sparse_model = model_type()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        sparse_model,
        context,
        positions=(0, 1, 2, 3),
        selected_token_ids=(0, 1, 2, 3),
        immediate_successor_token_ids=(1, 2, 3),
        chunk_sizes=(2, 2),
    )
    context.events.clear()
    sparse_telemetry = {}
    adopted = []
    original = NativeMTPRequestCache.adopt_sparse_target.__func__

    def capture(cls, *args, **kwargs):
        request = original(cls, *args, **kwargs)
        adopted.append(request)
        return request

    monkeypatch.setattr(
        NativeMTPRequestCache, "adopt_sparse_target", classmethod(capture)
    )

    sparse_output = list(
        mtp_generate_step(
            None,
            sparse_model,
            max_tokens=4,
            prefill_step_size=2,
            sparse_bootstrap=bootstrap,
            model_forward_context=context.context,
            telemetry=sparse_telemetry,
        )
    )

    assert [(token, drafted) for token, _, drafted in sparse_output] == [
        (token, drafted) for token, _, drafted in dense_output
    ]
    assert sparse_telemetry == dense_telemetry
    for (_, dense_logprobs, _), (_, sparse_logprobs, _) in zip(
        dense_output, sparse_output
    ):
        assert mx.allclose(dense_logprobs, sparse_logprobs).item()
    assert sparse_model.mtp_input_ids[:3] == [1, 2, 3]
    assert not any(
        event.phase is GenerationForwardPhase.PREFILL for event in context.events
    )
    dense_request = dense_model.requests[-1]
    sparse_request = adopted[0]
    _assert_arrays_cache_equal(dense_request.backbone[0], bootstrap.target_cache[0])
    _assert_kv_cache_equal(dense_request.backbone[1], bootstrap.target_cache[1])
    _assert_kv_cache_equal(dense_request.mtp[0], sparse_request.mtp[0])
    assert sparse_request.state.next_logical_position == 8


def _native_position_oracle(*, rejected):
    phases = [
        (GenerationForwardPhase.MTP_DRAFT, (0,)),
        (GenerationForwardPhase.MTP_DRAFT, (3,)),
        (GenerationForwardPhase.MTP_DRAFT, (7,)),
        (GenerationForwardPhase.VERIFY, (8, 9)),
    ]
    if rejected:
        phases.append((GenerationForwardPhase.DECODE, (8,)))
    return phases


@pytest.mark.parametrize("model_type", (_NativeMTPModel, _NativeMTPMoEModel))
@pytest.mark.parametrize("rejected", (False, True))
def test_attested_noncontiguous_positions_match_per_token_oracle(
    model_type, rejected, monkeypatch
):
    oracle = _per_token_native_oracle(
        model_type(wrong_draft=rejected),
        positions=(0, 3, 7),
        selected_token_ids=(0, 3, 5),
        immediate_successor_token_ids=(1, 4),
        rejected=rejected,
    )
    model = model_type(wrong_draft=rejected)
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0, 3, 7),
        selected_token_ids=(0, 3, 5),
        immediate_successor_token_ids=(1, 4),
        chunk_sizes=(1, 2),
    )
    context.events.clear()
    adopted = []
    original = NativeMTPRequestCache.adopt_sparse_target.__func__

    def capture(cls, *args, **kwargs):
        request = original(cls, *args, **kwargs)
        adopted.append(request)
        return request

    monkeypatch.setattr(
        NativeMTPRequestCache, "adopt_sparse_target", classmethod(capture)
    )

    output = list(
        mtp_generate_step(
            None,
            model,
            max_tokens=2,
            prefill_step_size=1,
            sparse_bootstrap=bootstrap,
            model_forward_context=context.context,
        )
    )

    oracle_tokens, oracle_logprobs, oracle_target, oracle_mtp = oracle
    assert tuple(token for token, _, _ in output) == oracle_tokens
    for (_, logprobs, _), expected in zip(output, oracle_logprobs):
        assert mx.allclose(logprobs, expected).item()
    assert _event_contract(context.events) == _native_position_oracle(rejected=rejected)
    assert model.mtp_input_ids[:2] == [1, 4]
    request = adopted[0]
    _assert_arrays_cache_equal(bootstrap.target_cache[0], oracle_target[0])
    _assert_kv_cache_equal(bootstrap.target_cache[1], oracle_target[1])
    _assert_kv_cache_equal(request.mtp[0], oracle_mtp[0])
    assert request.state.next_logical_position == 10


def test_attested_receipts_are_sealed_and_duplicate_authority_must_match():
    with pytest.raises(TypeError):
        GenerationForwardPositionReceipt()

    model = _NativeMTPModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0, 3),
        selected_token_ids=(0, 3),
        immediate_successor_token_ids=(1,),
        chunk_sizes=(1, 1),
    )
    forged = object.__new__(GenerationForwardPositionReceipt)
    for name in (
        "model_id",
        "cache_container_id",
        "cache_entry_ids",
        "capability",
        "phase",
        "logical_positions",
        "token_ids",
        "immediate_successor_token_ids",
        "logits",
        "hidden_rows",
        "_issuer_seal",
        "_record_token",
    ):
        object.__setattr__(forged, name, getattr(bootstrap.receipts[0], name))
    with pytest.raises(RuntimeError, match="receipt_canonical_mismatch"):
        replace(bootstrap, receipts=(forged, bootstrap.receipts[1])).validate(model)
    with pytest.raises(ValueError, match="receipt_successors_mismatch"):
        replace(bootstrap, immediate_successor_token_ids=(2,)).validate(model)


def test_sparse_bootstrap_requires_final_canonical_logits_evidence():
    model = _NativeMTPModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0, 3),
        selected_token_ids=(0, 3),
        immediate_successor_token_ids=(1,),
        chunk_sizes=(1, 1),
    )
    malformed = replace(
        bootstrap,
        receipts=tuple(reversed(bootstrap.receipts)),
    )
    with pytest.raises(RuntimeError, match="final_logits_evidence_missing"):
        malformed.validate(model)


@pytest.mark.parametrize(
    "mutation",
    (
        "host",
        "cache",
        "model_id",
        "cache_container_id",
        "cache_entry_ids",
        "capability",
        "phase",
        "logical_positions",
        "token_ids",
        "successors",
        "logits",
        "logits_content",
        "hidden_rows",
        "hidden_content",
        "record_token",
    ),
)
def test_sparse_bootstrap_canonical_authority_rejects_mutation(mutation):
    model = _NativeMTPModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0, 3),
        selected_token_ids=(0, 3),
        immediate_successor_token_ids=(1,),
        chunk_sizes=(1, 1),
    )
    if mutation == "host":
        candidate = replace(bootstrap, selected_token_ids=(0, 2))
    elif mutation == "cache":
        candidate = replace(bootstrap, target_cache=list(bootstrap.target_cache))
    else:
        receipt = bootstrap.receipts[-1]
        if mutation == "logits_content":
            receipt.logits[:, -1, 0] = receipt.logits[:, -1, 0] + 1
            candidate = bootstrap
        elif mutation == "hidden_content":
            receipt.hidden_rows[:, -1, 0] = receipt.hidden_rows[:, -1, 0] + 1
            candidate = bootstrap
        else:
            mutations = {
                "model_id": receipt.model_id + 1,
                "cache_container_id": receipt.cache_container_id + 1,
                "cache_entry_ids": tuple(reversed(receipt.cache_entry_ids)),
                "capability": replace(receipt.capability, reason="mutated"),
                "phase": GenerationForwardPhase.VERIFY,
                "logical_positions": (2,),
                "token_ids": (2,),
                "successors": (1,),
                "logits": receipt.logits + 0,
                "hidden_rows": receipt.hidden_rows + 0,
                "record_token": bootstrap.receipts[0]._record_token,
            }
            field_name = (
                "immediate_successor_token_ids"
                if mutation == "successors"
                else f"_{mutation}" if mutation == "record_token" else mutation
            )
            object.__setattr__(receipt, field_name, mutations[mutation])
            candidate = bootstrap

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        candidate.validate(model)


@pytest.mark.parametrize("cache_kind", ("arrays", "kv"))
def test_sparse_claim_rejects_same_identity_cache_content_mutation_permanently(
    cache_kind,
):
    model = _NativeMTPModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0, 3),
        selected_token_ids=(0, 3),
        immediate_successor_token_ids=(1,),
        chunk_sizes=(1, 1),
    )
    if cache_kind == "arrays":
        state = bootstrap.target_cache[0][0]
        state[0, 0] = state[0, 0] + 1
    else:
        keys = bootstrap.target_cache[1].keys
        keys[0, 0, 0, 0] = keys[0, 0, 0, 0] + 1

    with pytest.raises(RuntimeError, match="evidence_content_mismatch"):
        bootstrap.claim(model)
    with pytest.raises(RuntimeError, match="already_claimed"):
        bootstrap.claim(model)


def test_sparse_claim_detects_exact_active_quantized_packed_word_mutation():
    model = _NativeMTPQuantizedTargetModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0,),
        selected_token_ids=(0,),
        immediate_successor_token_ids=(),
        chunk_sizes=(1,),
    )
    packed_keys = bootstrap.target_cache[1].keys[0]
    assert packed_keys[0, 0, 0, 0].item() == 16777216
    packed_keys[0, 0, 0, 0] = mx.array(16777217, dtype=mx.uint32)

    with pytest.raises(RuntimeError, match="evidence_content_mismatch"):
        bootstrap.claim(model)
    with pytest.raises(RuntimeError, match="already_claimed"):
        bootstrap.claim(model)


def test_sparse_claim_uses_one_aggregate_host_decision_and_final_cache_scan(
    monkeypatch,
):
    scans = 0
    decisions = 0
    hash_reductions = 0
    original_digest = generate_module._cache_content_digest
    original_decision = generate_module._sparse_attestation_host_decision
    original_sum = mx.sum

    def count_digest(entries):
        nonlocal scans
        scans += 1
        return original_digest(entries)

    def count_decision(value):
        nonlocal decisions
        decisions += 1
        return original_decision(value)

    def count_sum(value, *args, **kwargs):
        nonlocal hash_reductions
        axis = kwargs.get("axis", args[0] if args else None)
        if (
            value.dtype == mx.uint64
            and value.ndim > 1
            and value.shape[-1] == 2
            and axis == tuple(range(value.ndim - 1))
        ):
            hash_reductions += 1
        return original_sum(value, *args, **kwargs)

    monkeypatch.setattr(generate_module, "_cache_content_digest", count_digest)
    monkeypatch.setattr(
        generate_module, "_sparse_attestation_host_decision", count_decision
    )
    monkeypatch.setattr(mx, "sum", count_sum)
    model = _NativeMTPModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0, 3),
        selected_token_ids=(0, 3),
        immediate_successor_token_ids=(1,),
        chunk_sizes=(1, 1),
    )
    assert scans == 1
    # Two recurrent arrays plus active K and V: one fixed-width reduction each.
    assert hash_reductions == 4
    bootstrap.claim(model)
    assert scans == 2
    assert decisions == 1
    assert hash_reductions == 8


def test_unrelated_sparse_claim_reserves_while_device_verification_is_blocked(
    monkeypatch,
):
    first_model = _NativeMTPModel()
    first_context = _PositionContext()
    first = _make_sparse_bootstrap(
        first_model,
        first_context,
        positions=(0,),
        selected_token_ids=(0,),
        immediate_successor_token_ids=(),
        chunk_sizes=(1,),
    )
    second_model = _NativeMTPModel()
    second_context = _PositionContext()
    second = _make_sparse_bootstrap(
        second_model,
        second_context,
        positions=(0,),
        selected_token_ids=(0,),
        immediate_successor_token_ids=(),
        chunk_sizes=(1,),
    )
    blocked = threading.Event()
    release = threading.Event()
    result = []
    original_verify = generate_module._verify_sparse_canonical_content
    first_receipt_id = id(first.receipts[0])

    def block_first(records):
        if records[0].receipt_id == first_receipt_id:
            blocked.set()
            assert release.wait(timeout=5)
        return original_verify(records)

    monkeypatch.setattr(
        generate_module, "_verify_sparse_canonical_content", block_first
    )

    def claim_first():
        result.append(first.claim(first_model))

    thread = threading.Thread(target=claim_first)
    thread.start()
    assert blocked.wait(timeout=5)
    second_claim = second.claim(second_model)
    release.set()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert len(result) == 1
    assert result[0].target_cache is first.target_cache
    assert second_claim.target_cache is second.target_cache


@pytest.mark.parametrize("failure_kind", ("telemetry", "eos"))
def test_immediate_post_adoption_failure_closes_and_consumes(failure_kind, monkeypatch):
    model = _NativeMTPModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0, 3),
        selected_token_ids=(0, 3),
        immediate_successor_token_ids=(1,),
        chunk_sizes=(1, 1),
    )
    adopted = []
    original = NativeMTPRequestCache.adopt_sparse_target.__func__

    def capture(cls, *args, **kwargs):
        request = original(cls, *args, **kwargs)
        adopted.append(request)
        return request

    class FailingTelemetry(dict):
        def update(self, *args, **kwargs):
            raise RuntimeError("synthetic immediate post-adoption failure")

    monkeypatch.setattr(
        NativeMTPRequestCache, "adopt_sparse_target", classmethod(capture)
    )
    kwargs = (
        {"telemetry": FailingTelemetry()}
        if failure_kind == "telemetry"
        else {"eos_token_ids": [[]]}
    )
    with pytest.raises((RuntimeError, TypeError)):
        next(
            mtp_generate_step(
                None,
                model,
                sparse_bootstrap=bootstrap,
                model_forward_context=context.context,
                **kwargs,
            )
        )
    request = adopted[0]
    assert request.closed
    assert not request.checkpoint_active
    assert request.backbone[1].offset == 2
    assert request.mtp[0].offset == 0
    with pytest.raises(RuntimeError, match="already_claimed"):
        next(
            mtp_generate_step(
                None,
                model,
                sparse_bootstrap=bootstrap,
                model_forward_context=context.context,
            )
        )


def test_baseexception_after_adoption_is_inside_generator_cleanup(monkeypatch):
    class SyntheticAbort(BaseException):
        pass

    model = _NativeMTPModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0, 3),
        selected_token_ids=(0, 3),
        immediate_successor_token_ids=(1,),
        chunk_sizes=(1, 1),
    )
    adopted = []
    original_adopt = NativeMTPRequestCache.adopt_sparse_target.__func__

    def capture(cls, *args, **kwargs):
        request = original_adopt(cls, *args, **kwargs)
        adopted.append(request)
        return request

    def abort_rng(*args, **kwargs):
        raise SyntheticAbort("synthetic former-gap abort")

    monkeypatch.setattr(
        NativeMTPRequestCache, "adopt_sparse_target", classmethod(capture)
    )
    monkeypatch.setattr(mx.random, "key", abort_rng)
    with pytest.raises(SyntheticAbort, match="former-gap abort"):
        next(
            mtp_generate_step(
                None,
                model,
                sparse_bootstrap=bootstrap,
                model_forward_context=context.context,
            )
        )

    assert len(adopted) == 1
    assert adopted[0].closed
    assert not adopted[0].checkpoint_active
    assert adopted[0].backbone[1].offset == 2
    assert adopted[0].mtp[0].offset == 0
    with pytest.raises(RuntimeError, match="already_claimed"):
        bootstrap.claim(model)


def test_sparse_claim_stays_consumed_when_adoption_fails(monkeypatch):
    model = _NativeMTPModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0,),
        selected_token_ids=(0,),
        immediate_successor_token_ids=(),
        chunk_sizes=(1,),
    )
    original = NativeMTPRequestCache.adopt_sparse_target.__func__

    def fail_adoption(*args, **kwargs):
        raise RuntimeError("synthetic adoption failure")

    monkeypatch.setattr(
        NativeMTPRequestCache,
        "adopt_sparse_target",
        classmethod(fail_adoption),
    )
    with pytest.raises(RuntimeError, match="synthetic adoption failure"):
        next(
            mtp_generate_step(
                None,
                model,
                sparse_bootstrap=bootstrap,
                model_forward_context=context.context,
            )
        )
    monkeypatch.setattr(
        NativeMTPRequestCache, "adopt_sparse_target", classmethod(original)
    )
    with pytest.raises(RuntimeError, match="already_claimed"):
        next(
            mtp_generate_step(
                None,
                model,
                sparse_bootstrap=bootstrap,
                model_forward_context=context.context,
            )
        )


@pytest.mark.parametrize("action", ("close", "cancel"))
def test_sparse_bootstrap_one_shot_double_iterator_and_post_yield_cleanup(
    action, monkeypatch
):
    model = _NativeMTPModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0, 3),
        selected_token_ids=(0, 3),
        immediate_successor_token_ids=(1,),
        chunk_sizes=(1, 1),
    )
    context.events.clear()
    adopted = []
    original = NativeMTPRequestCache.adopt_sparse_target.__func__

    def capture(cls, *args, **kwargs):
        request = original(cls, *args, **kwargs)
        adopted.append(request)
        return request

    monkeypatch.setattr(
        NativeMTPRequestCache, "adopt_sparse_target", classmethod(capture)
    )
    first = mtp_generate_step(
        None,
        model,
        max_tokens=4,
        sparse_bootstrap=bootstrap,
        model_forward_context=context.context,
    )
    second = mtp_generate_step(
        None,
        model,
        max_tokens=4,
        sparse_bootstrap=bootstrap,
        model_forward_context=context.context,
    )
    next(first)
    assert context.active == 0
    with pytest.raises(RuntimeError, match="already_claimed"):
        next(second)
    assert context.active == 0
    if action == "close":
        first.close()
    else:
        with pytest.raises(RuntimeError, match="synthetic cancellation"):
            first.throw(RuntimeError("synthetic cancellation"))
    assert adopted[0].closed
    assert not adopted[0].checkpoint_active
    assert context.active == 0


def test_sparse_bootstrap_concurrent_claim_is_exactly_once():
    model = _NativeMTPModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0, 3),
        selected_token_ids=(0, 3),
        immediate_successor_token_ids=(1,),
        chunk_sizes=(1, 1),
    )
    context.events.clear()
    claim_barrier = threading.Barrier(2)
    outcomes = []

    def claim():
        try:
            claim_barrier.wait(timeout=5)
            claimed = bootstrap.claim(model)
        except RuntimeError as error:
            outcomes.append(("error", str(error)))
        else:
            outcomes.append(("success", claimed))

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert [outcome[0] for outcome in outcomes].count("success") == 1
    assert [outcome[0] for outcome in outcomes].count("error") == 1
    assert any(
        status == "error" and "already_claimed" in value for status, value in outcomes
    )
    claimed = next(value for status, value in outcomes if status == "success")
    # Claiming only transfers opaque authority.  The losing owner must fail
    # before either the attested target cache or a fresh MTP cache is mutated.
    assert claimed.target_cache[1].offset == 2
    request = NativeMTPRequestCache.adopt_sparse_target(
        model,
        target_cache=claimed.target_cache,
        target_tokens=len(claimed.selected_token_ids),
        next_logical_position=claimed.next_logical_position,
    )
    request.finish("cancelled")
    assert request.closed
    assert not request.checkpoint_active
    assert request.backbone[1].offset == 2
    assert request.mtp[0].offset == 0
    assert context.active == 0


def test_sparse_bootstrap_admission_fails_before_request_adoption():
    model = _NativeMTPModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0,),
        selected_token_ids=(0,),
        immediate_successor_token_ids=(),
        chunk_sizes=(1,),
    )
    with pytest.raises(ValueError, match="requires_position_context"):
        list(mtp_generate_step(None, model, sparse_bootstrap=bootstrap))
    with pytest.raises(ValueError, match="owns_selected_tokens"):
        list(
            mtp_generate_step(
                mx.array([0], dtype=mx.uint32),
                model,
                sparse_bootstrap=bootstrap,
                model_forward_context=context.context,
            )
        )
    with pytest.raises(ValueError, match="prefill_step_size"):
        list(
            mtp_generate_step(
                None,
                model,
                sparse_bootstrap=bootstrap,
                model_forward_context=context.context,
                prefill_step_size=0,
            )
        )
    assert model.requests == []


def test_sparse_bootstrap_failure_rolls_back_and_closes(monkeypatch):
    model = _NativeMTPModel(fail_mtp_offsets={0})
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0, 3),
        selected_token_ids=(0, 3),
        immediate_successor_token_ids=(1,),
        chunk_sizes=(1, 1),
    )
    adopted = []
    original = NativeMTPRequestCache.adopt_sparse_target.__func__

    def capture(cls, *args, **kwargs):
        request = original(cls, *args, **kwargs)
        adopted.append(request)
        return request

    monkeypatch.setattr(
        NativeMTPRequestCache, "adopt_sparse_target", classmethod(capture)
    )
    with pytest.raises(RuntimeError, match="synthetic unexpected next draft"):
        list(
            mtp_generate_step(
                None,
                model,
                sparse_bootstrap=bootstrap,
                model_forward_context=context.context,
            )
        )
    request = adopted[0]
    assert request.closed
    assert not request.checkpoint_active
    assert request.backbone[1].offset == 2
    assert request.mtp[0].offset == 0


@pytest.mark.parametrize("failure_kind", ("processor", "sampler", "lazy_eval"))
def test_sparse_pre_output_failure_rolls_back_unpublished_bootstrap(
    failure_kind, monkeypatch
):
    model = _NativeMTPModel()
    context = _PositionContext()
    bootstrap = _make_sparse_bootstrap(
        model,
        context,
        positions=(0, 3),
        selected_token_ids=(0, 3),
        immediate_successor_token_ids=(1,),
        chunk_sizes=(1, 1),
    )
    adopted = []
    original_adopt = NativeMTPRequestCache.adopt_sparse_target.__func__

    def capture(cls, *args, **kwargs):
        request = original_adopt(cls, *args, **kwargs)
        adopted.append(request)
        return request

    monkeypatch.setattr(
        NativeMTPRequestCache, "adopt_sparse_target", classmethod(capture)
    )
    kwargs = {}
    if failure_kind == "processor":

        def processor(_tokens, _logits):
            raise RuntimeError("synthetic pre-output failure")

        processor.native_mtp_replay_safe = True
        kwargs["logits_processors"] = [processor]
    elif failure_kind == "sampler":

        def sampler(_logprobs):
            raise RuntimeError("synthetic pre-output failure")

        sampler.native_mtp_deterministic = True
        kwargs["sampler"] = sampler
    else:
        original_eval = mx.eval

        def fail_current_eval(*values):
            if len(values) == 2 and all(
                isinstance(value, mx.array) for value in values
            ):
                raise RuntimeError("synthetic pre-output failure")
            return original_eval(*values)

        monkeypatch.setattr(mx, "eval", fail_current_eval)

    with pytest.raises(RuntimeError, match="synthetic pre-output failure"):
        list(
            mtp_generate_step(
                None,
                model,
                sparse_bootstrap=bootstrap,
                model_forward_context=context.context,
                **kwargs,
            )
        )
    request = adopted[0]
    assert request.closed
    assert not request.checkpoint_active
    assert request.replay_required is None
    assert request.state.backbone_tokens == 2
    assert request.state.mtp_tokens == 0
    assert request.backbone[1].offset == 2
    assert request.mtp[0].offset == 0


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
