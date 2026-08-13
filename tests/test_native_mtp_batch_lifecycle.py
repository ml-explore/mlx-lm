# Copyright © 2026 Apple Inc.
"""Typed public lifecycle coverage for native-MTP cohort admission."""

from contextlib import contextmanager
from dataclasses import dataclass
from collections import deque

import mlx.core as mx
import pytest

from test_qwen3_5_mtp_model import _checkpoint, _model, _sanitize_and_load

from mlx_lm.generate import (
    GenerationForward,
    GenerationForwardPositionAck,
    GenerationForwardPhase,
    NativeMTPAdmission,
    NativeMTPBatchGenerator,
    NativeMTPEmission,
    NativeMTPRejectedEpoch,
    NativeMTPRowSpec,
    NativeMTPSamplingConfig,
    NativeMTPSparseBootstrap,
    attested_target_forward,
    mtp_generate_step,
)
from mlx_lm.generate import (
    _NativeMTPCohortCache,
    _NativeMTPCohortMutationDelta,
    _NativeMTPSamplingState,
    _native_mtp_position_context,
)
from mlx_lm.models.cache import ArraysCache, KVCache, NativeMTPRequestCache


@dataclass(frozen=True)
class _Capability:
    supported: bool = True
    reason: str = "supported"


class _Model:
    mtp_capability = _Capability()

    def __init__(self):
        self.layers = [type("_Attention", (), {"is_linear": False})()]
        self.mtp = type("_MTP", (), {"layers": [object()]})()


def _request(model):
    request = NativeMTPRequestCache(model, [KVCache()], [KVCache()])
    request.retain(backbone_tokens=0, mtp_tokens=0)
    return request


class _BatchModel(_Model):
    """Actual tiny dense target/MTP pair; each row has a known +1 oracle."""

    vocab_size = 7

    def make_cache(self):
        return [KVCache()]

    def make_mtp_cache(self):
        return [KVCache()]

    def make_mtp_request_cache(self):
        return NativeMTPRequestCache.create(self)

    @staticmethod
    def _write(entry, inputs):
        values = inputs.astype(mx.float32)[:, None, :, None]
        values = mx.broadcast_to(values, (*values.shape[:3], 32))
        entry.update_and_fetch(values, values)

    def _logits(self, inputs, delta=1):
        ids = (inputs.astype(mx.int32) + delta) % self.vocab_size
        logits = mx.full((*inputs.shape, self.vocab_size), -20.0)
        for row in range(inputs.shape[0]):
            for column in range(inputs.shape[1]):
                logits[row, column, ids[row, column]] = 20.0
        return logits

    def __call__(self, inputs, *, cache, return_hidden=False):
        self._write(cache[0], inputs)
        hidden = mx.stack(
            (inputs.astype(mx.float32), inputs.astype(mx.float32)), axis=-1
        )
        logits = self._logits(inputs)
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(self, hidden, next_tokens, cache):
        self._write(cache[0], next_tokens)
        return self._logits(next_tokens)


class _FailingBatchModel(_BatchModel):
    def mtp_forward(self, hidden, next_tokens, cache):
        raise RuntimeError("injected_mtp_failure")


class _PrefillSplitFailureModel(_BatchModel):
    def __call__(self, inputs, *, cache, return_hidden=False):
        raise RuntimeError("injected_prefill_target_failure")


class _MixedRerunFailureModel(_BatchModel):
    def __init__(self):
        super().__init__()
        self.fail_rerun = False

    def __call__(self, inputs, *, cache, return_hidden=False):
        if self.fail_rerun:
            raise RuntimeError("injected_mixed_rerun_failure")
        return super().__call__(inputs, cache=cache, return_hidden=return_hidden)


class _SparseBatchModel(_BatchModel):
    """Tiny B>1 target/MTP double with an acknowledged Qwen position seam."""

    def __init__(self):
        # Exercise the same topology checks as real Qwen rather than relying
        # on inherited class attributes that happen to look cache-compatible.
        self.layers = [type("_Attention", (), {"is_linear": False})()]
        self.mtp = type("_MTP", (), {"layers": [object()]})()
        self.forwards = []
        self.target_calls = 0
        self.mtp_calls = 0
        self._forward = None

    @contextmanager
    def generation_forward_context(self, forward):
        self.forwards.append(forward)
        previous, self._forward = self._forward, forward
        try:
            yield
        finally:
            self._forward = previous

    def _ack(self):
        if self._forward is not None and self._forward.logical_position_ack is not None:
            self._forward.logical_position_ack.acknowledge(
                self._forward.logical_positions
            )

    def __call__(self, inputs, *, cache, return_hidden=False):
        self.target_calls += 1
        self._ack()
        return super().__call__(inputs, cache=cache, return_hidden=return_hidden)

    def mtp_forward(self, hidden, next_tokens, cache):
        self.mtp_calls += 1
        self._ack()
        return super().mtp_forward(hidden, next_tokens, cache)


class _SparseMoEBatchModel(_SparseBatchModel):
    """Token-routed tiny MoE oracle; cache behavior remains identical."""

    architecture = "qwen3_5_moe"
    num_experts = 2

    def _logits(self, inputs, delta=1):
        expert = inputs.astype(mx.int32) % self.num_experts
        zero = inputs.astype(mx.int32) + delta
        one = inputs.astype(mx.int32) - (self.vocab_size - delta)
        ids = mx.where(expert == 0, zero, one) % self.vocab_size
        logits = mx.full((*inputs.shape, self.vocab_size), -20.0)
        for row in range(inputs.shape[0]):
            for column in range(inputs.shape[1]):
                logits[row, column, ids[row, column]] = 20.0
        return logits


def _sparse_bootstrap(model, tokens, positions, *, chunks=(1, 1)):
    cache = model.make_cache()
    successors = tuple((token + 1) % model.vocab_size for token in tokens[:-1])
    receipts = []
    start = 0
    for size in chunks:
        end = start + size
        (_, _), receipt = attested_target_forward(
            model,
            tuple(tokens[start:end]),
            cache,
            phase=GenerationForwardPhase.PREFILL,
            logical_positions=tuple(positions[start:end]),
            immediate_successor_token_ids=tuple(
                successors[start : min(end, len(successors))]
            ),
            model_forward_context=model.generation_forward_context,
        )
        receipts.append(receipt)
        start = end
    assert start == len(tokens)
    return NativeMTPSparseBootstrap(
        receipts=tuple(receipts),
        selected_logical_positions=tuple(positions),
        selected_token_ids=tuple(tokens),
        immediate_successor_token_ids=successors,
        target_cache=cache,
        next_logical_position=positions[-1] + 1,
    )


def _admission(*, max_tokens=8):
    model = _Model()
    rows = (
        NativeMTPRowSpec(7, (1,), max_tokens, seed=17),
        NativeMTPRowSpec(3, (2,), max_tokens, seed=29),
    )
    return NativeMTPAdmission.create(model, rows, (_request(model), _request(model)))


def _emission(uid, token, *, draft=False, finish_reason=None):
    return NativeMTPEmission(
        uid=uid,
        token=token,
        logprobs=mx.zeros((5,), dtype=mx.float32),
        from_draft=draft,
        finish_reason=finish_reason,
    )


@pytest.mark.parametrize(
    "kwargs, reason",
    (
        ({"prefix_cache": object()}, "prefix_reuse"),
        ({"media": object()}, "media"),
        ({"external_draft": object()}, "external_draft"),
        ({"sparse_bootstrap": object()}, "sparse_bootstrap"),
        ({"logits_processors": [object()]}, "logits_processors"),
        ({"kv_bits": 8}, "quantized_cache"),
        ({"max_kv_size": 16}, "rotating_cache"),
    ),
)
def test_admission_fails_closed_for_unqualified_composition(kwargs, reason):
    model = _Model()
    with pytest.raises(ValueError, match=reason):
        NativeMTPAdmission.create(
            model,
            (NativeMTPRowSpec(1, (1,), 2),),
            (_request(model),),
            **kwargs,
        )


def test_arrays_delta_matrix_matches_real_advance_for_optional_vectors():
    entry = ArraysCache(1, left_padding=[2, 0])
    entry.prepare(lengths=[5, 3])
    before = _NativeMTPCohortCache._host_metadata((entry,))
    entry.advance(1)
    after = _NativeMTPCohortCache._host_metadata((entry,))
    _NativeMTPCohortCache._validate_delta("backbone", before, after, ((1, 1),))

    # Both ArraysCache vectors are part of the live recurrent-cache contract.
    malformed = ((None, after[0][1], after[0][2][:-1]),)
    with pytest.raises(RuntimeError, match="arrays_metadata_changed"):
        _NativeMTPCohortCache._validate_delta("backbone", before, malformed, ((1, 1),))

    # Qwen's recurrent cache commonly provides left_padding without lengths.
    left_padding_only = ArraysCache(1, left_padding=[2, 0])
    before = _NativeMTPCohortCache._host_metadata((left_padding_only,))
    left_padding_only.advance(1)
    after = _NativeMTPCohortCache._host_metadata((left_padding_only,))
    _NativeMTPCohortCache._validate_delta("backbone", before, after, ((1, 1),))

    lengths_only = ArraysCache(1)
    lengths_only.prepare(lengths=[5, 3])
    before = _NativeMTPCohortCache._host_metadata((lengths_only,))
    lengths_only.advance(1)
    after = _NativeMTPCohortCache._host_metadata((lengths_only,))
    _NativeMTPCohortCache._validate_delta("backbone", before, after, ((1, 1),))

    unpadded = ArraysCache(1)
    before = _NativeMTPCohortCache._host_metadata((unpadded,))
    unpadded.advance(1)
    after = _NativeMTPCohortCache._host_metadata((unpadded,))
    _NativeMTPCohortCache._validate_delta("backbone", before, after, ((1,),))


def test_arrays_delta_allows_real_unpadded_advance_noop_seal():
    model = _Model()
    model.layers = [type("_Linear", (), {"is_linear": True})()]
    request = NativeMTPRequestCache(model, [ArraysCache(1)], [KVCache()])
    request.retain(backbone_tokens=0, mtp_tokens=0)
    cohort = _NativeMTPCohortCache(model, (request,), uids=(7,))
    for entry in cohort.backbone:
        entry.left_padding = None
        entry.lengths = None
    cohort._binding = cohort._make_binding()
    cohort.bind_before_mutation()
    for entry in cohort.backbone:
        entry.advance(3)
    cohort.seal_after_mutation(
        _NativeMTPCohortMutationDelta(backbone=((3,),), mtp=((0,),))
    )
    assert not cohort.poisoned


def _arrays_cohort_for_seal(*, left_padding=None, lengths=None):
    model = _Model()
    model.layers = [type("_Linear", (), {"is_linear": True})()]
    request = NativeMTPRequestCache(model, [ArraysCache(1)], [KVCache()])
    request.retain(backbone_tokens=0, mtp_tokens=0)
    cohort = _NativeMTPCohortCache(model, (request,), uids=(7,))
    entry = cohort.backbone[0]
    entry.left_padding = (
        None if left_padding is None else mx.array(left_padding, dtype=mx.int32)
    )
    entry.lengths = None if lengths is None else mx.array(lengths, dtype=mx.int32)
    cohort._binding = cohort._make_binding()
    return cohort


def _assert_arrays_seal_rejected_and_poisoned(cohort):
    with pytest.raises(RuntimeError, match="native_mtp_cohort"):
        cohort.seal_after_mutation(
            _NativeMTPCohortMutationDelta(backbone=((1,),), mtp=((0,),))
        )
    assert cohort.poisoned


def test_arrays_seal_rejects_and_poisons_absent_to_present_metadata():
    cohort = _arrays_cohort_for_seal()
    cohort.bind_before_mutation()
    entry = cohort.backbone[0]
    entry.left_padding = mx.array([0], dtype=mx.int32)
    entry.lengths = mx.array([1], dtype=mx.int32)
    _assert_arrays_seal_rejected_and_poisoned(cohort)


def test_arrays_seal_rejects_and_poisons_present_to_absent_metadata():
    cohort = _arrays_cohort_for_seal(left_padding=[2], lengths=[4])
    cohort.bind_before_mutation()
    entry = cohort.backbone[0]
    entry.left_padding = None
    entry.lengths = None
    _assert_arrays_seal_rejected_and_poisoned(cohort)


def test_arrays_seal_rejects_and_poisons_wrong_left_padding_delta():
    cohort = _arrays_cohort_for_seal(left_padding=[2], lengths=[4])
    cohort.bind_before_mutation()
    entry = cohort.backbone[0]
    entry.advance(1)
    entry.left_padding = mx.array([0], dtype=mx.int32)
    _assert_arrays_seal_rejected_and_poisoned(cohort)


def test_arrays_seal_rejects_and_poisons_wrong_lengths_delta():
    cohort = _arrays_cohort_for_seal(left_padding=[2], lengths=[4])
    cohort.bind_before_mutation()
    entry = cohort.backbone[0]
    entry.advance(1)
    entry.lengths = mx.array([2], dtype=mx.int32)
    _assert_arrays_seal_rejected_and_poisoned(cohort)


def test_public_decision_methods_reject_positional_and_keyword_forgery():
    model = _BatchModel()
    rows = (NativeMTPRowSpec(7, (1, 2), 8), NativeMTPRowSpec(3, (4,), 8))
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(model, rows, (_request(model), _request(model)))
    )
    assert not hasattr(generator, "initial")
    _, initial = generator.prefill(prefill_step_size=1)
    with pytest.raises(TypeError):
        initial.resume((7,))
    with pytest.raises(TypeError):
        initial.resume(draft_uids=(7,))
    ready = initial.resume()
    with pytest.raises(TypeError):
        ready.decide((7,))
    with pytest.raises(TypeError):
        ready.decide(accepted_uids=(7,))
    decision = ready.decide()
    with pytest.raises(TypeError):
        decision.accept((7,))
    with pytest.raises(TypeError):
        decision.accept(emissions=())
    with pytest.raises(TypeError):
        decision.reject((3,))
    with pytest.raises(TypeError):
        decision.reject(emissions=())
    _, accepted = decision.accept()
    with pytest.raises(TypeError):
        accepted.bonus(())
    with pytest.raises(TypeError):
        accepted.bonus(emissions=())
    bonus = accepted.bonus()
    with pytest.raises(TypeError):
        bonus.catch_up(())
    with pytest.raises(TypeError):
        bonus.catch_up(draft_uids=())
    rejected = NativeMTPRejectedEpoch(generator, "rejected", ())
    with pytest.raises(TypeError):
        rejected.redraft(())
    with pytest.raises(TypeError):
        rejected.redraft(draft_uids=())


def test_mixed_terminal_rows_are_removed_before_bonus_catchup_and_join():
    model = _BatchModel()
    rows = tuple(NativeMTPRowSpec(uid, (uid % 7,), 8) for uid in (7, 3, 11, 13))
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(model, rows, tuple(_request(model) for _ in rows))
    )
    rejected_owner = generator._move_to_front((7, 3))
    generator._mixed_accepted_owner = generator._cohort
    generator._mixed_rejected_owner = rejected_owner
    generator._prune_mixed_terminal_owners(
        (
            _emission(7, 1, finish_reason="eos"),
            _emission(3, 2),
            _emission(11, 3, finish_reason="length"),
            _emission(13, 4),
        ),
        (7, 3),
        (11, 13),
    )
    assert generator._mixed_accepted_uids == (3,)
    assert generator._mixed_rejected_uids == (13,)
    assert generator._mixed_accepted_owner.uids == (3,)
    assert generator._mixed_rejected_owner.uids == (13,)

    generator._mixed_accepted_ready = ()
    generator._mixed_rejected_ready = (13,)
    generator._mixed_accepted_uids = ()
    generator._mixed_accepted_owner = generator._filter_owner_uids(
        generator._mixed_accepted_owner, ()
    )
    assert generator._mixed_accepted_owner is None
    assert generator._mixed_rejected_owner.uids == (13,)
    ready = generator._mixed_after_bonus(None)
    assert ready.active_uids == (13,)
    assert generator._cohort.uids == ready.active_uids
    assert generator._cohort.backbone[0].offset.size == 1


def test_injected_mtp_failure_closes_the_consumed_cohort_owner():
    model = _FailingBatchModel()
    rows = (NativeMTPRowSpec(7, (1, 2), 4),)
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(model, rows, (_request(model),))
    )
    with pytest.raises(RuntimeError, match="injected_mtp_failure"):
        generator.prefill(prefill_step_size=1)
    assert generator.closed
    assert generator._cohort.poisoned


def test_prefill_target_failure_after_split_poison_selected_and_remaining_owners():
    model = _PrefillSplitFailureModel()
    rows = (NativeMTPRowSpec(7, (1, 2, 3), 4), NativeMTPRowSpec(3, (4,), 4))
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(model, rows, (_request(model), _request(model)))
    )
    with pytest.raises(RuntimeError, match="injected_prefill_target_failure"):
        generator.prefill(prefill_step_size=1)
    assert generator.closed
    assert len(generator._last_failed_owners) == 2
    assert all(owner.poisoned for owner in generator._last_failed_owners)


def test_mixed_rerun_failure_preserves_primary_and_poisoned_branch_owners():
    model = _MixedRerunFailureModel()
    rows = (NativeMTPRowSpec(7, (1, 2), 8), NativeMTPRowSpec(3, (4,), 8))
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(model, rows, (_request(model), _request(model)))
    )
    _, initial = generator.prefill(prefill_step_size=1)
    decision = initial.resume().decide()
    # The public decision is model-derived; this test only controls the
    # private branch fixture so the accepted rerun becomes injectable.
    decision.accepted_uids = (7,)
    decision.rejected_uids = (3,)
    model.fail_rerun = True
    with pytest.raises(RuntimeError, match="injected_mixed_rerun_failure"):
        decision.resolve()
    assert generator.closed
    assert len(generator._last_failed_owners) == 2
    assert all(owner.poisoned for owner in generator._last_failed_owners)


def test_actual_merged_chunked_prefill_and_all_accepted_round():
    model = _BatchModel()
    rows = (
        NativeMTPRowSpec(7, (1, 2, 3), 8, seed=17),
        NativeMTPRowSpec(3, (4,), 8, seed=29),
    )
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(model, rows, (_request(model), _request(model)))
    )
    emissions, initial = generator.prefill(prefill_step_size=2)
    assert tuple(emission.token for emission in emissions) == (4, 5)
    ready = initial.resume()
    decision = ready.decide()
    assert decision.accepted_uids == (7, 3)
    _, accepted = decision.accept()
    bonus = accepted.bonus()
    ready = bonus.catch_up()
    assert ready.phase == "ready"


def test_public_b1_all_accept_returns_one_resolution_emission_and_epoch():
    model = _BatchModel()
    row = NativeMTPRowSpec(7, (1, 2), 8)
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(model, (row,), (_request(model),))
    )

    initial_emissions, initial = generator.prefill(prefill_step_size=1)
    decision = initial.resume().decide()
    emissions, accepted = decision.accept()

    assert len(initial_emissions) == 1
    assert decision.accepted_uids == (7,)
    assert decision.rejected_uids == ()
    assert len(emissions) == 1
    assert emissions[0].uid == 7
    assert emissions[0].token == generator._draft[7].item()
    assert emissions[0].from_draft is True
    assert emissions[0].finish_reason is None
    assert accepted.phase == "accepted"
    assert accepted.active_uids == (7,)


def test_public_b1_all_reject_returns_one_resolution_emission_and_epoch(monkeypatch):
    sampling = NativeMTPSamplingConfig(temperature=0.7, seed=17)
    model = _BatchModel()
    row = NativeMTPRowSpec(7, (1, 2), 8, seed=17, sampling_config=sampling)
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(model, (row,), (_request(model),))
    )

    with monkeypatch.context() as local_monkeypatch:
        _force_acceptance_uniform(local_monkeypatch, (2.0,))
        _, initial = generator.prefill(prefill_step_size=1)
        decision = initial.resume().decide()
        emissions, rejected = decision.reject()

    assert decision.accepted_uids == ()
    assert decision.rejected_uids == (7,)
    assert len(emissions) == 1
    assert emissions[0].uid == 7
    assert emissions[0].token == generator._replacement[7].item()
    assert emissions[0].from_draft is False
    assert emissions[0].finish_reason is None
    assert rejected.phase == "rejected"
    assert rejected.active_uids == (7,)


@pytest.mark.parametrize("accepted", (True, False))
def test_public_b1_terminal_resolution_emission_is_returned(accepted, monkeypatch):
    sampling = NativeMTPSamplingConfig(temperature=0.7, seed=17)
    model = _BatchModel()
    row = NativeMTPRowSpec(7, (1, 2), 2, seed=17, sampling_config=sampling)
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(model, (row,), (_request(model),))
    )

    with monkeypatch.context() as local_monkeypatch:
        _force_acceptance_uniform(local_monkeypatch, ((-1.0,) if accepted else (2.0,)))
        _, initial = generator.prefill(prefill_step_size=1)
        decision = initial.resume().decide()
        emissions, epoch = decision.accept() if accepted else decision.reject()

    assert len(emissions) == 1
    assert emissions[0].uid == 7
    assert emissions[0].from_draft is accepted
    assert emissions[0].finish_reason == "length"
    assert epoch.active_uids == ()


def test_bonus_uses_verified_target_logits_not_hidden_width():
    model = _BatchModel()
    rows = (
        NativeMTPRowSpec(7, (1, 2, 3), 8, seed=17),
        NativeMTPRowSpec(3, (4,), 8, seed=29),
    )
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(model, rows, (_request(model), _request(model)))
    )
    _, initial = generator.prefill(prefill_step_size=2)
    decision = initial.resume().decide()
    _, accepted = decision.accept()
    bonus = accepted.bonus()
    # The tiny model's hidden width is 2 while its vocabulary is 7.  These
    # values can only be correct when bonus sampling reads verify logits[:, 1].
    assert bonus.active_uids == (7, 3)
    assert tuple(generator._head[uid].item() for uid in (7, 3)) == (6, 0)


def test_actual_dense_batch_initial_tokens_match_independent_b1_oracles():
    prompts = ((1, 2, 3), (4,))
    oracle = []
    for prompt in prompts:
        stream = mtp_generate_step(
            mx.array(prompt, dtype=mx.uint32), _BatchModel(), max_tokens=2
        )
        oracle.append(next(stream)[0])
        stream.close()

    model = _BatchModel()
    rows = tuple(
        NativeMTPRowSpec(uid, prompt, 2, seed=uid)
        for uid, prompt in zip((7, 3), prompts)
    )
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(model, rows, (_request(model), _request(model)))
    )
    emissions, _ = generator.prefill(prefill_step_size=2)
    assert tuple(emission.token for emission in emissions) == tuple(oracle)


def test_uid_rng_uses_post_initial_b1_sequential_draw_chain():
    model = _BatchModel()
    row = NativeMTPRowSpec(
        7,
        (1, 2, 3),
        8,
        seed=17,
        sampling_config=NativeMTPSamplingConfig(temperature=0.7, seed=17),
    )
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(model, (row,), (_request(model),))
    )
    _, initial = generator.prefill(prefill_step_size=2)
    decision = initial.resume().decide()
    _, accepted = decision.accept()
    accepted.bonus()

    # B=1 draws initial, draft, acceptance, then accepted bonus.  The retained
    # key remains UID-owned, so cohort reordering cannot affect this chain.
    expected = mx.random.key(17)
    for _ in range(4):
        expected, _ = mx.random.split(expected)
    mx.eval(expected, generator._rng_key[7])
    assert mx.array_equal(expected, generator._rng_key[7]).item()


@pytest.mark.parametrize("moe", (False, True))
def test_actual_qwen_dense_and_moe_variable_prefill_admission_fixture(moe):
    """Exercise real Qwen layers: MoE includes its recurrent+KV backbone."""
    model = _model(moe=moe)
    _sanitize_and_load(model, _checkpoint(model, moe=moe))
    target = model.language_model
    rows = (
        NativeMTPRowSpec(7, (1, 2, 3), 4, seed=17),
        NativeMTPRowSpec(3, (4,), 4, seed=29),
    )
    requests = tuple(model.make_mtp_request_cache() for _ in rows)
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(target, rows, requests)
    )
    emissions, initial = generator.prefill(prefill_step_size=1)
    assert len(emissions) == 2
    assert initial.active_uids == (7, 3)
    assert generator._cohort.backbone[-1].offset.tolist() == [3, 1]


def _force_acceptance_uniform(monkeypatch, values):
    values = deque(values)

    def forced_uniform(*args, **kwargs):
        return mx.array(values.popleft(), dtype=mx.float32)

    monkeypatch.setattr(mx.random, "uniform", forced_uniform)


def _shared_qwen_checkpoint(*, moe):
    """Freeze one synthetic backbone for every oracle/batch comparison."""

    reference = _model(moe=moe)
    return _checkpoint(reference, moe=moe)


def _load_shared_qwen_checkpoint(model, weights):
    """Give sanitize a fresh mapping while preserving the frozen MLX leaves."""

    _sanitize_and_load(model, dict(weights))


def _tiny_qwen_b1_emissions(
    monkeypatch, *, moe, weights, prompt, sampling, uniform, count
):
    model = _model(moe=moe)
    _load_shared_qwen_checkpoint(model, weights)
    with monkeypatch.context() as local_monkeypatch:
        requests = []
        make_request = model.language_model.make_mtp_request_cache

        def capture_request(*args, **kwargs):
            request = make_request(*args, **kwargs)
            requests.append(request)
            return request

        local_monkeypatch.setattr(
            model.language_model, "make_mtp_request_cache", capture_request
        )
        _force_acceptance_uniform(local_monkeypatch, uniform)
        stream = mtp_generate_step(
            mx.array(prompt, dtype=mx.uint32),
            model.language_model,
            max_tokens=8,
            prefill_step_size=1,
            sampling_config=sampling,
        )
        try:
            emissions = tuple(next(stream) for _ in range(count))
        finally:
            stream.close()
    return emissions, requests[0]


def _actual_sparse_bootstrap(target, token_ids, logical_positions, *, chunk_sizes):
    """Produce real Qwen sparse evidence without reconstructing hidden rows."""

    target_cache = target.make_cache()
    successors = tuple(token_ids[1:])
    receipts = []
    start = 0
    for size in chunk_sizes:
        end = start + size
        (_, _), receipt = attested_target_forward(
            target,
            tuple(token_ids[start:end]),
            target_cache,
            phase=GenerationForwardPhase.PREFILL,
            logical_positions=tuple(logical_positions[start:end]),
            immediate_successor_token_ids=tuple(
                successors[start : min(end, len(successors))]
            ),
            model_forward_context=target.generation_forward_context,
        )
        receipts.append(receipt)
        start = end
    assert start == len(token_ids)
    return NativeMTPSparseBootstrap(
        receipts=tuple(receipts),
        selected_logical_positions=tuple(logical_positions),
        selected_token_ids=tuple(token_ids),
        immediate_successor_token_ids=successors,
        target_cache=target_cache,
        next_logical_position=logical_positions[-1] + 1,
    )


def _tiny_qwen_b1_sparse_emissions(
    monkeypatch, *, moe, weights, tokens, positions, sampling, uniform, count
):
    """Pause a real sparse B1 stream at the same lifecycle boundary as batch."""

    model = _model(moe=moe)
    _load_shared_qwen_checkpoint(model, weights)
    bootstrap = _actual_sparse_bootstrap(
        model, tokens, positions, chunk_sizes=(1,) * len(tokens)
    )
    requests = []
    original_adopt = NativeMTPRequestCache.adopt_sparse_target.__func__

    def capture_adopt(cls, *args, **kwargs):
        request = original_adopt(cls, *args, **kwargs)
        requests.append(request)
        return request

    with monkeypatch.context() as local_monkeypatch:
        local_monkeypatch.setattr(
            NativeMTPRequestCache, "adopt_sparse_target", classmethod(capture_adopt)
        )
        _force_acceptance_uniform(local_monkeypatch, uniform)
        stream = mtp_generate_step(
            None,
            model,
            max_tokens=8,
            prefill_step_size=1,
            sampling_config=sampling,
            sparse_bootstrap=bootstrap,
        )
        try:
            emissions = tuple(next(stream) for _ in range(count))
        finally:
            stream.close()
    return emissions, requests[0]


def _assert_cohort_recurrent_row_matches_b1(owner, uid, request):
    row = owner.uids.index(uid)
    for cohort_entry, b1_entry in zip(owner.backbone, request.backbone):
        if not isinstance(cohort_entry, ArraysCache):
            continue
        assert isinstance(b1_entry, ArraysCache)
        for cohort_value, b1_value in zip(cohort_entry.cache, b1_entry.cache):
            if b1_value is None:
                assert cohort_value is None
            else:
                assert mx.allclose(
                    cohort_value[row : row + 1],
                    b1_value,
                    rtol=1e-4,
                    atol=1e-5,
                ).item()
        for name in ("left_padding", "lengths"):
            cohort_value = getattr(cohort_entry, name)
            b1_value = getattr(b1_entry, name)
            if b1_value is None:
                continue
            else:
                assert mx.array_equal(cohort_value[row : row + 1], b1_value).item()


def _assert_cohort_offsets_match_b1(owner, uid, request):
    row = owner.uids.index(uid)
    for cohort_entry, b1_entry in zip(
        owner.backbone + owner.mtp, request.backbone + request.mtp
    ):
        if isinstance(cohort_entry, ArraysCache):
            continue
        assert cohort_entry.offset[row].item() == b1_entry.offset


def _assert_cohort_cache_payload_row_matches_b1(owner, uid, request):
    """Compare the live KV and recurrent payloads, not merely their offsets."""

    row = owner.uids.index(uid)
    for cohort_entry, b1_entry in zip(
        owner.backbone + owner.mtp, request.backbone + request.mtp
    ):
        if isinstance(cohort_entry, ArraysCache):
            continue
        for name in ("keys", "values"):
            cohort_value = getattr(cohort_entry, name)
            b1_value = getattr(b1_entry, name)
            if b1_value is None:
                assert cohort_value is None
            else:
                offset = b1_entry.offset
                padding = cohort_entry.left_padding[row].item()
                assert mx.allclose(
                    cohort_value[row : row + 1, ..., padding : padding + offset, :],
                    b1_value[..., :offset, :],
                    rtol=1e-4,
                    atol=1e-5,
                ).item()


class _B1SparsePhaseOracle:
    """Manual sparse B1 transaction stopped exactly at the Ready boundary."""

    def __init__(self, model, tokens, positions, sampling, *, accepted):
        bootstrap = _actual_sparse_bootstrap(
            model, tokens, positions, chunk_sizes=(1,) * len(tokens)
        )
        claim = bootstrap.claim(model)
        self.request = NativeMTPRequestCache.adopt_sparse_target(
            model,
            target_cache=claim.target_cache,
            target_tokens=len(claim.selected_token_ids),
            next_logical_position=claim.next_logical_position,
        )
        self.context = _native_mtp_position_context(model, None)
        NativeMTPAdmission._initialize_sparse_mtp_request(
            model, self.request, claim, self.context
        )
        self.model = model
        self.sampling = _NativeMTPSamplingState(sampling, seed=sampling.seed)
        self.history = mx.array([tokens[-1]], dtype=mx.uint32)
        self.cursor = claim.next_logical_position
        self._initial(claim)
        self._draft_initial(positions[-1])
        self._verify(expected_accepted=accepted)
        if self.accepted:
            self.emissions = self._accept_bonus_catchup()
        else:
            self.emissions = (self._reject_replay_redraft(),)

    def _target(self, tokens, positions, phase):
        inputs = mx.array(tokens, dtype=mx.uint32)[None]
        ack = GenerationForwardPositionAck(
            tuple(positions), model=self.model, cache=self.request.backbone, phase=phase
        )
        forward = GenerationForward(
            model=self.model,
            input_tokens=inputs,
            cache=self.request.backbone,
            phase=phase,
            logical_positions=tuple(positions),
            logical_position_ack=ack,
        )
        with self.context(forward):
            ack._activate()
            try:
                result = self.model(
                    inputs, cache=self.request.backbone, return_hidden=True
                )
                ack._require_acknowledged()
                return result
            finally:
                ack._finish()

    def _mtp(self, hidden, tokens, positions):
        inputs = mx.array(tokens, dtype=mx.uint32)[None]
        ack = GenerationForwardPositionAck(
            tuple(positions),
            model=self.model,
            cache=self.request.mtp,
            phase=GenerationForwardPhase.MTP_DRAFT,
        )
        forward = GenerationForward(
            model=self.model,
            input_tokens=inputs,
            cache=self.request.mtp,
            phase=GenerationForwardPhase.MTP_DRAFT,
            logical_positions=tuple(positions),
            logical_position_ack=ack,
        )
        with self.context(forward):
            ack._activate()
            try:
                result = self.model.mtp_forward(hidden, inputs, self.request.mtp)
                ack._require_acknowledged()
                return result
            finally:
                ack._finish()

    def _sample(self, reported):
        return self.sampling.sample(
            self.sampling.sampling_distribution(reported),
            rng_key=(
                self.sampling.next_rng_key()
                if self.sampling.config.stochastic
                else None
            ),
        ).reshape(-1)

    def _initial(self, claim):
        reported = self.sampling.reported_logprobs(
            claim.final_target_logits[0, -1, :], self.history
        )
        self.head = self._sample(reported)
        self.hidden = claim.final_target_hidden
        mx.eval(self.head, reported)
        self.history = mx.concatenate([self.history, self.head])
        self.cursor += 1
        self.initial = (self.head.item(), reported, False)

    def _draft_initial(self, initial_mtp_position):
        logits = self._mtp(self.hidden, self.head, (initial_mtp_position,))[:, -1, :]
        self.request.retain(
            backbone_tokens=self.request.state.backbone_tokens,
            mtp_tokens=self.request.state.mtp_tokens + 1,
        )
        self.draft_logprobs = self.sampling.sampling_distribution(
            self.sampling.reported_logprobs(logits.squeeze(0), self.history)
        )
        self.draft = self.sampling.sample(
            self.draft_logprobs,
            rng_key=(
                self.sampling.next_rng_key()
                if self.sampling.config.stochastic
                else None
            ),
        ).reshape(-1)
        mx.eval(self.draft)

    def _verify(self, *, expected_accepted):
        self.request.checkpoint()
        positions = (self.cursor - 1, self.cursor)
        logits, hidden = self._target(
            mx.concatenate([self.head, self.draft]),
            positions,
            GenerationForwardPhase.VERIFY,
        )
        self.request.seal_verified(
            backbone_tokens=self.request.state.backbone_tokens + 2,
            mtp_tokens=self.request.state.mtp_tokens,
        )
        self.verify_positions = positions
        self.verify_logits, self.verify_hidden = logits, hidden
        self.verify_reported = self.sampling.reported_logprobs(
            logits[:, 0, :].squeeze(0), self.history
        )
        sampled = self.sampling.sampling_distribution(self.verify_reported)
        probability = mx.minimum(
            mx.exp(sampled[self.draft.item()] - self.draft_logprobs[self.draft.item()]),
            1.0,
        )
        accepted = mx.random.uniform(key=self.sampling.next_rng_key()) < probability
        mx.eval(accepted)
        self.accepted = bool(accepted.item())
        assert self.accepted is expected_accepted

    def _accept_bonus_catchup(self):
        self.request.commit(
            backbone_tokens=self.request.state.backbone_tokens + 2,
            mtp_tokens=self.request.state.mtp_tokens,
        )
        accepted = (self.draft.item(), self.verify_reported, True)
        self.cursor += 1
        history = mx.concatenate([self.history, self.draft])
        bonus_reported = self.sampling.reported_logprobs(
            self.verify_logits[:, 1, :].squeeze(0), history
        )
        self.head = self._sample(bonus_reported)
        mx.eval(self.head, bonus_reported)
        self.history = mx.concatenate([history, self.head])
        self.cursor += 1
        logits = self._mtp(
            self.verify_hidden,
            mx.concatenate([self.draft, self.head]),
            self.verify_positions,
        )[:, -1, :]
        self.request.retain(
            backbone_tokens=self.request.state.backbone_tokens,
            mtp_tokens=self.request.state.mtp_tokens + 2,
        )
        reported = self.sampling.reported_logprobs(logits.squeeze(0), self.history)
        self.draft_logprobs = self.sampling.sampling_distribution(reported)
        self.draft = self.sampling.sample(
            self.draft_logprobs,
            rng_key=(
                self.sampling.next_rng_key()
                if self.sampling.config.stochastic
                else None
            ),
        ).reshape(-1)
        mx.eval(self.draft)
        return accepted, (self.head.item(), bonus_reported, False)

    def _reject_replay_redraft(self):
        self.request.reject_partial(accepted_backbone_tokens=1, accepted_mtp_tokens=0)
        replay_position = self.cursor - 1
        _, hidden = self._target(
            self.head, (replay_position,), GenerationForwardPhase.DECODE
        )
        self.request.replay_retained(
            backbone_tokens=self.request.state.backbone_tokens + 1,
            mtp_tokens=self.request.state.mtp_tokens,
        )
        sampled = self.sampling.sampling_distribution(self.verify_reported)
        self.head, _ = self.sampling.residual_sample(
            sampled, self.draft_logprobs, rng_key=self.sampling.next_rng_key()
        )
        self.head = self.head.reshape(-1)
        mx.eval(self.head)
        self.history = mx.concatenate([self.history, self.head])
        self.cursor += 1
        logits = self._mtp(hidden[:, -1:, :], self.head, (replay_position,))[:, -1, :]
        self.request.retain(
            backbone_tokens=self.request.state.backbone_tokens,
            mtp_tokens=self.request.state.mtp_tokens + 1,
        )
        reported = self.sampling.reported_logprobs(logits.squeeze(0), self.history)
        self.draft_logprobs = self.sampling.sampling_distribution(reported)
        self.draft = self.sampling.sample(
            self.draft_logprobs,
            rng_key=(
                self.sampling.next_rng_key()
                if self.sampling.config.stochastic
                else None
            ),
        ).reshape(-1)
        mx.eval(self.draft)
        return self.head.item(), self.verify_reported, False


class _B1PhaseOracle:
    """Test-local pause-point oracle for B1 native-MTP cache transitions."""

    def __init__(self, model, prompt, sampling):
        self.model = model
        self.request = model.make_mtp_request_cache()
        self.sampling = _NativeMTPSamplingState(sampling, seed=sampling.seed)
        self.history = mx.array(prompt, dtype=mx.uint32)
        self._prefill()

    def _target(self, tokens):
        return self.model(
            mx.array(tokens, dtype=mx.uint32)[None],
            cache=self.request.backbone,
            return_hidden=True,
        )

    def _mtp(self, hidden, tokens):
        return self.model.mtp_forward(
            hidden, mx.array(tokens, dtype=mx.uint32)[None], self.request.mtp
        )

    def _retain(self, *, backbone=0, mtp=0):
        self.request.retain(
            backbone_tokens=self.request.state.backbone_tokens + backbone,
            mtp_tokens=self.request.state.mtp_tokens + mtp,
        )

    def _prefill(self):
        remaining = self.history
        while remaining.size > 1:
            count = remaining.size - 1
            _, hidden = self._target(remaining[:count])
            self._mtp(hidden, remaining[1 : count + 1])
            self._retain(backbone=count, mtp=count)
            remaining = remaining[count:]
        logits, hidden = self._target(remaining)
        self._retain(backbone=1)
        reported = self.sampling.reported_logprobs(
            logits[:, -1, :].squeeze(0), remaining
        )
        self.head = self.sampling.sample(
            self.sampling.sampling_distribution(reported)
        ).reshape(-1)
        mx.eval(self.head, reported)
        self.history = mx.concatenate([remaining, self.head])
        self.hidden = hidden[:, -1:, :]
        self.initial = (self.head.item(), reported, False)

    def draft(self):
        logits = self._mtp(self.hidden, self.head)[:, -1, :]
        self._retain(mtp=1)
        reported = self.sampling.reported_logprobs(logits.squeeze(0), self.history)
        self.draft_logprobs = self.sampling.sampling_distribution(reported)
        self.draft = self.sampling.sample(self.draft_logprobs).reshape(-1)
        mx.eval(self.draft, reported)

    def verify(self):
        self.request.checkpoint()
        logits, hidden = self._target(mx.concatenate([self.head, self.draft]))
        self.request.seal_verified(
            backbone_tokens=self.request.state.backbone_tokens + 2,
            mtp_tokens=self.request.state.mtp_tokens,
        )
        reported = self.sampling.reported_logprobs(
            logits[:, 0, :].squeeze(0), self.history
        )
        sampled = self.sampling.sampling_distribution(reported)
        probability = mx.minimum(
            mx.exp(sampled[self.draft.item()] - self.draft_logprobs[self.draft.item()]),
            1.0,
        )
        accepted = mx.random.uniform(key=self.sampling.next_rng_key()) < probability
        mx.eval(accepted)
        self.accepted = bool(accepted.item())
        self.verify_hidden, self.verify_logits, self.verify_reported = (
            hidden,
            logits,
            reported,
        )
        if not self.accepted:
            self.replacement, _ = self.sampling.residual_sample(
                sampled, self.draft_logprobs
            )
            self.replacement = self.replacement.reshape(-1)
            mx.eval(self.replacement)

    def accept_bonus_catchup(self):
        self.request.commit(
            backbone_tokens=self.request.state.backbone_tokens + 2,
            mtp_tokens=self.request.state.mtp_tokens,
        )
        accepted = (self.draft.item(), self.verify_reported, True)
        bonus_history = mx.concatenate([self.history, self.draft])
        bonus_reported = self.sampling.reported_logprobs(
            self.verify_logits[:, 1, :].squeeze(0), bonus_history
        )
        bonus = self.sampling.sample(
            self.sampling.sampling_distribution(bonus_reported)
        ).reshape(-1)
        mx.eval(bonus, bonus_reported)
        bonus_emission = (bonus.item(), bonus_reported, False)
        self.history = mx.concatenate([bonus_history, bonus])
        self.head = bonus
        self.hidden = self.verify_hidden[:, 1:2, :]
        self.draft = self.sampling.sample(
            self.sampling.sampling_distribution(
                self._mtp(
                    self.verify_hidden,
                    mx.concatenate([self.draft, bonus]),
                )[:, -1, :]
            )
        ).reshape(-1)
        self._retain(mtp=2)
        mx.eval(self.draft)
        return accepted, bonus_emission

    def reject_replay_redraft(self):
        self.request.reject_partial(accepted_backbone_tokens=1, accepted_mtp_tokens=0)
        _, replay_hidden = self._target(self.head)
        self.request.replay_retained(
            backbone_tokens=self.request.state.backbone_tokens + 1,
            mtp_tokens=self.request.state.mtp_tokens,
        )
        emission = (self.replacement.item(), self.verify_reported, False)
        self.history = mx.concatenate([self.history, self.replacement])
        self.head = self.replacement
        self.hidden = replay_hidden[:, -1:, :]
        logits = self._mtp(self.hidden, self.head)[:, -1, :]
        self._retain(mtp=1)
        self.draft = self.sampling.sample(
            self.sampling.sampling_distribution(
                self.sampling.reported_logprobs(logits.squeeze(0), self.history)
            )
        ).reshape(-1)
        mx.eval(self.draft)
        return emission


@pytest.mark.parametrize("moe", (False, True))
def test_actual_qwen_stochastic_mixed_public_lifecycle_matches_b1_emissions(
    moe, monkeypatch
):
    """Dense and recurrent+KV MoE rows resolve a forced mixed public epoch."""

    accepted_sampling = NativeMTPSamplingConfig(temperature=0.7, seed=17)
    rejected_sampling = NativeMTPSamplingConfig(temperature=0.7, seed=29)
    weights = _shared_qwen_checkpoint(moe=moe)
    accepted_b1, accepted_request = _tiny_qwen_b1_emissions(
        monkeypatch,
        moe=moe,
        weights=weights,
        prompt=(1, 2, 3),
        sampling=accepted_sampling,
        uniform=(-1.0,),
        count=3,
    )
    rejected_b1, _ = _tiny_qwen_b1_emissions(
        monkeypatch,
        moe=moe,
        weights=weights,
        prompt=(4,),
        sampling=rejected_sampling,
        uniform=(2.0,),
        count=2,
    )

    model = _model(moe=moe)
    _load_shared_qwen_checkpoint(model, weights)
    target = model.language_model
    rows = (
        NativeMTPRowSpec(7, (1, 2, 3), 8, seed=17, sampling_config=accepted_sampling),
        NativeMTPRowSpec(3, (4,), 8, seed=29, sampling_config=rejected_sampling),
    )
    requests = tuple(model.make_mtp_request_cache() for _ in rows)
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(target, rows, requests)
    )
    with monkeypatch.context() as local_monkeypatch:
        _force_acceptance_uniform(local_monkeypatch, (-1.0, 2.0))
        initial_emissions, initial = generator.prefill(prefill_step_size=1)
        decision = initial.resume().decide()
        assert decision.accepted_uids == (7,)
        assert decision.rejected_uids == (3,)
        resolution_emissions, continuation = decision.resolve()
        bonus_emissions, bonus_continuation = continuation.resume_after_resolution()
        _assert_cohort_offsets_match_b1(
            generator._mixed_accepted_owner, 7, accepted_request
        )
        _assert_cohort_recurrent_row_matches_b1(
            generator._mixed_accepted_owner, 7, accepted_request
        )
        ready = bonus_continuation.resume_after_bonus()

    by_uid = {emission.uid: emission for emission in initial_emissions}
    assert by_uid[7].token == accepted_b1[0][0]
    assert by_uid[3].token == rejected_b1[0][0]
    assert mx.allclose(by_uid[7].logprobs, accepted_b1[0][1]).item()
    assert mx.allclose(by_uid[3].logprobs, rejected_b1[0][1]).item()

    by_uid = {emission.uid: emission for emission in resolution_emissions}
    assert by_uid[7].token == accepted_b1[1][0]
    assert by_uid[3].token == rejected_b1[1][0]
    assert by_uid[7].from_draft is True
    assert by_uid[3].from_draft is False
    assert mx.allclose(by_uid[7].logprobs, accepted_b1[1][1]).item()
    assert mx.allclose(by_uid[3].logprobs, rejected_b1[1][1]).item()

    assert len(bonus_emissions) == 1
    assert bonus_emissions[0].uid == 7
    assert bonus_emissions[0].token == accepted_b1[2][0]
    assert mx.allclose(bonus_emissions[0].logprobs, accepted_b1[2][1]).item()
    assert ready.active_uids == (7, 3)
    assert ready._generator._cohort.uids == ready.active_uids
    with pytest.raises(RuntimeError, match="native_mtp_epoch_moved"):
        continuation.resume_after_resolution()
    with pytest.raises(RuntimeError, match="native_mtp_epoch_moved"):
        bonus_continuation.resume_after_bonus()
    ready.cancel()
    assert generator.closed


@pytest.mark.parametrize("moe", (False, True))
def test_actual_qwen_mixed_ready_matches_independent_b1_phase_oracle(moe, monkeypatch):
    accepted_sampling = NativeMTPSamplingConfig(temperature=0.7, seed=17)
    rejected_sampling = NativeMTPSamplingConfig(temperature=0.7, seed=29)
    weights = _shared_qwen_checkpoint(moe=moe)

    accepted_model = _model(moe=moe)
    _load_shared_qwen_checkpoint(accepted_model, weights)
    accepted = _B1PhaseOracle(
        accepted_model.language_model, (1, 2, 3), accepted_sampling
    )
    accepted.draft()
    with monkeypatch.context() as local_monkeypatch:
        _force_acceptance_uniform(local_monkeypatch, (-1.0,))
        accepted.verify()
    assert accepted.accepted
    accepted_emission, accepted_bonus = accepted.accept_bonus_catchup()

    rejected_model = _model(moe=moe)
    _load_shared_qwen_checkpoint(rejected_model, weights)
    rejected = _B1PhaseOracle(rejected_model.language_model, (4,), rejected_sampling)
    rejected.draft()
    with monkeypatch.context() as local_monkeypatch:
        _force_acceptance_uniform(local_monkeypatch, (2.0,))
        rejected.verify()
    assert not rejected.accepted
    rejected_emission = rejected.reject_replay_redraft()

    model = _model(moe=moe)
    _load_shared_qwen_checkpoint(model, weights)
    rows = (
        NativeMTPRowSpec(7, (1, 2, 3), 8, seed=17, sampling_config=accepted_sampling),
        NativeMTPRowSpec(3, (4,), 8, seed=29, sampling_config=rejected_sampling),
    )
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(
            model.language_model,
            rows,
            tuple(model.make_mtp_request_cache() for _ in rows),
        )
    )
    with monkeypatch.context() as local_monkeypatch:
        _force_acceptance_uniform(local_monkeypatch, (-1.0, 2.0))
        initial_emissions, initial = generator.prefill(prefill_step_size=1)
        decision = initial.resume().decide()
        resolution_emissions, continuation = decision.resolve()
        bonus_emissions, bonus_continuation = continuation.resume_after_resolution()
        ready = bonus_continuation.resume_after_bonus()

    assert decision.accepted_uids == (7,)
    assert decision.rejected_uids == (3,)
    assert tuple(emission.token for emission in initial_emissions) == (
        accepted.initial[0],
        rejected.initial[0],
    )
    by_uid = {emission.uid: emission for emission in resolution_emissions}
    assert by_uid[7].token == accepted_emission[0]
    assert by_uid[3].token == rejected_emission[0]
    assert mx.allclose(by_uid[7].logprobs, accepted_emission[1]).item()
    assert mx.allclose(by_uid[3].logprobs, rejected_emission[1]).item()
    assert bonus_emissions[0].token == accepted_bonus[0]
    assert mx.allclose(bonus_emissions[0].logprobs, accepted_bonus[1]).item()
    assert ready.active_uids == (7, 3)
    _assert_cohort_offsets_match_b1(ready._generator._cohort, 7, accepted.request)
    _assert_cohort_offsets_match_b1(ready._generator._cohort, 3, rejected.request)
    _assert_cohort_recurrent_row_matches_b1(
        ready._generator._cohort, 7, accepted.request
    )
    _assert_cohort_recurrent_row_matches_b1(
        ready._generator._cohort, 3, rejected.request
    )
    assert generator._draft[7].item() == accepted.draft.item()
    assert generator._draft[3].item() == rejected.draft.item()
    assert mx.array_equal(generator._rng_key[7], accepted.sampling._rng_key).item()
    assert mx.array_equal(generator._rng_key[3], rejected.sampling._rng_key).item()


@pytest.mark.parametrize("moe", (False, True))
def test_actual_qwen_sparse_mixed_lifecycle_matches_independent_b1_oracles(
    moe, monkeypatch
):
    """Sparse B>=2 Ready is identical to the manual B1 sparse transaction."""

    accepted_sampling = NativeMTPSamplingConfig(temperature=0.7, seed=17)
    rejected_sampling = NativeMTPSamplingConfig(temperature=0.7, seed=29)
    weights = _shared_qwen_checkpoint(moe=moe)
    accepted_tokens, accepted_positions = (1, 2, 3), (2, 7, 12)
    rejected_tokens, rejected_positions = (4, 5), (3, 11)
    accepted_model = _model(moe=moe)
    _load_shared_qwen_checkpoint(accepted_model, weights)
    with monkeypatch.context() as local_monkeypatch:
        _force_acceptance_uniform(local_monkeypatch, (-1.0,))
        accepted_b1 = _B1SparsePhaseOracle(
            accepted_model,
            accepted_tokens,
            accepted_positions,
            accepted_sampling,
            accepted=True,
        )
    rejected_model = _model(moe=moe)
    _load_shared_qwen_checkpoint(rejected_model, weights)
    with monkeypatch.context() as local_monkeypatch:
        _force_acceptance_uniform(local_monkeypatch, (2.0,))
        rejected_b1 = _B1SparsePhaseOracle(
            rejected_model,
            rejected_tokens,
            rejected_positions,
            rejected_sampling,
            accepted=False,
        )

    model = _model(moe=moe)
    _load_shared_qwen_checkpoint(model, weights)
    rows = (
        NativeMTPRowSpec(
            7, accepted_tokens, 8, seed=17, sampling_config=accepted_sampling
        ),
        NativeMTPRowSpec(
            3, rejected_tokens, 8, seed=29, sampling_config=rejected_sampling
        ),
    )
    admission = NativeMTPAdmission.create_from_sparse_bootstraps(
        model,
        rows,
        (
            _actual_sparse_bootstrap(
                model, accepted_tokens, accepted_positions, chunk_sizes=(1, 2)
            ),
            _actual_sparse_bootstrap(
                model, rejected_tokens, rejected_positions, chunk_sizes=(1, 1)
            ),
        ),
    )
    generator = NativeMTPBatchGenerator(admission)
    with monkeypatch.context() as local_monkeypatch:
        _force_acceptance_uniform(local_monkeypatch, (-1.0, 2.0))
        initial, initial_epoch = generator.start_sparse()
        decision = initial_epoch.resume().decide()
        resolution, continuation = decision.resolve()
        bonus, bonus_continuation = continuation.resume_after_resolution()
        ready = bonus_continuation.resume_after_bonus()

    assert decision.accepted_uids == (7,)
    assert decision.rejected_uids == (3,)
    for actual, expected in zip(initial, (accepted_b1.initial, rejected_b1.initial)):
        assert actual.token == expected[0]
        assert mx.allclose(actual.logprobs, expected[1]).item()
    resolved = {item.uid: item for item in resolution}
    assert resolved[7].token == accepted_b1.emissions[0][0]
    assert resolved[3].token == rejected_b1.emissions[0][0]
    assert mx.allclose(resolved[7].logprobs, accepted_b1.emissions[0][1]).item()
    assert mx.allclose(resolved[3].logprobs, rejected_b1.emissions[0][1]).item()
    assert bonus[0].token == accepted_b1.emissions[1][0]
    assert mx.allclose(bonus[0].logprobs, accepted_b1.emissions[1][1]).item()
    assert generator._history[7].tolist() == accepted_b1.history.tolist()
    assert generator._history[3].tolist() == rejected_b1.history.tolist()
    assert generator._verify_position_by_uid == {7: (13, 14), 3: (12, 13)}
    assert generator._logical_cursor == {7: 16, 3: 14}
    assert generator._logical_cursor == {7: accepted_b1.cursor, 3: rejected_b1.cursor}
    for uid, oracle in ((7, accepted_b1), (3, rejected_b1)):
        assert mx.array_equal(generator._rng_key[uid], oracle.sampling._rng_key).item()
        assert generator._head[uid].item() == oracle.head.item()
        assert generator._draft[uid].item() == oracle.draft.item()
        _assert_cohort_offsets_match_b1(ready._generator._cohort, uid, oracle.request)
        _assert_cohort_recurrent_row_matches_b1(
            ready._generator._cohort, uid, oracle.request
        )
        _assert_cohort_cache_payload_row_matches_b1(
            ready._generator._cohort, uid, oracle.request
        )
    ready.cancel()
    assert generator.closed


@pytest.mark.parametrize("moe", (False, True))
def test_actual_qwen_sparse_terminal_rows_are_filtered_before_ready_join(
    moe, monkeypatch
):
    """A sparse mixed epoch retains only the non-terminal survivor's owner."""

    weights = _shared_qwen_checkpoint(moe=moe)
    model = _model(moe=moe)
    _load_shared_qwen_checkpoint(model, weights)
    configs = {
        7: NativeMTPSamplingConfig(temperature=0.7, seed=17),
        3: NativeMTPSamplingConfig(temperature=0.7, seed=29),
        11: NativeMTPSamplingConfig(temperature=0.7, seed=43),
    }
    values = {
        7: ((1, 2), (2, 8)),
        3: ((4, 5), (3, 10)),
        11: ((6, 7), (4, 12)),
    }
    rows = tuple(
        NativeMTPRowSpec(
            uid,
            values[uid][0],
            2 if uid in (7, 3) else 8,
            seed=configs[uid].seed,
            sampling_config=configs[uid],
        )
        for uid in (7, 3, 11)
    )
    admission = NativeMTPAdmission.create_from_sparse_bootstraps(
        model,
        rows,
        tuple(
            _actual_sparse_bootstrap(
                model, values[uid][0], values[uid][1], chunk_sizes=(1, 1)
            )
            for uid in (7, 3, 11)
        ),
    )
    generator = NativeMTPBatchGenerator(admission)
    with monkeypatch.context() as local_monkeypatch:
        _force_acceptance_uniform(local_monkeypatch, (-1.0, 2.0, -1.0))
        _, initial = generator.start_sparse()
        resolution, continuation = initial.resume().decide().resolve()
        bonus, after_bonus = continuation.resume_after_resolution()
        ready = after_bonus.resume_after_bonus()
    by_uid = {item.uid: item for item in resolution}
    assert by_uid[7].finish_reason == "length"
    assert by_uid[3].finish_reason == "length"
    assert tuple(item.uid for item in bonus) == (11,)
    assert ready.active_uids == (11,)
    assert ready._generator._cohort.uids == (11,)


@pytest.mark.parametrize("moe", (False, True))
def test_actual_qwen_sparse_partial_admission_failure_consumes_every_bootstrap(
    moe, monkeypatch
):
    """A later B1 MTP-init failure cannot leave a claim or owner reusable."""

    weights = _shared_qwen_checkpoint(moe=moe)
    model = _model(moe=moe)
    _load_shared_qwen_checkpoint(model, weights)
    first = _actual_sparse_bootstrap(model, (1, 2), (2, 8), chunk_sizes=(1, 1))
    second = _actual_sparse_bootstrap(model, (4, 5), (3, 10), chunk_sizes=(1, 1))
    original_mtp = model.mtp_forward
    calls = 0

    def fail_second_mtp(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected_sparse_second_mtp_init_failure")
        return original_mtp(*args, **kwargs)

    rows = (NativeMTPRowSpec(7, (1, 2), 8), NativeMTPRowSpec(3, (4, 5), 8))
    with monkeypatch.context() as local_monkeypatch:
        local_monkeypatch.setattr(model, "mtp_forward", fail_second_mtp)
        with pytest.raises(
            RuntimeError, match="injected_sparse_second_mtp_init_failure"
        ):
            NativeMTPAdmission.create_from_sparse_bootstraps(
                model, rows, (first, second)
            )
    for bootstrap in (first, second):
        with pytest.raises(RuntimeError, match="already_claimed"):
            bootstrap.claim(model)


@pytest.mark.parametrize("after_bonus", (False, True))
def test_mixed_continuation_cancel_is_move_only_and_closes_all_owners(
    after_bonus, monkeypatch
):
    sampling = NativeMTPSamplingConfig(temperature=0.7, seed=17)
    model = _BatchModel()
    rows = (
        NativeMTPRowSpec(7, (1, 2), 8, seed=17, sampling_config=sampling),
        NativeMTPRowSpec(
            3,
            (4,),
            8,
            seed=29,
            sampling_config=NativeMTPSamplingConfig(temperature=0.7, seed=29),
        ),
    )
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(model, rows, (_request(model), _request(model)))
    )
    with monkeypatch.context() as local_monkeypatch:
        _force_acceptance_uniform(local_monkeypatch, (-1.0, 2.0))
        _, initial = generator.prefill(prefill_step_size=1)
        _, continuation = initial.resume().decide().resolve()
        handle = continuation
        if after_bonus:
            _, handle = continuation.resume_after_resolution()
        handle.cancel()
    assert generator.closed
    assert all(owner.poisoned for owner in generator._last_failed_owners)
    with pytest.raises(RuntimeError, match="native_mtp_epoch_moved"):
        handle.cancel()
    with pytest.raises(RuntimeError, match="native_mtp_epoch_moved"):
        (
            handle.resume_after_bonus()
            if after_bonus
            else handle.resume_after_resolution()
        )


def _visible_cohort_state(generator):
    """Return UID-keyed, order-independent state after an owner transition."""

    owner = generator._cohort
    result = {}
    for index, uid in enumerate(owner.uids):
        cache = []
        for entry in owner.backbone + owner.mtp:
            if isinstance(entry, ArraysCache):
                slots = tuple(
                    None if value is None else value[index : index + 1]
                    for value in entry.cache
                )
                cache.append(
                    (
                        "arrays",
                        slots,
                        (
                            entry.left_padding[index : index + 1]
                            if entry.left_padding is not None
                            else None
                        ),
                        (
                            entry.lengths[index : index + 1]
                            if entry.lengths is not None
                            else None
                        ),
                    )
                )
            else:
                cache.append(
                    (
                        "kv",
                        entry.offset[index : index + 1],
                        (
                            entry.left_padding[index : index + 1]
                            if entry.left_padding is not None
                            else None
                        ),
                    )
                )
        result[uid] = (
            generator._head[uid],
            generator._draft[uid],
            generator._rng_key[uid],
            generator._history[uid],
            tuple(cache),
        )
    return result


def _assert_visible_state_equal(left, right):
    assert left.keys() == right.keys()
    for uid in left:
        for expected, actual in zip(left[uid][:4], right[uid][:4]):
            assert mx.array_equal(expected, actual).item(), uid
        for expected_entry, actual_entry in zip(left[uid][4], right[uid][4]):
            assert expected_entry[0] == actual_entry[0]
            for expected, actual in zip(expected_entry[1:], actual_entry[1:]):
                if isinstance(expected, tuple):
                    assert isinstance(actual, tuple)
                    for expected_slot, actual_slot in zip(expected, actual):
                        if expected_slot is None:
                            assert actual_slot is None
                        else:
                            assert mx.array_equal(
                                expected_slot, actual_slot
                            ).item(), uid
                elif expected is None:
                    assert actual is None
                else:
                    assert mx.array_equal(expected, actual).item(), uid


def _run_actual_qwen_mixed_epoch(monkeypatch, *, moe, weights, row_order):
    """Resolve the same UID-owned mixed epoch under an arbitrary input order."""

    configs = {
        7: NativeMTPSamplingConfig(temperature=0.7, seed=17),
        3: NativeMTPSamplingConfig(temperature=0.7, seed=29),
    }
    prompts = {7: (1, 2, 3), 3: (4,)}
    model = _model(moe=moe)
    _load_shared_qwen_checkpoint(model, weights)
    rows = tuple(
        NativeMTPRowSpec(
            uid, prompts[uid], 8, seed=configs[uid].seed, sampling_config=configs[uid]
        )
        for uid in row_order
    )
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(
            model.language_model,
            rows,
            tuple(model.make_mtp_request_cache() for _ in rows),
        )
    )
    # Acceptance draws occur in current cohort order.  Bind values to UIDs,
    # rather than accidentally granting positional RNG semantics in the test.
    uniforms = tuple(-1.0 if uid == 7 else 2.0 for uid in row_order)
    with monkeypatch.context() as local_monkeypatch:
        _force_acceptance_uniform(local_monkeypatch, uniforms)
        initial, initial_epoch = generator.prefill(prefill_step_size=1)
        decision = initial_epoch.resume().decide()
        resolution, continuation = decision.resolve()
        bonus, after_bonus = continuation.resume_after_resolution()
        ready = after_bonus.resume_after_bonus()
    return generator, initial, resolution, bonus, ready


@pytest.mark.parametrize("moe", (False, True))
def test_actual_qwen_uid_stochastic_state_survives_reorder_split_filter_join(
    moe, monkeypatch
):
    """UID-owned stochastic state is invariant to batch order and branch moves."""

    weights = _shared_qwen_checkpoint(moe=moe)
    ordered = _run_actual_qwen_mixed_epoch(
        monkeypatch, moe=moe, weights=weights, row_order=(7, 3)
    )
    reordered = _run_actual_qwen_mixed_epoch(
        monkeypatch, moe=moe, weights=weights, row_order=(3, 7)
    )
    for boundary in range(3):
        left = {emission.uid: emission for emission in ordered[boundary + 1]}
        right = {emission.uid: emission for emission in reordered[boundary + 1]}
        assert left.keys() == right.keys()
        for uid in left:
            assert (left[uid].token, left[uid].from_draft, left[uid].finish_reason) == (
                right[uid].token,
                right[uid].from_draft,
                right[uid].finish_reason,
            )
            assert mx.allclose(left[uid].logprobs, right[uid].logprobs).item()
    assert set(ordered[-1].active_uids) == set(reordered[-1].active_uids) == {3, 7}
    _assert_visible_state_equal(
        _visible_cohort_state(ordered[0]), _visible_cohort_state(reordered[0])
    )


@pytest.mark.parametrize("moe", (False, True))
def test_actual_qwen_mixed_eos_and_length_rows_skip_later_branch_work(moe, monkeypatch):
    """Model-derived terminal resolution rows never replay, bonus, or catch up."""

    accepted_config = NativeMTPSamplingConfig(temperature=0.7, seed=17)
    weights = _shared_qwen_checkpoint(moe=moe)
    oracle_model = _model(moe=moe)
    _load_shared_qwen_checkpoint(oracle_model, weights)
    oracle = _B1PhaseOracle(oracle_model.language_model, (1, 2, 3), accepted_config)
    oracle.draft()
    eos_draft = oracle.draft.item()

    model = _model(moe=moe)
    _load_shared_qwen_checkpoint(model, weights)
    rows = (
        NativeMTPRowSpec(
            7,
            (1, 2, 3),
            8,
            seed=17,
            eos_token_ids=frozenset({eos_draft}),
            sampling_config=accepted_config,
        ),
        NativeMTPRowSpec(
            3,
            (4,),
            2,
            seed=29,
            sampling_config=NativeMTPSamplingConfig(temperature=0.7, seed=29),
        ),
        NativeMTPRowSpec(
            11,
            (2,),
            8,
            seed=43,
            sampling_config=NativeMTPSamplingConfig(temperature=0.7, seed=43),
        ),
    )
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(
            model.language_model,
            rows,
            tuple(model.make_mtp_request_cache() for _ in rows),
        )
    )
    with monkeypatch.context() as local_monkeypatch:
        _force_acceptance_uniform(local_monkeypatch, (-1.0, 2.0, -1.0))
        _, initial = generator.prefill(prefill_step_size=1)
        decision = initial.resume().decide()
        resolution, continuation = decision.resolve()
        terminal_rng = {uid: generator._rng_key[uid] for uid in (7, 3)}
        terminal_history = {uid: generator._history[uid] for uid in (7, 3)}
        terminal_draft = {uid: generator._draft[uid] for uid in (7, 3)}
        bonus, after_bonus = continuation.resume_after_resolution()
        ready = after_bonus.resume_after_bonus()

    by_uid = {emission.uid: emission for emission in resolution}
    assert by_uid[7].finish_reason == "eos"
    assert by_uid[3].finish_reason == "length"
    assert tuple(emission.uid for emission in bonus) == (11,)
    assert ready.active_uids == (11,)
    assert generator._cohort.uids == (11,)
    for uid in (7, 3):
        assert mx.array_equal(generator._rng_key[uid], terminal_rng[uid]).item()
        assert mx.array_equal(generator._history[uid], terminal_history[uid]).item()
        assert mx.array_equal(generator._draft[uid], terminal_draft[uid]).item()


class _FailureBoundaryModel(_BatchModel):
    def __init__(self):
        super().__init__()
        self.fail_target = False
        self.fail_mtp = False

    def __call__(self, inputs, *, cache, return_hidden=False):
        if self.fail_target:
            raise RuntimeError("injected_rejected_replay_target_failure")
        return super().__call__(inputs, cache=cache, return_hidden=return_hidden)

    def mtp_forward(self, hidden, next_tokens, cache):
        if self.fail_mtp:
            raise RuntimeError("injected_accepted_catchup_failure")
        return super().mtp_forward(hidden, next_tokens, cache)


def _mixed_failure_fixture(monkeypatch):
    model = _FailureBoundaryModel()
    rows = (
        NativeMTPRowSpec(
            7,
            (1, 2),
            8,
            seed=17,
            sampling_config=NativeMTPSamplingConfig(temperature=0.7, seed=17),
        ),
        NativeMTPRowSpec(
            3,
            (4,),
            8,
            seed=29,
            sampling_config=NativeMTPSamplingConfig(temperature=0.7, seed=29),
        ),
    )
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(model, rows, (_request(model), _request(model)))
    )
    _force_acceptance_uniform(monkeypatch, (-1.0, 2.0))
    _, initial = generator.prefill(prefill_step_size=1)
    _, continuation = initial.resume().decide().resolve()
    return model, generator, continuation


def _assert_mixed_failure_closed(generator, owners, message):
    assert generator.closed
    assert len(generator._last_failed_owners) == len({id(owner) for owner in owners})
    assert all(
        owner.poisoned and owner._checkpoint is None for owner in owners
    ), message


def test_rejected_replay_target_failure_poison_closes_and_clears_checkpoints(
    monkeypatch,
):
    model, generator, continuation = _mixed_failure_fixture(monkeypatch)
    owners = (generator._mixed_accepted_owner, generator._mixed_rejected_owner)
    model.fail_target = True
    with pytest.raises(RuntimeError, match="injected_rejected_replay_target_failure"):
        continuation.resume_after_resolution()
    _assert_mixed_failure_closed(generator, owners, "rejected replay")


def test_accepted_bonus_sampling_failure_poison_closes_and_preserves_primary(
    monkeypatch,
):
    model, generator, continuation = _mixed_failure_fixture(monkeypatch)
    del model
    owners = (generator._mixed_accepted_owner, generator._mixed_rejected_owner)
    accepted_state = generator._sampling[7]
    original_sample = accepted_state.sample

    def fail_bonus_sample(*args, **kwargs):
        raise RuntimeError("injected_accepted_bonus_sampling_failure")

    monkeypatch.setattr(accepted_state, "sample", fail_bonus_sample)
    with pytest.raises(RuntimeError, match="injected_accepted_bonus_sampling_failure"):
        continuation.resume_after_resolution()
    monkeypatch.setattr(accepted_state, "sample", original_sample)
    _assert_mixed_failure_closed(generator, owners, "accepted bonus")


def test_accepted_mtp_catchup_failure_poison_closes_and_clears_checkpoints(monkeypatch):
    model, generator, continuation = _mixed_failure_fixture(monkeypatch)
    owners = (generator._mixed_accepted_owner, generator._mixed_rejected_owner)
    _, after_bonus = continuation.resume_after_resolution()
    model.fail_mtp = True
    with pytest.raises(RuntimeError, match="injected_accepted_catchup_failure"):
        after_bonus.resume_after_bonus()
    _assert_mixed_failure_closed(generator, owners, "accepted catchup")


def test_ready_owner_join_failure_poison_closes_and_preserves_primary(monkeypatch):
    _, generator, continuation = _mixed_failure_fixture(monkeypatch)
    owners = (generator._mixed_accepted_owner, generator._mixed_rejected_owner)
    _, after_bonus = continuation.resume_after_resolution()

    def fail_join(self, other):
        raise RuntimeError("injected_ready_owner_join_failure")

    monkeypatch.setattr(type(owners[0]), "join", fail_join)
    with pytest.raises(RuntimeError, match="injected_ready_owner_join_failure"):
        after_bonus.resume_after_bonus()
    _assert_mixed_failure_closed(generator, owners, "ready join")


@pytest.mark.parametrize("model_type", (_SparseBatchModel, _SparseMoEBatchModel))
def test_sparse_admission_claims_b1_receipts_then_starts_batched_without_target_replay(
    model_type,
):
    model = model_type()
    bootstraps = (
        _sparse_bootstrap(model, (1, 2), (5, 8)),
        _sparse_bootstrap(model, (4, 5), (11, 14)),
    )
    target_calls_before = model.target_calls
    rows = (
        NativeMTPRowSpec(7, (1, 2), 8, seed=17),
        NativeMTPRowSpec(3, (4, 5), 8, seed=29),
    )

    admission = NativeMTPAdmission.create_from_sparse_bootstraps(
        model, rows, bootstraps
    )
    generator = NativeMTPBatchGenerator(admission)
    emissions, initial = generator.start_sparse()

    assert [item.uid for item in emissions] == [7, 3]
    assert initial.phase == "initial"
    # Sparse admission performs MTP initialization only; sampling initial
    # heads must retain the receipts' compact final target logits.
    assert model.target_calls == target_calls_before
    assert model.mtp_calls == 2
    # ``start_sparse`` emits the initial head, so each cursor has already
    # advanced once beyond its bootstrap's attested next logical position.
    assert generator._logical_cursor == {7: 10, 3: 16}
    assert admission.cohort.uids == (7, 3)


def test_sparse_ready_uses_explicit_per_row_target_position_matrix():
    model = _SparseBatchModel()
    rows = (
        NativeMTPRowSpec(7, (1, 2), 8),
        NativeMTPRowSpec(3, (4, 5), 8),
    )
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create_from_sparse_bootstraps(
            model,
            rows,
            (
                _sparse_bootstrap(model, (1, 2), (2, 9)),
                _sparse_bootstrap(model, (4, 5), (11, 14)),
            ),
        )
    )
    _, initial = generator.start_sparse()
    decision = initial.resume().decide()

    assert decision.active_uids == (7, 3)
    verify = [
        forward
        for forward in model.forwards
        if forward.phase is GenerationForwardPhase.VERIFY
    ]
    assert verify[-1].logical_positions == ((10, 11), (15, 16))


def test_sparse_admission_uses_original_noncontiguous_successors_and_exact_cursor():
    model = _SparseBatchModel()
    cache = model.make_cache()
    # The successor is intentionally not derived from the sparse token IDs.
    (_, _), first = attested_target_forward(
        model,
        (1,),
        cache,
        phase=GenerationForwardPhase.PREFILL,
        logical_positions=(2,),
        immediate_successor_token_ids=(6,),
        model_forward_context=model.generation_forward_context,
    )
    (_, _), final = attested_target_forward(
        model,
        (6,),
        cache,
        phase=GenerationForwardPhase.PREFILL,
        logical_positions=(9,),
        immediate_successor_token_ids=(),
        model_forward_context=model.generation_forward_context,
    )
    bootstrap = NativeMTPSparseBootstrap(
        receipts=(first, final),
        selected_logical_positions=(2, 9),
        selected_token_ids=(1, 6),
        immediate_successor_token_ids=(6,),
        target_cache=cache,
        next_logical_position=10,
    )
    admission = NativeMTPAdmission.create_from_sparse_bootstraps(
        model, (NativeMTPRowSpec(9, (1, 6), 8),), (bootstrap,)
    )
    generator = NativeMTPBatchGenerator(admission)
    _, initial = generator.start_sparse()
    ready = initial.resume()

    mtp_forwards = [
        forward
        for forward in model.forwards
        if forward.phase is GenerationForwardPhase.MTP_DRAFT
    ]
    assert mtp_forwards[0].logical_positions == (2,)
    assert mtp_forwards[-1].logical_positions == (9,)
    assert generator._logical_cursor == {9: 11}
    assert ready.active_uids == (9,)


def test_sparse_admission_prevalidation_preserves_unclaimed_receipts():
    model = _SparseBatchModel()
    first = _sparse_bootstrap(model, (1, 2), (0, 3))
    second = _sparse_bootstrap(model, (3, 4), (1, 4))
    # Row metadata is rejected before any bootstrap is claimed or adopted.
    rows = (NativeMTPRowSpec(1, (1, 2), 8), NativeMTPRowSpec(2, (0,), 8))
    with pytest.raises(ValueError, match="row_prompt_mismatch"):
        NativeMTPAdmission.create_from_sparse_bootstraps(model, rows, (first, second))
    assert first.claim(model).selected_token_ids == (1, 2)
    assert second.claim(model).selected_token_ids == (3, 4)
