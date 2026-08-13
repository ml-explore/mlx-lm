# Copyright © 2026 Apple Inc.
"""Typed public lifecycle coverage for native-MTP cohort admission."""

from dataclasses import dataclass
from collections import deque

import mlx.core as mx
import pytest

from test_qwen3_5_mtp_model import _checkpoint, _model, _sanitize_and_load

from mlx_lm.generate import (
    NativeMTPAdmission,
    NativeMTPBatchGenerator,
    NativeMTPEmission,
    NativeMTPRejectedEpoch,
    NativeMTPRowSpec,
    NativeMTPSamplingConfig,
    mtp_generate_step,
)
from mlx_lm.generate import (
    _NativeMTPCohortCache,
    _NativeMTPCohortMutationDelta,
    _NativeMTPSamplingState,
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
    accepted = decision.accept()
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
    accepted = decision.accept()
    bonus = accepted.bonus()
    ready = bonus.catch_up()
    assert ready.phase == "ready"


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
    bonus = decision.accept().bonus()
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
    decision.accept().bonus()

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
