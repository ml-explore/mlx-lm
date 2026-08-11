# Copyright © 2026 Apple Inc.
"""Synthetic transaction coverage for request-local native Qwen MTP caches."""

from dataclasses import dataclass

import mlx.core as mx
import pytest

import mlx_lm.models.cache as cache_module
from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    NativeMTPRequestCache,
    QuantizedKVCache,
)


@dataclass(frozen=True)
class _Capability:
    supported: bool = True
    reason: str = "supported"


class _Model:
    mtp_capability = _Capability()

    def __init__(self, backbone=None, mtp=None):
        self._backbone = backbone
        self._mtp = mtp
        self.layers = [
            type("_Layer", (), {"is_linear": isinstance(entry, ArraysCache)})()
            for entry in backbone
        ]
        self.mtp = type("_MTP", (), {"layers": [object() for _ in mtp]})()

    def make_cache(self):
        return self._backbone

    def make_mtp_cache(self):
        return self._mtp


def _tokens(count, dim=32):
    values = mx.arange(count * dim, dtype=mx.float32).reshape(1, 1, count, dim)
    return values, values + 100


def _advance(cache, count, dim=32):
    keys, values = _tokens(count, dim)
    cache.update_and_fetch(keys, values)


def _cache_pair():
    recurrent = ArraysCache(2)
    recurrent[0] = mx.array([[1.0, 2.0]])
    recurrent[1] = mx.array([[3.0, 4.0]])
    backbone_attention = KVCache()
    mtp_attention = KVCache()
    return [recurrent, backbone_attention], [mtp_attention]


def test_commit_asserts_every_attention_cache_matches_its_logical_position():
    backbone, mtp = _cache_pair()
    request = NativeMTPRequestCache(_Model(backbone, mtp), backbone, mtp)

    _advance(backbone[1], 2)
    _advance(mtp[0], 1)
    request.retain(backbone_tokens=2, mtp_tokens=1)
    assert request.state.backbone_tokens == 2
    assert request.state.mtp_tokens == 1

    request.checkpoint()
    _advance(backbone[1], 1)
    _advance(mtp[0], 1)
    request.seal_verified(backbone_tokens=3, mtp_tokens=2)
    request.commit(backbone_tokens=3, mtp_tokens=2)
    assert request.state.backbone_tokens == 3
    assert request.state.mtp_tokens == 2

    _advance(backbone[1], 1)
    with pytest.raises(RuntimeError, match="backbone_cache_position_mismatch"):
        request.retain(backbone_tokens=3, mtp_tokens=2)


def test_rollback_restores_recurrent_and_attention_state_without_copying_kv_prefix():
    backbone, mtp = _cache_pair()
    request = NativeMTPRequestCache(_Model(backbone, mtp), backbone, mtp)
    _advance(backbone[1], 2)
    _advance(mtp[0], 1)
    request.retain(backbone_tokens=2, mtp_tokens=1)
    initial_recurrent = backbone[0][0]

    request.checkpoint()
    _advance(backbone[1], 1)
    _advance(mtp[0], 1)
    backbone[0][0] = mx.array([[9.0, 9.0]])
    request.seal_verified(backbone_tokens=3, mtp_tokens=2)
    request.rollback()
    mx.eval(backbone[0][0])

    assert backbone[1].offset == 2
    assert mtp[0].offset == 1
    assert mx.array_equal(backbone[0][0], initial_recurrent).item()
    assert request.state.backbone_tokens == 2
    assert request.state.mtp_tokens == 1


def test_quantization_replaces_entries_through_the_caller_owned_containers():
    backbone, mtp = _cache_pair()
    request = NativeMTPRequestCache(_Model(backbone, mtp), backbone, mtp)
    _advance(backbone[1], 2)
    _advance(mtp[0], 1)
    request.retain(backbone_tokens=2, mtp_tokens=1)
    caller_backbone = backbone
    caller_mtp = mtp

    assert request.quantize(kv_bits=8, kv_group_size=32) == 2
    assert request.backbone is caller_backbone
    assert request.mtp is caller_mtp
    assert isinstance(backbone[1], QuantizedKVCache)
    assert isinstance(mtp[0], QuantizedKVCache)
    request.assert_aligned(backbone_tokens=2, mtp_tokens=1)


def test_rollback_restores_pre_quantization_entry_objects_and_positions():
    backbone, mtp = _cache_pair()
    request = NativeMTPRequestCache(_Model(backbone, mtp), backbone, mtp)
    _advance(backbone[1], 2)
    _advance(mtp[0], 2)
    request.retain(backbone_tokens=2, mtp_tokens=2)
    original_backbone = backbone[1]
    original_mtp = mtp[0]

    request.checkpoint()
    request.quantize(kv_bits=8, kv_group_size=32)
    _advance(backbone[1], 1)
    _advance(mtp[0], 1)
    request.seal_verified(backbone_tokens=3, mtp_tokens=3)
    request.rollback()

    assert backbone[1] is original_backbone
    assert mtp[0] is original_mtp
    assert backbone[1].offset == 2
    assert mtp[0].offset == 2


def test_prefix_reuse_batch_and_external_replacement_fail_closed():
    backbone, mtp = _cache_pair()
    model = _Model(backbone, mtp)
    with pytest.raises(ValueError, match="native_mtp_prefix_reuse_unsupported"):
        NativeMTPRequestCache.create(model, prompt_cache=backbone)

    request = NativeMTPRequestCache(model, backbone, mtp)
    backbone[1] = KVCache()
    with pytest.raises(RuntimeError, match="entry_replaced_externally"):
        request.assert_aligned(backbone_tokens=0, mtp_tokens=0)

    multi_batch = KVCache()
    keys = mx.zeros((2, 1, 1, 32))
    multi_batch.update_and_fetch(keys, keys)
    with pytest.raises(RuntimeError, match="native_mtp_batch_size_unsupported"):
        NativeMTPRequestCache(
            _Model([multi_batch], [KVCache()]), [multi_batch], [KVCache()]
        )


def test_pipeline_parallelism_is_rejected_before_any_cache_transaction():
    backbone, mtp = _cache_pair()
    model = _Model(backbone, mtp)
    model.model = type("_Pipeline", (), {"pipeline_size": 2})()
    with pytest.raises(
        RuntimeError, match="native_mtp_pipeline_parallelism_unsupported"
    ):
        NativeMTPRequestCache(model, backbone, mtp)


@pytest.mark.parametrize("accepted", (1, 2))
def test_partial_acceptance_requires_rollback_and_sequential_replay(accepted):
    """Replaying the accepted prefix matches a synthetic recurrent dense path."""

    backbone, mtp = _cache_pair()
    request = NativeMTPRequestCache(_Model(backbone, mtp), backbone, mtp)
    request.checkpoint()
    for _ in range(3):
        _advance(backbone[1], 1)
        _advance(mtp[0], 1)
        backbone[0][0] = backbone[0][0] + 1
    request.seal_verified(backbone_tokens=3, mtp_tokens=3)

    # Direct attention trimming cannot make the recurrent state a valid
    # accepted-prefix state; commit must still force full rollback + replay.
    backbone[1].trim(3 - accepted)
    mtp[0].trim(3 - accepted)
    with pytest.raises(RuntimeError, match="native_mtp_partial_commit_requires_replay"):
        request.commit(backbone_tokens=accepted, mtp_tokens=accepted)
    replay = request.reject_partial(
        accepted_backbone_tokens=accepted, accepted_mtp_tokens=accepted
    )
    assert replay.replay_to_backbone_tokens == accepted
    assert request.replay_required is replay
    assert mx.array_equal(backbone[0][0], mx.array([[1.0, 2.0]])).item()

    for token in range(1, accepted + 1):
        _advance(backbone[1], 1)
        _advance(mtp[0], 1)
        backbone[0][0] = backbone[0][0] + 1
        request.replay_retained(backbone_tokens=token, mtp_tokens=token)
    assert request.replay_required is None
    assert mx.array_equal(
        backbone[0][0], mx.array([[1.0 + accepted, 2.0 + accepted]])
    ).item()


def test_topology_rejects_truncated_swapped_duplicate_and_cross_alias_caches():
    backbone, mtp = _cache_pair()
    model = _Model(backbone, mtp)
    with pytest.raises(ValueError, match="backbone_cache_topology_mismatch"):
        NativeMTPRequestCache(model, backbone[:1], mtp)
    with pytest.raises(TypeError, match="backbone_cache_type_mismatch"):
        NativeMTPRequestCache(model, [backbone[1], backbone[0]], mtp)
    duplicate = [backbone[0], backbone[0]]
    duplicate_model = _Model(duplicate, mtp)
    with pytest.raises(ValueError, match="entries_must_be_unique"):
        NativeMTPRequestCache(duplicate_model, duplicate, mtp)
    alias_backbone, alias_mtp = _cache_pair()
    alias_mtp[0] = alias_backbone[1]
    alias_model = _Model(alias_backbone, alias_mtp)
    with pytest.raises(ValueError, match="entries_must_be_unique"):
        NativeMTPRequestCache(alias_model, alias_backbone, alias_mtp)


class _FailingKVCache(KVCache):
    def to_quantized(self, *args, **kwargs):
        raise ValueError("synthetic quantization conversion failure")


def _quantization_failure_caches(*, fail_in):
    recurrent = ArraysCache(2)
    recurrent[0] = mx.array([[1.0]])
    recurrent[1] = mx.array([[2.0]])
    first = KVCache()
    second = _FailingKVCache() if fail_in == "backbone" else KVCache()
    head = _FailingKVCache() if fail_in == "mtp" else KVCache()
    for entry in (first, second, head):
        _advance(entry, 1)
    backbone, mtp = [recurrent, first, second], [head]
    request = NativeMTPRequestCache(_Model(backbone, mtp), backbone, mtp)
    request.retain(backbone_tokens=1, mtp_tokens=1)
    return request, backbone, mtp


@pytest.mark.parametrize("fail_in", ("backbone", "mtp"))
def test_quantization_conversion_failure_never_publishes_a_partial_container(fail_in):
    request, backbone, mtp = _quantization_failure_caches(fail_in=fail_in)
    before_backbone = tuple(backbone)
    before_mtp = tuple(mtp)

    with pytest.raises(ValueError, match="synthetic quantization conversion failure"):
        request.quantize(kv_bits=8, kv_group_size=32)

    assert tuple(backbone) == before_backbone
    assert tuple(mtp) == before_mtp
    assert all(isinstance(entry, KVCache) for entry in (*backbone[1:], *mtp))


def test_quantization_evaluation_failure_never_publishes_staged_replacements(
    monkeypatch,
):
    backbone, mtp = _cache_pair()
    request = NativeMTPRequestCache(_Model(backbone, mtp), backbone, mtp)
    _advance(backbone[1], 1)
    _advance(mtp[0], 1)
    request.retain(backbone_tokens=1, mtp_tokens=1)
    before_backbone = tuple(backbone)
    before_mtp = tuple(mtp)

    def fail_eval(*_args, **_kwargs):
        raise RuntimeError("synthetic lazy evaluation failure")

    monkeypatch.setattr(cache_module.mx, "eval", fail_eval)
    with pytest.raises(RuntimeError, match="native_mtp_cache_quantization_failed"):
        request.quantize(kv_bits=8, kv_group_size=32)

    assert tuple(backbone) == before_backbone
    assert tuple(mtp) == before_mtp
    assert isinstance(backbone[1], KVCache)
    assert isinstance(mtp[0], KVCache)


@pytest.mark.parametrize("reason", ("eos", "length", "cancelled", "generator_closed"))
def test_finish_rolls_back_active_draft_and_is_idempotent(reason):
    backbone, mtp = _cache_pair()
    request = NativeMTPRequestCache(_Model(backbone, mtp), backbone, mtp)
    _advance(backbone[1], 1)
    _advance(mtp[0], 1)
    request.retain(backbone_tokens=1, mtp_tokens=1)
    request.checkpoint()
    _advance(backbone[1], 1)
    _advance(mtp[0], 1)

    request.finish(reason)
    assert request.closed
    assert backbone[1].offset == 1
    assert mtp[0].offset == 1
    request.finish(reason)
