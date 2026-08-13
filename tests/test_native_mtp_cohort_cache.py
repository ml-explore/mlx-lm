# Copyright © 2026 Apple Inc.
"""Move-only cache ownership coverage for batched native Qwen MTP."""

from dataclasses import dataclass
import inspect

import mlx.core as mx
import pytest

from mlx_lm.generate import _NativeMTPCohortCache, _NativeMTPCohortMutationDelta
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
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

    def __init__(self):
        self.layers = [
            type("_Linear", (), {"is_linear": True})(),
            type("_Attention", (), {"is_linear": False})(),
        ]
        self.mtp = type("_MTP", (), {"layers": [object()]})()


def _advance(entry, count, *, heads=1):
    if entry.keys is None and isinstance(entry, BatchKVCache):
        assert entry.offset.size == entry.left_padding.size
        batch_size = entry.offset.size
    else:
        batch_size = entry.keys.shape[0] if entry.keys is not None else 1
    values = mx.arange(batch_size * count * heads * 32, dtype=mx.float32).reshape(
        batch_size, heads, count, 32
    )
    entry.update_and_fetch(values, values)


def _request(model, value, *, backbone_heads=1, recurrent_shape=(1, 1), token_count=2):
    if recurrent_shape is not None and recurrent_shape[0] != 1:
        raise ValueError("cohort fixtures must enter through B=1 owners")
    if token_count < 0:
        raise ValueError("cohort fixtures require non-negative token counts")
    recurrent = ArraysCache(1)
    if recurrent_shape is not None:
        recurrent[0] = mx.zeros(recurrent_shape, dtype=mx.float32) + value
    backbone = [recurrent, KVCache()]
    mtp = [KVCache()]
    if token_count:
        _advance(backbone[1], token_count, heads=backbone_heads)
        _advance(mtp[0], token_count)
    request = NativeMTPRequestCache(model, backbone, mtp)
    request.retain(backbone_tokens=token_count, mtp_tokens=token_count)
    return request


def _empty_two_slot_request(model):
    recurrent = ArraysCache(2)
    request = NativeMTPRequestCache(model, [recurrent, KVCache()], [KVCache()])
    request.retain(backbone_tokens=0, mtp_tokens=0)
    return request


def _assert_valid_b1_owner(request):
    request.assert_aligned(
        backbone_tokens=request.state.backbone_tokens,
        mtp_tokens=request.state.mtp_tokens,
    )
    for entry in request.backbone + request.mtp:
        if not isinstance(entry, ArraysCache) and entry.keys is not None:
            assert entry.keys.shape[0] == 1


def test_cohort_rollback_restores_recurrent_values_and_kv_offsets():
    model = _Model()
    cohort = _NativeMTPCohortCache(model, [_request(model, 1), _request(model, 2)])
    before = cohort.backbone[0][0]
    backbone_keys = cohort.backbone[1].keys
    mtp_keys = cohort.mtp[0].keys
    backbone_entry = cohort.backbone[1]
    mtp_entry = cohort.mtp[0]

    cohort.checkpoint()
    _advance(cohort.backbone[1], 1)
    _advance(cohort.mtp[0], 1)
    cohort.backbone[0][0] = cohort.backbone[0][0] + 10
    cohort.rollback()
    mx.eval(before, cohort.backbone[0][0])

    assert cohort.size == 2
    assert cohort.backbone[1]._idx == 2
    assert cohort.mtp[0]._idx == 2
    assert cohort.backbone[1] is backbone_entry
    assert cohort.mtp[0] is mtp_entry
    assert mx.array_equal(cohort.backbone[1].keys[..., :2, :], backbone_keys).item()
    assert mx.array_equal(cohort.mtp[0].keys[..., :2, :], mtp_keys).item()
    assert mx.array_equal(cohort.backbone[0][0], before).item()


def test_cohort_split_join_filters_without_cache_aliasing():
    model = _Model()
    rows = [_request(model, 1), _request(model, 2)]
    cohort = _NativeMTPCohortCache(model, rows, uids=(11, 12))
    assert all(row.closed for row in rows)

    selected = cohort.split([0])
    assert cohort.size == selected.size == 1
    assert cohort.uids == (12,)
    assert selected.uids == (11,)
    assert cohort.backbone[1] is not selected.backbone[1]

    cohort.join(selected)
    assert cohort.size == 2
    with pytest.raises(RuntimeError, match="native_mtp_cohort_moved"):
        selected.bind_before_mutation()


def test_cohort_empty_batch_cache_preserves_merged_batch_width():
    model = _Model()
    rows = [
        _request(model, 1, token_count=0),
        _request(model, 2, token_count=0),
    ]
    for row in rows:
        _assert_valid_b1_owner(row)

    cohort = _NativeMTPCohortCache(model, rows)
    for entry in (cohort.backbone[1], cohort.mtp[0]):
        assert isinstance(entry, BatchKVCache)
        assert entry.keys is None
        assert entry.offset.size == entry.left_padding.size == cohort.size == 2
        assert _NativeMTPCohortCache._entry_schema(entry)[1] == cohort.size

    cohort.checkpoint()
    _advance(cohort.backbone[1], 1)
    _advance(cohort.mtp[0], 1)
    assert cohort.backbone[1].keys.shape[0] == cohort.mtp[0].keys.shape[0] == 2
    cohort.seal_after_mutation(
        _NativeMTPCohortMutationDelta(
            backbone=((), (1, 1)),
            mtp=((1, 1),),
        )
    )
    cohort.commit()


def test_cohort_allows_first_population_of_empty_recurrent_slots():
    model = _Model()
    rows = [
        _request(model, 1, recurrent_shape=None, token_count=0),
        _request(model, 2, recurrent_shape=None, token_count=0),
    ]
    for row in rows:
        _assert_valid_b1_owner(row)

    cohort = _NativeMTPCohortCache(model, rows)
    recurrent = cohort.backbone[0]
    assert recurrent.batch_size == cohort.size == 2
    assert recurrent[0] is None
    cohort.checkpoint()
    recurrent[0] = mx.zeros((2, 3), dtype=mx.float32)
    cohort.seal_after_mutation(
        _NativeMTPCohortMutationDelta(
            backbone=((), (0, 0)),
            mtp=((0, 0),),
        )
    )
    cohort.commit()
    cohort.bind_before_mutation()
    assert _NativeMTPCohortCache._entry_schema(recurrent)[2][0] == (
        2,
        2,
        (3,),
        mx.float32,
    )


def test_cohort_rejects_mismatched_batch_first_population_of_empty_arrays_slots():
    model = _Model()
    rows = [_empty_two_slot_request(model), _empty_two_slot_request(model)]
    for row in rows:
        _assert_valid_b1_owner(row)

    cohort = _NativeMTPCohortCache(model, rows)
    recurrent = cohort.backbone[0]
    assert recurrent.batch_size == cohort.size == 2
    assert recurrent.cache == [None, None]
    cohort.checkpoint()
    recurrent[0] = mx.zeros((2, 3), dtype=mx.float32)
    recurrent[1] = mx.zeros((1, 3), dtype=mx.float32)

    with pytest.raises(RuntimeError, match="unexpected_layout_change"):
        cohort.seal_after_mutation(
            _NativeMTPCohortMutationDelta(
                backbone=((), (0, 0)),
                mtp=((0, 0),),
            )
        )
    assert cohort.poisoned


@pytest.mark.parametrize("transition", ("partial", "mismatched_batch"))
def test_cohort_rejects_invalid_empty_kv_schema_transitions(transition):
    model = _Model()
    rows = [
        _request(model, 1, token_count=0),
        _request(model, 2, token_count=0),
    ]
    for row in rows:
        _assert_valid_b1_owner(row)

    cohort = _NativeMTPCohortCache(model, rows)
    cohort.checkpoint()
    entry = cohort.backbone[1]
    if transition == "partial":
        entry.keys = mx.zeros((2, 1, 1, 32), dtype=mx.float32)
    else:
        entry.keys = mx.zeros((1, 1, 1, 32), dtype=mx.float32)
        entry.values = mx.zeros((1, 1, 1, 32), dtype=mx.float32)

    with pytest.raises(RuntimeError, match="unexpected_layout_change"):
        cohort.seal_after_mutation(
            _NativeMTPCohortMutationDelta(
                backbone=((), (0, 0)),
                mtp=((0, 0),),
            )
        )
    assert cohort.poisoned


def test_cohort_refuses_quantized_row_cache_and_poisoned_rebinding():
    model = _Model()
    request = _request(model, 1)
    request.quantize(kv_bits=8, kv_group_size=32)
    with pytest.raises(RuntimeError, match="cache_layout_unsupported"):
        _NativeMTPCohortCache(model, [request])

    cohort = _NativeMTPCohortCache(model, [_request(model, 1)])
    cohort.backbone[1] = KVCache()
    with pytest.raises(RuntimeError, match="cache_binding_changed"):
        cohort.bind_before_mutation()
    assert cohort.poisoned


def test_cohort_partition_never_calls_row_extraction(monkeypatch):
    model = _Model()
    cohort = _NativeMTPCohortCache(model, [_request(model, 1), _request(model, 2)])

    with monkeypatch.context() as partition_patch:
        partition_patch.setattr(
            BatchKVCache,
            "extract",
            lambda *_: pytest.fail("cohort partition must not extract cache rows"),
        )
        partition_patch.setattr(
            ArraysCache,
            "extract",
            lambda *_: pytest.fail("cohort partition must not extract recurrent rows"),
        )
        partition_patch.setattr(
            mx,
            "eval",
            lambda *_: pytest.fail("cohort partition must not synchronize"),
        )
        partition_patch.setattr(
            mx,
            "contiguous",
            lambda *_: pytest.fail("cohort partition must not copy complete rows"),
        )
        cohort._partition((0,))
    selected = cohort.split([0])
    cohort.join(selected)
    assert cohort.size == 2
    source = inspect.getsource(_NativeMTPCohortCache._partition)
    assert ".item(" not in source
    assert "mx.eval" not in source
    source = inspect.getsource(_NativeMTPCohortCache._partition_entry)
    assert ".item(" not in source
    assert "mx.eval" not in source
    assert "mx.contiguous" not in source


def test_cohort_binding_tracks_batch_metadata_and_recurrent_size_bound():
    model = _Model()
    cohort = _NativeMTPCohortCache(model, [_request(model, 1)])
    cohort.backbone[1].offset = mx.array([7])
    with pytest.raises(RuntimeError, match="cache_binding_changed"):
        cohort.bind_before_mutation()

    recurrent = ArraysCache(1)
    recurrent[0] = mx.zeros((1, 65537), dtype=mx.float32)
    oversized = NativeMTPRequestCache(model, [recurrent, KVCache()], [KVCache()])
    with pytest.raises(RuntimeError, match="recurrent_state_too_large"):
        _NativeMTPCohortCache(model, [oversized])


def test_cohort_binding_rejects_arrays_metadata_and_entry_replacement():
    model = _Model()
    cohort = _NativeMTPCohortCache(model, [_request(model, 1)])
    cohort.backbone[0].left_padding = mx.array([0])
    cohort._binding = cohort._make_binding()
    cohort.backbone[0].left_padding = mx.array([1])
    with pytest.raises(RuntimeError, match="cache_binding_changed"):
        cohort.bind_before_mutation()

    cohort = _NativeMTPCohortCache(model, [_request(model, 1)])
    assert isinstance(cohort.backbone[1], BatchKVCache)
    replacement = BatchKVCache([0])
    replacement.offset = object()
    cohort.backbone[1] = replacement
    with pytest.raises(RuntimeError, match="cache_binding_changed"):
        cohort.bind_before_mutation()
    assert cohort.poisoned


def test_cohort_rejects_incompatible_kv_and_recurrent_storage_schemas():
    model = _Model()
    kv_a = _request(model, 1)
    kv_b = _request(model, 2, backbone_heads=2)
    _assert_valid_b1_owner(kv_a)
    _assert_valid_b1_owner(kv_b)
    with pytest.raises(TypeError, match="storage_schema_mismatch"):
        _NativeMTPCohortCache(model, [kv_a, kv_b])

    recurrent_a = _request(model, 1)
    recurrent_b = _request(model, 2, recurrent_shape=(1, 2))
    _assert_valid_b1_owner(recurrent_a)
    _assert_valid_b1_owner(recurrent_b)
    with pytest.raises(TypeError, match="storage_schema_mismatch"):
        _NativeMTPCohortCache(model, [recurrent_a, recurrent_b])


def test_cohort_seal_poison_on_recurrent_padding_or_lengths_drift():
    model = _Model()
    cohort = _NativeMTPCohortCache(model, [_request(model, 1)])
    cohort.backbone[0].left_padding = mx.array([0])
    cohort.backbone[0].lengths = mx.array([1])
    cohort._binding = cohort._make_binding()
    cohort.checkpoint()
    cohort.backbone[0].left_padding = mx.array([1])
    with pytest.raises(RuntimeError, match="arrays_metadata_changed"):
        cohort.seal_after_mutation(
            _NativeMTPCohortMutationDelta(backbone=((), (0,)), mtp=((0,),))
        )
    assert cohort.poisoned


def test_cohort_seal_requires_declared_layer_row_deltas_before_commit():
    model = _Model()
    cohort = _NativeMTPCohortCache(model, [_request(model, 1), _request(model, 2)])
    cohort.checkpoint()
    with pytest.raises(RuntimeError, match="mutation_not_sealed"):
        cohort.commit()
    _advance(cohort.backbone[1], 1)
    _advance(cohort.mtp[0], 1)
    cohort.seal_after_mutation(
        _NativeMTPCohortMutationDelta(
            backbone=((), (1, 1)),
            mtp=((1, 1),),
        )
    )
    cohort.commit()


def test_cohort_seal_allows_recurrent_replacement_and_kv_capacity_growth():
    model = _Model()
    cohort = _NativeMTPCohortCache(model, [_request(model, 1), _request(model, 2)])
    cohort.checkpoint()
    cohort.backbone[0][0] = cohort.backbone[0][0] + 3
    _advance(cohort.backbone[1], 300)
    _advance(cohort.mtp[0], 300)
    cohort.seal_after_mutation(
        _NativeMTPCohortMutationDelta(
            backbone=((), (300, 300)),
            mtp=((300, 300),),
        )
    )
    cohort.commit()


def test_cohort_source_consume_failure_closes_every_source_and_poison_cohort(
    monkeypatch,
):
    model = _Model()
    rows = [_request(model, 1), _request(model, 2)]
    original_finish = NativeMTPRequestCache.finish
    calls = []

    def fail_second(self, reason):
        calls.append(self)
        if len(calls) == 2:
            raise RuntimeError("finish failed")
        return original_finish(self, reason)

    monkeypatch.setattr(NativeMTPRequestCache, "finish", fail_second)
    with pytest.raises(RuntimeError, match="finish failed"):
        _NativeMTPCohortCache(model, rows)
    assert all(row.closed for row in rows)
