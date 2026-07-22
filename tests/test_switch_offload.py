# Copyright © 2024 Apple Inc.

"""Tests for opt-in disk-backed expert offloading on SwitchLinear /
QuantizedSwitchLinear (enable_offload).

Self-contained: the "disk" is an in-memory numpy fetch_fn returning the same
raw (np.ndarray, dtype_str) shape a real storage-backed fetch_fn returns, so no
model download or on-disk shard is needed. The core guarantee under test is
that an offloaded module produces output bit-identical to the unmodified
fully-resident path, on both a cold cache (every call misses) and a warm one
(repeated calls hit the LRU), across enough evictions that the dict cache and
the chunked gather are actually exercised.
"""
import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

import mlx_lm.models.switch_layers as sl
from mlx_lm.models.switch_layers import (
    SwitchGLU,
    SwitchLinear,
    _offload_data_nbytes,
    _offload_fetched_to_data,
)

INPUT_DIMS = 32
OUTPUT_DIMS = 48
NUM_EXPERTS = 16
RESIDENT_SLOTS = 3  # << NUM_EXPERTS, so most calls miss and force eviction


def _dense_fetch(weight):
    """fetch_fn(e) -> (np.ndarray, dtype_str) for one expert row of a dense
    (num_experts, out, in) weight. Built before enable_offload replaces the
    module's weight with a 1-row stand-in."""
    w = np.array(weight)

    def fetch(e):
        return (w[e], "F32")

    return fetch


def _quant_fetch(module):
    """fetch_fn(e) -> ((w,dt), (s,dt), (b,dt)|None) for one expert of a
    QuantizedSwitchLinear, mirroring the 3-part quantized fetch shape."""
    w = np.array(module.weight)
    s = np.array(module.scales)
    b = np.array(module.biases) if module.biases is not None else None

    def fetch(e):
        return (
            (w[e], "U32"),
            (s[e], "F32"),
            (b[e], "F32") if b is not None else None,
        )

    return fetch


def _random_indices(n_calls=6, tokens=5, top_k=1, seed=0):
    key = mx.random.key(seed)
    out = []
    for _ in range(n_calls):
        key, sub = mx.random.split(key)
        out.append(mx.random.randint(0, NUM_EXPERTS, (tokens, top_k), key=sub))
    return out


def _copy_params(dst, src, attrs):
    for a in attrs:
        if src.get(a) is not None:
            dst[a] = src[a]
    mx.eval(dst.parameters())


def _per_expert_bytes(fetch, quantized):
    return _offload_data_nbytes(_offload_fetched_to_data(fetch(0), quantized))


class TestSwitchOffload(unittest.TestCase):
    # --- Core bit-identity guarantee -------------------------------------

    def test_dense_offload_matches_baseline(self):
        baseline = SwitchLinear(INPUT_DIMS, OUTPUT_DIMS, NUM_EXPERTS, bias=False)
        mx.eval(baseline.parameters())

        offloaded = SwitchLinear(INPUT_DIMS, OUTPUT_DIMS, NUM_EXPERTS, bias=False)
        _copy_params(offloaded, baseline, ["weight"])
        offloaded.enable_offload(RESIDENT_SLOTS, _dense_fetch(baseline.weight))

        x = mx.random.normal((5, 1, 1, INPUT_DIMS))
        calls = _random_indices()

        for indices in calls:  # cold: first sighting of each expert misses
            self.assertTrue(mx.array_equal(baseline(x, indices), offloaded(x, indices)))
        for indices in calls:  # warm: replay hits the LRU and re-misses evicted ids
            self.assertTrue(mx.array_equal(baseline(x, indices), offloaded(x, indices)))

    def test_quantized_offload_matches_baseline(self):
        baseline = SwitchLinear(INPUT_DIMS, OUTPUT_DIMS, NUM_EXPERTS, bias=False).to_quantized(
            group_size=32, bits=4
        )
        mx.eval(baseline.parameters())

        offloaded = SwitchLinear(INPUT_DIMS, OUTPUT_DIMS, NUM_EXPERTS, bias=False).to_quantized(
            group_size=32, bits=4
        )
        _copy_params(offloaded, baseline, ["weight", "scales", "biases"])
        offloaded.enable_offload(RESIDENT_SLOTS, _quant_fetch(baseline))

        x = mx.random.normal((5, 1, 1, INPUT_DIMS))
        calls = _random_indices()

        for indices in calls:
            self.assertTrue(mx.array_equal(baseline(x, indices), offloaded(x, indices)))
        for indices in calls:
            self.assertTrue(mx.array_equal(baseline(x, indices), offloaded(x, indices)))

    def _offloaded_glu(self, baseline, hidden, max_stack_size=4):
        offloaded = SwitchGLU(INPUT_DIMS, hidden, NUM_EXPERTS)
        for name in ("gate_proj", "up_proj", "down_proj"):
            _copy_params(getattr(offloaded, name), getattr(baseline, name), ["weight"])
            getattr(offloaded, name).enable_offload(
                RESIDENT_SLOTS,
                _dense_fetch(getattr(baseline, name).weight),
                max_stack_size=max_stack_size,
            )
        return offloaded

    def test_switch_glu_decode_matches_baseline(self):
        """A decode-shaped SwitchGLU call routes to a single gather group and is
        bit-identical to the full-resident baseline (no sort, so no
        sorted_indices kernel-hint mismatch)."""
        hidden = 40
        baseline = SwitchGLU(INPUT_DIMS, hidden, NUM_EXPERTS)
        mx.eval(baseline.parameters())
        offloaded = self._offloaded_glu(baseline, hidden)

        x = mx.random.normal((1, INPUT_DIMS))
        indices = mx.random.randint(0, NUM_EXPERTS, (1, 2))  # size 2 < 64, no sort
        self.assertTrue(mx.array_equal(baseline(x, indices), offloaded(x, indices)))

    def test_switch_glu_compact_matches_mask(self):
        """Prefill across all three projections: SwitchGLU's own _gather_sort
        feeds sorted indices to the sublayers, so the compact gather path runs
        with a small max_stack_size forcing multiple groups. Compact must be
        bit-identical to the always-correct mask path. (Not compared to the
        full-resident SwitchGLU: it dispatches gather with sorted_indices=True,
        whose kernel hint is not bit-identical to the offload path's False.)"""
        hidden = 40
        baseline = SwitchGLU(INPUT_DIMS, hidden, NUM_EXPERTS)
        mx.eval(baseline.parameters())
        offloaded = self._offloaded_glu(baseline, hidden)

        x = mx.random.normal((16, INPUT_DIMS))
        indices = mx.random.randint(0, NUM_EXPERTS, (16, 8))  # 128 >= 64 -> sorted dispatch
        self.assertGreaterEqual(indices.size, 64)

        compact = offloaded(x, indices)  # compact path in all three sublayers

        orig = sl._offload_chunked_gather
        with mock.patch.object(
            sl,
            "_offload_chunked_gather",
            lambda m, xx, ii, g, sorted_indices=False: orig(m, xx, ii, g, sorted_indices=False),
        ):
            mask = offloaded(x, indices)  # same module forced onto the mask path

        self.assertEqual(compact.shape, mask.shape)
        self.assertTrue(mx.array_equal(compact, mask))

    # --- resident_bytes byte-budgeted residency --------------------------

    def test_resident_bytes_derives_slot_count_dense(self):
        module = SwitchLinear(INPUT_DIMS, OUTPUT_DIMS, NUM_EXPERTS, bias=False)
        mx.eval(module.parameters())
        fetch = _dense_fetch(module.weight)
        per_expert = _per_expert_bytes(fetch, quantized=False)

        module.enable_offload(resident_bytes=5 * per_expert, fetch_fn=fetch)
        self.assertEqual(module._offload_capacity, 5)

    def test_resident_bytes_derives_slot_count_quantized(self):
        """Per-expert footprint of a quantized entry is weight + scales +
        biases, so the byte budget must account for all three, not just the
        weight."""
        module = SwitchLinear(INPUT_DIMS, OUTPUT_DIMS, NUM_EXPERTS, bias=False).to_quantized(
            group_size=32, bits=4
        )
        mx.eval(module.parameters())
        fetch = _quant_fetch(module)
        per_expert = _per_expert_bytes(fetch, quantized=True)

        module.enable_offload(resident_bytes=4 * per_expert, fetch_fn=fetch)
        self.assertEqual(module._offload_capacity, 4)

    def test_resident_bytes_and_slots_takes_tighter(self):
        m0 = SwitchLinear(INPUT_DIMS, OUTPUT_DIMS, NUM_EXPERTS, bias=False)
        per_expert = _per_expert_bytes(_dense_fetch(m0.weight), quantized=False)

        # bytes budget (3) tighter than the slot count (8) -> 3 wins
        m1 = SwitchLinear(INPUT_DIMS, OUTPUT_DIMS, NUM_EXPERTS, bias=False)
        m1.enable_offload(
            resident_slots=8, fetch_fn=_dense_fetch(m1.weight), resident_bytes=3 * per_expert
        )
        self.assertEqual(m1._offload_capacity, 3)

        # slot count (2) tighter than the bytes budget (9) -> 2 wins
        m2 = SwitchLinear(INPUT_DIMS, OUTPUT_DIMS, NUM_EXPERTS, bias=False)
        m2.enable_offload(
            resident_slots=2, fetch_fn=_dense_fetch(m2.weight), resident_bytes=9 * per_expert
        )
        self.assertEqual(m2._offload_capacity, 2)

    def test_resident_bytes_result_still_bit_identical(self):
        """A byte-derived slot count runs the same offload path as an equivalent
        slot count -- derivation only picks the number."""
        baseline = SwitchLinear(INPUT_DIMS, OUTPUT_DIMS, NUM_EXPERTS, bias=False)
        mx.eval(baseline.parameters())

        offloaded = SwitchLinear(INPUT_DIMS, OUTPUT_DIMS, NUM_EXPERTS, bias=False)
        _copy_params(offloaded, baseline, ["weight"])
        fetch = _dense_fetch(baseline.weight)
        per_expert = _per_expert_bytes(fetch, quantized=False)
        offloaded.enable_offload(resident_bytes=RESIDENT_SLOTS * per_expert, fetch_fn=fetch)
        self.assertEqual(offloaded._offload_capacity, RESIDENT_SLOTS)

        x = mx.random.normal((5, 1, 1, INPUT_DIMS))
        for indices in _random_indices():
            self.assertTrue(mx.array_equal(baseline(x, indices), offloaded(x, indices)))

    # --- Guards and the unchanged default path ---------------------------

    def test_missing_fetch_fn_raises(self):
        module = SwitchLinear(INPUT_DIMS, OUTPUT_DIMS, NUM_EXPERTS, bias=False)
        with self.assertRaises(ValueError):
            module.enable_offload(resident_slots=RESIDENT_SLOTS)

    def test_no_budget_raises(self):
        module = SwitchLinear(INPUT_DIMS, OUTPUT_DIMS, NUM_EXPERTS, bias=False)
        with self.assertRaises(ValueError):
            module.enable_offload(fetch_fn=_dense_fetch(module.weight))

    def test_noop_when_resident_slots_covers_all_experts(self):
        module = SwitchLinear(INPUT_DIMS, OUTPUT_DIMS, NUM_EXPERTS, bias=False)

        def _unused_fetch(e):  # must never be called on the no-op path
            raise AssertionError("fetch_fn called on a no-op enable_offload")

        module.enable_offload(NUM_EXPERTS, _unused_fetch)
        self.assertFalse(hasattr(module, "_offload_fetch"))
        # Still the plain resident path: full weight, no stand-in, no offload attrs.
        self.assertEqual(module.num_experts, NUM_EXPERTS)
        self.assertEqual(module.weight.shape[0], NUM_EXPERTS)


if __name__ == "__main__":
    unittest.main()
