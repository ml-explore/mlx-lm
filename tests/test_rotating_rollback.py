# Copyright © 2026 Apple Inc.

"""Exact speculative rollback for RotatingKVCache (sliding-window KV).

The sibling of the ArraysCache (GDN) exact-replay rollback: once a rotating
cache wraps (offset >= max_size) it is not directly trimmable, so speculative
decoding needs an exact rollback. These tests pin the key property —
bit-exact tail-independence: two verify chunks that share an m-token prefix
but have different tails must leave IDENTICAL windows after rolling back to m
(any off-by-one in the restore/re-append leaks the tail).
"""
import unittest

import mlx.core as mx
from mlx_lm.models.cache import RotatingKVCache

W = 4  # tiny window so the cache wraps
H, D = 2, 8


def _kv(n, seed):
    mx.random.seed(seed)
    return (mx.random.normal((1, H, n, D)), mx.random.normal((1, H, n, D)))


def _prefill(cache, n, seed=0):
    k, v = _kv(n, seed)
    cache.update_and_fetch(k, v)
    return k, v


def _window(cache):
    return (
        mx.contiguous(cache._temporal_order(cache.keys)),
        mx.contiguous(cache._temporal_order(cache.values)),
    )


class TestRotatingRollback(unittest.TestCase):
    def test_tail_independence(self):
        # Reference: prefill P (wraps), then append only the shared prefix m.
        P, m, tail = 6, 3, 3
        pk, pv = _kv(m, seed=1)  # shared prefix K/V

        ref = RotatingKVCache(max_size=W)
        _prefill(ref, P, seed=9)
        ref.update_and_fetch(pk, pv)
        rk, rv = _window(ref)

        for tail_seed in (2, 3):  # two DIFFERENT tails
            tk, tv = _kv(tail, seed=tail_seed)
            c = RotatingKVCache(max_size=W)
            _prefill(c, P, seed=9)
            c.start_speculation()
            # one verify forward of [prefix ++ tail]
            c.update_and_fetch(
                mx.concatenate([pk, tk], axis=2),
                mx.concatenate([pv, tv], axis=2),
            )
            c.trim(tail)  # roll back to the m-token prefix
            wk, wv = _window(c)
            self.assertTrue(mx.allclose(wk, rk, atol=1e-5).item(),
                            f"keys leak tail (seed {tail_seed})")
            self.assertTrue(mx.allclose(wv, rv, atol=1e-5).item(),
                            f"values leak tail (seed {tail_seed})")
            self.assertEqual(c.offset, ref.offset)

    def test_trim_to_various_prefixes(self):
        P, block = 5, 4
        c = RotatingKVCache(max_size=W)
        _prefill(c, P, seed=7)
        bk, bv = _kv(block, seed=8)
        for keep in range(block + 1):
            ref = RotatingKVCache(max_size=W)
            _prefill(ref, P, seed=7)
            if keep > 0:
                ref.update_and_fetch(bk[..., :keep, :], bv[..., :keep, :])
            rk, rv = _window(ref)

            t = RotatingKVCache(max_size=W)
            _prefill(t, P, seed=7)
            t.start_speculation()
            t.update_and_fetch(bk, bv)
            t.trim(block - keep)
            wk, wv = _window(t)
            self.assertTrue(mx.allclose(wk, rk, atol=1e-5).item(),
                            f"keys mismatch at keep={keep}")
            self.assertTrue(mx.allclose(wv, rv, atol=1e-5).item(),
                            f"values mismatch at keep={keep}")
            self.assertEqual(t.offset, ref.offset, f"offset at keep={keep}")

    def test_stacked_draft_records(self):
        # Mimic a draft: several single-token forwards, then roll them all back.
        P = 5
        c = RotatingKVCache(max_size=W)
        _prefill(c, P, seed=4)
        base_k, base_v = _window(c)
        base_off = c.offset
        c.start_speculation()
        for i in range(3):
            k, v = _kv(1, seed=100 + i)
            c.update_and_fetch(k, v)
        c.trim(3)  # rewind all three draft tokens
        wk, wv = _window(c)
        self.assertTrue(mx.allclose(wk, base_k, atol=1e-5).item())
        self.assertTrue(mx.allclose(wv, base_v, atol=1e-5).item())
        self.assertEqual(c.offset, base_off)

    def test_not_speculating_uses_plain_trim(self):
        c = RotatingKVCache(max_size=64)
        _prefill(c, 10, seed=0)  # offset < max_size -> directly trimmable
        self.assertTrue(c.is_trimmable())
        c.trim(3)
        self.assertEqual(c.offset, 7)


if __name__ == "__main__":
    unittest.main()
