# Copyright © 2026 Apple Inc.

import copy
import unittest

import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
from mlx_lm.prefix_forks import (
    ForkedKVCache,
    FrozenPrefixSnapshot,
    PrefixForkRegistry,
    compute_cid,
)
from mlx_lm.utils import load

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


def _random_kv(n, dims=(1, 2, 8)):
    B, H, D = dims
    k = mx.random.normal((B, H, n, D)).astype(mx.float16)
    v = mx.random.normal((B, H, n, D)).astype(mx.float16)
    mx.eval(k, v)
    return k, v


def _prime_cache(tokens, n_layers=2, dims=(1, 2, 8)):
    """A KVCache stack filled with len(tokens) of random KV."""
    cache = [KVCache() for _ in range(n_layers)]
    for c in cache:
        k, v = _random_kv(len(tokens), dims)
        c.update_and_fetch(k, v)
    mx.eval(*(x for c in cache for x in c.state))
    return cache


class TestForkedKVCache(unittest.TestCase):
    """Pure-array tests: a ForkedKVCache must be observationally identical to
    a plain KVCache fed the same KV stream."""

    def test_update_and_fetch_matches_kvcache(self):
        mx.random.seed(0)
        prefix_len = 300  # crosses the step=256 boundary and is not a multiple
        full = KVCache()
        pk, pv = _random_kv(prefix_len)
        full.update_and_fetch(pk, pv)

        snapshot = FrozenPrefixSnapshot(
            "cid", "m", tuple(range(prefix_len)), [mx.array(pk)], [mx.array(pv)]
        )
        fork = snapshot.fork()[0]

        # Mixed prefill/decode chunks; total tail crosses another step boundary
        for n in (1, 7, 1, 1, 256, 1, 40):
            k, v = _random_kv(n)
            fk, fv = full.update_and_fetch(k, v)
            gk, gv = fork.update_and_fetch(k, v)
            self.assertEqual(fork.offset, full.offset)
            self.assertEqual(fork.size(), full.size())
            self.assertTrue(mx.array_equal(fk, gk))
            self.assertTrue(mx.array_equal(fv, gv))

        # state is the joined exact-length view
        sk, sv = fork.state
        self.assertTrue(mx.array_equal(sk, full.state[0]))
        self.assertTrue(mx.array_equal(sv, full.state[1]))

        # detaching gives an equal, independent KVCache
        detached = fork.to_kv_cache()
        self.assertIsInstance(detached, KVCache)
        self.assertEqual(detached.offset, full.offset)
        self.assertTrue(mx.array_equal(detached.state[0], full.state[0]))

    def test_trim_is_clamped_to_tail(self):
        pk, pv = _random_kv(64)
        fork = ForkedKVCache(pk, pv)
        k, v = _random_kv(10)
        fork.update_and_fetch(k, v)
        self.assertEqual(fork.offset, 74)
        # Trim deeper than the tail: only the 10 private tokens go
        self.assertTrue(fork.is_trimmable())
        self.assertEqual(fork.trim(30), 10)
        self.assertEqual(fork.offset, 64)
        self.assertEqual(fork.trim(5), 0)

    def test_nbytes_counts_private_tail_only(self):
        pk, pv = _random_kv(128)
        fork = ForkedKVCache(pk, pv)
        self.assertEqual(fork.nbytes, 0)  # fork creation costs no private bytes
        self.assertEqual(fork.shared_nbytes, pk.nbytes + pv.nbytes)
        k, v = _random_kv(4)
        fork.update_and_fetch(k, v)
        self.assertEqual(fork.nbytes, fork.tail.nbytes)
        self.assertGreater(fork.nbytes, 0)

    def test_make_mask_matches_kvcache(self):
        pk, pv = _random_kv(32)
        fork = ForkedKVCache(pk, pv)
        k, v = _random_kv(8)
        fork.update_and_fetch(k, v)
        ref = KVCache()
        rk, rv = _random_kv(40)
        ref.update_and_fetch(rk, rv)
        for N, kwargs in [
            (1, dict(return_array=False, window_size=None)),
            (5, dict(return_array=False, window_size=None)),
            (5, dict(return_array=True, window_size=None)),
            (5, dict(return_array=False, window_size=16)),
        ]:
            a = fork.make_mask(N, **kwargs)
            b = ref.make_mask(N, **kwargs)
            if isinstance(b, (str, type(None))):
                self.assertEqual(a, b)
            else:
                self.assertTrue(mx.array_equal(a, b))

    def test_state_is_read_only(self):
        pk, pv = _random_kv(8)
        fork = ForkedKVCache(pk, pv)
        with self.assertRaises(ValueError):
            fork.state = (pk, pv)

    def test_parent_arrays_untouched_by_tail_writes(self):
        prefix_len = 260
        pk, pv = _random_kv(prefix_len)
        before_k, before_v = mx.array(pk), mx.array(pv)
        mx.eval(before_k, before_v)
        fork = ForkedKVCache(pk, pv)
        for _ in range(300):  # force several tail buffer growths
            k, v = _random_kv(1)
            fork.update_and_fetch(k, v)
        mx.eval(*fork.state)
        self.assertTrue(mx.array_equal(pk, before_k))
        self.assertTrue(mx.array_equal(pv, before_v))


class TestPrefixForkRegistry(unittest.TestCase):

    def test_freeze_rejects_unsupported_caches(self):
        registry = PrefixForkRegistry()
        tokens = list(range(16))
        # Wrong cache type
        rot = [RotatingKVCache(max_size=8)]
        k, v = _random_kv(16)
        rot[0].update_and_fetch(k, v)
        self.assertIsNone(registry.freeze("m", tokens, rot))
        # Offset mismatch
        cache = _prime_cache(tokens)
        self.assertIsNone(registry.freeze("m", tokens + [1, 2], cache))
        # Empty
        self.assertIsNone(registry.freeze("m", [], []))
        self.assertEqual(len(registry), 0)

    def test_cid_dedup(self):
        registry = PrefixForkRegistry()
        tokens = list(range(32))
        mx.random.seed(3)
        cache_a = _prime_cache(tokens)
        mx.random.seed(4)
        cache_b = _prime_cache(tokens)

        cid_a = registry.freeze("m", tokens, cache_a)
        nbytes = registry.nbytes
        cid_b = registry.freeze("m", tokens, cache_b)
        # Same content (model, tokens) -> same cid, ONE snapshot, bytes once
        self.assertEqual(cid_a, cid_b)
        self.assertEqual(len(registry), 1)
        self.assertEqual(registry.nbytes, nbytes)

        # Different tokens or model namespace -> different cid
        self.assertNotEqual(compute_cid("m", tokens), compute_cid("m", tokens[:-1]))
        self.assertNotEqual(compute_cid("m", tokens), compute_cid("m2", tokens))

    def test_fetch_longest_prefix_and_false_miss(self):
        registry = PrefixForkRegistry()
        short, long_ = list(range(8)), list(range(20))
        registry.freeze("m", short, _prime_cache(short))
        cid_long = registry.freeze("m", long_, _prime_cache(long_))

        request = long_ + [99, 100]
        forks, remaining = registry.fetch_fork("m", request)
        self.assertIsNotNone(forks)
        self.assertEqual(remaining, [99, 100])  # longest match won
        self.assertEqual(forks[0].prefix_length, len(long_))

        # A snapshot that only EXTENDS the request is a deliberate miss
        forks2, remaining2 = registry.fetch_fork("m", long_[:15])
        self.assertIsNotNone(forks2)  # falls back to the shorter snapshot
        self.assertEqual(forks2[0].prefix_length, len(short))

        # Unknown model / tokens: miss returns the tokens unchanged
        forks3, remaining3 = registry.fetch_fork("other", request)
        self.assertIsNone(forks3)
        self.assertEqual(remaining3, request)

        # Invalidation errs toward false MISS: fetch misses afterwards...
        self.assertTrue(registry.invalidate(cid_long))
        self.assertFalse(registry.invalidate(cid_long))
        forks4, _ = registry.fetch_fork("m", request)
        self.assertEqual(forks4[0].prefix_length, len(short))
        # ...but live forks made earlier still hold their arrays
        k, v = _random_kv(1)
        out_k, _ = forks[0].update_and_fetch(k, v)
        mx.eval(out_k)
        self.assertEqual(out_k.shape[2], len(long_) + 1)

    def test_lru_eviction(self):
        registry = PrefixForkRegistry(max_snapshots=2)
        toks = [list(range(i, i + 8)) for i in (0, 100, 200)]
        cids = [registry.freeze("m", t, _prime_cache(t)) for t in toks]
        self.assertEqual(len(registry), 2)
        # Oldest evicted -> miss; newer ones still hit
        self.assertIsNone(registry.get(cids[0]))
        self.assertIsNone(registry.fetch_fork("m", toks[0])[0])
        self.assertIsNotNone(registry.fetch_fork("m", toks[1])[0])
        self.assertIsNotNone(registry.fetch_fork("m", toks[2])[0])


class TestPrefixForksWithModel(unittest.TestCase):
    """End-to-end correctness against the real model used by the prompt-cache
    suite. Bar: a fork must be indistinguishable (greedy, token-exact) from a
    full deepcopy of the same cache, while never touching the parent."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = load(HF_MODEL_PATH)
        text = (
            "The lighthouse keeper counted ships every morning. " * 40
            + "One day the fog rolled in and"
        )
        cls.tokens = cls.tokenizer.encode(text)
        assert len(cls.tokens) > 300  # prefix must cross the step=256 boundary
        cls.prefix = cls.tokens[:-1]
        cls.cache = make_prompt_cache(cls.model)
        cls.model(mx.array(cls.prefix)[None], cache=cls.cache)
        mx.eval(*(x for c in cls.cache for x in c.state))

    def _greedy(self, tokens, cache, n):
        out = []
        for token, _ in generate_step(
            mx.array(tokens), self.model, prompt_cache=cache, max_tokens=n
        ):
            out.append(int(token))
        return out

    def test_fork_generation_matches_deepcopy(self):
        registry = PrefixForkRegistry()
        cid = registry.freeze(self.model, self.prefix, self.cache)
        self.assertIsNotNone(cid)
        snapshot = registry.get(cid)

        # Fork creation must be O(tail): no prefix-sized materialization
        mx.eval(*(a for s in [snapshot] for a in s.keys + s.values))
        base_mem = mx.get_active_memory()
        forks, remaining = registry.fetch_fork(self.model, self.tokens)
        fork_mem = mx.get_active_memory() - base_mem
        self.assertIsNotNone(forks)
        self.assertEqual(remaining, [self.tokens[-1]])
        self.assertLess(fork_mem, max(1 << 20, snapshot.nbytes // 20))
        self.assertEqual(sum(f.nbytes for f in forks), 0)

        # Snapshot state before anyone generates
        before = [mx.array(a) for a in snapshot.keys + snapshot.values]
        mx.eval(*before)

        # (1) Token-exact equivalence with a full deepcopy continuation
        ref_cache = copy.deepcopy(self.cache)
        ref = self._greedy(remaining, ref_cache, 110)
        got = self._greedy(remaining, forks, 110)
        self.assertEqual(ref, got)

        # (2) Parent immutability after 110 generated tokens
        for arr, prev in zip(snapshot.keys + snapshot.values, before):
            self.assertTrue(mx.array_equal(arr, prev))

        # The original primed cache is also untouched (freeze copied it)
        self.assertEqual(self.cache[0].offset, len(self.prefix))

    def test_sibling_forks_do_not_contaminate(self):
        registry = PrefixForkRegistry()
        cid = registry.freeze(self.model, self.prefix, self.cache)
        snapshot = registry.get(cid)

        suffix_a = [self.tokens[-1]]
        suffix_b = self.tokenizer.encode(" thunder shook the")

        forks_a, _ = registry.fetch_fork(self.model, self.prefix + suffix_a)
        forks_b, _ = registry.fetch_fork(self.model, self.prefix + suffix_b)

        # Independent single-user references for both continuations
        ref_a = self._greedy(suffix_a, copy.deepcopy(self.cache), 40)
        ref_b = self._greedy(suffix_b, copy.deepcopy(self.cache), 40)

        # Generate on A first, then B: if A's writes leaked into the shared
        # prefix (or into B), B would diverge from its clean reference.
        got_a = self._greedy(suffix_a, forks_a, 40)
        got_b = self._greedy(suffix_b, forks_b, 40)
        self.assertEqual(ref_a, got_a)
        self.assertEqual(ref_b, got_b)
        self.assertNotEqual(got_a, got_b)  # sanity: they really diverged

        # And the shared parent still froze exactly the prefix
        self.assertEqual(len(snapshot), len(self.prefix))
        for f_a, f_b in zip(forks_a, forks_b):
            self.assertIs(f_a._prefix_keys, f_b._prefix_keys)  # truly shared


if __name__ == "__main__":
    unittest.main()
