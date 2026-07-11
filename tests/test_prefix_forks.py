# Copyright © 2026 Apple Inc.

import copy
import gc
import os
import tempfile
import unittest

import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.models.cache import (
    KVCache,
    RotatingKVCache,
    make_prompt_cache,
    save_prompt_cache,
    trim_prompt_cache,
)
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

    def test_trim_within_tail_and_raises_beyond(self):
        pk, pv = _random_kv(64)
        fork = ForkedKVCache(pk, pv)
        k, v = _random_kv(10)
        fork.update_and_fetch(k, v)
        self.assertEqual(fork.offset, 74)
        self.assertTrue(fork.is_trimmable())
        # Rollback within the private tail works like KVCache
        self.assertEqual(fork.trim(4), 4)
        self.assertEqual(fork.offset, 70)
        # Trimming into the frozen prefix must FAIL LOUDLY (F1): callers of
        # trim_prompt_cache ignore the return value, so a silent clamp would
        # leave generation running against skewed KV.
        with self.assertRaises(ValueError):
            fork.trim(30)
        self.assertEqual(fork.offset, 70)  # untouched by the failed trim
        self.assertEqual(fork.trim(0), 0)

    def test_trim_prompt_cache_fails_loudly_not_skewed(self):
        # F1 regression (reviewer's exact scenario): a fork inside the
        # trim-longer path of a request-level cache. trim_prompt_cache used
        # to under-trim silently (offset skew); now it raises.
        tokens = list(range(300))
        registry = PrefixForkRegistry()
        registry.freeze("m", tokens, _prime_cache(tokens))
        forks, _ = registry.fetch_fork("m", tokens)
        for f in forks:
            f.update_and_fetch(*_random_kv(10))
        with self.assertRaises(ValueError):
            trim_prompt_cache(forks, 50)  # 50 > tail of 10

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

    def test_empty_tail_state_is_sealed(self):
        # F2 regression: .state with an EMPTY tail must never hand out the
        # raw frozen array objects — a consumer __setitem__ would corrupt
        # the snapshot and every sibling fork.
        tokens = list(range(32))
        registry = PrefixForkRegistry()
        cid = registry.freeze("m", tokens, _prime_cache(tokens))
        snapshot = registry.get(cid)
        fork_a = registry.fetch_fork("m", tokens)[0]
        fork_b = registry.fetch_fork("m", tokens)[0]
        before_k = mx.array(snapshot.keys[0])
        before_v = mx.array(snapshot.values[0])
        mx.eval(before_k, before_v)

        k, v = fork_a[0].state  # tail empty
        self.assertIsNot(k, snapshot.keys[0])
        k[..., 0, :] = 999.0  # hostile consumer edits .state in place
        v[..., 0, :] = -999.0
        mx.eval(k, v)

        self.assertTrue(mx.array_equal(snapshot.keys[0], before_k))
        self.assertTrue(mx.array_equal(snapshot.values[0], before_v))
        kb, vb = fork_b[0].state  # sibling unaffected
        self.assertTrue(mx.array_equal(kb, before_k))
        self.assertTrue(mx.array_equal(vb, before_v))

    def test_snapshot_init_seals_caller_arrays(self):
        # F2 (defensive): a caller that retains its arrays and later mutates
        # them via __setitem__ must not reach the snapshot's stored nodes.
        pk, pv = _random_kv(16)
        before = mx.array(pk)
        mx.eval(before)
        snapshot = FrozenPrefixSnapshot("cid", "m", tuple(range(16)), [pk], [pv])
        fork = ForkedKVCache(pk, pv)
        pk[..., 0, :] = 123.0  # caller scribbles its own reference
        mx.eval(pk)
        self.assertTrue(mx.array_equal(snapshot.keys[0], before))
        self.assertTrue(mx.array_equal(fork.state[0], before))

    def test_meta_state_and_save_refuse(self):
        # N4: save_prompt_cache would write a file load_prompt_cache cannot
        # reconstruct (class lookup is models/cache.py globals); refuse loudly.
        pk, pv = _random_kv(8)
        forks = [ForkedKVCache(pk, pv)]
        with self.assertRaises(ValueError):
            _ = forks[0].meta_state
        path = os.path.join(tempfile.mkdtemp(), "fork.safetensors")
        with self.assertRaises(ValueError):
            save_prompt_cache(path, forks)

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

    def test_model_key_must_be_str(self):
        # B1 regression: id()-derived and object keys are forbidden. An
        # in-place weight swap keeps the same object, and CPython recycles
        # addresses of dead objects — both turned into stale-KV false HITs.
        registry = PrefixForkRegistry()
        tokens = list(range(8))
        cache = _prime_cache(tokens)

        class FakeModel:
            pass

        for bad in (FakeModel(), ("path", "adapter", None), 42, None, b"key"):
            with self.assertRaises(TypeError):
                registry.freeze(bad, tokens, cache)
            with self.assertRaises(TypeError):
                registry.fetch_fork(bad, tokens)
            with self.assertRaises(TypeError):
                compute_cid(bad, tokens)
        self.assertEqual(len(registry), 0)

    def test_distinct_string_keys_never_collide(self):
        registry = PrefixForkRegistry()
        tokens = list(range(16))
        mx.random.seed(11)
        cache_a = _prime_cache(tokens)
        mx.random.seed(12)
        cache_b = _prime_cache(tokens)
        cid_a = registry.freeze("org/model@rev1", tokens, cache_a)
        cid_b = registry.freeze("org/model@rev2", tokens, cache_b)
        self.assertNotEqual(cid_a, cid_b)
        self.assertEqual(len(registry), 2)
        got_a = registry.fetch_fork("org/model@rev1", tokens)[0]
        got_b = registry.fetch_fork("org/model@rev2", tokens)[0]
        self.assertTrue(mx.array_equal(got_a[0].state[0], cache_a[0].state[0]))
        self.assertTrue(mx.array_equal(got_b[0].state[0], cache_b[0].state[0]))

    def test_key_contract_weight_swap_misses(self):
        # B1 contract, documented: the key MUST change when the weights
        # change (adapter load). The caller changing the key converts what
        # was a deterministic stale-KV HIT into a safe MISS.
        registry = PrefixForkRegistry()
        tokens = list(range(24))
        registry.freeze("org/model@base", tokens, _prime_cache(tokens))
        forks, remaining = registry.fetch_fork("org/model@base+adapterX", tokens)
        self.assertIsNone(forks)
        self.assertEqual(remaining, tokens)

    def test_dedup_mismatch_beyond_layer0_raises(self):
        """A layer-1-only KV difference (layer 0 bit-identical — the
        realistic LoRA-that-skips-early-layers shape) must be caught by
        the all-layer dedup spot-check."""
        registry = PrefixForkRegistry()
        tokens = list(range(40))
        mx.random.seed(31)
        base = _prime_cache(tokens)
        registry.freeze("m@v1", tokens, base)

        # Rebuild with layer 0 bit-identical and layer 1 perturbed only
        mx.random.seed(31)
        altered = _prime_cache(tokens)  # same seed -> same content
        k1, v1 = altered[1].state
        k1_mod = mx.array(k1)
        k1_mod[..., -1, :] = 123.0
        altered[1].keys[..., : altered[1].offset, :] = k1_mod
        mx.eval(altered[1].keys)
        with self.assertRaises(ValueError):
            registry.freeze("m@v1", tokens, altered)

    def test_dedup_content_mismatch_raises(self):
        # F3 regression: same key + tokens but DIFFERENT KV content (weights
        # changed under a stale key) must raise, not silently dedup.
        registry = PrefixForkRegistry()
        tokens = list(range(32))
        mx.random.seed(21)
        registry.freeze("org/model@base", tokens, _prime_cache(tokens))
        mx.random.seed(22)
        post_adapter = _prime_cache(tokens)
        with self.assertRaises(ValueError):
            registry.freeze("org/model@base", tokens, post_adapter)

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
        cache_a = _prime_cache(tokens)
        cache_b = copy.deepcopy(cache_a)  # same content, different arrays

        cid_a = registry.freeze("m", tokens, cache_a)
        nbytes = registry.nbytes
        cid_b = registry.freeze("m", tokens, cache_b)
        # Same content (model key, tokens, KV) -> same cid, ONE snapshot,
        # bytes counted once
        self.assertEqual(cid_a, cid_b)
        self.assertEqual(len(registry), 1)
        self.assertEqual(registry.nbytes, nbytes)

        # Different tokens or model key -> different cid
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

        # Unknown model key / tokens: miss returns the tokens unchanged
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

    def test_pinned_nbytes_tracks_live_forks(self):
        # F4 regression: nbytes drops on invalidate while live forks still
        # pin the snapshot's memory. pinned_nbytes reports that residency.
        registry = PrefixForkRegistry()
        tokens = list(range(400))
        cid = registry.freeze("m", tokens, _prime_cache(tokens))
        snap_bytes = registry.get(cid).nbytes
        self.assertEqual(registry.nbytes, snap_bytes)
        self.assertEqual(registry.pinned_nbytes, 0)  # no forks yet

        forks, _ = registry.fetch_fork("m", tokens)
        self.assertEqual(registry.pinned_nbytes, snap_bytes)

        registry.invalidate(cid)
        # Discoverability gone, residency not: max_bytes bounds the
        # discoverable set only.
        self.assertEqual(registry.nbytes, 0)
        self.assertEqual(registry.pinned_nbytes, snap_bytes)

        del forks
        gc.collect()
        self.assertEqual(registry.pinned_nbytes, 0)


class TestPrefixForksWithModel(unittest.TestCase):
    """End-to-end correctness against the real model used by the prompt-cache
    suite. Bar: a fork must be indistinguishable (greedy, token-exact) from a
    full deepcopy of the same cache, while never touching the parent."""

    # The stable weight-identity string the registry requires (see the
    # MODEL-KEY CONTRACT in prefix_forks.py).
    MODEL_KEY = HF_MODEL_PATH + "@main"

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = load(HF_MODEL_PATH)
        text = (
            "The lighthouse keeper counted ships every morning. " * 40
            + "One day the fog rolled in and"
        )
        cls.tokens = cls.tokenizer.encode(text)
        assert len(cls.tokens) > 300  # prefix must cross the step=256 boundary
        # Multi-token suffix: N>1 prefill through the fork exercises the
        # causal mask / position handling (a flipped prefix/tail concat is
        # invisible to N=1 decode, which is permutation-invariant over keys).
        cls.prefix = cls.tokens[:-4]
        cls.suffix = cls.tokens[-4:]
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
        cid = registry.freeze(self.MODEL_KEY, self.prefix, self.cache)
        self.assertIsNotNone(cid)
        snapshot = registry.get(cid)

        # Fork creation must be O(tail): no prefix-sized materialization.
        # Measure AFTER evaluating everything the fork exposes, so a lazy
        # O(prefix) copy cannot hide behind deferred execution.
        mx.eval(*(a for s in [snapshot] for a in s.keys + s.values))
        base_mem = mx.get_active_memory()
        forks, remaining = registry.fetch_fork(self.MODEL_KEY, self.tokens)
        self.assertIsNotNone(forks)
        self.assertEqual(remaining, self.suffix)
        mx.eval(*(a for f in forks for a in f.state))
        fork_mem = mx.get_active_memory() - base_mem
        self.assertLess(fork_mem, max(1 << 20, snapshot.nbytes // 20))
        self.assertEqual(sum(f.nbytes for f in forks), 0)

        # Snapshot state before anyone generates
        before = [mx.array(a) for a in snapshot.keys + snapshot.values]
        mx.eval(*before)

        # (1) Token-exact equivalence with a full deepcopy continuation,
        # through an N>1 prefill (position-sensitive) then 110 decode steps
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
        cid = registry.freeze(self.MODEL_KEY, self.prefix, self.cache)
        snapshot = registry.get(cid)

        suffix_a = self.suffix
        suffix_b = self.tokenizer.encode(" thunder shook the")

        forks_a, _ = registry.fetch_fork(self.MODEL_KEY, self.prefix + suffix_a)
        forks_b, _ = registry.fetch_fork(self.MODEL_KEY, self.prefix + suffix_b)

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

        # And the shared parent still froze exactly the prefix, with both
        # forks referencing the SAME snapshot (no per-fork copies)
        self.assertEqual(len(snapshot), len(self.prefix))
        for f_a, f_b in zip(forks_a, forks_b):
            self.assertIs(f_a._snapshot, f_b._snapshot)
            self.assertIs(f_a._snapshot, snapshot)


if __name__ == "__main__":
    unittest.main()
