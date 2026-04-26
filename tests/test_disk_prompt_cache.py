# Copyright © 2026 Apple Inc.

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from mlx_lm.disk_prompt_cache import compute_model_id, hash_tokens


class TestHashTokens(unittest.TestCase):

    def test_basic_deterministic(self):
        h1 = hash_tokens([1, 2, 3])
        h2 = hash_tokens([1, 2, 3])
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)
        self.assertTrue(all(c in "0123456789abcdef" for c in h1))

    def test_order_matters(self):
        self.assertNotEqual(hash_tokens([1, 2, 3]), hash_tokens([3, 2, 1]))

    def test_empty(self):
        h = hash_tokens([])
        self.assertEqual(len(h), 16)

    def test_large(self):
        h = hash_tokens(list(range(50_000)))
        self.assertEqual(len(h), 16)


class TestComputeModelId(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.tmpdir.name) / "model"
        self.model_path.mkdir()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write(self, name, content):
        (self.model_path / name).write_text(content)

    def test_deterministic(self):
        self._write("model.safetensors.index.json", '{"weight_map": {}}')
        self._write("tokenizer.json", '{"model": "test"}')
        a = compute_model_id(self.model_path)
        b = compute_model_id(self.model_path)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 16)

    def test_changes_with_weights(self):
        self._write("model.safetensors.index.json", '{"weight_map": {}}')
        self._write("tokenizer.json", '{"model": "test"}')
        a = compute_model_id(self.model_path)
        self._write("model.safetensors.index.json", '{"weight_map": {"x": "y"}}')
        b = compute_model_id(self.model_path)
        self.assertNotEqual(a, b)

    def test_changes_with_tokenizer(self):
        self._write("model.safetensors.index.json", '{"weight_map": {}}')
        self._write("tokenizer.json", '{"v": 1}')
        a = compute_model_id(self.model_path)
        self._write("tokenizer.json", '{"v": 2}')
        b = compute_model_id(self.model_path)
        self.assertNotEqual(a, b)

    def test_missing_optional_files_handled(self):
        # Only weights index present; tokenizer and chat template missing.
        # Should still produce a valid id, not raise.
        self._write("model.safetensors.index.json", '{"weight_map": {}}')
        h = compute_model_id(self.model_path)
        self.assertEqual(len(h), 16)


from mlx_lm.disk_prompt_cache import (
    DiskCacheLockError,
    acquire_disk_dir_lock,
)


class TestDiskDirLock(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_lock_acquires(self):
        with acquire_disk_dir_lock(self.dir) as fd:
            self.assertIsNotNone(fd)
            # Lock file exists
            self.assertTrue((self.dir / ".lock").exists())
        # After release we can re-acquire
        with acquire_disk_dir_lock(self.dir):
            pass

    def test_double_lock_raises(self):
        with acquire_disk_dir_lock(self.dir):
            with self.assertRaises(DiskCacheLockError) as cm:
                with acquire_disk_dir_lock(self.dir):
                    pass
            self.assertIn("another", str(cm.exception).lower())

    def test_creates_dir_if_missing(self):
        nested = self.dir / "deeply" / "nested"
        with acquire_disk_dir_lock(nested):
            self.assertTrue(nested.exists())


import mlx.core as mx

from mlx_lm.disk_prompt_cache import (
    WriteJob,
    read_entry_metadata,
    write_entry_atomic,
)
from mlx_lm.models.cache import KVCache, load_prompt_cache


def _make_dummy_kvcache(num_layers=2, ntokens=4, dim=8):
    """Build a tiny KVCache list for tests — no model needed."""
    cache = [KVCache() for _ in range(num_layers)]
    for c in cache:
        k = mx.random.normal((1, 1, ntokens, dim))
        v = mx.random.normal((1, 1, ntokens, dim))
        c.update_and_fetch(k, v)
    mx.eval([(c.state[0], c.state[1]) for c in cache])
    return cache


class TestWriteEntryAtomic(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.entries_dir = Path(self.tmpdir.name) / "entries"
        self.entries_dir.mkdir()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_writes_file_atomically(self):
        cache = _make_dummy_kvcache()
        tokens = [10, 20, 30, 40]
        job = WriteJob(
            token_hash="abc123def4567890",
            tokens=tokens,
            prompt_cache=cache,
            cache_type_classes=["KVCache", "KVCache"],
            trimmable=True,
            parents_to_evict=[],
            model_id="modelid0123456",
        )
        write_entry_atomic(self.entries_dir, job, fsync=False)
        final = self.entries_dir / f"{job.token_hash}.safetensors"
        self.assertTrue(final.exists())
        # No tmp file leftover
        self.assertFalse(
            (self.entries_dir / f"{job.token_hash}.safetensors.tmp").exists()
        )

    def test_metadata_round_trip(self):
        cache = _make_dummy_kvcache()
        tokens = list(range(64))
        job = WriteJob(
            token_hash="ffffffffffffffff",
            tokens=tokens,
            prompt_cache=cache,
            cache_type_classes=["KVCache", "KVCache"],
            trimmable=True,
            parents_to_evict=[],
            model_id="m0001",
        )
        write_entry_atomic(self.entries_dir, job, fsync=False)
        final = self.entries_dir / f"{job.token_hash}.safetensors"
        meta = read_entry_metadata(final)
        self.assertEqual(meta["length"], 64)
        self.assertEqual(meta["trimmable"], True)
        self.assertEqual(meta["model_id"], "m0001")
        self.assertEqual(meta["format_version"], 1)
        self.assertEqual(meta["tokens"], tokens)
        self.assertEqual(meta["cache_type_classes"], ["KVCache", "KVCache"])

    def test_tensor_round_trip(self):
        original = _make_dummy_kvcache(num_layers=2, ntokens=8, dim=16)
        original_keys = [c.state[0] for c in original]
        mx.eval(original_keys)
        tokens = list(range(8))
        job = WriteJob(
            token_hash="aaaaaaaaaaaaaaaa",
            tokens=tokens,
            prompt_cache=original,
            cache_type_classes=["KVCache", "KVCache"],
            trimmable=True,
            parents_to_evict=[],
            model_id="m",
        )
        write_entry_atomic(self.entries_dir, job, fsync=False)
        final = self.entries_dir / f"{job.token_hash}.safetensors"
        loaded = load_prompt_cache(str(final))
        self.assertEqual(len(loaded), 2)
        loaded_keys = [c.state[0] for c in loaded]
        mx.eval(loaded_keys)
        for orig_k, loaded_k in zip(original_keys, loaded_keys):
            self.assertTrue(mx.allclose(orig_k, loaded_k).item())


from mlx_lm.disk_prompt_cache import load_entry


class TestLoadEntry(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.entries_dir = Path(self.tmpdir.name) / "entries"
        self.entries_dir.mkdir()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_load_returns_cache_and_metadata(self):
        cache = _make_dummy_kvcache(num_layers=2, ntokens=4, dim=8)
        tokens = [1, 2, 3, 4]
        job = WriteJob(
            token_hash="0123456789abcdef",
            tokens=tokens,
            prompt_cache=cache,
            cache_type_classes=["KVCache", "KVCache"],
            trimmable=True,
            parents_to_evict=[],
            model_id="m",
        )
        write_entry_atomic(self.entries_dir, job, fsync=False)
        final = self.entries_dir / f"{job.token_hash}.safetensors"
        loaded_cache, meta = load_entry(final)
        self.assertEqual(len(loaded_cache), 2)
        self.assertEqual(meta["length"], 4)
        self.assertEqual(meta["tokens"], tokens)


from mlx_lm.disk_prompt_cache import (
    DiskLeaf,
    reconstruct_disk_trie,
)


class TestReconstructDiskTrie(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.entries_dir = Path(self.tmpdir.name) / "entries"
        self.entries_dir.mkdir()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _seed_entry(self, tokens, *, trimmable=True):
        cache = _make_dummy_kvcache(num_layers=2, ntokens=len(tokens), dim=8)
        from mlx_lm.disk_prompt_cache import hash_tokens

        job = WriteJob(
            token_hash=hash_tokens(tokens),
            tokens=tokens,
            prompt_cache=cache,
            cache_type_classes=["KVCache", "KVCache"],
            trimmable=trimmable,
            parents_to_evict=[],
            model_id="m",
        )
        write_entry_atomic(self.entries_dir, job, fsync=False)
        return job.token_hash

    def test_reconstructs_empty(self):
        trie, total_bytes, h2t = reconstruct_disk_trie(self.entries_dir, model_id="m")
        self.assertEqual(total_bytes, 0)
        self.assertEqual(h2t, {})
        self.assertEqual(trie.search("m", []).common_prefix, 0)

    def test_reconstructs_single_entry(self):
        tokens = [1, 2, 3, 4]
        h = self._seed_entry(tokens)
        trie, total_bytes, h2t = reconstruct_disk_trie(self.entries_dir, model_id="m")
        result = trie.search("m", tokens)
        self.assertEqual(result.exact, tokens)
        leaf = trie.get("m", tokens)
        self.assertIsInstance(leaf, DiskLeaf)
        self.assertEqual(leaf.token_hash, h)
        self.assertTrue(leaf.trimmable)
        self.assertGreater(total_bytes, 0)
        self.assertEqual(h2t[h], tokens)

    def test_reconstructs_multiple(self):
        tokens_a = [1, 2, 3]
        tokens_b = [1, 2, 3, 4, 5]
        self._seed_entry(tokens_a)
        self._seed_entry(tokens_b)
        trie, total, h2t = reconstruct_disk_trie(self.entries_dir, model_id="m")
        self.assertEqual(trie.search("m", tokens_a).exact, tokens_a)
        self.assertEqual(trie.search("m", tokens_b).exact, tokens_b)
        self.assertEqual(len(h2t), 2)

    def test_cleans_tmp_files(self):
        # tmp files are named {hash}.tmp.safetensors
        tmp_path = self.entries_dir / "deadbeefdeadbeef.tmp.safetensors"
        tmp_path.write_bytes(b"partial")
        reconstruct_disk_trie(self.entries_dir, model_id="m")
        self.assertFalse(tmp_path.exists())

    def test_skips_tmp_files_in_main_glob(self):
        # If a .tmp.safetensors exists, it should NOT be loaded as a real entry.
        tmp_path = self.entries_dir / "abadabadabadabad.tmp.safetensors"
        tmp_path.write_bytes(b"partial")  # not valid safetensors
        # Should not crash trying to load it as a real entry
        trie, total, h2t = reconstruct_disk_trie(self.entries_dir, model_id="m")
        # tmp file deleted, no entries loaded
        self.assertEqual(len(h2t), 0)
        self.assertFalse(tmp_path.exists())


from mlx_lm.disk_prompt_cache import DiskPromptCache


class TestDiskPromptCacheInit(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)
        self.fake_model = self.dir / "fake_model"
        self.fake_model.mkdir()
        (self.fake_model / "model.safetensors.index.json").write_text("{}")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_creates_layout(self):
        cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        try:
            self.assertTrue((self.dir / "cache" / "format-version").exists())
            self.assertTrue((self.dir / "cache" / ".lock").exists())
            self.assertEqual(
                (self.dir / "cache" / "format-version").read_text().strip(), "1"
            )
            self.assertTrue((self.dir / "cache" / "models").exists())
            # Per-model subdir
            model_dirs = list((self.dir / "cache" / "models").iterdir())
            self.assertEqual(len(model_dirs), 1)
            self.assertTrue((model_dirs[0] / "info.json").exists())
            self.assertTrue((model_dirs[0] / "entries").exists())
        finally:
            cache.shutdown(timeout=2.0)

    def test_double_init_raises(self):
        cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        try:
            with self.assertRaises(DiskCacheLockError):
                DiskPromptCache(
                    root=self.dir / "cache",
                    model_path=self.fake_model,
                    max_bytes=1 << 30,
                )
        finally:
            cache.shutdown(timeout=2.0)

    def test_format_version_mismatch_refuses(self):
        # Pre-create dir with a wrong version
        (self.dir / "cache").mkdir()
        (self.dir / "cache" / "format-version").write_text("999\n")
        with self.assertRaises(RuntimeError) as cm:
            DiskPromptCache(
                root=self.dir / "cache",
                model_path=self.fake_model,
                max_bytes=1 << 30,
            )
        self.assertIn("format", str(cm.exception).lower())

    def test_per_model_isolation(self):
        # Two different "models" → two different model_ids → two subdirs
        model_b = self.dir / "fake_model_b"
        model_b.mkdir()
        (model_b / "model.safetensors.index.json").write_text('{"x":"y"}')
        cache_a = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        cache_a.shutdown(timeout=2.0)
        cache_b = DiskPromptCache(
            root=self.dir / "cache",
            model_path=model_b,
            max_bytes=1 << 30,
        )
        try:
            model_dirs = sorted((self.dir / "cache" / "models").iterdir())
            self.assertEqual(len(model_dirs), 2)
        finally:
            cache_b.shutdown(timeout=2.0)


class TestDiskPromptCacheSearch(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)
        self.fake_model = self.dir / "fake_model"
        self.fake_model.mkdir()
        (self.fake_model / "model.safetensors.index.json").write_text("{}")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _seed(self, dpc, tokens):
        cache = _make_dummy_kvcache(num_layers=2, ntokens=len(tokens), dim=8)
        from mlx_lm.disk_prompt_cache import hash_tokens

        job = WriteJob(
            token_hash=hash_tokens(tokens),
            tokens=tokens,
            prompt_cache=cache,
            cache_type_classes=["KVCache", "KVCache"],
            trimmable=True,
            parents_to_evict=[],
            model_id=dpc.model_id,
        )
        write_entry_atomic(dpc.entries_dir, job, fsync=False)
        # Manually update trie for tests (writer thread isn't running yet)
        st = (dpc.entries_dir / f"{job.token_hash}.safetensors").stat()
        leaf = DiskLeaf(
            token_hash=job.token_hash,
            length=len(tokens),
            nbytes=st.st_size,
            mtime_ns=st.st_mtime_ns,
            trimmable=True,
        )
        with dpc._trie_lock:
            dpc._trie.add(dpc.model_id, tokens, leaf)
            dpc._hash_to_tokens[job.token_hash] = tokens
            dpc._total_bytes += st.st_size
        return leaf

    def test_search_empty(self):
        cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        try:
            result = cache.search([1, 2, 3])
            self.assertIsNone(result.exact)
            self.assertIsNone(result.shorter)
            self.assertIsNone(result.longer)
            self.assertEqual(result.common_prefix, 0)
        finally:
            cache.shutdown(timeout=2.0)

    def test_search_exact(self):
        cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        try:
            self._seed(cache, [1, 2, 3, 4])
            result = cache.search([1, 2, 3, 4])
            self.assertEqual(result.exact, [1, 2, 3, 4])
        finally:
            cache.shutdown(timeout=2.0)

    def test_search_longer(self):
        cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        try:
            self._seed(cache, [1, 2, 3, 4, 5, 6])
            result = cache.search([1, 2, 3])
            # Some kind of match info should be present
            self.assertTrue(
                result.longer is not None
                or result.exact is not None
                or result.common_prefix > 0
            )
        finally:
            cache.shutdown(timeout=2.0)

    def test_get_leaf(self):
        cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        try:
            seeded = self._seed(cache, [10, 20, 30])
            leaf = cache.get_leaf([10, 20, 30])
            self.assertEqual(leaf.token_hash, seeded.token_hash)
        finally:
            cache.shutdown(timeout=2.0)

    def test_entry_path(self):
        cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        try:
            p = cache.entry_path("deadbeefcafef00d")
            self.assertEqual(p.name, "deadbeefcafef00d.safetensors")
            self.assertEqual(p.parent, cache.entries_dir)
        finally:
            cache.shutdown(timeout=2.0)


class TestDiskPromptCacheLoad(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)
        self.fake_model = self.dir / "fake_model"
        self.fake_model.mkdir()
        (self.fake_model / "model.safetensors.index.json").write_text("{}")
        self.cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        # Seed an entry on disk + add to trie
        from mlx_lm.disk_prompt_cache import hash_tokens

        kv = _make_dummy_kvcache(num_layers=2, ntokens=4, dim=8)
        self.tokens = [1, 2, 3, 4]
        self.token_hash = hash_tokens(self.tokens)
        job = WriteJob(
            token_hash=self.token_hash,
            tokens=self.tokens,
            prompt_cache=kv,
            cache_type_classes=["KVCache", "KVCache"],
            trimmable=True,
            parents_to_evict=[],
            model_id=self.cache.model_id,
        )
        write_entry_atomic(self.cache.entries_dir, job, fsync=False)
        st = (self.cache.entries_dir / f"{job.token_hash}.safetensors").stat()
        leaf = DiskLeaf(
            token_hash=job.token_hash,
            length=len(self.tokens),
            nbytes=st.st_size,
            mtime_ns=st.st_mtime_ns,
            trimmable=True,
        )
        with self.cache._trie_lock:
            self.cache._trie.add(self.cache.model_id, self.tokens, leaf)
            self.cache._hash_to_tokens[job.token_hash] = self.tokens

    def tearDown(self):
        self.cache.shutdown(timeout=2.0)
        self.tmpdir.cleanup()

    def test_load_returns_cache(self):
        prompt_cache, meta = self.cache.load(self.token_hash)
        self.assertEqual(len(prompt_cache), 2)
        self.assertEqual(meta["length"], 4)

    def test_concurrent_load_dedup(self):
        # Patch load_entry to count calls
        from mlx_lm import disk_prompt_cache as mod

        original = mod.load_entry
        call_count = [0]
        import threading as _t

        gate = _t.Event()

        def slow_load(path):
            call_count[0] += 1
            gate.wait(timeout=2.0)
            return original(path)

        mod.load_entry = slow_load
        try:
            results = []
            errors = []

            def worker():
                try:
                    results.append(self.cache.load(self.token_hash))
                except Exception as e:
                    errors.append(e)

            threads = [_t.Thread(target=worker) for _ in range(5)]
            for t in threads:
                t.start()
            time.sleep(0.05)  # let all threads enter load()
            gate.set()
            for t in threads:
                t.join(timeout=5.0)
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 5)
            self.assertEqual(
                call_count[0], 1, "only one disk read should have happened"
            )
        finally:
            mod.load_entry = original


from mlx_lm.disk_prompt_cache import DiskLeaf


class TestDiskPromptCacheTouch(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)
        self.fake_model = self.dir / "fake_model"
        self.fake_model.mkdir()
        (self.fake_model / "model.safetensors.index.json").write_text("{}")
        self.cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        kv = _make_dummy_kvcache(num_layers=1, ntokens=4, dim=8)
        from mlx_lm.disk_prompt_cache import hash_tokens

        self.tokens = [1, 2, 3, 4]
        self.token_hash = hash_tokens(self.tokens)
        job = WriteJob(
            token_hash=self.token_hash,
            tokens=self.tokens,
            prompt_cache=kv,
            cache_type_classes=["KVCache"],
            trimmable=True,
            parents_to_evict=[],
            model_id=self.cache.model_id,
        )
        write_entry_atomic(self.cache.entries_dir, job, fsync=False)
        st = self.cache.entry_path(self.token_hash).stat()
        leaf = DiskLeaf(
            token_hash=self.token_hash,
            length=4,
            nbytes=st.st_size,
            mtime_ns=st.st_mtime_ns,
            trimmable=True,
        )
        with self.cache._trie_lock:
            self.cache._trie.add(self.cache.model_id, self.tokens, leaf)
            self.cache._hash_to_tokens[self.token_hash] = self.tokens
        self.original_mtime = st.st_mtime_ns
        self.cache.start()  # start the writer + touch threads

    def tearDown(self):
        self.cache.shutdown(timeout=5.0)
        self.tmpdir.cleanup()

    def test_touch_updates_mtime(self):
        time.sleep(0.05)  # so mtime changes detectably
        self.cache.touch_async(self.token_hash)
        # Allow background thread to flush
        time.sleep(0.5)
        new_mtime = self.cache.entry_path(self.token_hash).stat().st_mtime_ns
        self.assertGreater(new_mtime, self.original_mtime)

    def test_touch_updates_in_memory_leaf(self):
        time.sleep(0.05)
        self.cache.touch_async(self.token_hash)
        # In-memory mtime updates synchronously
        with self.cache._trie_lock:
            leaf = self.cache._trie.get(self.cache.model_id, self.tokens)
        self.assertGreater(leaf.mtime_ns, self.original_mtime)

    def test_touch_unknown_hash_no_op(self):
        # touch_async on a hash we've never seen should not raise
        self.cache.touch_async("nonexistenthash")
        # No assertion; just confirming no exception


class TestDominatedPrefixes(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)
        self.fake_model = self.dir / "fake_model"
        self.fake_model.mkdir()
        (self.fake_model / "model.safetensors.index.json").write_text("{}")
        self.cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )

    def tearDown(self):
        self.cache.shutdown(timeout=2.0)
        self.tmpdir.cleanup()

    def _seed(self, tokens, trimmable=True):
        from mlx_lm.disk_prompt_cache import hash_tokens

        kv = _make_dummy_kvcache(num_layers=1, ntokens=len(tokens), dim=8)
        h = hash_tokens(tokens)
        job = WriteJob(
            token_hash=h,
            tokens=tokens,
            prompt_cache=kv,
            cache_type_classes=["KVCache"],
            trimmable=trimmable,
            parents_to_evict=[],
            model_id=self.cache.model_id,
        )
        write_entry_atomic(self.cache.entries_dir, job, fsync=False)
        st = self.cache.entry_path(h).stat()
        leaf = DiskLeaf(
            token_hash=h,
            length=len(tokens),
            nbytes=st.st_size,
            mtime_ns=st.st_mtime_ns,
            trimmable=trimmable,
        )
        with self.cache._trie_lock:
            self.cache._trie.add(self.cache.model_id, tokens, leaf)
            self.cache._hash_to_tokens[h] = tokens
        return h

    def test_no_dominated_when_empty(self):
        result = self.cache.find_dominated_prefixes([1, 2, 3])
        self.assertEqual(result, [])

    def test_dominates_shorter_trimmable(self):
        h_short = self._seed([1, 2, 3])
        result = self.cache.find_dominated_prefixes([1, 2, 3, 4, 5])
        self.assertEqual(result, [h_short])

    def test_skips_non_trimmable_shorter(self):
        h_short = self._seed([1, 2, 3], trimmable=False)
        result = self.cache.find_dominated_prefixes([1, 2, 3, 4, 5])
        self.assertEqual(result, [])

    def test_dominates_chain(self):
        h_a = self._seed([1, 2])
        h_b = self._seed([1, 2, 3])
        h_c = self._seed([1, 2, 3, 4])
        result = self.cache.find_dominated_prefixes([1, 2, 3, 4, 5, 6])
        # All three shorter, all trimmable → all dominated
        self.assertCountEqual(result, [h_a, h_b, h_c])

    def test_does_not_dominate_off_path(self):
        h_off = self._seed([9, 9, 9])  # totally different path
        result = self.cache.find_dominated_prefixes([1, 2, 3, 4])
        self.assertEqual(result, [])


class TestWriterLoop(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)
        self.fake_model = self.dir / "fake_model"
        self.fake_model.mkdir()
        (self.fake_model / "model.safetensors.index.json").write_text("{}")
        self.cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        self.cache.start()

    def tearDown(self):
        self.cache.shutdown(timeout=5.0)
        self.tmpdir.cleanup()

    def _job(self, tokens, trimmable=True):
        from mlx_lm.disk_prompt_cache import hash_tokens

        kv = _make_dummy_kvcache(num_layers=1, ntokens=len(tokens), dim=8)
        return WriteJob(
            token_hash=hash_tokens(tokens),
            tokens=tokens,
            prompt_cache=kv,
            cache_type_classes=["KVCache"],
            trimmable=trimmable,
            parents_to_evict=[],
            model_id=self.cache.model_id,
        )

    def test_basic_write_lands_on_disk(self):
        job = self._job([1, 2, 3, 4])
        self.cache.enqueue_write(job)
        self.cache._queue.join()
        self.assertTrue(self.cache.entry_path(job.token_hash).exists())
        # Trie updated
        self.assertEqual(self.cache.search([1, 2, 3, 4]).exact, [1, 2, 3, 4])

    def test_dominator_replacement_deletes_shorter(self):
        job_a = self._job([1, 2, 3])
        self.cache.enqueue_write(job_a)
        self.cache._queue.join()
        # Now insert a deeper leaf that dominates [1,2,3]
        job_b = self._job([1, 2, 3, 4, 5])
        job_b.parents_to_evict = self.cache.find_dominated_prefixes(job_b.tokens)
        self.assertEqual(job_b.parents_to_evict, [job_a.token_hash])
        self.cache.enqueue_write(job_b)
        self.cache._queue.join()
        self.assertFalse(self.cache.entry_path(job_a.token_hash).exists())
        self.assertTrue(self.cache.entry_path(job_b.token_hash).exists())
        # Trie no longer has the shorter
        with self.cache._trie_lock:
            self.assertNotIn(job_a.token_hash, self.cache._hash_to_tokens)

    def test_total_bytes_tracking(self):
        before = self.cache._total_bytes
        job = self._job([10, 20, 30])
        self.cache.enqueue_write(job)
        self.cache._queue.join()
        after = self.cache._total_bytes
        self.assertGreater(after, before)


class TestEviction(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)
        self.fake_model = self.dir / "fake_model"
        self.fake_model.mkdir()
        (self.fake_model / "model.safetensors.index.json").write_text("{}")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _job(self, cache, tokens):
        from mlx_lm.disk_prompt_cache import hash_tokens

        kv = _make_dummy_kvcache(num_layers=1, ntokens=len(tokens), dim=64)
        return WriteJob(
            token_hash=hash_tokens(tokens),
            tokens=tokens,
            prompt_cache=kv,
            cache_type_classes=["KVCache"],
            trimmable=True,
            parents_to_evict=[],
            model_id=cache.model_id,
        )

    def test_evicts_oldest_first(self):
        # Tight cap: ~3 entries fit
        cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=200_000,
            eviction_headroom=0.95,
        )
        cache.start()
        try:
            entries = []
            for i in range(8):
                tokens = [i * 10 + j for j in range(50)]
                job = self._job(cache, tokens)
                entries.append(job.token_hash)
                cache.enqueue_write(job)
                cache._queue.join()
                # Brief sleep so mtime ordering is unambiguous
                time.sleep(0.01)
            # Some early entries should have been evicted
            remaining = [h for h in entries if cache.entry_path(h).exists()]
            self.assertLess(len(remaining), len(entries))
            # First entries (oldest mtime) should be the ones evicted
            self.assertNotIn(entries[0], remaining)
            # Cap respected
            self.assertLessEqual(cache._total_bytes, cache.max_bytes)
        finally:
            cache.shutdown(timeout=5.0)

    def test_eviction_headroom(self):
        cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=200_000,
            eviction_headroom=0.80,
        )
        cache.start()
        try:
            for i in range(10):
                tokens = [i * 100 + j for j in range(50)]
                job = self._job(cache, tokens)
                cache.enqueue_write(job)
                cache._queue.join()
                time.sleep(0.01)
            # After eviction, must be ≤ headroom * cap
            self.assertLessEqual(cache._total_bytes, int(cache.max_bytes * 0.80) + 1)
        finally:
            cache.shutdown(timeout=5.0)


class TestShutdown(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)
        self.fake_model = self.dir / "fake_model"
        self.fake_model.mkdir()
        (self.fake_model / "model.safetensors.index.json").write_text("{}")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_shutdown_drains_queue(self):
        cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        cache.start()
        token_hashes = []
        from mlx_lm.disk_prompt_cache import hash_tokens

        for i in range(5):
            tokens = list(range(i * 10, i * 10 + 8))
            kv = _make_dummy_kvcache(num_layers=1, ntokens=8, dim=8)
            job = WriteJob(
                token_hash=hash_tokens(tokens),
                tokens=tokens,
                prompt_cache=kv,
                cache_type_classes=["KVCache"],
                trimmable=True,
                parents_to_evict=[],
                model_id=cache.model_id,
            )
            token_hashes.append(job.token_hash)
            cache.enqueue_write(job)
        cache.shutdown(timeout=10.0)
        # All 5 entries on disk
        for h in token_hashes:
            self.assertTrue(cache.entry_path(h).exists(), f"missing {h}")

    def test_shutdown_releases_lock(self):
        cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        cache.start()
        cache.shutdown(timeout=2.0)
        # Should be able to re-create
        cache2 = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        cache2.shutdown(timeout=2.0)

    def test_shutdown_idempotent(self):
        cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        cache.start()
        cache.shutdown(timeout=2.0)
        cache.shutdown(timeout=2.0)  # second call is a no-op


class TestFailureHandling(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)
        self.fake_model = self.dir / "fake_model"
        self.fake_model.mkdir()
        (self.fake_model / "model.safetensors.index.json").write_text("{}")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_eaccess_disables_writes(self):
        cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        cache.start()
        try:
            from mlx_lm import disk_prompt_cache as mod
            from mlx_lm.disk_prompt_cache import hash_tokens

            original = mod.write_entry_atomic

            def fail(*a, **kw):
                raise PermissionError(13, "Permission denied", "x")

            mod.write_entry_atomic = fail
            try:
                tokens = [1, 2, 3]
                kv = _make_dummy_kvcache(num_layers=1, ntokens=3, dim=8)
                job = WriteJob(
                    token_hash=hash_tokens(tokens),
                    tokens=tokens,
                    prompt_cache=kv,
                    cache_type_classes=["KVCache"],
                    trimmable=True,
                    parents_to_evict=[],
                    model_id=cache.model_id,
                )
                cache.enqueue_write(job)
                cache._queue.join()
                # Writes are now disabled
                self.assertTrue(cache._writes_disabled)
            finally:
                mod.write_entry_atomic = original
        finally:
            cache.shutdown(timeout=2.0)

    def test_enospc_triggers_emergency_eviction(self):
        import errno

        cache = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        cache.start()
        try:
            from mlx_lm import disk_prompt_cache as mod
            from mlx_lm.disk_prompt_cache import hash_tokens

            # Seed some entries first so emergency eviction has something to evict
            for i in range(3):
                tokens = [i, i, i, i]
                kv = _make_dummy_kvcache(num_layers=1, ntokens=4, dim=8)
                job = WriteJob(
                    token_hash=hash_tokens(tokens),
                    tokens=tokens,
                    prompt_cache=kv,
                    cache_type_classes=["KVCache"],
                    trimmable=True,
                    parents_to_evict=[],
                    model_id=cache.model_id,
                )
                cache.enqueue_write(job)
                cache._queue.join()
                time.sleep(0.01)
            initial_count = len(cache._hash_to_tokens)

            original = mod.write_entry_atomic
            call_count = [0]

            def fail_then_succeed(*a, **kw):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise OSError(errno.ENOSPC, "No space left", "x")
                return original(*a, **kw)

            mod.write_entry_atomic = fail_then_succeed
            try:
                tokens = [99, 99, 99]
                kv = _make_dummy_kvcache(num_layers=1, ntokens=3, dim=8)
                job = WriteJob(
                    token_hash=hash_tokens(tokens),
                    tokens=tokens,
                    prompt_cache=kv,
                    cache_type_classes=["KVCache"],
                    trimmable=True,
                    parents_to_evict=[],
                    model_id=cache.model_id,
                )
                cache.enqueue_write(job)
                cache._queue.join()
                # Emergency eviction should have removed at least one earlier entry
                self.assertLess(len(cache._hash_to_tokens), initial_count + 1)
                # The new entry should be present (retry succeeded)
                self.assertTrue(cache.entry_path(job.token_hash).exists())
                # Writes still enabled
                self.assertFalse(cache._writes_disabled)
            finally:
                mod.write_entry_atomic = original
        finally:
            cache.shutdown(timeout=2.0)


class TestLRUPromptCacheIntegration(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)
        self.fake_model = self.dir / "fake_model"
        self.fake_model.mkdir()
        (self.fake_model / "model.safetensors.index.json").write_text("{}")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_disk_param_default_none(self):
        # Constructor still works without disk (existing API unchanged)
        from mlx_lm.models.cache import LRUPromptCache

        ram = LRUPromptCache(max_size=10)
        self.assertIsNone(ram.disk)

    def test_disk_param_accepted(self):
        from mlx_lm.disk_prompt_cache import DiskPromptCache
        from mlx_lm.models.cache import LRUPromptCache

        disk = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        try:
            disk.start()
            ram = LRUPromptCache(max_size=10, disk=disk)
            self.assertIs(ram.disk, disk)
        finally:
            disk.shutdown(timeout=2.0)

    def test_existing_two_arg_constructor_still_works(self):
        # Verify the pre-PR API call signature works unchanged
        from mlx_lm.models.cache import LRUPromptCache

        ram = LRUPromptCache(10, 1 << 30)
        self.assertEqual(ram.max_size, 10)
        self.assertEqual(ram.max_bytes, 1 << 30)
        self.assertIsNone(ram.disk)

    def test_insert_cache_writes_to_disk(self):
        from mlx_lm.models.cache import LRUPromptCache

        disk = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        disk.start()
        try:
            ram = LRUPromptCache(max_size=10, disk=disk)
            tokens = [1, 2, 3, 4]
            kv = _make_dummy_kvcache(num_layers=1, ntokens=4, dim=8)
            ram.insert_cache(disk.model_id, tokens, kv, cache_type="user")
            disk._queue.join()
            from mlx_lm.disk_prompt_cache import hash_tokens

            self.assertTrue(disk.entry_path(hash_tokens(tokens)).exists())
        finally:
            disk.shutdown(timeout=2.0)

    def test_insert_cache_disk_dominator_replacement(self):
        from mlx_lm.disk_prompt_cache import hash_tokens
        from mlx_lm.models.cache import LRUPromptCache

        disk = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        disk.start()
        try:
            ram = LRUPromptCache(max_size=10, disk=disk)
            short = [1, 2, 3]
            long = [1, 2, 3, 4, 5]
            kv_s = _make_dummy_kvcache(num_layers=1, ntokens=3, dim=8)
            kv_l = _make_dummy_kvcache(num_layers=1, ntokens=5, dim=8)
            ram.insert_cache(disk.model_id, short, kv_s, cache_type="user")
            disk._queue.join()
            self.assertTrue(disk.entry_path(hash_tokens(short)).exists())
            ram.insert_cache(disk.model_id, long, kv_l, cache_type="user")
            disk._queue.join()
            # Shorter trimmable file should now be deleted
            self.assertFalse(disk.entry_path(hash_tokens(short)).exists())
            self.assertTrue(disk.entry_path(hash_tokens(long)).exists())
        finally:
            disk.shutdown(timeout=2.0)

    def test_fetch_loads_from_disk_when_ram_misses(self):
        from mlx_lm.disk_prompt_cache import hash_tokens
        from mlx_lm.models.cache import LRUPromptCache, PromptTrie

        disk = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        disk.start()
        try:
            ram = LRUPromptCache(max_size=10, disk=disk)
            tokens = [1, 2, 3, 4]
            kv = _make_dummy_kvcache(num_layers=1, ntokens=4, dim=8)
            ram.insert_cache(disk.model_id, tokens, kv, cache_type="user")
            disk._queue.join()

            # Simulate RAM eviction by clearing the RAM trie
            ram._trie = PromptTrie()
            ram._lru = type(ram._lru)()
            ram._n_bytes = 0
            ram._n_bytes_by_type = {k: 0 for k in ram._lru._ordering}

            # Fetch should now load from disk
            cache, rest = ram.fetch_nearest_cache(disk.model_id, tokens)
            self.assertIsNotNone(cache)
            self.assertEqual(rest, [])
            self.assertEqual(len(cache), 1)
            # Now in RAM trie too (promoted)
            self.assertEqual(ram._trie.search(disk.model_id, tokens).exact, tokens)
        finally:
            disk.shutdown(timeout=2.0)

    def test_fetch_touches_disk_mtime_on_ram_hit(self):
        from mlx_lm.disk_prompt_cache import hash_tokens
        from mlx_lm.models.cache import LRUPromptCache

        disk = DiskPromptCache(
            root=self.dir / "cache",
            model_path=self.fake_model,
            max_bytes=1 << 30,
        )
        disk.start()
        try:
            ram = LRUPromptCache(max_size=10, disk=disk)
            tokens = [1, 2, 3, 4]
            kv = _make_dummy_kvcache(num_layers=1, ntokens=4, dim=8)
            ram.insert_cache(disk.model_id, tokens, kv, cache_type="user")
            disk._queue.join()
            original_mtime = disk.entry_path(hash_tokens(tokens)).stat().st_mtime_ns

            time.sleep(0.05)
            # Pure RAM hit
            cache, rest = ram.fetch_nearest_cache(disk.model_id, tokens)
            # Allow async touch to flush
            time.sleep(0.5)
            new_mtime = disk.entry_path(hash_tokens(tokens)).stat().st_mtime_ns
            self.assertGreater(new_mtime, original_mtime)
        finally:
            disk.shutdown(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
