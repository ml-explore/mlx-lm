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


if __name__ == "__main__":
    unittest.main()
