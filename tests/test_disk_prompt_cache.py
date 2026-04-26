# Copyright © 2026 Apple Inc.

import json
import os
import tempfile
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


if __name__ == "__main__":
    unittest.main()
