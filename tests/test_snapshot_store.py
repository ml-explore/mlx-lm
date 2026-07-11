# Copyright © 2026 Apple Inc.

import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.models.cache import KVCache
from mlx_lm.prefix_forks import FrozenPrefixSnapshot, PrefixForkRegistry, compute_cid
from mlx_lm.server import PersistentPromptCache, ResponseGenerator
from mlx_lm.snapshot_store import SnapshotStore, _array_digest, _read_header


def _cache(length, seed=1, layers=2):
    mx.random.seed(seed)
    out = []
    for _ in range(layers):
        cache = KVCache()
        key = mx.random.normal((1, 2, length, 4))
        value = mx.random.normal((1, 2, length, 4))
        cache.update_and_fetch(key, value)
        out.append(cache)
    mx.eval(*(array for cache in out for array in cache.state))
    return out


def _prefix_cache(length):
    cache = KVCache()
    positions = mx.arange(length, dtype=mx.float32)[None, None, :, None]
    heads = mx.arange(2, dtype=mx.float32)[None, :, None, None] * 100
    dims = mx.arange(4, dtype=mx.float32)[None, None, None, :]
    key = mx.broadcast_to(positions + heads + dims, (1, 2, length, 4))
    value = key + 1000
    cache.update_and_fetch(key, value)
    mx.eval(*cache.state)
    return [cache]


def _freeze(registry, model_key, tokens, seed=1):
    cid = registry.freeze(model_key, tokens, _cache(len(tokens), seed=seed))
    return registry.get(cid)


class TestSnapshotStore(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_round_trip_restart_and_model_key_guard(self):
        registry = PrefixForkRegistry()
        tokens = list(range(16))
        snapshot = _freeze(registry, "model@rev-a", tokens)
        before = snapshot.keys + snapshot.values
        SnapshotStore(self.path).save(snapshot)

        restarted = PrefixForkRegistry()
        store = SnapshotStore(self.path, registry=restarted)
        forks, rest = restarted.fetch_fork("model@rev-a", tokens + [99])
        self.assertEqual(rest, [99])
        self.assertEqual(store.stats["disk_hits"], 1)
        for got, expected in zip(
            [array for fork in forks for array in fork.state],
            [item for pair in zip(before[:2], before[2:]) for item in pair],
        ):
            self.assertTrue(mx.array_equal(got, expected))
        miss, rest = restarted.fetch_fork("model@rev-b", tokens)
        self.assertIsNone(miss)
        self.assertEqual(rest, tokens)

    def test_chain_round_trip_and_depth_cap_self_roots(self):
        registry = PrefixForkRegistry()
        root = registry.get(registry.freeze("m@r", list(range(8)), _prefix_cache(8)))
        child = registry.get(registry.freeze("m@r", list(range(16)), _prefix_cache(16)))
        grand = registry.get(registry.freeze("m@r", list(range(24)), _prefix_cache(24)))
        store = SnapshotStore(self.path, max_chain_depth=1)
        store.save(root)
        store.save(child)
        store.save(grand)
        self.assertIsNone(store._records[root.cid].parent_cid)
        self.assertEqual(store._records[child.cid].parent_cid, root.cid)
        self.assertIsNone(store._records[grand.cid].parent_cid)
        restored = store.restore(child.cid)
        self.assertEqual(restored.tokens, child.tokens)
        for got, expected in zip(
            restored.keys + restored.values, child.keys + child.values
        ):
            self.assertTrue(mx.array_equal(got, expected))

    def test_delta_refuses_mismatched_kv_prefix(self):
        registry = PrefixForkRegistry()
        root = _freeze(registry, "m@r", list(range(8)), seed=1)
        mismatch = _freeze(registry, "m@r", list(range(16)), seed=2)
        store = SnapshotStore(self.path)
        store.save(root)
        store.save(mismatch)
        self.assertIsNone(store._records[mismatch.cid].parent_cid)

    def test_existing_cid_rejects_mismatched_payload(self):
        first_registry = PrefixForkRegistry()
        first = _freeze(first_registry, "m@r", list(range(8)), seed=1)
        store = SnapshotStore(self.path)
        store.save(first)
        second_registry = PrefixForkRegistry()
        mismatch = _freeze(second_registry, "m@r", list(range(8)), seed=2)
        with self.assertRaises(ValueError):
            store.save(mismatch)

    def test_rebuild_is_header_only_until_fetch(self):
        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "m@r", list(range(64)))
        SnapshotStore(self.path).save(snapshot)
        restarted = PrefixForkRegistry()
        with patch("mlx_lm.snapshot_store.mx.load") as load:
            SnapshotStore(self.path, registry=restarted)
            load.assert_not_called()
        self.assertEqual(len(restarted), 1)

    def test_temp_cleaned_and_truncated_quarantined(self):
        (self.path / ".dead.tmp-crash.safetensors").write_bytes(b"partial")
        (self.path / ("a" * 64 + ".safetensors")).write_bytes(b"short")
        store = SnapshotStore(self.path)
        self.assertFalse(any("tmp-crash" in p.name for p in self.path.iterdir()))
        self.assertTrue(any(p.suffix == ".quarantined" for p in self.path.iterdir()))
        self.assertEqual(store.stats["quarantines"], 1)

    def test_header_valid_truncated_payload_quarantined_at_rebuild(self):
        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "m@r", list(range(8)))
        SnapshotStore(self.path).save(snapshot)
        path = self.path / f"{snapshot.cid}.safetensors"
        with path.open("r+b") as handle:
            handle.truncate(path.stat().st_size - 1)
        restarted = PrefixForkRegistry()
        SnapshotStore(self.path, registry=restarted)
        self.assertFalse(path.exists())
        self.assertEqual(len(restarted), 0)

    def test_filename_mismatch_and_unknown_feature_quarantine(self):
        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "m@r", list(range(8)))
        store = SnapshotStore(self.path)
        store.save(snapshot)
        original = self.path / f"{snapshot.cid}.safetensors"
        wrong = self.path / ("f" * 64 + ".safetensors")
        os.replace(original, wrong)
        SnapshotStore(self.path)
        self.assertTrue(any(p.suffix == ".quarantined" for p in self.path.iterdir()))

        other = self.path / f"{snapshot.cid}.safetensors"
        mx.save_safetensors(
            str(other),
            {"k.0": mx.zeros((1, 1, 8, 1)), "v.0": mx.zeros((1, 1, 8, 1))},
            {
                "format_version": "1",
                "features": json.dumps(["future-feature"]),
                "model_key": "m@r",
                "token_ids": json.dumps(list(range(8))),
            },
        )
        SnapshotStore(self.path)
        self.assertFalse(other.exists())

    def test_digest_guard_on_fetch_and_scrub(self):
        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "m@r", list(range(16)))
        store = SnapshotStore(self.path)
        store.save(snapshot)
        path = self.path / f"{snapshot.cid}.safetensors"
        header, data_start = _read_header(path)
        offset = data_start + header["k.0"]["data_offsets"][0]
        with path.open("r+b") as handle:
            handle.seek(offset)
            byte = handle.read(1)
            handle.seek(offset)
            handle.write(bytes([byte[0] ^ 1]))
        restarted = PrefixForkRegistry()
        corrupt = SnapshotStore(self.path, registry=restarted)
        forks, rest = restarted.fetch_fork("m@r", list(range(16)))
        self.assertIsNone(forks)
        self.assertEqual(rest, list(range(16)))
        self.assertGreaterEqual(corrupt.stats["quarantines"], 1)

        clean_dir = self.path / "clean"
        clean_store = SnapshotStore(clean_dir)
        clean_store.save(snapshot)
        clean_path = clean_dir / f"{snapshot.cid}.safetensors"
        header, data_start = _read_header(clean_path)
        offset = data_start + header["v.0"]["data_offsets"][0]
        with clean_path.open("r+b") as handle:
            handle.seek(offset)
            byte = handle.read(1)
            handle.seek(offset)
            handle.write(bytes([byte[0] ^ 1]))
        self.assertFalse(clean_store.scrub_one())
        self.assertFalse(clean_path.exists())

    def test_gc_is_lru_leaves_first_and_resave_works(self):
        registry = PrefixForkRegistry()
        root = registry.get(registry.freeze("m@r", list(range(8)), _prefix_cache(8)))
        child = registry.get(registry.freeze("m@r", list(range(16)), _prefix_cache(16)))
        store = SnapshotStore(self.path)
        store.save(root)
        store.save(child)
        child_size = store._records[child.cid].file_bytes
        root_size = store._records[root.cid].file_bytes
        store.max_bytes = root_size + child_size - 1
        store.gc()
        self.assertIn(root.cid, store._records)
        self.assertNotIn(child.cid, store._records)
        self.assertEqual(store.save(child), child.cid)

    def test_missing_parent_poisons_descendant(self):
        registry = PrefixForkRegistry()
        root = registry.get(registry.freeze("m@r", list(range(8)), _prefix_cache(8)))
        child = registry.get(registry.freeze("m@r", list(range(16)), _prefix_cache(16)))
        store = SnapshotStore(self.path)
        store.save(root)
        store.save(child)
        (self.path / f"{root.cid}.safetensors").unlink()
        restarted = PrefixForkRegistry()
        SnapshotStore(self.path, registry=restarted)
        forks, _ = restarted.fetch_fork("m@r", list(range(16)))
        self.assertIsNone(forks)

    def test_forged_parent_identity_quarantines_child(self):
        registry = PrefixForkRegistry()
        root = registry.get(registry.freeze("m@r", list(range(8)), _prefix_cache(8)))
        child = registry.get(registry.freeze("m@r", list(range(16)), _prefix_cache(16)))
        other = registry.get(
            registry.freeze("m@r", list(range(100, 108)), _prefix_cache(8))
        )
        store = SnapshotStore(self.path)
        store.save(root)
        store.save(other)
        store.save(child)
        path = self.path / f"{child.cid}.safetensors"
        arrays, metadata = mx.load(str(path), return_metadata=True)
        metadata["parent_cid"] = other.cid
        mx.save_safetensors(str(path), arrays, metadata)
        SnapshotStore(self.path)
        self.assertFalse(path.exists())

    def test_incompatible_chain_geometry_quarantines_descendant(self):
        registry = PrefixForkRegistry()
        root = registry.get(registry.freeze("m@r", list(range(8)), _prefix_cache(8)))
        child = registry.get(registry.freeze("m@r", list(range(16)), _prefix_cache(16)))
        store = SnapshotStore(self.path)
        store.save(root)
        store.save(child)
        path = self.path / f"{child.cid}.safetensors"
        arrays, metadata = mx.load(str(path), return_metadata=True)
        arrays["k.0"] = mx.zeros((1, 3, 8, 4))
        metadata["sha256_k_0"] = _array_digest(arrays["k.0"])
        mx.save_safetensors(str(path), arrays, metadata)
        restarted = PrefixForkRegistry()
        rebuilt = SnapshotStore(self.path, registry=restarted)
        forks, rest = restarted.fetch_fork("m@r", list(range(16)))
        self.assertIsNone(forks)
        self.assertEqual(rest, list(range(16)))
        self.assertGreaterEqual(rebuilt.stats["quarantines"], 1)

    def test_permissions_and_ghost_telemetry(self):
        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "m@r", list(range(8)))
        store = SnapshotStore(self.path)
        store.save(snapshot)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o700)
        self.assertEqual(
            (self.path / f"{snapshot.cid}.safetensors").stat().st_mode & 0o777,
            0o600,
        )
        store.note_ram_eviction(snapshot.cid)
        store.note_request_cid(snapshot.cid)
        self.assertEqual(store.stats["ghost_hits"], 1)

    def test_cid_filename_contract(self):
        self.assertEqual(
            compute_cid("m@r", list(range(8))),
            compute_cid("m@r", list(range(8))),
        )


class TestPersistentServerBridge(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temp.name) / "model"
        self.store_dir = Path(self.temp.name) / "store"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text("{}")

    def tearDown(self):
        self.temp.cleanup()

    def test_insert_flush_restart_fetch_and_promote(self):
        model = (str(self.model_dir), None, None)
        tokens = list(range(8))
        first = PersistentPromptCache(
            4, self.store_dir, max_bytes=1 << 20, max_files=32
        )
        first.insert_cache(model, tokens, _prefix_cache(8))
        self.assertEqual(first.flush_persistent(scrub=False), 1)

        restarted = PersistentPromptCache(
            4, self.store_dir, max_bytes=1 << 20, max_files=32
        )
        forks, rest = restarted.fetch_nearest_cache(model, tokens + [9])
        self.assertIsNotNone(forks)
        self.assertEqual(rest, [9])
        self.assertEqual(restarted.persistent_stats["disk_hits"], 1)
        # The disk proxy was re-promoted to an in-RAM immutable snapshot.
        self.assertTrue(
            any(
                isinstance(snapshot, FrozenPrefixSnapshot)
                for snapshot in restarted._registry.snapshots()
            )
        )

    def test_opt_in_disables_incompatible_batch_path_only(self):
        provider = type(
            "Provider",
            (),
            {
                "is_batchable": True,
                "cli_args": Namespace(prompt_cache_persist_dir=str(self.store_dir)),
            },
        )()
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.model_provider = provider
        args = Namespace(seed=None)
        self.assertFalse(generator._is_batchable(args))
        provider.cli_args.prompt_cache_persist_dir = None
        self.assertTrue(generator._is_batchable(args))

    def test_unsupported_cache_falls_back_without_disk_file(self):
        class Unsupported:
            offset = 4
            nbytes = 0

            def is_trimmable(self):
                return False

        cache = PersistentPromptCache(
            4, self.store_dir, max_bytes=1 << 20, max_files=32
        )
        item = Unsupported()
        cache.insert_cache((str(self.model_dir), None, None), list(range(4)), [item])
        self.assertEqual(len(cache._fallback), 1)
        self.assertFalse(list(self.store_dir.glob("*.safetensors")))

    def test_ram_cap_demotes_and_disk_hit_repromotes_safely(self):
        model = (str(self.model_dir), None, None)
        cache = PersistentPromptCache(
            0, self.store_dir, max_bytes=1 << 20, max_files=32
        )
        tokens = list(range(8))
        cache.insert_cache(model, tokens, _prefix_cache(8))
        self.assertFalse(
            any(
                isinstance(snapshot, FrozenPrefixSnapshot)
                for snapshot in cache._registry.snapshots()
            )
        )
        forks, rest = cache.fetch_nearest_cache(model, tokens + [9])
        self.assertIsNotNone(forks)
        self.assertEqual(rest, [9])
        self.assertGreaterEqual(cache.persistent_stats["promotions"], 1)
        self.assertGreaterEqual(cache.persistent_stats["demotions"], 2)


if __name__ == "__main__":
    unittest.main()
