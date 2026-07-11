# Copyright © 2026 Apple Inc.

import copy
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, LRUPromptCache, make_prompt_cache
from mlx_lm.prefix_forks import FrozenPrefixSnapshot, PrefixForkRegistry, compute_cid
from mlx_lm.server import (
    PersistentPromptCache,
    ResponseGenerator,
    _persistent_model_key,
    _run_http_server,
    run,
)
from mlx_lm.snapshot_store import (
    SnapshotCorruptError,
    SnapshotStore,
    _array_digest,
    _read_header,
)
from mlx_lm.utils import load

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


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
        with self.assertRaises(SnapshotCorruptError):
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
        store.close()
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
        store.close()
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

    def test_quarantine_rename_failure_blocks_proxy_without_losing_record(self):
        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "m@r", list(range(8)))
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
        store.close()
        restarted = PrefixForkRegistry()
        reopened = SnapshotStore(self.path, registry=restarted)
        original = os.replace

        def refuse(source, target):
            if Path(source) == path:
                raise PermissionError("injected")
            return original(source, target)

        with patch("mlx_lm.snapshot_store.os.replace", side_effect=refuse):
            forks, _ = restarted.fetch_fork("m@r", list(range(8)))
        self.assertIsNone(forks)
        self.assertTrue(path.exists())
        self.assertIn(snapshot.cid, reopened._records)
        self.assertIsNone(restarted.get(snapshot.cid))

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
        store.close()
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
        store.close()
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
        store.close()
        restarted = PrefixForkRegistry()
        rebuilt = SnapshotStore(self.path, registry=restarted)
        forks, rest = restarted.fetch_fork("m@r", list(range(16)))
        # Corrupt child is gone; the valid root remains a safe shorter hit.
        self.assertIsNotNone(forks)
        self.assertEqual(forks[0].prefix_length, 8)
        self.assertEqual(rest, list(range(8, 16)))
        self.assertGreaterEqual(rebuilt.stats["quarantines"], 1)

    def test_forged_logical_nbytes_quarantined(self):
        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "m@r", list(range(8)))
        store = SnapshotStore(self.path)
        store.save(snapshot)
        path = self.path / f"{snapshot.cid}.safetensors"
        arrays, metadata = mx.load(str(path), return_metadata=True)
        metadata["logical_nbytes"] = str(1 << 40)
        mx.save_safetensors(str(path), arrays, metadata)
        store.close()
        restarted = PrefixForkRegistry()
        reopened = SnapshotStore(self.path, registry=restarted)
        self.assertEqual(len(restarted), 0)
        self.assertFalse(path.exists())
        reopened.close()

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
        store.note_request("m@r", list(range(8)) + [9])
        self.assertEqual(store.stats["ghost_hits"], 1)

    def test_single_owner_close_and_reopen(self):
        store = SnapshotStore(self.path)
        with self.assertRaises(RuntimeError):
            SnapshotStore(self.path)
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "from mlx_lm.snapshot_store import SnapshotStore; "
                f"SnapshotStore({str(self.path)!r})",
            ],
            env={**os.environ, "PYTHONPATH": "."},
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(child.returncode, 0)
        self.assertIn("live writer", child.stderr)
        if hasattr(os, "fork"):
            pid = os.fork()
            if pid == 0:
                try:
                    SnapshotStore(self.path)
                except RuntimeError:
                    os._exit(0)
                os._exit(1)
            _, status = os.waitpid(pid, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)
        store.close()
        reopened = SnapshotStore(self.path)
        reopened.close()

    def test_disk_proxy_freeze_materializes_cleanly(self):
        registry = PrefixForkRegistry()
        tokens = list(range(8))
        snapshot = _freeze(registry, "m@r", tokens)
        store = SnapshotStore(self.path)
        store.save(snapshot)
        store.close()
        restarted = PrefixForkRegistry()
        reopened = SnapshotStore(self.path, registry=restarted)
        self.assertEqual(restarted.freeze("m@r", tokens, _cache(8)), snapshot.cid)
        reopened.close()

    def test_gc_unlink_failure_retains_bookkeeping(self):
        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "m@r", list(range(8)))
        store = SnapshotStore(self.path)
        store.save(snapshot)
        path = self.path / f"{snapshot.cid}.safetensors"
        store.max_bytes = path.stat().st_size - 1
        original = Path.unlink

        def refuse(target, *args, **kwargs):
            if target == path:
                raise PermissionError("injected")
            return original(target, *args, **kwargs)

        with patch.object(Path, "unlink", refuse), self.assertRaises(RuntimeError):
            store.gc()
        self.assertTrue(path.exists())
        self.assertIn(snapshot.cid, store._records)

    def test_orphan_safetensors_is_gc_candidate(self):
        store = SnapshotStore(self.path)
        orphan = self.path / ("f" * 64 + ".safetensors")
        orphan.write_bytes(b"x" * 4096)
        store.max_bytes = 1
        store.gc()
        self.assertFalse(orphan.exists())

    def test_hit_lru_persists_on_flush_index(self):
        registry = PrefixForkRegistry()
        a = _freeze(registry, "m@r", list(range(8)), seed=1)
        b = _freeze(registry, "m@r", list(range(100, 108)), seed=2)
        store = SnapshotStore(self.path)
        store.save(a)
        store.save(b)
        store.restore(a.cid)
        runtime = list(store._records)
        store.flush_index()
        store.close()
        rebooted = SnapshotStore(self.path)
        self.assertEqual(list(rebooted._records), runtime)

    def test_mmap_survives_unlink_before_touch(self):
        path = self.path / "mmap.safetensors"
        mx.save_safetensors(str(path), {"x": mx.arange(4096)}, {})
        arrays = mx.load(str(path))
        path.unlink()
        self.assertEqual(int(mx.sum(arrays["x"]).item()), sum(range(4096)))

    def test_flush_and_scrub_respect_byte_budgets(self):
        registry = PrefixForkRegistry()
        store = SnapshotStore(self.path, registry=registry)
        snapshot = _freeze(registry, "m@r", list(range(64)))
        self.assertEqual(
            store.flush_registry(registry, max_snapshots=1, max_bytes=1), 0
        )
        self.assertFalse(list(self.path.glob("*.safetensors")))
        store.save(snapshot)
        self.assertFalse(store.scrub_increment(max_bytes=17))
        self.assertIsNotNone(store._scrub_state)

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
        self.assertEqual(first.flush_persistent(scrub=False, shutdown=True), 1)
        first.close()

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
        restarted.close()

    def test_local_identity_hashes_content_not_size_or_mtime(self):
        weights = self.model_dir / "weights.safetensors"
        weights.write_bytes(b"A" * 4096)
        stamp = weights.stat().st_mtime_ns
        first = _persistent_model_key((str(self.model_dir), None, None))
        weights.write_bytes(b"B" * 4096)
        os.utime(weights, ns=(stamp, stamp))
        second = _persistent_model_key((str(self.model_dir), None, None))
        self.assertNotEqual(first, second)

    def test_hf_snapshot_identity_opens_no_payload_and_tracks_revision(self):
        repo = Path(self.temp.name) / "models--org--model" / "snapshots"
        (repo.parent / "blobs").mkdir(parents=True)
        revision_a = "a" * 40
        revision_b = "b" * 40
        snapshot_a = repo / revision_a
        snapshot_b = repo / revision_b
        snapshot_a.mkdir(parents=True)
        snapshot_b.mkdir()
        (snapshot_a / "weights.safetensors").write_bytes(b"A" * 4096)
        (snapshot_b / "weights.safetensors").write_bytes(b"A" * 4096)
        with patch.object(
            Path,
            "open",
            side_effect=AssertionError("HF payload must not be opened for identity"),
        ):
            identity_a = _persistent_model_key((str(snapshot_a), None, None))
            identity_b = _persistent_model_key((str(snapshot_b), None, None))
        self.assertNotEqual(identity_a, identity_b)
        self.assertIn(revision_a, identity_a)
        self.assertIn(revision_b, identity_b)

    def test_local_snapshots_lookalike_still_hashes_bytes(self):
        lookalike = Path(self.temp.name) / "ordinary-local" / "snapshots" / ("c" * 40)
        lookalike.mkdir(parents=True)
        weights = lookalike / "adapter.safetensors"
        weights.write_bytes(b"A" * 4096)
        stamp = weights.stat().st_mtime_ns
        first = _persistent_model_key((str(lookalike), None, None))
        weights.write_bytes(b"B" * 4096)
        os.utime(weights, ns=(stamp, stamp))
        second = _persistent_model_key((str(lookalike), None, None))
        self.assertNotEqual(first, second)

    def test_unresolved_hf_identity_fails_closed(self):
        with self.assertRaises(ValueError):
            _persistent_model_key(("org/not-loader-resolved", None, None))

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

    def test_off_switch_constructs_plain_cache_and_no_files(self):
        provider = type(
            "Provider",
            (),
            {"cli_args": Namespace(prompt_cache_size=3)},
        )()
        with patch("mlx_lm.server.ResponseGenerator") as response, patch(
            "mlx_lm.server._run_http_server"
        ):
            run("127.0.0.1", 0, provider)
        self.assertIsInstance(response.call_args.args[1], LRUPromptCache)
        self.assertFalse(self.store_dir.exists())

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
        cache.close()

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
        cache.close()

    def test_maintenance_failure_is_contained(self):
        cache = PersistentPromptCache(
            4, self.store_dir, max_bytes=1 << 20, max_files=32
        )
        with patch.object(
            cache, "_maintenance_tick", side_effect=RuntimeError("disk full")
        ):
            cache.flush_persistent()
            while not cache._maintenance_future.done():
                time.sleep(0.001)
            self.assertEqual(cache.flush_persistent(), 0)
        cache.close()

    def test_idle_maintenance_never_blocks_request_thread(self):
        cache = PersistentPromptCache(
            4, self.store_dir, max_bytes=1 << 20, max_files=32
        )
        with patch.object(
            cache, "_maintenance_tick", side_effect=lambda _: time.sleep(0.2)
        ):
            start = time.monotonic()
            cache.flush_persistent()
            self.assertLess(time.monotonic() - start, 0.05)
        cache.close()

    def test_non_keyboard_server_exit_always_stops_generator(self):
        class Generator:
            stopped = False

            def stop_and_join(self):
                self.stopped = True

        class BoomServer:
            address_family = socket.AF_INET

            def __init__(self, *args, **kwargs):
                pass

            def serve_forever(self):
                raise RuntimeError("injected")

            def shutdown(self):
                pass

        generator = Generator()
        with patch(
            "mlx_lm.server.socket.getaddrinfo",
            return_value=[(socket.AF_INET, None, None, None, ("127.0.0.1", 0))],
        ), self.assertRaises(RuntimeError):
            _run_http_server("127.0.0.1", 0, generator, server_class=BoomServer)
        self.assertTrue(generator.stopped)


class TestPersistentRoundTripWithModel(unittest.TestCase):
    def test_restart_continuation_matches_never_persisted(self):
        model, tokenizer = load(HF_MODEL_PATH)
        tokens = tokenizer.encode("The careful archivist checked every page. " * 12)
        prefix, suffix = tokens[:-3], tokens[-3:]
        cache = make_prompt_cache(model)
        model(mx.array(prefix)[None], cache=cache)
        mx.eval(*(array for layer in cache for array in layer.state))
        baseline = copy.deepcopy(cache)
        registry = PrefixForkRegistry()
        cid = registry.freeze(HF_MODEL_PATH + "@resolved-test", prefix, cache)
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotStore(directory)
            store.save(registry.get(cid))
            store.close()
            restarted = PrefixForkRegistry()
            reopened = SnapshotStore(directory, registry=restarted)
            forks, remaining = restarted.fetch_fork(
                HF_MODEL_PATH + "@resolved-test", tokens
            )
            expected = [
                int(token)
                for token, _ in generate_step(
                    mx.array(suffix), model, prompt_cache=baseline, max_tokens=8
                )
            ]
            actual = [
                int(token)
                for token, _ in generate_step(
                    mx.array(remaining), model, prompt_cache=forks, max_tokens=8
                )
            ]
            self.assertEqual(actual, expected)
            reopened.close()


if __name__ == "__main__":
    unittest.main()
