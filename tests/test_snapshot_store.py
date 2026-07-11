# Copyright © 2026 Apple Inc.

import copy
import hashlib
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
    _payload_digests,
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


def _prefix_cache(length, dtype=mx.float32):
    cache = KVCache()
    positions = mx.arange(length, dtype=mx.float32)[None, None, :, None]
    heads = mx.arange(2, dtype=mx.float32)[None, :, None, None] * 100
    dims = mx.arange(4, dtype=mx.float32)[None, None, None, :]
    key = mx.broadcast_to(positions + heads + dims, (1, 2, length, 4)).astype(dtype)
    value = key + 1000
    cache.update_and_fetch(key, value)
    mx.eval(*cache.state)
    return [cache]


def _save_with_payload_digests(path, arrays, metadata):
    layers = int(metadata["layers"])
    for layer in range(layers):
        metadata[f"sha256_k_{layer}"] = "0" * 64
        metadata[f"sha256_v_{layer}"] = "0" * 64
    mx.save_safetensors(str(path), arrays, metadata)
    for layer, (key_digest, value_digest) in enumerate(_payload_digests(path, layers)):
        metadata[f"sha256_k_{layer}"] = key_digest
        metadata[f"sha256_v_{layer}"] = value_digest
    mx.save_safetensors(str(path), arrays, metadata)


def _rewrite_header(path, mutate):
    raw = path.read_bytes()
    header_size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_size])
    payload = raw[8 + header_size :]
    mutate(header)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def _save_legacy_v1(path, snapshot):
    tensors = {}
    metadata = {
        "format_version": "1",
        "features": json.dumps(["delta-chain", "payload-sha256"]),
        "model_key": snapshot.model_key,
        "token_ids": json.dumps(list(snapshot.tokens), separators=(",", ":")),
        "created_at": "1.0",
        "parent_cid": "",
        "parent_length": "0",
        "chain_depth": "0",
        "logical_nbytes": str(snapshot.nbytes),
        "layers": str(len(snapshot.keys)),
    }
    for layer, (key, value) in enumerate(zip(snapshot.keys, snapshot.values)):
        tensors[f"k.{layer}"] = key
        tensors[f"v.{layer}"] = value
        metadata[f"sha256_k_{layer}"] = "0" * 64
        metadata[f"sha256_v_{layer}"] = "0" * 64
    mx.save_safetensors(str(path), tensors, metadata)
    header, data_start = _read_header(path)
    raw = path.read_bytes()
    for layer in range(len(snapshot.keys)):
        for side in "kv":
            start, end = header[f"{side}.{layer}"]["data_offsets"]
            metadata[f"sha256_{side}_{layer}"] = hashlib.sha256(
                raw[data_start + start : data_start + end]
            ).hexdigest()
    mx.save_safetensors(str(path), tensors, metadata)


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

    def test_two_pass_failure_windows_leave_no_visible_or_transient_snapshot(self):
        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "m@r", list(range(8)))
        original_save = mx.save_safetensors
        original_digests = _payload_digests
        original_replace = os.replace

        def exercise(label, save_effect=None, digest_effect=None, replace_effect=None):
            directory = self.path / label
            store = SnapshotStore(directory)
            save_calls = 0
            digest_calls = 0

            def save_wrapper(*args, **kwargs):
                nonlocal save_calls
                save_calls += 1
                if save_effect is not None:
                    save_effect(save_calls)
                return original_save(*args, **kwargs)

            def digest_wrapper(*args, **kwargs):
                nonlocal digest_calls
                digest_calls += 1
                result = original_digests(*args, **kwargs)
                return digest_effect(digest_calls, result) if digest_effect else result

            def replace_wrapper(source, target):
                if replace_effect is not None:
                    replace_effect(Path(source), Path(target))
                return original_replace(source, target)

            with patch(
                "mlx_lm.snapshot_store.mx.save_safetensors", side_effect=save_wrapper
            ), patch(
                "mlx_lm.snapshot_store._payload_digests",
                side_effect=digest_wrapper,
            ), patch(
                "mlx_lm.snapshot_store.os.replace", side_effect=replace_wrapper
            ), self.assertRaises(
                (OSError, RuntimeError, SnapshotCorruptError)
            ):
                store.save(snapshot)
            self.assertFalse((directory / f"{snapshot.cid}.safetensors").exists())
            self.assertNotIn(snapshot.cid, store._records)
            self.assertFalse(
                [
                    path
                    for path in directory.iterdir()
                    if ".digest-" in path.name or ".tmp-" in path.name
                ]
            )
            store.close()

        exercise(
            "stage-save",
            save_effect=lambda call: (
                (_ for _ in ()).throw(RuntimeError("stage")) if call == 1 else None
            ),
        )
        exercise(
            "stage-digest",
            digest_effect=lambda call, result: (
                (_ for _ in ()).throw(SnapshotCorruptError("stage digest"))
                if call == 1
                else result
            ),
        )
        exercise(
            "final-save",
            save_effect=lambda call: (
                (_ for _ in ()).throw(RuntimeError("final")) if call == 2 else None
            ),
        )

        def mismatch(call, result):
            if call != 2:
                return result
            altered = [list(pair) for pair in result]
            altered[0][0] = "f" * 64
            return tuple(tuple(pair) for pair in altered)

        exercise("final-digest", digest_effect=mismatch)
        exercise(
            "final-digest-error",
            digest_effect=lambda call, result: (
                (_ for _ in ()).throw(SnapshotCorruptError("final digest"))
                if call == 2
                else result
            ),
        )
        exercise(
            "replace",
            replace_effect=lambda source, target: (
                (_ for _ in ()).throw(OSError("replace"))
                if ".tmp-" in source.name
                else None
            ),
        )

    def test_transient_unlink_retry_cleans_stage(self):
        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "m@r", list(range(8)))
        store = SnapshotStore(self.path)
        original = Path.unlink
        failures = 0

        def flaky(path, *args, **kwargs):
            nonlocal failures
            if ".digest-" in path.name and failures == 0:
                failures += 1
                raise PermissionError("retry me")
            return original(path, *args, **kwargs)

        with patch.object(Path, "unlink", flaky):
            store.save(snapshot)
        self.assertEqual(failures, 1)
        self.assertFalse([p for p in self.path.iterdir() if ".digest-" in p.name])

    def test_persistent_transient_unlink_failure_is_cleaned_on_restart(self):
        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "m@r", list(range(8)))
        store = SnapshotStore(self.path)
        original = Path.unlink

        def refuse_stage(path, *args, **kwargs):
            if ".digest-" in path.name:
                raise PermissionError("persistent injected failure")
            return original(path, *args, **kwargs)

        with patch.object(Path, "unlink", refuse_stage), self.assertRaises(
            RuntimeError
        ):
            store.save(snapshot)
        self.assertFalse((self.path / f"{snapshot.cid}.safetensors").exists())
        self.assertNotIn(snapshot.cid, store._records)
        self.assertEqual(len(list(self.path.glob("*.digest-*.safetensors"))), 1)
        store.close()

        restarted = SnapshotStore(self.path)
        self.assertFalse(list(self.path.glob("*.digest-*.safetensors")))
        self.assertEqual(restarted.stats["temps_cleaned"], 1)
        restarted.close()

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
                "format_version": "2",
                "features": json.dumps(["future-feature"]),
                "model_key": "m@r",
                "token_ids": json.dumps(list(range(8))),
            },
        )
        SnapshotStore(self.path)
        self.assertFalse(other.exists())

    def test_legacy_v1_old_writer_is_deliberate_safe_miss(self):
        registry = PrefixForkRegistry()
        tokens = list(range(8))
        snapshot = _freeze(registry, "legacy@r", tokens)
        path = self.path / f"{snapshot.cid}.safetensors"
        _save_legacy_v1(path, snapshot)
        restarted = PrefixForkRegistry()
        store = SnapshotStore(self.path, registry=restarted)
        forks, remaining = restarted.fetch_fork("legacy@r", tokens)
        self.assertIsNone(forks)
        self.assertEqual(remaining, tokens)
        self.assertFalse(path.exists())
        self.assertGreaterEqual(store.stats["quarantines"], 1)

    def test_malformed_descriptors_quarantine_at_startup_and_restore(self):
        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "m@r", list(range(8)))
        store = SnapshotStore(self.path)
        store.save(snapshot)
        path = self.path / f"{snapshot.cid}.safetensors"
        store.close()
        _rewrite_header(path, lambda header: header["k.0"].pop("dtype"))
        restarted = PrefixForkRegistry()
        reopened = SnapshotStore(self.path, registry=restarted)
        self.assertEqual(len(restarted), 0)
        self.assertFalse(path.exists())
        reopened.close()

        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "bad-dtype@r", list(range(8)))
        store = SnapshotStore(self.path)
        store.save(snapshot)
        path = self.path / f"{snapshot.cid}.safetensors"
        store.close()
        _rewrite_header(path, lambda header: header["k.0"].update(dtype=7))
        restarted = PrefixForkRegistry()
        reopened = SnapshotStore(self.path, registry=restarted)
        self.assertEqual(len(restarted), 0)
        self.assertFalse(path.exists())
        reopened.close()

        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "m2@r", list(range(8)))
        store = SnapshotStore(self.path)
        store.save(snapshot)
        path = self.path / f"{snapshot.cid}.safetensors"
        store.close()
        restarted = PrefixForkRegistry()
        reopened = SnapshotStore(self.path, registry=restarted)
        _rewrite_header(path, lambda header: header["v.0"].update(shape=[1, 2, 9, 4]))
        forks, remaining = restarted.fetch_fork("m2@r", list(range(8)))
        self.assertIsNone(forks)
        self.assertEqual(remaining, list(range(8)))
        self.assertFalse(path.exists())
        reopened.close()

        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "bad-layers@r", list(range(8)))
        store = SnapshotStore(self.path)
        store.save(snapshot)
        path = self.path / f"{snapshot.cid}.safetensors"
        store.close()
        _rewrite_header(
            path,
            lambda header: header["__metadata__"].update(layers="1000000000"),
        )
        restarted = PrefixForkRegistry()
        reopened = SnapshotStore(self.path, registry=restarted)
        self.assertEqual(len(restarted), 0)
        self.assertFalse(path.exists())
        reopened.close()

        registry = PrefixForkRegistry()
        snapshot = _freeze(registry, "bad-shape@r", list(range(8)))
        store = SnapshotStore(self.path)
        store.save(snapshot)
        path = self.path / f"{snapshot.cid}.safetensors"
        store.close()
        restarted = PrefixForkRegistry()
        reopened = SnapshotStore(self.path, registry=restarted)
        _rewrite_header(path, lambda header: header["v.0"].update(shape="oops"))
        forks, remaining = restarted.fetch_fork("bad-shape@r", list(range(8)))
        self.assertIsNone(forks)
        self.assertEqual(remaining, list(range(8)))
        self.assertFalse(path.exists())
        reopened.close()

    def test_over_limit_json_integer_is_normalized_to_corruption(self):
        path = self.path / "malicious.safetensors"
        encoded = (
            b'{"x":{"dtype":"F32","shape":[' + b"9" * 5000 + b'],"data_offsets":[0,0]}}'
        )
        path.write_bytes(len(encoded).to_bytes(8, "little") + encoded)

        with self.assertRaises(SnapshotCorruptError):
            _read_header(path)

        nested = self.path / "deeply-nested.safetensors"
        depth = 2000
        encoded = b"[" * depth + b"0" + b"]" * depth
        nested.write_bytes(len(encoded).to_bytes(8, "little") + encoded)
        with self.assertRaises(SnapshotCorruptError):
            _read_header(nested)

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

    def test_bfloat16_round_trip_corruption_and_incremental_scrub(self):
        registry = PrefixForkRegistry()
        tokens = list(range(8))
        cid = registry.freeze("bf16@r", tokens, _prefix_cache(8, mx.bfloat16))
        snapshot = registry.get(cid)
        store = SnapshotStore(self.path)
        store.save(snapshot)
        while not store.scrub_increment(max_bytes=13):
            self.assertIsNotNone(store._scrub_state)
        store.close()

        restarted = PrefixForkRegistry()
        reopened = SnapshotStore(self.path, registry=restarted)
        forks, rest = restarted.fetch_fork("bf16@r", tokens + [9])
        self.assertEqual(rest, [9])
        self.assertEqual(forks[0].state[0].dtype, mx.bfloat16)
        self.assertTrue(mx.array_equal(forks[0].state[0], snapshot.keys[0]))
        reopened.demote_from_ram(cid)
        path = self.path / f"{cid}.safetensors"
        header, data_start = _read_header(path)
        offset = data_start + header["v.0"]["data_offsets"][0]
        with path.open("r+b") as handle:
            handle.seek(offset)
            byte = handle.read(1)
            handle.seek(offset)
            handle.write(bytes([byte[0] ^ 1]))
        corrupt, remaining = restarted.fetch_fork("bf16@r", tokens)
        self.assertIsNone(corrupt)
        self.assertEqual(remaining, tokens)

    def test_payload_digest_distinguishes_dtype_with_same_zero_bytes(self):
        bf_registry = PrefixForkRegistry()
        u16_registry = PrefixForkRegistry()
        tokens = list(range(8))
        bf_cache = _prefix_cache(8, mx.bfloat16)
        u16_cache = _prefix_cache(8, mx.uint16)
        # Force identical all-zero payload bytes; only dtype metadata differs.
        for cache in bf_cache + u16_cache:
            cache.keys = mx.zeros_like(cache.keys)
            cache.values = mx.zeros_like(cache.values)
        bf_cid = bf_registry.freeze("bf@r", tokens, bf_cache)
        u16_cid = u16_registry.freeze("u16@r", tokens, u16_cache)
        store = SnapshotStore(self.path)
        store.save(bf_registry.get(bf_cid))
        store.save(u16_registry.get(u16_cid))
        self.assertNotEqual(
            store._records[bf_cid].digests, store._records[u16_cid].digests
        )

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
        store.max_bytes = 32 << 30
        self.assertEqual(store.save(child), child.cid)

    def test_reservation_evicts_deterministic_lru_leaf(self):
        registry = PrefixForkRegistry()
        older = _freeze(registry, "older@r", list(range(8)), seed=1)
        newer = _freeze(registry, "newer@r", list(range(100, 108)), seed=2)
        incoming = _freeze(registry, "incoming@r", list(range(200, 208)), seed=3)
        store = SnapshotStore(self.path)
        store.save(older)
        store.save(newer)
        # Make recency disagree with insertion order so this detects a
        # first-eligible implementation instead of explicit LRU selection.
        store._records[older.cid].last_use = 2.0
        store._records[newer.cid].last_use = 1.0
        store.max_files = 2

        store.save(incoming)

        self.assertIn(older.cid, store._records)
        self.assertNotIn(newer.cid, store._records)
        self.assertIn(incoming.cid, store._records)

    def test_impossible_reservation_does_not_evict_existing_snapshot(self):
        registry = PrefixForkRegistry()
        existing = _freeze(registry, "existing@r", list(range(8)), seed=1)
        incoming = _freeze(registry, "incoming@r", list(range(100, 108)), seed=2)
        store = SnapshotStore(self.path)
        store.save(existing)
        existing_path = self.path / f"{existing.cid}.safetensors"
        store.max_bytes = 1

        with self.assertRaisesRegex(RuntimeError, "exceeds max_bytes"):
            store.save(incoming)

        self.assertTrue(existing_path.exists())
        self.assertIn(existing.cid, store._records)
        self.assertFalse((self.path / f"{incoming.cid}.safetensors").exists())

        store.close()
        file_cap_path = self.path / "file-cap"
        store = SnapshotStore(file_cap_path)
        store.save(existing)
        existing_path = file_cap_path / f"{existing.cid}.safetensors"
        store.max_files = 0

        with self.assertRaisesRegex(RuntimeError, "max_files >= 1"):
            store.save(incoming)

        self.assertTrue(existing_path.exists())
        self.assertIn(existing.cid, store._records)
        self.assertFalse((file_cap_path / f"{incoming.cid}.safetensors").exists())

    def test_infeasible_child_reservation_preserves_all_existing_state(self):
        source = PrefixForkRegistry()
        root = source.get(source.freeze("m@r", list(range(8)), _prefix_cache(8)))
        child = source.get(source.freeze("m@r", list(range(16)), _prefix_cache(16)))
        unrelated = source.get(
            source.freeze("other@r", list(range(100, 108)), _prefix_cache(8))
        )
        first = SnapshotStore(self.path)
        first.save(root)
        first.save(unrelated)
        first.close()

        registry = PrefixForkRegistry()
        store = SnapshotStore(self.path, registry=registry)
        captured = {}

        class ReservationProbe(RuntimeError):
            pass

        def capture_reservation(reservation, protected_cids=None):
            captured["reservation"] = reservation
            captured["protected"] = set(protected_cids or ())
            raise ReservationProbe

        with patch.object(
            store, "_reserve_space", side_effect=capture_reservation
        ), self.assertRaises(ReservationProbe):
            store.save(child)
        self.assertEqual(captured["protected"], {root.cid})
        store.max_bytes = captured["reservation"]

        low_free_files = {
            path.name: path.read_bytes()
            for path in self.path.iterdir()
            if path.is_file()
        }
        low_free_records = copy.deepcopy(tuple(store._records.items()))
        low_free_ghosts = tuple(store._disk_ghosts)
        low_free_stats = store.stats
        low_free_index_dirty = store._index_dirty
        with patch(
            "mlx_lm.snapshot_store.shutil.disk_usage",
            return_value=Namespace(total=100, used=100, free=0),
        ), self.assertRaisesRegex(RuntimeError, "insufficient free disk space"):
            store._reserve_space(captured["reservation"], {root.cid})
        self.assertEqual(
            {
                path.name: path.read_bytes()
                for path in self.path.iterdir()
                if path.is_file()
            },
            low_free_files,
        )
        self.assertEqual(tuple(store._records.items()), low_free_records)
        self.assertEqual(tuple(store._disk_ghosts), low_free_ghosts)
        self.assertEqual(store.stats, low_free_stats)
        self.assertEqual(store._index_dirty, low_free_index_dirty)

        files_before = {
            path.name: path.read_bytes()
            for path in self.path.iterdir()
            if path.is_file()
        }
        records_before = copy.deepcopy(tuple(store._records.items()))
        proxies_before = {cid: registry.get(cid) for cid in (root.cid, unrelated.cid)}
        index_before = (self.path / "index.json").read_bytes()
        ghosts_before = tuple(store._disk_ghosts)
        stats_before = store.stats
        index_dirty_before = store._index_dirty

        original_reserve = store._reserve_space

        def reserve_at_exact_cap(reservation, protected_cids=None):
            store.max_bytes = reservation
            return original_reserve(reservation, protected_cids)

        with patch.object(
            store, "_reserve_space", side_effect=reserve_at_exact_cap
        ), self.assertRaisesRegex(RuntimeError, "no evictable leaf"):
            store.save(child)

        self.assertEqual(
            {
                path.name: path.read_bytes()
                for path in self.path.iterdir()
                if path.is_file()
            },
            files_before,
        )
        self.assertEqual(tuple(store._records.items()), records_before)
        for cid, proxy in proxies_before.items():
            self.assertIs(registry.get(cid), proxy)
        self.assertEqual((self.path / "index.json").read_bytes(), index_before)
        self.assertEqual(tuple(store._disk_ghosts), ghosts_before)
        self.assertEqual(store.stats, stats_before)
        self.assertEqual(store._index_dirty, index_dirty_before)
        self.assertNotIn(child.cid, store._records)
        self.assertFalse((self.path / f"{child.cid}.safetensors").exists())
        self.assertFalse(
            [
                path
                for path in self.path.iterdir()
                if ".digest-" in path.name or ".tmp-" in path.name
            ]
        )
        store.close()

    def test_reservation_protects_incoming_parent_chain(self):
        registry = PrefixForkRegistry()
        root = registry.get(registry.freeze("m@r", list(range(8)), _prefix_cache(8)))
        child = registry.get(registry.freeze("m@r", list(range(16)), _prefix_cache(16)))
        store = SnapshotStore(self.path)
        store.save(root)
        root_path = self.path / f"{root.cid}.safetensors"
        store.max_files = 1

        with self.assertRaisesRegex(RuntimeError, "no evictable leaf"):
            store.save(child)

        self.assertTrue(root_path.exists())
        self.assertIn(root.cid, store._records)
        self.assertNotIn(child.cid, store._records)
        self.assertFalse(
            [
                path
                for path in self.path.iterdir()
                if ".digest-" in path.name or ".tmp-" in path.name
            ]
        )

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
        _save_with_payload_digests(path, arrays, metadata)
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
        with patch(
            "mlx_lm.server.hf_constants.HF_HUB_CACHE", str(Path(self.temp.name))
        ), patch.object(
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
        lookalike = (
            Path(self.temp.name)
            / "ordinary-local"
            / "models--org--model"
            / "snapshots"
            / ("c" * 40)
        )
        lookalike.mkdir(parents=True)
        (lookalike.parent.parent / "blobs").mkdir()
        weights = lookalike / "adapter.safetensors"
        weights.write_bytes(b"A" * 4096)
        stamp = weights.stat().st_mtime_ns
        trusted_root = Path(self.temp.name) / "configured-hf-cache"
        trusted_root.mkdir()
        with patch("mlx_lm.server.hf_constants.HF_HUB_CACHE", str(trusted_root)):
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
