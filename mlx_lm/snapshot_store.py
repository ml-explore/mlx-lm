# Copyright © 2026 Apple Inc.

"""Fail-closed disk tier for content-addressed prefix-fork snapshots.

Snapshots can contain conversation content and are protected like logs:
directories are mode 0700 and files are mode 0600.  Tier 1 is deliberately
uncompressed because compression prevents ``mx.load`` from memory mapping the
payload.  The store provides no repair, redundancy, lossy re-quantization,
OS-sleep handling, or adaptive sizing: cached KV is recomputable, so every
integrity doubt becomes quarantine plus a safe miss.
"""

import fcntl
import hashlib
import json
import logging
import math
import os
import time
import uuid
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .prefix_forks import FrozenPrefixSnapshot, compute_cid

FORMAT_VERSION = "1"
KNOWN_FEATURES = frozenset({"delta-chain", "payload-sha256"})
_PROCESS_LOCKS = {}


class SnapshotCorruptError(RuntimeError):
    """A snapshot cannot be trusted and must be treated as a miss."""


def _array_digest(array: mx.array) -> str:
    return hashlib.sha256(np.asarray(array).tobytes(order="C")).hexdigest()


def _read_header(path: Path) -> Tuple[dict, int]:
    """Read and validate only the safetensors JSON header."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise SnapshotCorruptError("truncated safetensors length")
        header_size = int.from_bytes(raw, "little")
        if header_size <= 0 or header_size > min(size - 8, 64 << 20):
            raise SnapshotCorruptError("invalid safetensors header length")
        try:
            header = json.loads(handle.read(header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SnapshotCorruptError("invalid safetensors header JSON") from exc
    if not isinstance(header, dict):
        raise SnapshotCorruptError("safetensors header is not an object")
    data_start = 8 + header_size
    spans = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        try:
            start, end = tensor["data_offsets"]
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotCorruptError("invalid tensor offsets") from exc
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end < start
            or end > size - data_start
        ):
            raise SnapshotCorruptError("tensor payload extends past file")
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise SnapshotCorruptError("gapped or overlapping tensor payload")
        cursor = end
    if cursor != size - data_start:
        raise SnapshotCorruptError("trailing or truncated tensor payload")
    return header, data_start


@dataclass
class SnapshotRecord:
    cid: str
    path: Path
    model_key: str
    tokens: Tuple[int, ...]
    created_at: float
    parent_cid: Optional[str]
    parent_length: int
    depth: int
    logical_nbytes: int
    file_bytes: int
    last_use: float
    digests: Tuple[Tuple[str, str], ...]


class DiskBackedSnapshot:
    """Header-only registry entry; KV payload is restored on first fork."""

    def __init__(self, store: "SnapshotStore", record: SnapshotRecord):
        self._store = store
        self._record = record
        self._forks = ()

    @property
    def cid(self):
        return self._record.cid

    @property
    def model_key(self):
        return self._record.model_key

    @property
    def tokens(self):
        return self._record.tokens

    @property
    def nbytes(self):
        return self._record.logical_nbytes

    @property
    def pinned(self):
        return False

    def __len__(self):
        return len(self.tokens)

    def fork(self):
        snapshot = self._store.restore(self.cid)
        return None if snapshot is None else snapshot.fork()


class SnapshotStore:
    """Content-addressed, delta-chained safetensors snapshot store."""

    def __init__(
        self,
        directory,
        *,
        max_bytes: int = 32 << 30,
        max_files: int = 4096,
        max_chain_depth: int = 8,
        registry=None,
    ):
        for name, value in (
            ("max_bytes", max_bytes),
            ("max_files", max_files),
            ("max_chain_depth", max_chain_depth),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be a non-negative integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        lock_key = str(self.directory.resolve())
        lock_fd = _PROCESS_LOCKS.get(lock_key)
        if lock_fd is None:
            lock_path = self.directory / ".writer.lock"
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(lock_fd)
                raise RuntimeError(
                    f"snapshot directory already has a live writer: {self.directory}"
                ) from exc
            os.chmod(lock_path, 0o600)
            _PROCESS_LOCKS[lock_key] = lock_fd
        self._lock_fd = lock_fd
        self.max_bytes = int(max_bytes)
        self.max_files = int(max_files)
        self.max_chain_depth = int(max_chain_depth)
        self.registry = registry
        self._records: "OrderedDict[str, SnapshotRecord]" = OrderedDict()
        self._ram_ghosts = deque(maxlen=256)
        self._disk_ghosts = deque(maxlen=256)
        self._scrub_cursor = 0
        self._stats = Counter()
        self.rebuild(registry)

    @property
    def stats(self):
        out = dict(self._stats)
        out.update(
            files=len(self._records),
            bytes=sum(r.file_bytes for r in self._records.values()),
        )
        return out

    def _parse_record(self, path: Path) -> SnapshotRecord:
        os.chmod(path, 0o600)
        header, _ = _read_header(path)
        metadata = header.get("__metadata__")
        if not isinstance(metadata, dict):
            raise SnapshotCorruptError("missing metadata")
        if metadata.get("format_version") != FORMAT_VERSION:
            raise SnapshotCorruptError("unknown format version")
        try:
            raw_features = json.loads(metadata.get("features", "[]"))
        except (TypeError, json.JSONDecodeError) as exc:
            raise SnapshotCorruptError("invalid feature flags") from exc
        if not isinstance(raw_features, list) or any(
            not isinstance(feature, str) for feature in raw_features
        ):
            raise SnapshotCorruptError("invalid feature flags")
        features = set(raw_features)
        unknown = features - KNOWN_FEATURES
        if unknown:
            raise SnapshotCorruptError(f"unknown features: {sorted(unknown)}")
        try:
            model_key = metadata["model_key"]
            tokens = tuple(json.loads(metadata["token_ids"]))
            created_at = float(metadata["created_at"])
            parent_cid = metadata.get("parent_cid") or None
            parent_length = int(metadata.get("parent_length", "0"))
            depth = int(metadata.get("chain_depth", "0"))
            logical_nbytes = int(metadata["logical_nbytes"])
            layers = int(metadata["layers"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SnapshotCorruptError("invalid snapshot metadata") from exc
        if not isinstance(model_key, str) or any(type(t) is not int for t in tokens):
            raise SnapshotCorruptError("invalid identity metadata")
        if (
            not math.isfinite(created_at)
            or created_at < 0
            or logical_nbytes < 0
            or layers <= 0
            or depth < 0
        ):
            raise SnapshotCorruptError("invalid numeric metadata")
        cid = path.stem
        if len(cid) != 64 or cid != compute_cid(model_key, tokens):
            raise SnapshotCorruptError("filename/cid identity mismatch")
        if parent_length < 0 or parent_length > len(tokens):
            raise SnapshotCorruptError("invalid parent length")
        if bool(parent_cid) != bool(parent_length):
            raise SnapshotCorruptError("inconsistent parent metadata")
        if parent_cid is not None and (
            len(parent_cid) != 64
            or any(char not in "0123456789abcdef" for char in parent_cid)
        ):
            raise SnapshotCorruptError("invalid parent cid")
        digests = []
        expected_names = {f"{side}.{layer}" for layer in range(layers) for side in "kv"}
        actual_names = {name for name in header if name != "__metadata__"}
        if actual_names != expected_names:
            raise SnapshotCorruptError("unexpected or missing layer tensor")
        for layer in range(layers):
            k_name, v_name = f"k.{layer}", f"v.{layer}"
            if k_name not in header or v_name not in header:
                raise SnapshotCorruptError("missing layer tensor")
            try:
                pair = (
                    metadata[f"sha256_k_{layer}"],
                    metadata[f"sha256_v_{layer}"],
                )
            except KeyError as exc:
                raise SnapshotCorruptError("missing payload digest") from exc
            if any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                for digest in pair
            ):
                raise SnapshotCorruptError("invalid payload digest")
            digests.append(pair)
        return SnapshotRecord(
            cid=cid,
            path=path,
            model_key=model_key,
            tokens=tokens,
            created_at=created_at,
            parent_cid=parent_cid,
            parent_length=parent_length,
            depth=depth,
            logical_nbytes=logical_nbytes,
            file_bytes=path.stat().st_size,
            last_use=created_at,
            digests=tuple(digests),
        )

    def _quarantine(self, path: Path, reason: str):
        if not path.exists():
            return
        target = path.with_suffix(".quarantined")
        if target.exists():
            target = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.quarantined")
        try:
            os.replace(path, target)
            os.chmod(target, 0o600)
        except OSError:
            logging.warning("Snapshot quarantine failed for %s", path)
        self._stats["quarantines"] += 1
        logging.warning("Snapshot %s quarantined: %s", path.stem, reason)

    def rebuild(self, registry=None):
        """Header-only directory scan; tensor payloads remain untouched."""
        if registry is not None:
            self.registry = registry
        self._records.clear()
        # A temp is never authoritative.  A previous crash may leave one.
        for path in self.directory.iterdir():
            if ".tmp-" in path.name:
                try:
                    path.unlink()
                    self._stats["temps_cleaned"] += 1
                except OSError:
                    pass
        parsed = []
        for path in self.directory.glob("*.safetensors"):
            try:
                parsed.append(self._parse_record(path))
            except (OSError, SnapshotCorruptError) as exc:
                self._quarantine(path, str(exc))
        for record in sorted(parsed, key=lambda r: r.last_use):
            self._records[record.cid] = record
        self._poison_missing_or_cyclic()
        self._load_index_lru()
        if self.registry is not None:
            for record in self._records.values():
                self.registry.register_snapshot(DiskBackedSnapshot(self, record))
        self._write_index()
        self.gc()
        return len(self._records)

    def _poison_missing_or_cyclic(self):
        bad = set()
        for cid, record in list(self._records.items()):
            if record.parent_cid is None:
                if record.parent_length != 0 or record.depth != 0:
                    bad.add(cid)
            else:
                parent = self._records.get(record.parent_cid)
                if (
                    parent is None
                    or parent.model_key != record.model_key
                    or record.parent_length != len(parent.tokens)
                    or parent.tokens != record.tokens[: record.parent_length]
                    or record.depth != parent.depth + 1
                    or record.depth > self.max_chain_depth
                ):
                    bad.add(cid)
            seen = set()
            current = cid
            while current:
                if current in seen or current not in self._records:
                    bad.add(cid)
                    break
                seen.add(current)
                current = self._records[current].parent_cid
        # Descendants of a bad node are bad too; repeat to a fixed point.
        changed = True
        while changed:
            changed = False
            for cid, record in self._records.items():
                if record.parent_cid in bad and cid not in bad:
                    bad.add(cid)
                    changed = True
        for cid in bad:
            record = self._records.pop(cid, None)
            if record is not None:
                self._quarantine(record.path, "missing or cyclic parent chain")

    def _load_index_lru(self):
        path = self.directory / "index.json"
        try:
            data = json.loads(path.read_text())
            lru = data.get("last_use", {})
        except (OSError, json.JSONDecodeError, AttributeError):
            return
        for cid, stamp in lru.items():
            if (
                cid in self._records
                and isinstance(stamp, (int, float))
                and math.isfinite(stamp)
            ):
                self._records[cid].last_use = float(stamp)
        self._records = OrderedDict(
            sorted(self._records.items(), key=lambda item: item[1].last_use)
        )

    def _write_index(self):
        path = self.directory / "index.json"
        if self.max_files == 0 or self.max_bytes == 0:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return
        temp = self.directory / f"index.json.tmp-{uuid.uuid4().hex}"
        data = {
            "format_version": FORMAT_VERSION,
            "last_use": {cid: rec.last_use for cid, rec in self._records.items()},
        }
        with temp.open("w") as handle:
            json.dump(data, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, 0o600)
        os.replace(temp, path)
        os.chmod(path, 0o600)

    def _choose_parent(self, snapshot) -> Optional[SnapshotRecord]:
        best = None
        tokens = snapshot.tokens
        for record in self._records.values():
            if (
                record.model_key == snapshot.model_key
                and len(record.tokens) < len(tokens)
                and tokens[: len(record.tokens)] == record.tokens
                and (best is None or len(record.tokens) > len(best.tokens))
            ):
                best = record
        # Do not silently fall back to a shorter ancestor: once the natural
        # longest chain reaches the cap, this write is a new whole root.
        return None if best is not None and best.depth >= self.max_chain_depth else best

    def save(self, snapshot: FrozenPrefixSnapshot) -> str:
        """Atomically save one RAM snapshot, delta-chained when profitable."""
        cid = snapshot.cid
        if cid in self._records:
            stored = self._restore(cid, record_hit=False, promote=False)
            if stored is None or not self._snapshots_equal(snapshot, stored):
                raise ValueError(
                    f"snapshot content mismatch for existing cid {cid}; "
                    "the model key must change whenever weights change"
                )
            self._touch(cid)
            return cid
        parent = self._choose_parent(snapshot)
        if parent is not None and not self._parent_matches(snapshot, parent):
            # KV can be chunk- or implementation-dependent even for the same
            # nominal weights/tokens.  Delta only when the actual bytes share
            # the claimed prefix; otherwise self-root to prevent a false hit.
            parent = None
        parent_length = len(parent.tokens) if parent is not None else 0
        keys = snapshot.keys
        values = snapshot.values
        tensors = {}
        metadata = {
            "format_version": FORMAT_VERSION,
            "features": json.dumps(sorted(KNOWN_FEATURES)),
            "model_key": snapshot.model_key,
            "token_ids": json.dumps(list(snapshot.tokens), separators=(",", ":")),
            "created_at": repr(time.time()),
            "parent_cid": parent.cid if parent is not None else "",
            "parent_length": str(parent_length),
            "chain_depth": str(parent.depth + 1 if parent is not None else 0),
            "logical_nbytes": str(snapshot.nbytes),
            "layers": str(len(keys)),
        }
        for layer, (key, value) in enumerate(zip(keys, values)):
            key = key[..., parent_length:, :]
            value = value[..., parent_length:, :]
            mx.eval(key, value)
            tensors[f"k.{layer}"] = key
            tensors[f"v.{layer}"] = value
            metadata[f"sha256_k_{layer}"] = _array_digest(key)
            metadata[f"sha256_v_{layer}"] = _array_digest(value)
        # Keep the safetensors suffix: MLX otherwise appends one, which would
        # make the path passed to chmod/replace differ from the file written.
        temp = self.directory / f".{cid}.tmp-{uuid.uuid4().hex}.safetensors"
        final = self.directory / f"{cid}.safetensors"
        try:
            mx.save_safetensors(str(temp), tensors, metadata)
            os.chmod(temp, 0o600)
            os.replace(temp, final)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        os.chmod(final, 0o600)
        record = self._parse_record(final)
        self._records[cid] = record
        self._stats["demotions"] += 1
        self._write_index()
        self.gc()
        return cid

    def _parent_matches(self, snapshot, parent: SnapshotRecord) -> bool:
        restored = self._restore(parent.cid, record_hit=False, promote=False)
        if restored is None:
            return False
        child_keys, child_values = snapshot.keys, snapshot.values
        if len(child_keys) != len(restored.keys) or len(child_values) != len(
            restored.values
        ):
            return False
        n = len(parent.tokens)
        for child, old in zip(
            child_keys + child_values, restored.keys + restored.values
        ):
            if (
                child.ndim != old.ndim
                or child.dtype != old.dtype
                or child.shape[:2] + child.shape[3:] != old.shape[:2] + old.shape[3:]
                or not bool(mx.array_equal(child[..., :n, :], old))
            ):
                return False
        return True

    @staticmethod
    def _snapshots_equal(left, right) -> bool:
        if (
            left.model_key != right.model_key
            or tuple(left.tokens) != tuple(right.tokens)
            or len(left.keys) != len(right.keys)
            or len(left.values) != len(right.values)
        ):
            return False
        return all(
            a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b))
            for a, b in zip(left.keys + left.values, right.keys + right.values)
        )

    def _load_delta(self, record: SnapshotRecord):
        try:
            arrays, _ = mx.load(str(record.path), return_metadata=True)
            keys, values = [], []
            for layer, (expected_k, expected_v) in enumerate(record.digests):
                key, value = arrays[f"k.{layer}"], arrays[f"v.{layer}"]
                tail_length = len(record.tokens) - record.parent_length
                if (
                    key.ndim != 4
                    or value.ndim != 4
                    or key.shape[:3] != value.shape[:3]
                    or key.shape[2] != tail_length
                ):
                    raise SnapshotCorruptError(
                        f"invalid tensor geometry at layer {layer}"
                    )
                # Verification is layer-incremental and occurs only when this
                # disk hit is materialized, never during startup indexing.
                mx.eval(key, value)
                if (
                    _array_digest(key) != expected_k
                    or _array_digest(value) != expected_v
                ):
                    raise SnapshotCorruptError(
                        f"payload digest mismatch at layer {layer}"
                    )
                keys.append(key)
                values.append(value)
            return keys, values
        except (
            OSError,
            KeyError,
            ValueError,
            RuntimeError,
            SnapshotCorruptError,
        ) as exc:
            raise SnapshotCorruptError(str(exc)) from exc

    def restore(self, cid: str) -> Optional[FrozenPrefixSnapshot]:
        """Walk and verify root→leaf, returning a materialized snapshot."""
        return self._restore(cid, record_hit=True, promote=True)

    def _restore(
        self, cid: str, *, record_hit: bool, promote: bool
    ) -> Optional[FrozenPrefixSnapshot]:
        if cid in self._disk_ghosts:
            self._stats["ghost_hits"] += 1
        record = self._records.get(cid)
        if record is None:
            return None
        chain, seen = [], set()
        current = record
        while current is not None:
            if current.cid in seen or len(chain) > self.max_chain_depth:
                self._quarantine_chain(chain + [current], "cyclic/deep chain")
                return None
            seen.add(current.cid)
            chain.append(current)
            current = (
                self._records.get(current.parent_cid) if current.parent_cid else None
            )
            if chain[-1].parent_cid and current is None:
                self._quarantine_chain(chain, "missing parent")
                return None
        keys = values = None
        try:
            for part in reversed(chain):
                delta_keys, delta_values = self._load_delta(part)
                if keys is None:
                    keys, values = delta_keys, delta_values
                else:
                    if (
                        not keys
                        or len(keys) != len(delta_keys)
                        or len(values) != len(delta_values)
                    ):
                        raise SnapshotCorruptError("incompatible layer counts in chain")
                    for old_k, old_v, new_k, new_v in zip(
                        keys, values, delta_keys, delta_values
                    ):
                        if (
                            old_k.ndim != 4
                            or old_v.ndim != 4
                            or new_k.ndim != 4
                            or new_v.ndim != 4
                            or old_k.dtype != new_k.dtype
                            or old_v.dtype != new_v.dtype
                            or old_k.shape[:2] + old_k.shape[3:]
                            != new_k.shape[:2] + new_k.shape[3:]
                            or old_v.shape[:2] + old_v.shape[3:]
                            != new_v.shape[:2] + new_v.shape[3:]
                        ):
                            raise SnapshotCorruptError(
                                "incompatible tensor geometry in chain"
                            )
                    keys = [
                        mx.concatenate([a, b], axis=2) for a, b in zip(keys, delta_keys)
                    ]
                    values = [
                        mx.concatenate([a, b], axis=2)
                        for a, b in zip(values, delta_values)
                    ]
            mx.eval(*keys, *values)
        except SnapshotCorruptError as exc:
            self._quarantine_descendants(part.cid, str(exc))
            return None
        if any(
            key.shape[2] != len(record.tokens) or value.shape[2] != len(record.tokens)
            for key, value in zip(keys, values)
        ):
            self._quarantine_chain(chain, "restored length mismatch")
            return None
        self._touch(cid)
        if record_hit:
            self._stats["disk_hits"] += 1
        if promote:
            self._stats["promotions"] += 1
        snapshot = FrozenPrefixSnapshot(
            cid, record.model_key, record.tokens, keys, values
        )
        if promote and self.registry is not None:
            self.registry.replace_snapshot(snapshot)
        return snapshot

    def _quarantine_chain(self, chain, reason):
        for record in chain:
            self._records.pop(record.cid, None)
            self._quarantine(record.path, reason)
            if self.registry is not None:
                self.registry.invalidate(record.cid)
        self._write_index()
        self.gc()

    def _quarantine_descendants(self, cid, reason):
        poisoned = {cid}
        changed = True
        while changed:
            changed = False
            for child, record in self._records.items():
                if record.parent_cid in poisoned and child not in poisoned:
                    poisoned.add(child)
                    changed = True
        self._quarantine_chain(
            [self._records[x] for x in poisoned if x in self._records], reason
        )

    def _touch(self, cid):
        record = self._records.pop(cid, None)
        if record is not None:
            record.last_use = time.time()
            self._records[cid] = record

    def flush_registry(self, registry=None):
        """Write every unsaved RAM snapshot; safe to call only while idle."""
        registry = registry or self.registry
        if registry is None:
            return 0
        count = 0
        for snapshot in registry.snapshots():
            if (
                isinstance(snapshot, FrozenPrefixSnapshot)
                and snapshot.cid not in self._records
            ):
                self.save(snapshot)
                count += 1
        return count

    def demote_from_ram(self, cid: str) -> bool:
        """Replace a discoverable RAM snapshot with its header-only proxy."""
        if self.registry is None:
            return False
        snapshot = self.registry.get(cid)
        if not isinstance(snapshot, FrozenPrefixSnapshot):
            return False
        already_on_disk = cid in self._records
        if not already_on_disk:
            self.save(snapshot)
        record = self._records.get(cid)
        if record is None:
            return False
        replaced = self.registry.replace_snapshot(DiskBackedSnapshot(self, record))
        if replaced:
            self._ram_ghosts.append(cid)
            if already_on_disk:
                self._stats["demotions"] += 1
        return replaced

    def scrub_one(self) -> bool:
        """Verify one snapshot's complete per-layer payload, round-robin."""
        if not self._records:
            return False
        cids = list(self._records)
        cid = cids[self._scrub_cursor % len(cids)]
        self._scrub_cursor += 1
        record = self._records[cid]
        try:
            self._load_delta(record)
            self._stats["scrubs"] += 1
            return True
        except SnapshotCorruptError as exc:
            self._quarantine_descendants(record.cid, str(exc))
            return False

    def note_ram_eviction(self, cid: str):
        """Record ARC-style RAM ghost telemetry (sizing remains static)."""
        self._ram_ghosts.append(cid)

    def note_request_cid(self, cid: str):
        """Count requests that would have hit either telemetry ghost list."""
        if cid in self._ram_ghosts or cid in self._disk_ghosts:
            self._stats["ghost_hits"] += 1

    def gc(self):
        """Enforce both caps, evicting least-recently-used leaves first."""

        def all_files():
            return [
                p
                for p in self.directory.iterdir()
                if p.is_file() and p.name != ".writer.lock"
            ]

        while True:
            files = all_files()
            total = sum(p.stat().st_size for p in files)
            if len(files) <= self.max_files and total <= self.max_bytes:
                break
            # Rebuildable/transient artifacts never deserve eviction
            # protection over valid KV data.
            candidates = [p for p in files if p.suffix != ".safetensors"]
            if candidates:
                victim = min(candidates, key=lambda p: p.stat().st_mtime)
                try:
                    victim.unlink()
                    continue
                except OSError:
                    pass
            parents = {r.parent_cid for r in self._records.values() if r.parent_cid}
            leaf = next(
                (r for r in self._records.values() if r.cid not in parents), None
            )
            if leaf is not None:
                self._records.pop(leaf.cid, None)
                try:
                    leaf.path.unlink()
                except OSError:
                    pass
                self._disk_ghosts.append(leaf.cid)
                self._stats["gc_files"] += 1
                if self.registry is not None:
                    self.registry.invalidate(leaf.cid)
                continue
            break
        return len(self._records)
