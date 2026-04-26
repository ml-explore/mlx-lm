# Copyright © 2026 Apple Inc.
"""Disk-backed L2 prompt cache for mlx_lm.

See docs/disk_prompt_cache.md for the design overview.

This module is opt-in: it is only constructed when mlx_lm.server is given
``--prompt-cache-disk-dir``. With ``disk=None`` (the default) ``LRUPromptCache``
behaves identically to the pre-PR baseline.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import errno
import fcntl
import hashlib
import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

from .models.cache import (
    can_trim_prompt_cache,
    load_prompt_cache,
    save_prompt_cache,
)

logger = logging.getLogger("mlx_lm.prompt_cache.disk")

FORMAT_VERSION = 1
"""On-disk schema version. Bumped if the schema changes."""

_SHUTDOWN_SENTINEL = object()
"""Sentinel object pushed into the writer queue to signal graceful drain."""


def hash_tokens(tokens: List[int]) -> str:
    """Stable 16-hex-char content hash of a token sequence.

    Used as the on-disk filename for a leaf. Collision probability for ~10K
    entries is ~10^-15, negligible.
    """
    import struct

    # Use int32 little-endian to match safetensors / numpy default
    buf = struct.pack(f"<{len(tokens)}i", *tokens) if tokens else b""
    return hashlib.sha256(buf).hexdigest()[:16]


def _read_bytes_or_empty(path: Path) -> bytes:
    """Read file bytes; return b'' if file does not exist."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""


def compute_model_id(model_path: Path) -> str:
    """Stable 16-hex-char identity for a model.

    Hashes the concatenation of:
      - model.safetensors.index.json (the weight manifest)
      - tokenizer.json
      - tokenizer_config.json
      - chat_template.jinja  (if present)
      - mlx_lm.__version__ major (e.g., "0")

    Any missing optional file contributes an empty bytestring. Switching
    weights, tokenizer, or chat template all yield distinct ids.
    """
    from . import __version__ as mlx_lm_version

    model_path = Path(model_path)
    parts: List[bytes] = [
        b"weights_index:",
        _read_bytes_or_empty(model_path / "model.safetensors.index.json"),
        b"\ntokenizer:",
        _read_bytes_or_empty(model_path / "tokenizer.json"),
        b"\ntokenizer_config:",
        _read_bytes_or_empty(model_path / "tokenizer_config.json"),
        b"\nchat_template:",
        _read_bytes_or_empty(model_path / "chat_template.jinja"),
        b"\nmlx_lm_major:",
        mlx_lm_version.split(".")[0].encode(),
    ]
    return hashlib.sha256(b"".join(parts)).hexdigest()[:16]


class DiskCacheLockError(RuntimeError):
    """Raised when the disk cache directory is already locked by another process."""


@contextlib.contextmanager
def acquire_disk_dir_lock(disk_dir: Path):
    """Exclusive non-blocking flock on `${disk_dir}/.lock`.

    Creates the directory if missing. Raises DiskCacheLockError if another
    process holds the lock. Lock is released when the context exits.
    """
    disk_dir = Path(disk_dir)
    disk_dir.mkdir(parents=True, exist_ok=True)
    lock_path = disk_dir / ".lock"
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    fd_closed = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            os.close(fd)
            fd_closed = True
            raise DiskCacheLockError(
                f"Disk cache directory {disk_dir} is locked by another mlx_lm process. "
                f"Stop the other process or choose a different --prompt-cache-disk-dir."
            ) from e
        yield fd
    finally:
        if not fd_closed:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)


@dataclasses.dataclass
class WriteJob:
    """A pending disk-cache write enqueued from `LRUPromptCache.insert_cache`.

    Holds the prompt_cache reference (and therefore the underlying mx.array
    GPU buffers) until the write completes — pinning that memory. The bounded
    queue size in DiskPromptCache limits how much can be pinned at once.
    """

    token_hash: str  # 16 hex chars; used as filename
    tokens: List[int]  # the token sequence (stored in metadata)
    prompt_cache: List[Any]  # KV state to serialize
    cache_type_classes: List[str]  # one per layer
    trimmable: bool  # can_trim_prompt_cache result
    parents_to_evict: List[str]  # token_hashes whose disk files this dominates
    model_id: str  # for metadata


def _serialize_metadata(job: "WriteJob") -> Dict[str, str]:
    """Pack a WriteJob's metadata into the Dict[str, str] safetensors header.

    Complex values (lists, ints, bools) are JSON-encoded.
    """
    return {
        "tokens_b64": base64.b64encode(
            b"".join(t.to_bytes(4, "little", signed=True) for t in job.tokens)
        ).decode(),
        "length": str(len(job.tokens)),
        "cache_type_classes": json.dumps(job.cache_type_classes),
        "trimmable": "true" if job.trimmable else "false",
        "parent_token_hash": "",  # reserved; unused in v1
        "model_id": job.model_id,
        "format_version": str(FORMAT_VERSION),
        "written_at": str(int(time.time())),
    }


def _deserialize_metadata(raw: Dict[str, str]) -> Dict[str, Any]:
    """Inverse of _serialize_metadata. Raises KeyError if a required key is missing."""
    tokens_bytes = base64.b64decode(raw["tokens_b64"])
    tokens = [
        int.from_bytes(tokens_bytes[i : i + 4], "little", signed=True)
        for i in range(0, len(tokens_bytes), 4)
    ]
    return {
        "tokens": tokens,
        "length": int(raw["length"]),
        "cache_type_classes": json.loads(raw["cache_type_classes"]),
        "trimmable": raw["trimmable"] == "true",
        "parent_token_hash": raw.get("parent_token_hash") or None,
        "model_id": raw["model_id"],
        "format_version": int(raw["format_version"]),
        "written_at": int(raw["written_at"]),
    }


def write_entry_atomic(
    entries_dir: Path, job: "WriteJob", *, fsync: bool = False
) -> None:
    """Write one disk-cache entry atomically.

    Sequence:
      1. write to ${entries_dir}/${token_hash}.safetensors.tmp
      2. (optional) fsync tmp + parent dir
      3. os.rename tmp -> .safetensors  (atomic on POSIX)

    ``save_prompt_cache`` always appends ``.safetensors`` to whatever stem is
    passed, so we pass ``${token_hash}.tmp`` and it creates
    ``${token_hash}.tmp.safetensors`` — that is the file we fsync and rename.

    If any step before rename raises, the .tmp file is left behind for the next
    startup's debris cleanup to remove.
    """
    final = entries_dir / f"{job.token_hash}.safetensors"
    tmp_stem = entries_dir / f"{job.token_hash}.tmp"
    tmp = entries_dir / f"{job.token_hash}.tmp.safetensors"
    metadata = _serialize_metadata(job)
    save_prompt_cache(str(tmp_stem), job.prompt_cache, metadata=metadata)
    if fsync:
        fd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        # parent dir fsync to commit the rename
        fd = os.open(str(entries_dir), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    os.rename(str(tmp), str(final))


def read_entry_metadata(path: Path) -> Dict[str, Any]:
    """Load metadata from a disk cache entry without materializing tensors.

    Uses ``mx.load(..., return_metadata=True)`` which is the only metadata-read
    path supported by the mlx Python API today. (mx.load does read tensor
    headers but does not materialize tensor data into Python until accessed.)

    ``save_prompt_cache`` stores the user metadata dict as index-1 in a
    tree-flattened list, so every key is prefixed ``"1."`` in the raw header.
    We strip that prefix before handing off to ``_deserialize_metadata``.
    """
    _arrays, raw = mx.load(str(path), return_metadata=True)
    # User metadata lives at prefix "1." in the flat safetensors header.
    user_meta = {k[2:]: v for k, v in raw.items() if k.startswith("1.")}
    return _deserialize_metadata(user_meta)


def load_entry(path: Path) -> Tuple[List[Any], Dict[str, Any]]:
    """Load full KV state + metadata from a disk cache entry.

    Returns:
        (prompt_cache, metadata_dict)

    The metadata dict has the same shape as ``read_entry_metadata`` returns
    (``tokens``, ``length``, ``cache_type_classes``, ``trimmable``,
    ``parent_token_hash``, ``model_id``, ``format_version``, ``written_at``).

    Note: ``load_prompt_cache`` with ``return_metadata=True`` calls
    ``tree_unflatten`` on the raw header dict internally, so ``raw`` is already
    the extracted user metadata dict (without the ``"1."`` prefix).
    ``_deserialize_metadata`` can therefore be called on it directly.
    """
    cache, raw = load_prompt_cache(str(path), return_metadata=True)
    return cache, _deserialize_metadata(raw)


@dataclasses.dataclass
class DiskLeaf:
    """In-memory record for one on-disk cache entry.

    Stored as the value at trie leaves on the disk-side trie.
    """

    token_hash: str
    length: int
    nbytes: int
    mtime_ns: int
    trimmable: bool


def reconstruct_disk_trie(
    entries_dir: Path, *, model_id: str
) -> Tuple["PromptTrie", int, Dict[str, List[int]]]:
    """Walk entries_dir, read each entry's metadata, build a PromptTrie.

    Also removes any leftover ``*.tmp.safetensors`` files (crash debris from
    interrupted writes).

    Returns:
        (trie, total_bytes, hash_to_tokens)
        - trie: PromptTrie keyed by ``model_id`` with DiskLeaf values
        - total_bytes: sum of file sizes for valid entries
        - hash_to_tokens: token_hash -> tokens list, for fast mtime-touch lookups

    Skipped (logged + ignored): files with mismatched ``format_version`` or
    unreadable metadata.
    """
    from .models.cache import PromptTrie

    entries_dir = Path(entries_dir)
    entries_dir.mkdir(parents=True, exist_ok=True)
    trie = PromptTrie()
    total_bytes = 0
    hash_to_tokens: Dict[str, List[int]] = {}

    # 1. Crash debris: ${hash}.tmp.safetensors
    for tmp in entries_dir.glob("*.tmp.safetensors"):
        try:
            tmp.unlink()
            logger.info("Removed crash-debris %s", tmp.name)
        except OSError as e:
            logger.warning("Could not remove %s: %s", tmp, e)

    # 2. Real entries — no .tmp.safetensors files remain at this point
    for path in sorted(entries_dir.glob("*.safetensors")):
        # Race-safety: skip any tmp that snuck through
        if ".tmp." in path.name:
            continue
        try:
            meta = read_entry_metadata(path)
        except Exception as e:
            logger.warning("Could not read metadata from %s: %s; skipping", path, e)
            continue
        if meta["format_version"] != FORMAT_VERSION:
            logger.warning(
                "Skipping %s: format_version=%s, expected %s",
                path.name,
                meta["format_version"],
                FORMAT_VERSION,
            )
            continue
        # token_hash is the filename stem
        token_hash = path.stem
        st = path.stat()
        leaf = DiskLeaf(
            token_hash=token_hash,
            length=meta["length"],
            nbytes=st.st_size,
            mtime_ns=st.st_mtime_ns,
            trimmable=meta["trimmable"],
        )
        trie.add(model_id, meta["tokens"], leaf)
        hash_to_tokens[token_hash] = meta["tokens"]
        total_bytes += st.st_size

    return trie, total_bytes, hash_to_tokens
