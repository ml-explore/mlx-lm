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
