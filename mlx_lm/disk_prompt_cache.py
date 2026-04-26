# Copyright © 2026 Apple Inc.
"""Disk-backed L2 prompt cache for mlx_lm.

See docs/disk_prompt_cache.md for the design overview.

This module is opt-in: it is only constructed when mlx_lm.server is given
``--prompt-cache-disk-dir``. With ``disk=None`` (the default) ``LRUPromptCache``
behaves identically to the pre-PR baseline.
"""

from __future__ import annotations

import base64
import dataclasses
import errno
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
