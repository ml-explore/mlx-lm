# Copyright © 2026 Apple Inc.
"""Minimal example: programmatic use of LRUPromptCache with disk tier.

Run:
    python examples/disk_prompt_cache.py /tmp/disk-cache /path/to/local/model

The disk tier is opt-in: pass ``disk=DiskPromptCache(...)`` to
``LRUPromptCache``. With ``disk=None`` (the default), behavior is
byte-identical to the in-memory-only baseline.
"""

import sys
from pathlib import Path

from mlx_lm.disk_prompt_cache import DiskPromptCache
from mlx_lm.models.cache import LRUPromptCache


def main(disk_dir: str, model_path: str) -> None:
    disk = DiskPromptCache(
        root=Path(disk_dir),
        model_path=Path(model_path),
        max_bytes=100 * (10**9),  # 100 GB
        fsync=False,  # async durability is fine
    )
    disk.start()
    try:
        ram = LRUPromptCache(max_size=10, disk=disk)
        # Use ``ram`` exactly as you would an in-memory LRUPromptCache.
        # Every insert_cache() write-throughs to disk; every
        # fetch_nearest_cache() consults both tiers transparently.
        print(f"RAM+disk cache ready. Disk: {disk.root} " f"(model_id={disk.model_id})")
    finally:
        disk.shutdown(timeout=30.0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
