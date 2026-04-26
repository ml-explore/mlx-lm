# Copyright © 2026 Apple Inc.
"""Inspection / pruning CLI for the disk-backed prompt cache.

Usage:
    python -m mlx_lm.cache_admin stats <disk-dir>
    python -m mlx_lm.cache_admin list <disk-dir> [--model <model-id>]
    python -m mlx_lm.cache_admin prune <disk-dir> --older-than <duration>
    python -m mlx_lm.cache_admin verify <disk-dir>
    python -m mlx_lm.cache_admin remove <disk-dir> --model <model-id>

Reads disk dirs without importing the model loading machinery, so it's safe
to run when no model is configured. Subcommands ``prune``, ``verify``, and
``remove`` are wired in Task 23.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List


def _format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n = n / 1024
    return f"{n}B"


def cmd_stats(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    if not (root / "format-version").exists():
        print(
            f"error: {root} is not a disk-cache root (missing format-version)",
            file=sys.stderr,
        )
        return 1
    models_dir = root / "models"
    if not models_dir.exists():
        print(f"{root}: empty (no models)")
        return 0
    print(f"Disk cache root: {root}")
    print(f"Format version: {(root / 'format-version').read_text().strip()}")
    print()
    total_entries = 0
    total_bytes = 0
    for m in sorted(models_dir.iterdir()):
        if not m.is_dir():
            continue
        info_path = m / "info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        entries_dir = m / "entries"
        entries = (
            list(entries_dir.glob("*.safetensors")) if entries_dir.exists() else []
        )
        # Skip .tmp.safetensors files — they're crash debris
        entries = [p for p in entries if not p.name.endswith(".tmp.safetensors")]
        nbytes = sum(p.stat().st_size for p in entries)
        total_entries += len(entries)
        total_bytes += nbytes
        oldest = min((p.stat().st_mtime for p in entries), default=0)
        newest = max((p.stat().st_mtime for p in entries), default=0)
        print(f"  model: {m.name}")
        print(f"    path_hint: {info.get('model_path_hint', '?')}")
        print(f"    entries: {len(entries)}")
        print(f"    size: {_format_bytes(nbytes)}")
        if entries:
            print(f"    oldest: {time.ctime(oldest)}")
            print(f"    newest: {time.ctime(newest)}")
        print()
    print(f"Total: {total_entries} entries, {_format_bytes(total_bytes)}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    models_dir = root / "models"
    if args.model:
        targets = [models_dir / args.model]
    else:
        targets = sorted(models_dir.iterdir()) if models_dir.exists() else []
    print(f"{'token_hash':<18} {'size':>10} {'mtime':<24}")
    print("-" * 56)
    for m in targets:
        if not m.is_dir():
            continue
        entries_dir = m / "entries"
        if not entries_dir.exists():
            continue
        for p in sorted(
            entries_dir.glob("*.safetensors"), key=lambda x: x.stat().st_mtime
        ):
            if p.name.endswith(".tmp.safetensors"):
                continue
            st = p.stat()
            print(
                f"{p.stem:<18} {_format_bytes(st.st_size):>10} "
                f"{time.ctime(st.st_mtime)}"
            )
    return 0


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(
        prog="mlx_lm.cache_admin",
        description="Disk prompt cache admin tool",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_stats = sub.add_parser("stats", help="Show summary stats")
    p_stats.add_argument("dir")
    p_stats.set_defaults(func=cmd_stats)

    p_list = sub.add_parser("list", help="List entries")
    p_list.add_argument("dir")
    p_list.add_argument("--model", default=None, help="Limit to a single model_id")
    p_list.set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
