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


def _parse_age(s: str) -> float:
    """Parse '30d', '12h', '600m', '5s' as seconds."""
    s = s.strip()
    if s.endswith("d"):
        return float(s[:-1]) * 86400
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


def cmd_prune(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    cutoff = time.time() - _parse_age(args.older_than)
    to_delete = []
    models_dir = root / "models"
    if models_dir.exists():
        for m in models_dir.iterdir():
            entries_dir = m / "entries"
            if not entries_dir.exists():
                continue
            for p in entries_dir.glob("*.safetensors"):
                if p.name.endswith(".tmp.safetensors"):
                    continue
                if p.stat().st_mtime < cutoff:
                    to_delete.append(p)
    if not to_delete:
        print("Nothing to prune.")
        return 0
    print(f"Will delete {len(to_delete)} entries older than {args.older_than}:")
    for p in to_delete[:10]:
        print(f"  {p}")
    if len(to_delete) > 10:
        print(f"  ... and {len(to_delete) - 10} more")
    if not args.yes:
        confirm = input("Continue? [y/N] ")
        if confirm.strip().lower() != "y":
            return 1
    for p in to_delete:
        try:
            p.unlink()
        except OSError as e:
            print(f"warning: could not delete {p}: {e}", file=sys.stderr)
    print(f"Deleted {len(to_delete)} entries.")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Validate every entry: metadata parses, file size matches, no orphan
    .tmp.safetensors files.
    """
    root = Path(args.dir)
    models_dir = root / "models"
    bad = []
    orphan_tmps = []
    n_models = 0
    if models_dir.exists():
        for m in models_dir.iterdir():
            if not m.is_dir():
                continue
            n_models += 1
            entries_dir = m / "entries"
            if not entries_dir.exists():
                continue
            for tmp in entries_dir.glob("*.tmp.safetensors"):
                orphan_tmps.append(tmp)
            for p in entries_dir.glob("*.safetensors"):
                if p.name.endswith(".tmp.safetensors"):
                    continue
                try:
                    import mlx.core as mx

                    _arrays, _raw = mx.load(str(p), return_metadata=True)
                except Exception as e:
                    bad.append((p, repr(e)))
    print(f"Models: {n_models}")
    print(f"Orphan .tmp.safetensors files: {len(orphan_tmps)}")
    if bad:
        print(f"BAD entries ({len(bad)}):")
        for p, err in bad:
            print(f"  {p}: {err}")
        return 1
    print("OK")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    target = root / "models" / args.model
    if not target.exists():
        print(
            f"error: model {args.model} not found in {root}",
            file=sys.stderr,
        )
        return 1
    if not args.yes:
        confirm = input(f"Delete entire model dir {target}? [y/N] ")
        if confirm.strip().lower() != "y":
            return 1
    import shutil

    shutil.rmtree(str(target))
    print(f"Removed {target}")
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

    p_prune = sub.add_parser("prune", help="Delete entries older than N")
    p_prune.add_argument("dir")
    p_prune.add_argument(
        "--older-than",
        required=True,
        help="Duration: 30d, 12h, 600m, or seconds as a number",
    )
    p_prune.add_argument("--yes", action="store_true", help="Skip confirmation")
    p_prune.set_defaults(func=cmd_prune)

    p_verify = sub.add_parser("verify", help="Validate all entries")
    p_verify.add_argument("dir")
    p_verify.set_defaults(func=cmd_verify)

    p_remove = sub.add_parser("remove", help="Delete a model's entire subdir")
    p_remove.add_argument("dir")
    p_remove.add_argument("--model", required=True)
    p_remove.add_argument("--yes", action="store_true", help="Skip confirmation")
    p_remove.set_defaults(func=cmd_remove)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
