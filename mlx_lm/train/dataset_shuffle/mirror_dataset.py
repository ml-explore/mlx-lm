import argparse
import logging
import os

import storage


def _source_prefix(input_path, files):
    """The leading part of each source path that the destination layout drops."""
    if input_path.endswith("/"):
        return input_path
    if storage.is_hf(input_path):
        # A bare hf:// repo URI is still a prefix.
        return input_path + "/"
    # A file list: keep whatever directory structure its entries share.
    common = os.path.commonprefix(files)
    return common[: common.rfind("/") + 1]


def _relative(src, base):
    if src.startswith(base):
        return src[len(base) :].lstrip("/")
    return os.path.basename(src)


def _human(num_bytes):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024 or unit == "TiB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


def check(input_path, output):
    """Validate the pair without copying anything. Raises SystemExit if unusable.

    Worth running once before fanning out: otherwise every worker rediscovers the
    same bad endpoint or empty input and reports it separately.
    """
    storage.check_reachable(output)
    logging.info("listing %s", input_path)
    files = storage.read_file_list(input_path)
    if not files:
        raise SystemExit(f"no files found under {input_path}")
    logging.info("ready: %d files, %s -> %s", len(files), input_path, output)
    return files


def mirror(input_path, output, num_workers, worker_id, overwrite=False, dry_run=False):
    # A dry run is for planning, so it should still work when the destination is
    # not set up yet -- but then it cannot know what is already there.
    check_destination = True
    if dry_run:
        check_destination = storage.reachable(output)
        if not check_destination:
            logging.warning(
                "%s is not reachable; listing the plan without checking what is "
                "already there",
                output,
            )
    else:
        storage.check_reachable(output)

    logging.info("listing %s", input_path)
    files = storage.read_file_list(input_path)
    if not files:
        raise SystemExit(f"no files found under {input_path}")
    base = _source_prefix(input_path, files)
    if not output.endswith("/"):
        output += "/"

    # Plain striding, no shuffling: this only has to balance the work, and
    # sharded corpora come in similarly sized files.
    mine = files[worker_id::num_workers]
    logging.info(
        "worker %d/%d: %d of %d files -> %s",
        worker_id,
        num_workers,
        len(mine),
        len(files),
        output,
    )

    copied = skipped = 0
    copied_bytes = 0
    for i, src in enumerate(mine, 1):
        rel = _relative(src, base)
        dst = output + rel
        if not overwrite and check_destination and storage.exists(dst):
            logging.info("%d/%d present %s", i, len(mine), rel)
            skipped += 1
            continue
        if dry_run:
            logging.info("%d/%d would copy %s -> %s", i, len(mine), src, dst)
            copied += 1
            continue
        logging.info("%d/%d copying %s", i, len(mine), rel)
        size = storage.copy(src, dst)
        copied_bytes += size
        copied += 1
        logging.info(
            "%d/%d done %s (%s, %s so far)",
            i,
            len(mine),
            rel,
            _human(size),
            _human(copied_bytes),
        )

    logging.info(
        "worker %d done: %d %s (%s), %d already present",
        worker_id,
        copied,
        "would be copied" if dry_run else "copied",
        _human(copied_bytes),
        skipped,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="what to mirror: an hf://datasets/<owner>/<name>[@revision]/<path> "
        "Hub prefix, an s3:// or local prefix ending in /, or a text file "
        "listing paths",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="destination root, s3:// or local; the source layout is recreated "
        "underneath it",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="number of workers to split the file list across",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="0-based id of this worker; run one process per id",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-copy files that are already at the destination",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be copied without transferring anything",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the destination is reachable and the input is non-empty, "
        "then exit. Run this once before launching many workers",
    )
    parser.add_argument(
        "--s3-endpoint-url",
        help="for S3-compatible stores; defaults to $S3_ENDPOINT_URL, else AWS",
    )
    args = parser.parse_args()

    if not 0 <= args.worker_id < args.num_workers:
        parser.error(f"--worker-id must be in [0, {args.num_workers})")
    if storage.is_hf(args.output):
        parser.error(
            "--output cannot be a hf:// URI: the Hub is a source only. Mirror to "
            "a local or s3:// prefix, then push that with `hf upload`."
        )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    storage.set_endpoint_url(args.s3_endpoint_url)
    if args.check:
        check(args.input, args.output)
        return
    mirror(
        args.input,
        args.output,
        args.num_workers,
        args.worker_id,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
