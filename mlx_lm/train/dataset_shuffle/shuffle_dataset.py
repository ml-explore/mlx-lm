"""Two-stage shuffle for large sharded datasets on S3 or local disk.

A dataset that arrives grouped by source, crawl or topic has to be shuffled
before it can be streamed into training, and it is usually far too large to
shuffle in one machine's memory. This does it in two passes so that no worker
ever holds more than 1/W of the corpus, while every output shard still ends up a
uniform random sample of the whole corpus:

    stage 1   Worker w takes its 1/W slice of the input files, shuffles the
              records it read, and scatters them into W buckets. Bucket (w, j)
              therefore holds a random ~1/W**2 of all records.

    stage 2   Worker j reads bucket (w, j) for every w -- one bucket from every
              stage-1 worker, which together are a uniform 1/W sample of the
              *whole* corpus -- shuffles again, and writes the final shards.

Run all W workers of stage 1 (in any order, in parallel),
wait for every one of them to finish, then run all W workers of stage 2.
"""

import argparse
import gzip
import io
import json
import logging
import random

import storage

# Data files are picked out of the input listing by extension, so anything else
# sitting under the prefix (READMEs, checksums, .index files) is ignored. A
# metadata file that itself ends in .json would be read as data; pass an explicit
# file list for a layout like that.
SUFFIXES = {
    "jsonl": (
        ".jsonl",
        ".json",
        ".jsonl.gz",
        ".json.gz",
        ".jsonl.zst",
        ".json.zst",
        ".jsonl.zstd",
        ".json.zstd",
    ),
    "parquet": (".parquet",),
}

OUTPUT_EXT = {"parquet": ".parquet", "jsonl": ".json.gz"}

# Stage 2 records the layout it wrote here so a reader does not have to be told
# it, or list a million keys to infer it. See mlx_lm.train.data.s3.read_manifest.
MANIFEST_NAME = "manifest.json"

GZIP_LEVEL = 6
PARQUET_COMPRESSION = "zstd"


# --------------------------------------------------------------- JSON --


def _read_jsonl(path):
    """The raw lines of one JSON Lines file, transparently decompressed."""
    with storage.fetch(path) as local:
        if path.endswith((".zst", ".zstd")):
            import zstandard

            with open(local, "rb") as raw:
                reader = zstandard.ZstdDecompressor().stream_reader(raw)
                lines = io.BufferedReader(reader).readlines()
        else:
            open_file = gzip.open if path.endswith(".gz") else open
            with open_file(local, "rb") as fin:
                lines = fin.readlines()
    if lines and not lines[-1].endswith(b"\n"):
        lines[-1] += b"\n"
    return lines


def _tag_source(lines, name):
    """Add a ``source_file`` field recording where each record came from."""
    tagged = []
    for i, line in enumerate(lines):
        record = json.loads(line)
        record["source_file"] = f"{name}:{i}"
        tagged.append(json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n")
    return tagged


def _load_jsonl(files, add_source):
    records = []
    for i, f in enumerate(files):
        logging.info("reading %d/%d %s", i + 1, len(files), f)
        lines = _read_jsonl(f)
        records.extend(_tag_source(lines, _source_name(f)) if add_source else lines)
    logging.info("read %d records", len(records))
    return records


def _scatter_jsonl(records, prefix, num_shards, rng):
    rng.shuffle(records)
    for i in range(num_shards):
        path = f"{prefix}{_shard_name(i, num_shards)}{OUTPUT_EXT['jsonl']}"
        logging.info("writing %s", path)
        with storage.publish(path) as tmp:
            with open(tmp, "wb") as raw:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=GZIP_LEVEL,
                    mtime=0,
                    fileobj=raw,
                ) as fout:
                    # Strided slicing turns one shuffle into a random partition.
                    fout.writelines(records[i::num_shards])


# ------------------------------------------------------------------ Parquet --


def _load_parquet(files, add_source):
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = []
    for i, f in enumerate(files):
        logging.info("reading %d/%d %s", i + 1, len(files), f)
        with storage.fetch(f) as local:
            table = pq.read_table(local)
        if add_source:
            name = _source_name(f)
            table = table.append_column(
                "source_file",
                pa.array([f"{name}:{j}" for j in range(table.num_rows)], pa.string()),
            )
        tables.append(table)
    # "permissive" unifies columns that are missing from, or differently typed
    # in, some files -- common across a corpus assembled from several dumps.
    table = pa.concat_tables(tables, promote_options="permissive")
    logging.info("read %d rows", table.num_rows)
    return table


def _widen(typ):
    """The large_* twin of a string or binary type, recursing into nested types.

    ``string`` addresses its values with int32 offsets, so an array holding more
    than 2 GiB of text cannot exist; ``large_string`` uses int64. Parquet encodes
    both as BYTE_ARRAY, so widening changes the Arrow schema and not the file.
    """
    import pyarrow as pa

    if typ == pa.string():
        return pa.large_string()
    if typ == pa.binary():
        return pa.large_binary()
    if isinstance(typ, pa.ListType):
        return pa.list_(_widen(typ.value_type))
    if isinstance(typ, pa.LargeListType):
        return pa.large_list(_widen(typ.value_type))
    if isinstance(typ, pa.MapType):
        return pa.map_(_widen(typ.key_type), _widen(typ.item_type))
    if isinstance(typ, pa.StructType):
        return pa.struct([(f.name, _widen(f.type)) for f in typ])
    return typ


def _scatter_parquet(table, prefix, num_shards, rng):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # take() gathers by concatenating the column's chunks first, so once a
    # worker's slice holds more than 2 GiB of text the int32 offsets overflow --
    # for any indices at all, even a single row. Widen them, and concatenate once
    # here rather than once per shard.
    table = table.cast(
        pa.schema([f.with_type(_widen(f.type)) for f in table.schema])
    ).combine_chunks()

    order = list(range(table.num_rows))
    rng.shuffle(order)
    for i in range(num_shards):
        path = f"{prefix}{_shard_name(i, num_shards)}.parquet"
        logging.info("writing %s", path)
        indices = pa.array(order[i::num_shards], type=pa.int64())
        with storage.publish(path) as tmp:
            pq.write_table(table.take(indices), tmp, compression=PARQUET_COMPRESSION)


LOAD = {"jsonl": _load_jsonl, "parquet": _load_parquet}
SCATTER = {"jsonl": _scatter_jsonl, "parquet": _scatter_parquet}


# ------------------------------------------------------------------- stages --


def _shard_name(shard_id, num_shards):
    return f"{shard_id:05d}-of-{num_shards:05d}"


def _source_name(path):
    """A short, stable name for a file: its last three path components."""
    return "/".join(path.split("/")[-3:])


def _as_directory(output):
    """Treat ``--output`` as a directory, whether or not it was spelled with a "/".

    Without this, an output of ".../data" would concatenate straight onto the
    shard name and write ".../data00000-of-00032/", which is never what anyone
    means. Applied by both stages, so they always agree on where the buckets are.
    """
    return output if output.endswith("/") else output + "/"


def _bucket_prefix(output, num_workers, worker_id):
    """Where stage 1 writes and stage 2 reads. Delete this once stage 2 is done."""
    return f"{_as_directory(output)}_tmp/{_shard_name(worker_id, num_workers)}/"


def _shard_files(files, num_workers, worker_id, seed):
    """This worker's slice of the input files.

    Shuffling before striding keeps a worker's slice free of the size and topic
    correlations that adjacent input files almost always have. Every worker
    derives the same permutation from ``seed``, so the slices tile the input
    exactly once with no coordination.
    """
    files = list(files)
    random.Random(f"{seed}:files").shuffle(files)
    mine = files[worker_id::num_workers]
    logging.info(
        "worker %d/%d: %d of %d input files",
        worker_id,
        num_workers,
        len(mine),
        len(files),
    )
    if not mine:
        raise SystemExit(
            f"worker {worker_id} got 0 of {len(files)} input files; "
            f"--num-workers must not exceed the number of input files"
        )
    return mine


def _detect_format(files):
    kinds = set()
    for f in files:
        for kind, suffixes in SUFFIXES.items():
            if f.endswith(suffixes):
                kinds.add(kind)
    if len(kinds) != 1:
        raise SystemExit(
            f"cannot infer --format: found {sorted(kinds) or 'no recognised data files'}; "
            f"pass --format explicitly"
        )
    return kinds.pop()


def _detect_bucket_format(output, num_workers, worker_id):
    """Infer the format of stage 1's output by looking for one of its buckets."""
    base = _bucket_prefix(output, num_workers, 0) + _shard_name(worker_id, num_workers)
    for kind, ext in OUTPUT_EXT.items():
        if storage.exists(base + ext):
            return kind
    raise SystemExit(
        f"no stage-1 bucket at {base}[{'|'.join(OUTPUT_EXT.values())}]; "
        f"did every stage-1 worker finish, and do --output/--num-workers match?"
    )


def stage_1(input_path, output, num_workers, worker_id, fmt, seed, add_source):
    """Read this worker's input files; scatter their records into W buckets."""
    files = storage.read_file_list(input_path)
    if fmt == "auto":
        fmt = _detect_format(files)
    data_files = [
        f
        for f in files
        if f.endswith(SUFFIXES[fmt]) and f.rsplit("/", 1)[-1] != MANIFEST_NAME
    ]
    if not data_files:
        raise SystemExit(
            f"no {fmt} files among the {len(files)} paths under {input_path}; "
            f"expected one of {' '.join(SUFFIXES[fmt])}"
        )
    logging.info(
        "format %s: %d of %d listed files are data", fmt, len(data_files), len(files)
    )
    files = _shard_files(data_files, num_workers, worker_id, seed)

    records = LOAD[fmt](files, add_source)
    rng = random.Random(f"{seed}:1:{worker_id}")
    SCATTER[fmt](
        records, _bucket_prefix(output, num_workers, worker_id), num_workers, rng
    )


def _write_manifest(output, num_workers, num_output_shards, fmt, seed):
    manifest = {
        "num_groups": num_workers,
        "shards_per_group": num_output_shards,
        "suffix": OUTPUT_EXT[fmt],
        "format": fmt,
        "seed": seed,
    }
    path = f"{_as_directory(output)}{MANIFEST_NAME}"
    with storage.publish(path) as tmp:
        with open(tmp, "w") as fid:
            json.dump(manifest, fid, indent=2, sort_keys=True)
    logging.info("wrote %s: %s", path, manifest)


def stage_2(output, num_workers, worker_id, num_output_shards, fmt, seed):
    """Gather one bucket from every stage-1 worker; write the final shards."""
    if fmt == "auto":
        fmt = _detect_bucket_format(output, num_workers, worker_id)
    files = [
        _bucket_prefix(output, num_workers, w)
        + _shard_name(worker_id, num_workers)
        + OUTPUT_EXT[fmt]
        for w in range(num_workers)
    ]

    records = LOAD[fmt](files, add_source=False)  # stage 1 already tagged them
    rng = random.Random(f"{seed}:2:{worker_id}")
    prefix = f"{_as_directory(output)}{_shard_name(worker_id, num_workers)}/"
    SCATTER[fmt](records, prefix, num_output_shards, rng)
    if worker_id == 0:
        _write_manifest(output, num_workers, num_output_shards, fmt, seed)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2),
        required=True,
        help="1 scatters into buckets, 2 gathers them; all of stage 1 must finish first",
    )
    parser.add_argument(
        "--input",
        help="prefix ending in / to list recursively, a hf://datasets/<owner>/"
        "<name>[@revision]/<path> Hub prefix, or a text file of paths relative "
        "to its own directory. Stage 1 only; ignored by stage 2",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output directory, local or s3://; a trailing / is optional",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        required=True,
        help="number of workers W; each holds ~1/W of the corpus in memory. "
        "Must be identical in both stages",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        required=True,
        help="0-based id of this worker; run one process per id",
    )
    parser.add_argument(
        "--num-output-shards",
        type=int,
        default=1,
        help="final files per stage-2 worker, so W * this many files in total",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "jsonl", "parquet"),
        default="auto",
        help="auto infers from the input file extensions",
    )
    parser.add_argument(
        "--seed",
        default="1711498900",
        help="any string; the same seed, inputs and shard counts give the same output",
    )
    parser.add_argument(
        "--no-source-column",
        action="store_true",
        help="skip the source_file field recording each record's origin. Faster "
        "for JSON Lines, which otherwise has to reserialise every record",
    )
    parser.add_argument(
        "--s3-endpoint-url",
        help="for S3-compatible stores; defaults to $S3_ENDPOINT_URL, else AWS",
    )
    args = parser.parse_args()

    if not 0 <= args.worker_id < args.num_workers:
        parser.error(f"--worker-id must be in [0, {args.num_workers})")
    if args.num_output_shards < 1:
        parser.error("--num-output-shards must be at least 1")
    if args.stage == 1 and not args.input:
        parser.error("--input is required for stage 1")
    if storage.is_hf(args.output):
        parser.error(
            "--output cannot be a hf:// URI: the Hub is supported for --input "
            "only. Shuffle to a local or s3:// prefix, then push that."
        )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    storage.set_endpoint_url(args.s3_endpoint_url)
    storage.check_reachable(args.output)

    if args.stage == 1:
        stage_1(
            args.input,
            args.output,
            args.num_workers,
            args.worker_id,
            args.format,
            args.seed,
            add_source=not args.no_source_column,
        )
    else:
        stage_2(
            args.output,
            args.num_workers,
            args.worker_id,
            args.num_output_shards,
            args.format,
            args.seed,
        )
    logging.info("stage %d worker %d done", args.stage, args.worker_id)


if __name__ == "__main__":
    main()
