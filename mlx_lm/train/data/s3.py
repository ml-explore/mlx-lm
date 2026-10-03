# Copyright © 2026 Apple Inc.

"""Pretraining documents read from pre-shuffled shards in S3.

The corpus is shuffled once, offline, by ``mlx_lm/train/dataset_shuffle``, so a
rank reads its own files start to finish: no online shuffle buffer, and a
position is just a key and a line number. Shard counts come from the manifest the
shuffler wrote.
"""

import gzip
import io
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import boto3
import mlx.core as mx
import zstandard
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

from mlx_lm.train.data.batching import tokenized_data

S3_MAX_ATTEMPTS = int(os.environ.get("S3_MAX_ATTEMPTS", 10))
S3_DOWNLOAD_ATTEMPTS = int(os.environ.get("S3_DOWNLOAD_ATTEMPTS", 10_000))
S3_DOWNLOAD_DIR = os.environ.get("S3_DOWNLOAD_DIR") or os.path.join(
    tempfile.gettempdir(), "mlx_lm_s3"
)
MANIFEST_NAME = "manifest.json"

DOLMA = {
    "pre": {
        "uri": "s3://smollm/dolma3-mix-6T-shuffled/data/",
        "num_groups": 1024,
    },
    "mid": {
        "uri": "s3://smollm/dolma3_dolmino_mix-100B-1025-shuffled/data/",
        "num_groups": 64,
    },
}


def s3_client(endpoint_url=None):
    return boto3.session.Session().client(
        "s3",
        endpoint_url=endpoint_url or os.environ.get("S3_ENDPOINT_URL") or None,
        config=Config(retries={"max_attempts": S3_MAX_ATTEMPTS, "mode": "standard"}),
    )


def split_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"expected an s3:// uri, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def read_manifest(uri, client=None):
    """The layout dataset_shuffle recorded, or None for a corpus written before it."""
    from botocore.exceptions import ClientError

    bucket, prefix = split_uri(uri)
    client = client or s3_client()
    try:
        body = client.get_object(Bucket=bucket, Key=prefix + MANIFEST_NAME)["Body"]
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(body.read())


def shard_keys(uri, num_groups, shards_per_group, suffix=".json.gz"):
    bucket, prefix = split_uri(uri)
    keys = [
        f"{prefix}{group:05d}-of-{num_groups:05d}/{shard:05d}-of-{shards_per_group:05d}{suffix}"
        for group in range(num_groups)
        for shard in range(shards_per_group)
    ]
    return bucket, keys


def _open_stream(key, body):
    if key.endswith(".zst"):
        return zstandard.ZstdDecompressor().stream_reader(body)
    if key.endswith(".gz"):
        return gzip.GzipFile(fileobj=body)
    return body


def download(bucket, key, client, local_dir):
    path = os.path.join(local_dir, key)
    # Reuse a file a stopped run left; download_file writes only complete files.
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        client.download_file(
            bucket,
            key,
            path,
            Config=TransferConfig(num_download_attempts=S3_DOWNLOAD_ATTEMPTS),
        )
    return path


def s3_lines(key, path, skip=0):
    try:
        with open(path, "rb") as fid:
            stream = io.BufferedReader(_open_stream(key, fid), buffer_size=2**20)
            for idx, line in enumerate(stream):
                if idx >= skip:
                    yield idx, line
    finally:
        os.remove(path)


def s3_data(
    uri,
    num_groups=None,
    shards_per_group=None,
    rank=0,
    size=1,
    suffix=None,
    start_file_name=None,
    start_sample_idx=0,
    client=None,
):
    client = client or s3_client()
    manifest = read_manifest(uri, client)
    if manifest is not None:
        num_groups = manifest["num_groups"]
        shards_per_group = manifest["shards_per_group"]
        suffix = manifest["suffix"]
        logging.info(
            "%s%s: %d x %d %s", uri, MANIFEST_NAME, num_groups, shards_per_group, suffix
        )
    else:
        if num_groups is None or shards_per_group is None:
            raise ValueError(
                f"no {MANIFEST_NAME} under {uri}, so the layout must come from the "
                "config: set dataset.num_groups and dataset.shards_per_group"
            )
        suffix = suffix or ".json.gz"
        logging.warning(
            "no %s under %s; using the layout from the config: %d x %d %s",
            MANIFEST_NAME,
            uri,
            num_groups,
            shards_per_group,
            suffix,
        )
    bucket, keys = shard_keys(uri, num_groups, shards_per_group, suffix)
    logging.info(
        "rank %d: %d of %d files under %s", rank, len(keys[rank::size]), len(keys), uri
    )
    keys = keys[rank::size]
    if start_file_name is not None:
        if start_file_name not in keys:
            raise ValueError(
                f"{start_file_name} is not among rank {rank}'s files under {uri}"
            )
        keys = keys[keys.index(start_file_name) :]
    if not keys:
        return
    # Download the next file while the current one is read.
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(download, bucket, keys[0], client, S3_DOWNLOAD_DIR)
        for i, key in enumerate(keys):
            path = future.result()
            if i + 1 < len(keys):
                future = pool.submit(
                    download, bucket, keys[i + 1], client, S3_DOWNLOAD_DIR
                )
            logging.info("reading s3://%s/%s", bucket, key)
            yield {
                "lines": s3_lines(key, path, skip=start_sample_idx if i == 0 else 0),
                "file_name": key,
            }


def jsonl_data(dataset, shard=False):
    g = mx.distributed.init()
    count = 0
    for data in dataset:
        for sample_idx, line in data["lines"]:
            count += 1
            if shard and (count % g.size() != g.rank()):
                continue
            yield {
                **json.loads(line),
                "file_name": data["file_name"],
                "sample_idx": sample_idx,
            }


# Dataset for pretraining and midtraining
def load_s3(
    tokenizer,
    rank,
    size,
    uri,
    num_groups=None,
    shards_per_group=None,
    suffix=None,
    start_file_name=None,
    start_sample_idx=0,
    client=None,
):
    return tokenized_data(
        tokenizer,
        jsonl_data(
            s3_data(
                uri,
                num_groups,
                shards_per_group,
                rank=rank,
                size=size,
                suffix=suffix,
                start_file_name=start_file_name,
                start_sample_idx=start_sample_idx,
                client=client,
            )
        ),
    )
