import glob
import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager

S3_PREFIX = "s3://"
HF_PREFIX = "hf://"

# S3-compatible stores (MinIO, Ceph, R2, ...) need an explicit endpoint. Set it
# with --s3-endpoint-url or $S3_ENDPOINT_URL; unset means the default AWS
# endpoint for the ambient region.
_endpoint_url = os.environ.get("S3_ENDPOINT_URL") or None
_client = None


def set_endpoint_url(url):
    global _endpoint_url, _client
    if url:
        _endpoint_url, _client = url, None


def is_remote(path):
    """Anything that has to be pulled to a local file before it can be read."""
    return path.startswith((S3_PREFIX, HF_PREFIX))


def is_hf(path):
    return path.startswith(HF_PREFIX)


def _split(path):
    """Split ``s3://bucket/some/key`` into ``("bucket", "some/key")``."""
    bucket, _, key = path[len(S3_PREFIX) :].partition("/")
    return bucket, key


def _hf_split(path):
    """Split ``hf://datasets/<owner>/<name>[@revision]/<key>``.

    Same spelling as huggingface_hub's own filesystem, so a URI can be pasted
    between the two. A revision is a tag, branch or commit sha without a "/".
    """
    rest = path[len(HF_PREFIX) :]
    if not rest.startswith("datasets/"):
        raise SystemExit(
            f"only dataset repos are supported: expected "
            f"hf://datasets/<owner>/<name>/..., got {path}"
        )
    parts = rest[len("datasets/") :].split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise SystemExit(f"expected hf://datasets/<owner>/<name>/..., got {path}")
    repo, _, revision = "/".join(parts[:2]).partition("@")
    return repo, revision or None, "/".join(parts[2:])


def _hf_uri(repo, revision, key):
    at = f"{repo}@{revision}" if revision else repo
    return f"{HF_PREFIX}datasets/{at}/{key}"


def _hf_api():
    from huggingface_hub import HfApi

    # Reads $HF_TOKEN, or whatever `hf auth login` stored, for gated datasets.
    return HfApi()


def _s3():
    global _client
    if _client is None:
        import boto3.session
        import botocore.client

        _client = boto3.session.Session().client(
            "s3",
            endpoint_url=_endpoint_url,
            config=botocore.client.Config(
                # Connecting is quick or not happening: a long connect timeout
                # times 12 retries turns a wrong endpoint into an hour of
                # silence. Reading is the part that is legitimately slow.
                connect_timeout=15,
                read_timeout=300,
                retries={"max_attempts": 12, "mode": "standard"},
            ),
        )
    return _client


def reachable(path):
    """Whether ``path``'s bucket answers, without raising. See check_reachable."""
    try:
        check_reachable(path)
    except SystemExit:
        return False
    return True


def check_reachable(path):
    """Fail fast, and loudly, if we cannot talk to ``path``'s bucket.

    Called once before a long run. Without it the first real request retries a
    bad endpoint for the better part of an hour with nothing on stdout, which
    is impossible to tell apart from slow progress.
    """
    if not is_remote(path) or is_hf(path):
        return
    import boto3.session
    import botocore.client
    import botocore.exceptions

    bucket, _ = _split(path)
    probe = boto3.session.Session().client(
        "s3",
        endpoint_url=_endpoint_url,
        config=botocore.client.Config(
            connect_timeout=10, read_timeout=10, retries={"max_attempts": 1}
        ),
    )
    logging.info("checking %s at %s", bucket, probe.meta.endpoint_url)
    try:
        probe.head_bucket(Bucket=bucket)
    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("403", "AccessDenied"):
            return  # reachable, and write-only credentials are legitimate
        if code in ("404", "NoSuchBucket"):
            raise SystemExit(
                f"bucket {bucket!r} does not exist at {probe.meta.endpoint_url}"
            )
        raise
    except botocore.exceptions.EndpointConnectionError as e:
        raise SystemExit(_unreachable_message(bucket, probe.meta.endpoint_url, e))
    except botocore.exceptions.ProxyConnectionError as e:
        raise SystemExit(
            f"cannot reach {probe.meta.endpoint_url} through the proxy in "
            f"$HTTPS_PROXY: {e}\nUnset the proxy, or add the endpoint to $NO_PROXY."
        )
    except botocore.exceptions.ConnectTimeoutError as e:
        raise SystemExit(_unreachable_message(bucket, probe.meta.endpoint_url, e))


def _unreachable_message(bucket, endpoint, error):
    hint = ""
    if _endpoint_url is None:
        # boto3 derived the endpoint from the region, which is only right for
        # real AWS. Any other S3-compatible store has to be named explicitly.
        hint = (
            f"\n{endpoint} was derived from your AWS region rather than given. "
            f"If {bucket!r} lives on an S3-compatible store, pass its address as "
            f"--s3-endpoint-url or $S3_ENDPOINT_URL."
        )
    return f"cannot reach {endpoint}: {error}{hint}"


def _retry(fn, *args, attempts=5, delay=30):
    """Retry around botocore's own retries.

    Thousands of multi-gigabyte transfers reliably turn up failures botocore
    does not retry for us, such as connections reset mid-download. Losing a
    whole worker to one of those is much more expensive than waiting.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001 - anything transient is worth a retry
            if attempt == attempts:
                raise
            logging.warning(
                "attempt %d/%d failed (%s), retrying in %ds",
                attempt,
                attempts,
                e,
                delay,
            )
            time.sleep(delay)


def list_files(prefix):
    """Every file under ``prefix``, sorted. ``prefix`` is a directory or key prefix."""
    if is_hf(prefix):
        repo, revision, key_prefix = _hf_split(prefix)
        files = _hf_api().list_repo_files(repo, repo_type="dataset", revision=revision)
        return sorted(
            _hf_uri(repo, revision, f) for f in files if f.startswith(key_prefix)
        )
    if not is_remote(prefix):
        return sorted(
            f
            for f in glob.glob(os.path.join(prefix, "**"), recursive=True)
            if os.path.isfile(f)
        )
    bucket, key_prefix = _split(prefix)
    keys = []
    pages = (
        _s3()
        .get_paginator("list_objects_v2")
        .paginate(Bucket=bucket, Prefix=key_prefix)
    )
    for page in pages:
        keys.extend(obj["Key"] for obj in page.get("Contents", ()))
    return sorted(f"{S3_PREFIX}{bucket}/{key}" for key in keys)


def read_file_list(path):
    """Resolve ``path`` to a list of input files.

    A trailing ``/`` means "everything under this prefix", and so does any
    ``hf://`` URI -- listing a Hub repo is one cheap call, and a prefix is what
    you almost always mean there. Anything else is read as a text file listing
    one path per line, relative to its own directory unless the line is itself
    absolute or carries a scheme.
    """
    if path.endswith("/") or is_hf(path):
        return list_files(path)
    with fetch(path) as local:
        with open(local) as fin:
            names = [line.strip() for line in fin if line.strip()]
    # A bare "files.txt" has no directory part, so its entries are relative to
    # the working directory, not to a directory called "files.txt".
    prefix = path.rsplit("/", 1)[0] + "/" if "/" in path else ""
    return sorted(name if _is_absolute(name) else prefix + name for name in names)


def _is_absolute(name):
    return name.startswith((S3_PREFIX, HF_PREFIX, "/"))


def exists(path):
    if is_hf(path):
        repo, revision, key = _hf_split(path)
        return _hf_api().file_exists(repo, key, repo_type="dataset", revision=revision)
    if not is_remote(path):
        return os.path.exists(path)
    import botocore.exceptions

    bucket, key = _split(path)
    try:
        _s3().head_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
    return True


def download(src, dst):
    if is_hf(src):
        _hf_download(src, dst)
        return
    if not is_remote(src):
        shutil.copyfile(src, dst)
        return
    bucket, key = _split(src)
    _retry(_s3().download_file, bucket, key, dst)
    logging.info("downloaded %s", src)


def _hf_download(src, dst):
    """Fetch one file from the Hub.

    ``local_dir`` keeps the download out of the shared cache under $HF_HOME: a
    corpus this size, read once and thrown away, would otherwise fill it. The
    Hub client does its own retrying, so this does not go through _retry.
    """
    from huggingface_hub import hf_hub_download

    repo, revision, key = _hf_split(src)
    with tempfile.TemporaryDirectory(prefix="shuffle-hf-") as scratch:
        path = hf_hub_download(
            repo,
            key,
            repo_type="dataset",
            revision=revision,
            local_dir=scratch,
        )
        shutil.move(path, dst)
    logging.info("downloaded %s", src)


def upload(src, dst):
    """Move/upload the local file ``src`` to ``dst``. ``src`` is consumed."""
    if is_hf(dst):
        raise SystemExit(
            f"cannot write to {dst}: the Hub is supported as a source only. "
            f"Write to a local or s3:// destination, then push that with "
            f"`hf upload` if you want the result on the Hub"
        )
    if not is_remote(dst):
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.move(src, dst)
        os.chmod(dst, 0o644)  # mkstemp files are 0600, which is wrong for data
        return
    bucket, key = _split(dst)
    _retry(_s3().upload_file, src, bucket, key)
    logging.info("uploaded %s", dst)


@contextmanager
def fetch(path):
    """Yield a readable local path for ``path``, downloading it if remote.

    Remote files land in $TMPDIR, so point that at a disk with room for the
    largest single input file.
    """
    if not is_remote(path):
        yield path
        return
    tmp = _mkstemp("shuffle-in-", os.path.splitext(path)[1])
    try:
        download(path, tmp)
        yield tmp
    finally:
        _unlink(tmp)


@contextmanager
def publish(path):
    """Yield a local temp path to write; on clean exit, move/upload it to ``path``.

    Nothing appears at ``path`` if the body raises, so a crashed worker never
    leaves a half-written shard behind for the next stage to read.
    """
    tmp = _mkstemp("shuffle-out-", os.path.splitext(path)[1])
    try:
        yield tmp
        upload(tmp, path)
    finally:
        _unlink(tmp)


def copy(src, dst):
    """Copy one file between any two supported locations. Returns bytes copied.

    Unlike upload(), ``src`` survives: a remote source is staged through a temp
    file that we own, a local one is read in place.
    """
    if is_remote(src):
        with fetch(src) as local:
            size = os.path.getsize(local)
            upload(local, dst)  # consumes the temp file, which is ours to spend
        return size
    size = os.path.getsize(src)
    if is_remote(dst):
        upload(src, dst)  # s3 upload does not consume its source
        return size
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copyfile(src, dst)
    return size


def _mkstemp(prefix, suffix):
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return name


def _unlink(path):
    if os.path.exists(path):
        os.remove(path)
