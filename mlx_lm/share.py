# Copyright © 2026 Apple Inc.

import argparse
import os
import sys
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
from huggingface_hub.errors import LocalEntryNotFoundError
from mlx._distributed_utils.common import Hostfile
from mlx._distributed_utils.launch import launch_jaccl, launch_ring
from tqdm import tqdm

from .utils import hf_repo_to_path

CHUNK_SIZE = 100 * 1024 * 1024


def error(*args, **kwargs):
    kwargs["file"] = sys.stderr
    print("\033[31m[ERROR]", *args, "\033[0m", **kwargs)


def launch(args):
    if args.hostfile is None:
        raise ValueError("No hostfile provided")

    hostfile = Hostfile.from_file(args.hostfile)
    if hostfile.backend == "":
        raise ValueError("Backend needs to be defined in the hostfile.")
    if len(hostfile.hosts) == 1:
        raise ValueError("More than one node needs to be in the hostfile")

    launch_args = argparse.Namespace(
        backend=hostfile.backend,
        cwd=str(Path.cwd()),
        env=hostfile.envs,
        verbose=False,
        python=None,
        starting_port=32323,
        connections_per_ip=1,
    )
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "share",
        "--path",
        args.path,
    ]
    if args.tmpdir is not None:
        cmd += ["--tmpdir", args.tmpdir]
    if args.dst is not None:
        cmd += ["--dst", args.dst]

    if hostfile.backend == "ring":
        launch_ring(None, hostfile.hosts, launch_args, cmd)
    elif hostfile.backend == "jaccl" or hostfile.backend == "jaccl-ring":
        launch_jaccl(None, hostfile.hosts, launch_args, cmd)
    else:
        raise ValueError("Only ring, jaccl and jaccl-ring backends are supported.")


def get_files(path):
    files = [str(f.relative_to(path)) for f in path.rglob("*") if f.is_file()]
    return sorted(files)


def share_file(path, file, src, group=None):
    group = group or mx.distributed.init()
    all_sum = partial(mx.distributed.all_sum, group=group)

    if group.rank() == src:
        with open(path / file, "rb") as f:
            f.seek(0, 2)
            total_size = f.tell()
            f.seek(0)

            pbar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=file,
            )
            while True:
                data = f.read(CHUNK_SIZE)
                if not data:
                    mx.eval(all_sum(0))
                    break

                mx.eval(all_sum(len(data)))
                mx.async_eval(all_sum(data))
                pbar.update(len(data))
            pbar.close()

    else:
        with open(path / file, "wb") as f:
            data = None
            chunk_size = all_sum(0).item()
            if chunk_size > 0:
                data = all_sum(mx.zeros(chunk_size, dtype=mx.uint8))
                mx.eval(data)

            while chunk_size > 0:
                next_data = None
                chunk_size = all_sum(0).item()
                if chunk_size > 0:
                    next_data = all_sum(mx.zeros(chunk_size, dtype=mx.uint8))
                    mx.async_eval(next_data)

                f.write(bytes(data))
                data = next_data


def share_files(path, files, src, group=None):
    group = group or mx.distributed.init()
    all_sum = partial(mx.distributed.all_sum, group=group)

    if group.rank() == src:
        # Share the list first
        file_list = "|".join(files).encode("utf-8")
        mx.eval(all_sum(len(file_list)))
        mx.eval(all_sum(file_list))

    else:
        # Get the list first
        file_list_size = all_sum(0).item()
        data = all_sum(mx.zeros(file_list_size, dtype=mx.uint8))
        files = bytes(data).decode("utf-8").split("|")

    for file in files:
        (path / file).parent.mkdir(parents=True, exist_ok=True)
        share_file(path, file, src, group)


def main():
    parser = argparse.ArgumentParser(
        description="Distribute a model to other nodes using MLX distributed."
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the MLX model."
    )
    parser.add_argument(
        "--hostfile",
        type=str,
        help="The file containing the hosts and connection information",
    )
    parser.add_argument(
        "--dst",
        type=str,
        help="The destination path in other nodes (defaults to --path)",
    )
    parser.add_argument(
        "--tmpdir",
        type=str,
        help="Intermediate temporary directory to ensure successfull transfer",
    )

    args = parser.parse_args()

    mx.set_default_device(mx.cpu)
    world = mx.distributed.init()

    if world.size() == 1:
        launch(args)
        return

    # Check if any node has the file
    path = None
    files = []
    try:
        path = Path(args.path)
        if path.exists():
            if path.is_file():
                files = [path.name]
                path = path.parent
            else:
                files = get_files(path)
        else:
            path = hf_repo_to_path(args.path)
            files = get_files(path)
    except:
        pass
    has_file = mx.distributed.all_gather(len(files) > 0)
    src = has_file.argmax().item()
    has_file = has_file.any().item()

    if not has_file:
        error("The --path needs to exist in at least one node.")
        error("If it is a remote repository download it first with `hf download`")
        sys.exit(1)

    # Share the path that is resolved
    if args.dst is None:
        if world.rank() == src:
            data = str(path).encode("utf-8")
            mx.eval(mx.distributed.all_sum(len(data)))
            mx.eval(mx.distributed.all_sum(data))
        else:
            data_size = mx.distributed.all_sum(0).item()
            data = mx.distributed.all_sum(mx.zeros(data_size, dtype=mx.uint8))
            path = Path(bytes(data).decode("utf-8"))
    elif world.rank() != src:
        path = Path(args.dst)

    with TemporaryDirectory(dir=args.tmpdir) as tmp:
        if world.rank() == src:
            share_files(path, files, src, world)
        else:
            share_files(Path(tmp), files, src, world)
            path.mkdir(parents=True, exist_ok=True)
            os.rename(tmp, path)
