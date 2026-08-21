# Copyright © 2026 Apple Inc.

import argparse
import os
import pickle
import sys
import time
from dataclasses import dataclass
from functools import partial, total_ordering
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Optional

import mlx.core as mx
from huggingface_hub.errors import LocalEntryNotFoundError
from mlx._distributed_utils.common import Hostfile
from mlx._distributed_utils.launch import launch_jaccl, launch_ring
from tqdm import tqdm

from .utils import hf_repo_to_path

CHUNK_SIZE = 100 * 1024 * 1024


@total_ordering
@dataclass
class DirectoryEntry:
    entry_type: Literal["directory", "symlink", "file"]
    path: str
    dst: Optional[str]

    def __lt__(self, other):
        order_type = dict(directory=0, symlink=1, file=2)
        o1 = order_type[self.entry_type]
        o2 = order_type[other.entry_type]
        return o1 < o2 or (o1 == o2 and self.path < other.path)

    def __eq__(self, other):
        return (
            self.entry_type == other.entry_type
            and self.path == other.path
            and self.dst == other.dst
        )

    @classmethod
    def from_path(cls, root, path):
        entry_type = {
            (True, False): "directory",
            (False, True): "symlink",
            (False, False): "file",
        }[path.is_dir(), path.is_symlink()]
        dst = path.readlink() if path.is_symlink() else None

        return cls(entry_type, str(path.relative_to(root)), str(dst))


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
    ]
    if args.path is not None:
        cmd += ["--path", args.path]
    if args.model is not None:
        cmd += ["--model", args.model]
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
    if not path.is_dir():
        return path.parent, [DirectoryEntry.from_path(path.parent, path)]

    files = [DirectoryEntry.from_path(path, f) for f in path.rglob("*")]
    return path, sorted(files)


def format_bw(x):
    if x >= 1e9:
        return f"{x / 1e9:.2} GB/s"
    if x >= 1e6:
        return f"{x / 1e6:.2} MB/s"
    if x >= 1e3:
        return f"{x / 1e3:.2} KB/s"
    return f"{x:.2} B/s"


def share_file(path, file, src, group=None):
    group = group or mx.distributed.init()
    all_sum = partial(mx.distributed.all_sum, group=group)
    total_size = 0
    start_time = time.time()

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
                position=1,
                leave=False,
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

    return total_size, time.time() - start_time


def share_files(path, files, src, group=None):
    group = group or mx.distributed.init()
    all_sum = partial(mx.distributed.all_sum, group=group)

    if group.rank() == src:
        # Share the list first
        file_list = pickle.dumps(files)
        mx.eval(all_sum(len(file_list)))
        mx.eval(all_sum(file_list))

    else:
        # Get the list first
        file_list_size = all_sum(0).item()
        data = all_sum(mx.zeros(file_list_size, dtype=mx.uint8))
        files = pickle.loads(bytes(data))

        # Make the directories and symlinks
        for file in files:
            if file.entry_type == "directory":
                (path / file.path).mkdir()
            elif file.entry_type == "symlink":
                (path / file.path).symlink_to(file.dst)

    # Everybody shares the files
    total_size = 0
    total_time = 1e-6
    pbar = tqdm(total=len(files), desc="Files", position=0, disable=group.rank() != src)
    for file in files:
        if file.entry_type == "file":
            s, t = share_file(path, file.path, src, group)
            total_size += s
            total_time += t
        pbar.update(1)
        pbar.set_postfix(speed=format_bw(total_size / total_time))
    pbar.close()


def main():
    parser = argparse.ArgumentParser(
        description="Distribute a model to other nodes using MLX distributed."
    )
    parser.add_argument("--path", type=str, help="Path to a file or folder to share.")
    parser.add_argument(
        "--model", type=str, help="The path to a local model or Hugging Face repo"
    )
    parser.add_argument(
        "--hostfile",
        type=str,
        help="The file containing the hosts and connection information",
    )
    parser.add_argument(
        "--dst",
        type=str,
        help="The destination path in other nodes (defaults to --path or --model)",
    )
    parser.add_argument(
        "--tmpdir",
        type=str,
        help="Intermediate temporary directory to ensure successfull transfer",
    )

    args = parser.parse_args()

    if args.path is args.model is None:
        parser.error("One of --path or --model must be provided")

    mx.set_default_device(mx.cpu)
    world = mx.distributed.init()

    if world.size() == 1:
        launch(args)
        return

    # Check if any node has the data
    path = None
    files = []
    if args.path is not None and (path := Path(args.path)).exists():
        path, files = get_files(path)
    elif args.model is not None:
        try:
            path = hf_repo_to_path(args.model)
            if path.parent.name != "snapshots":
                raise ValueError(
                    f"The model repository appears to be corrupted, it resolved to {str(path)}"
                )
            path, files = get_files(path.parent.parent)
        except Exception as e:
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
