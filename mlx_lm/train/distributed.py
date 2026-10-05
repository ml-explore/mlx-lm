# Copyright © 2026 Apple Inc.

import logging
from dataclasses import dataclass

import mlx.core as mx
from mlx.nn.utils import average_gradients


class DistributedGroup:
    """
    A wrapper class for a distributed group to fallback to single process if no group is provided.
    """

    def __init__(self, group=None):
        self.group = group

    @property
    def rank(self):
        return self.group.rank() if self.group is not None else 0

    @property
    def size(self):
        return self.group.size() if self.group is not None else 1

    @property
    def is_master(self):
        return self.rank == 0

    @property
    def is_leader(self):
        return self.rank == 0

    def all_gather(self, x):
        if self.group is None:
            return x
        return mx.distributed.all_gather(x, group=self.group)

    def average_gradients(self, grads, all_reduce_size):
        # mlx uses the global group when it gets None, so skip it here.
        if self.group is None:
            return grads
        return average_gradients(grads, self.group, all_reduce_size=all_reduce_size)


@dataclass(frozen=True)
class Mesh:

    world: DistributedGroup
    fsdp: DistributedGroup
    ddp: DistributedGroup

    @property
    def is_master(self) -> bool:
        return self.world.is_master


def init_fsdp_world(group) -> Mesh:
    """Shard over all ranks of ``group``, without a group split."""
    return Mesh(
        world=DistributedGroup(group),
        fsdp=DistributedGroup(group),
        ddp=DistributedGroup(None),
    )


def init_distributed(fsdp_dim: int = 1) -> Mesh:
    if mx.cuda.is_available():
        backend = "nccl" if mx.cuda.is_available() else "jaccl"
    elif mx.metal.is_available():
        backend = "jaccl"
    else:
        raise RuntimeError(
            "No supported distributed backend available. Please ensure that you have either CUDA or Metal support."
        )
    g = mx.distributed.init(backend=backend)
    rank, size = g.rank(), g.size()

    if backend == "jaccl" and fsdp_dim > 1:
        # jaccl has no group split, so FSDP must use all ranks.
        if fsdp_dim != size and rank == 0:
            logging.info(
                "jaccl has no group split, fsdp_dim %d set to world size %d",
                fsdp_dim,
                size,
            )
        mesh = init_fsdp_world(g)
    else:
        if size % fsdp_dim != 0:
            raise ValueError(
                f"world size {size} is not divisible by fsdp_dim={fsdp_dim}"
            )

        intra = lambda dim: g.split(rank // dim) if dim > 1 else None
        inter = lambda dim: g.split(rank % dim) if dim > 1 else g

        mesh = Mesh(
            world=DistributedGroup(g),
            fsdp=DistributedGroup(intra(fsdp_dim)),
            ddp=DistributedGroup(inter(fsdp_dim)),
        )
    if mesh.is_master:
        logging.info(
            "distributed: world=%d fsdp=%d ddp=%d",
            mesh.world.size,
            mesh.fsdp.size,
            mesh.ddp.size,
        )
    return mesh
