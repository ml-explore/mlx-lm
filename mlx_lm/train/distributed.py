# Copyright © 2026 Apple Inc.

import logging
from dataclasses import dataclass

import mlx.core as mx


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


@dataclass(frozen=True)
class Mesh:

    world: DistributedGroup
    fsdp: DistributedGroup
    ddp: DistributedGroup

    @property
    def is_master(self) -> bool:
        return self.world.is_master


def init_distributed(fsdp_dim: int = 1) -> Mesh:

    g = mx.distributed.init(backend="nccl")
    rank, size = g.rank(), g.size()

    if size % fsdp_dim != 0:
        raise ValueError(f"world size {size} is not divisible by fsdp_dim={fsdp_dim}")

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
