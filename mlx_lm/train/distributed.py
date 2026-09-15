# Copyright © 2026 Apple Inc.

import logging
from dataclasses import dataclass

import mlx.core as mx
from mlx.nn.utils import average_gradients, clip_grad_norm_sharded
from mlx.optimizers import clip_grad_norm


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
        # mlx's average_gradients falls back to the global group when it is
        # handed None, so the no-group case has to short circuit here.
        if self.group is None:
            return grads
        return average_gradients(grads, self.group, all_reduce_size=all_reduce_size)

    def clip_grad_norm(self, grads, max_norm):
        # Without a group the gradients are whole rather than sharded, so the
        # norm is local; clip_grad_norm_sharded would sum it over the global
        # group and inflate it by sqrt(world size).
        if self.group is None:
            return clip_grad_norm(grads, max_norm)
        return clip_grad_norm_sharded(grads, max_norm, group=self.group)


@dataclass(frozen=True)
class Mesh:

    world: DistributedGroup
    fsdp: DistributedGroup
    ddp: DistributedGroup

    @property
    def is_master(self) -> bool:
        return self.world.is_master


def _backend() -> str:
    if mx.cuda.is_available():
        return "nccl"
    if mx.metal.is_available():
        return "jaccl"
    raise RuntimeError(
        "No supported distributed backend available. Please ensure that you have either CUDA or Metal support."
    )


def _init_distributed(fsdp_dim: int = 1) -> Mesh:

    g = mx.distributed.init(backend=_backend())
    if fsdp_dim != g.size():
        raise ValueError(
            "without group split the fsdp group is the whole world, so "
            f"fsdp_dim has to be {g.size()}, got {fsdp_dim}"
        )
    return Mesh(
        world=DistributedGroup(g),
        fsdp=DistributedGroup(g),
        ddp=DistributedGroup(None),
    )


def init_distributed(fsdp_dim: int = 1) -> Mesh:
    backend = _backend()
    if backend == "jaccl" and fsdp_dim > 1:
        # jaccl has no group split, so fsdp has to span the whole world.
        mesh = _init_distributed(fsdp_dim)
    else:
        g = mx.distributed.init(backend=backend)
        rank, size = g.rank(), g.size()

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
            "distributed: backend=%s world=%d fsdp=%d ddp=%d",
            backend,
            mesh.world.size,
            mesh.fsdp.size,
            mesh.ddp.size,
        )
    return mesh
