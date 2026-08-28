from functools import reduce
from typing import Optional

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.nn.layers.distributed import _shard
from mlx.utils import tree_flatten, tree_unflatten


def _make_gather_fn(group, full_shapes, shard_sizes, cast_dtype):
    S = group.size()
    indices = reduce(lambda acc, w: acc + [acc[-1] + w], shard_sizes, [0])
    split_indices = indices[1:-1]
    shard_shapes = [(shape[0] // S,) + tuple(shape[1:]) for shape in full_shapes]

    def _maybe_cast(x, dtype):
        if dtype is None or x.dtype == dtype:
            return x
        return x.astype(dtype)

    @mx.custom_function
    def gather(shards):
        big_shard = mx.concatenate(
            [_maybe_cast(s.reshape(1, -1), cast_dtype) for s in shards], axis=1
        )
        big_full = mx.distributed.all_gather(big_shard, group=group)
        parts = mx.split(big_full, split_indices, axis=1)
        return [p.reshape(shape) for p, shape in zip(parts, full_shapes)]

    @gather.vjp
    def gather_vjp(shards, cotangents, _):
        big_cot_full = mx.concatenate([c.reshape(S, -1) for c in cotangents], axis=1)
        big_cot_shard = mx.distributed.sum_scatter(big_cot_full, group=group) / S
        parts = mx.split(big_cot_shard, split_indices, axis=1)
        return [p.reshape(shape) for p, shape in zip(parts, shard_shapes)]

    return gather


def _maybe_shard(m, k, v):
    if isinstance(v, FullyShardedModule):
        return False
    return Module.valid_parameter_filter(m, k, v)


class FullyShardedModule(Module):
    def __init__(self, module, group, cast_dtype):
        super().__init__()
        group = group or mx.distributed.init()
        N = group.size()

        shard_params = module.filter_and_map(_maybe_shard)
        flat = tree_flatten(shard_params)
        for path, a in flat:
            if a.ndim == 0:
                raise ValueError(
                    f"FSDP: parameter {path} is a 0-D scalar and cannot be sharded."
                )
            if a.shape[0] % N != 0:
                raise ValueError(
                    f"FSDP: parameter {path} has shape {a.shape}; axis 0 must "
                    f"be divisible by the FSDP group size {N}."
                )

        self._paths = [k for k, _ in flat]
        full_shapes = [a.shape for _, a in flat]
        shard_sizes = [a.size // N for _, a in flat]

        module.update(_shard(shard_params, lambda p, w: 0, group))

        self.module = module
        self._gather_fn = _make_gather_fn(group, full_shapes, shard_sizes, cast_dtype)

    def _gathered_call(self, fn, *args, **kwargs):
        shard_tree = self.module.filter_and_map(_maybe_shard)
        shards = [a for _, a in tree_flatten(shard_tree)]
        fulls = self._gather_fn(shards)
        self.module.update(tree_unflatten(list(zip(self._paths, fulls))))
        try:
            return fn(*args, **kwargs)
        finally:
            self.module.update(shard_tree)

    def __call__(self, *args, **kwargs):
        return self._gathered_call(self.module, *args, **kwargs)

    def as_linear(self, *args, **kwargs):
        return self._gathered_call(self.module.as_linear, *args, **kwargs)


def fully_shard(
    module: Module,
    group: Optional["mx.distributed.Group"] = None,
    cast_dtype: Optional[mx.Dtype] = None,
) -> Module:
    group = group or mx.distributed.init()
    if group.size() == 1:
        return module
    if isinstance(module, FullyShardedModule):
        return module

    wrapped = FullyShardedModule(module, group, cast_dtype)
    return wrapped if wrapped._paths else module


def shard_model(model, group, dtype):

    if group is None or group.size() == 1:
        return model

    def shard(m):
        return fully_shard(
            m,
            group=group,
            cast_dtype=dtype,
        )

    for i, layer in enumerate(model.layers):
        model.model.layers[i] = shard(layer)
    model.model.norm = shard(model.model.norm)
    model.model.embed_tokens = shard(model.model.embed_tokens)
    if not model.args.tie_word_embeddings:
        model.lm_head = shard(model.lm_head)
    return model
