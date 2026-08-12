# Copyright © 2025 Apple Inc.

"""Checkpoint save / resume.

Layout under ``save_dir``::

    <save_dir>/
      config.json
      <step:012d>/
        model.safetensors            full, unsharded weights (master only)
        opt_state_<fsdp_rank>.safetensors
        data_state_<world_rank>.json

``step`` is the number of steps *completed*, so ``000000000100`` holds the state
after 100 steps and ``init_step: 100`` resumes with the loop's next index at 100.
:func:`save_checkpoint` is therefore called with ``it + 1``.

The weights written are always *full*: when FSDP is active the parameter tree
holds 1/N shards under paths containing a ``module`` segment, so
:func:`gather_full_params` all-gathers them and strips that segment before
writing. Skipping that step produces a well-formed file containing one rank's
slice under full-model key names -- a failure that is silent until you resume.
"""

import json
import logging
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

_WRAPPER_SEGMENT = ".module."


def gather_full_params(params, mesh):
    """Reconstruct the full parameter tree from FSDP shards.

    Leaves whose path contains a ``module`` segment were sharded on axis 0 by
    ``fully_shard``; they are all-gathered over the FSDP group. The segment is
    then stripped so the saved keys match an unwrapped model.
    """
    if mesh.fsdp.size == 1:
        return params

    out = {}
    for path, x in tree_flatten(params):
        if _WRAPPER_SEGMENT in path:
            x = mesh.fsdp.all_gather(x)
            path = path.replace(_WRAPPER_SEGMENT, ".")
        out[path] = x
    mx.eval(list(out.values()))
    return tree_unflatten(list(out.items()))


def save_checkpoint(save_dir, step, params, optimizer, data_state, mesh):
    """Write weights, optimizer state and data position for one step.

    ``step`` of ``None`` writes into ``save_dir`` itself (the final save).
    """
    checkpoint_dir = Path(save_dir)
    if step is not None:
        checkpoint_dir /= f"{step:012d}"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    full = gather_full_params(params, mesh)
    if mesh.is_master:
        mx.save_safetensors(
            str(checkpoint_dir / "model.safetensors"), dict(tree_flatten(full))
        )

    # Optimizer state is sharded across the FSDP group and replicated across the
    # DDP axis, so exactly one replica of each shard is written.
    if mesh.ddp.is_leader:
        mx.save_safetensors(
            str(checkpoint_dir / f"opt_state_{mesh.fsdp.rank}.safetensors"),
            dict(tree_flatten(optimizer.state)),
        )

    # The data position is per rank: every rank reads a different slice.
    if data_state:
        with open(checkpoint_dir / f"data_state_{mesh.world.rank}.json", "w") as fid:
            json.dump(data_state, fid)

    mx.clear_cache()
    return checkpoint_dir


def load_weights(model, weight_path):
    weights = mx.load(str(weight_path))
    model.update(tree_unflatten(list(weights.items())))
    return model


def _convert_int_keys(tree):
    """safetensors keys are strings; optimizer state uses int list indices."""
    if isinstance(tree, dict):
        out = {}
        for k, v in tree.items():
            v = _convert_int_keys(v)
            out[int(k) if isinstance(k, str) and k.isdigit() else k] = v
        return out
    if isinstance(tree, list):
        return [_convert_int_keys(v) for v in tree]
    return tree


def init_state(model, optimizer, config, save_dir, mesh):
    """Restore weights, optimizer state and data position, if resuming.

    Returns ``(model, optimizer, data_state)``. ``data_state`` is empty when
    starting fresh or when ``no_data_state`` is set.

    Must run *before* :func:`~mlx_lm.train.sharding.shard_model`: the saved
    weights are full, and loading them into a wrapped model would mismatch both
    the shapes and the key names.
    """
    init_step = config.get("init_step", 0)
    if init_step == 0 and not config.get("resume", False):
        return model, optimizer, {}

    checkpoint_dir = Path(save_dir)
    if init_step > 0:
        checkpoint_dir /= f"{init_step:012d}"

    no_data_state = config.get("no_data_state", False)

    load_weights(model, checkpoint_dir / "model.safetensors")

    opt_path = checkpoint_dir / f"opt_state_{mesh.fsdp.rank}.safetensors"
    optimizer.state = _convert_int_keys(
        tree_unflatten(list(mx.load(str(opt_path)).items()))
    )
    optimizer._initialized = True
    if config.get("reset_schedule", no_data_state):
        # Restart the LR schedule from zero (midtraining, context extension).
        optimizer.state["step"] = mx.array(0)

    data_state = {}
    if not no_data_state:
        data_path = checkpoint_dir / f"data_state_{mesh.world.rank}.json"
        if data_path.exists():
            with open(data_path) as fid:
                data_state = json.load(fid)
        else:
            logging.warning("no data state at %s; starting the stream fresh", data_path)

    if mesh.is_master:
        logging.info("resumed from %s at step %d", checkpoint_dir, init_step)
    return model, optimizer, data_state
