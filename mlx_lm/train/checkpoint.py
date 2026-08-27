# Copyright © 2026 Apple Inc.

import json
import logging
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

_WRAPPER_SEGMENT = ".module."
RESTORE_MODES = ("all", "optimizer", "model")


def gather_full_params(params, mesh):
    """Reconstruct the full parameter tree from FSDP shards, flat for saving.

    Leaves whose path contains a ``module`` segment were sharded on axis 0 by
    ``fully_shard``; they are all-gathered over the FSDP group. The segment is
    then stripped so the saved keys match an unwrapped model.
    """
    if mesh.fsdp.size == 1:
        return dict(tree_flatten(params))

    out = {}
    for path, x in tree_flatten(params):
        if _WRAPPER_SEGMENT in path:
            x = mesh.fsdp.all_gather(x)
            path = path.replace(_WRAPPER_SEGMENT, ".")
        out[path] = x
    mx.eval(list(out.values()))
    return out


def save_checkpoint(save_dir, step, params, optimizer, data_state, mesh):
    """Write weights, optimizer state and data position for one step."""
    checkpoint_dir = Path(save_dir) / f"{step:012d}"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    full = gather_full_params(params, mesh)
    if mesh.is_master:
        mx.save_safetensors(str(checkpoint_dir / "model.safetensors"), full)

    # Optimizer state is sharded across the FSDP group and replicated across the
    # DDP axis, so exactly one replica of each shard is written.
    if mesh.ddp.is_leader:
        mx.save_safetensors(
            str(checkpoint_dir / f"opt_state_{mesh.fsdp.rank}.safetensors"),
            dict(tree_flatten(optimizer.state)),
        )

    # The data position is per rank: every rank reads a different slice.
    with open(checkpoint_dir / f"data_state_{mesh.world.rank}.json", "w") as fid:
        json.dump(data_state, fid)

    mx.clear_cache()
    return checkpoint_dir


def _mode(config):
    mode = config.get("restore", "all")
    if mode not in RESTORE_MODES:
        raise SystemExit(
            f"unknown config.restore {mode!r}; expected one of "
            f"{', '.join(RESTORE_MODES)}"
        )
    return mode


def _shards(checkpoint_dir, mesh):
    """This rank's optimizer shard, once the saved width is known to match."""
    written = list(checkpoint_dir.glob("opt_state_*.safetensors"))
    if len(written) != mesh.fsdp.size:
        # TODO: reshard instead
        raise SystemExit(
            f"{checkpoint_dir} holds {len(written)} optimizer shards but this run "
            f"has fsdp_dim {mesh.fsdp.size}; rerun with --fsdp-dim {len(written)}, "
            "or use restore 'model' to start from the weights alone"
        )
    return checkpoint_dir / f"opt_state_{mesh.fsdp.rank}.safetensors"


def load_training_state(model, optimizer, config, mesh):

    init_from = config.get("init_from")
    mode = _mode(config)
    resume_from_step = config.get("resume_from_step", 0)
    if init_from is None:
        if resume_from_step:
            raise SystemExit(
                "config.resume_from_step holds the schedule back by the updates a "
                "checkpoint carries; it needs config.init_from"
            )
        return 0, {}

    checkpoint_dir = Path(init_from)
    weights = checkpoint_dir / "model.safetensors"
    if not weights.exists():
        raise SystemExit(f"no checkpoint to load at {weights}")
    try:
        model.load_weights(str(weights))
    except ValueError as error:
        raise SystemExit(f"cannot load {weights}: {error}") from error

    if mode == "model":
        if resume_from_step:
            raise SystemExit(
                "restore 'model' starts a fresh optimizer, so there are no "
                "updates to hold the schedule back by; leave "
                "config.resume_from_step at 0"
            )
        if mesh.is_master:
            logging.info("restored the weights from %s", checkpoint_dir)
        return 0, {}

    shard = _shards(checkpoint_dir, mesh)
    optimizer.state = tree_unflatten(list(mx.load(str(shard)).items()))
    updates = int(optimizer.state["step"].item())
    if resume_from_step > updates:
        raise SystemExit(
            f"config.resume_from_step is {resume_from_step} but {shard.name} "
            f"holds only {updates} updates"
        )
    step = updates - resume_from_step

    if mode == "optimizer":
        # A new phase starts at zero, so the offset has to cancel every update
        # the optimizer inherits, or the schedule lands mid-decay.
        if step != 0:
            raise SystemExit(
                f"config.resume_from_step is {resume_from_step} but "
                f"{shard.name} holds {updates} updates; set it to {updates}"
            )
        if mesh.is_master:
            logging.info(
                "restored the weights and optimizer from %s, at update %d",
                checkpoint_dir,
                updates,
            )
        return 0, {}

    positions = list(checkpoint_dir.glob("data_state_*.json"))
    if len(positions) != mesh.world.size:
        raise SystemExit(
            f"{checkpoint_dir} holds {len(positions)} data positions but this run "
            f"has {mesh.world.size} ranks; the saved position only covers the "
            "ranks that wrote it, so use restore 'optimizer' to start the data "
            "over instead"
        )
    with open(checkpoint_dir / f"data_state_{mesh.world.rank}.json") as fid:
        data_state = json.load(fid)

    if mesh.is_master:
        logging.info("resumed from %s at step %d", checkpoint_dir, step)
        if config.get("dataset", {}).get("source") == "hf" and data_state.get(
            "sample_idx"
        ):
            logging.warning(
                "the hf loader has no seek, so each rank replays about %d "
                "documents before the first batch",
                data_state["sample_idx"],
            )
    return step, data_state
