# Copyright © 2026 Apple Inc.

import argparse
import logging
import os
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients, clip_grad_norm_sharded
from mlx.utils import tree_map

from mlx_lm.train import data, fsdp, optim, utils
from mlx_lm.train.checkpoint import load_training_state, save_checkpoint
from mlx_lm.train.distributed import init_distributed
from mlx_lm.train.metrics import Losses, Metrics, init_wandb, log_metrics


def main(config, save_dir):

    np.random.seed(config.seed)
    mx.random.seed(config.seed)

    fsdp_dim = config.get("fsdp_dim", 1)

    # Initialize distributed mesh and process groups
    mesh = init_distributed(fsdp_dim)

    grad_accum_steps = config.get("grad_accum_steps", 1)
    max_grad_norm = config.get("max_grad_norm", None)

    dtype = getattr(mx, config.data_type)

    save_dir = Path(save_dir)

    if mesh.is_master:
        save_dir.mkdir(parents=True, exist_ok=True)
        utils.save_config(save_dir, config)

    model = utils.build_model(config.model)
    optimizer = optim.build_optimizer(
        config.optimizer, config.num_steps, config.get("resume_from_step", 0)
    )
    init_step, data_state = load_training_state(model, optimizer, config, mesh)
    tokenizer = utils.load_tokenizer(config.get("tokenizer", "allenai/Olmo-3-1025-7B"))

    documents = data.get_documents(
        config.dataset, tokenizer, mesh, data_state, seed=config.seed
    )

    stream = data.iterate_batches(
        documents,
        context_size=config.context_size,
        batch_size=config.batch_size,
        resume_state=data_state,
    )

    if config.get("grad_checkpoint", False):
        utils.grad_checkpoint(model.layers[0], dtype=dtype)

    if fsdp_dim > 1:
        fsdp.shard_model(model, mesh.fsdp.group, dtype)

    params = model.trainable_parameters()
    mx.eval(params)
    z_loss_weight = config.get("z_loss_weight", 0.0)

    def loss_fn(params, sample):

        model.update(
            tree_map(lambda x: x.astype(dtype), params) if fsdp_dim == 1 else params
        )
        inputs = sample[:, :-1]
        targets = sample[:, 1:]

        logits = model(inputs)
        ce = nn.losses.cross_entropy(logits, targets, reduction="none")
        losses = {"ce_loss": ce.sum() / targets.size}

        if z_loss_weight > 0:
            log_z = mx.logsumexp(logits, axis=-1)
            losses["z_loss"] = z_loss_weight * (log_z**2).mean()

        # value_and_grad differentiates the first element
        return sum(losses.values()), losses

    state = [optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(sample, params, grad_accum, do_update):
        (_, losses), grads = mx.value_and_grad(loss_fn)(params, sample)
        # accumulate
        if grad_accum is not None:
            grads = tree_map(lambda x, y: x + y, grads, grad_accum)
        # update
        if do_update:
            grads = average_gradients(
                tree_map(lambda x: x / grad_accum_steps, grads), mesh.ddp.group
            )
            grad_norm = None
            if max_grad_norm is not None:
                grads, grad_norm = clip_grad_norm_sharded(
                    grads, max_norm=max_grad_norm, group=mesh.fsdp.group
                )
            params = optimizer.apply_gradients(grads, params)

        return (
            losses,
            grad_norm if do_update else None,
            None if do_update else grads,
            params,
        )

    tokens_per_step = (
        config.context_size * config.batch_size * mesh.world.size * grad_accum_steps
    )
    metrics = Metrics(
        tokens_per_step=tokens_per_step,
        steps_per_report=config.steps_per_report,
    )
    losses_sum = Losses()
    batches = data.prefetch(stream)
    exhausted = False
    step_done = init_step

    for it in range(init_step, config.num_steps):
        grads = None
        for micro in range(grad_accum_steps):
            sample = next(batches, None)
            if sample is None:
                exhausted = True
                break
            data_state = sample["_data_state"]

            losses, grad_norm, grads, params = step(
                mx.array(sample["input_ids"]),
                params,
                grads,
                micro + 1 == grad_accum_steps,
            )

            losses_sum = losses_sum.plus(losses)
            mx.eval(losses_sum)
            mx.eval(grads, params, optimizer.state)

        if exhausted:
            break
        step_done = it + 1
        reduced = losses_sum.all_reduce(mesh.world)
        losses_sum = Losses()
        mx.eval(reduced, grad_norm)

        metrics.append(
            **reduced,
            train_loss=reduced.total(),
            learning_rate=optimizer.learning_rate,
            grad_norm=grad_norm,
        )

        if (it + 1) % config.steps_per_report == 0:
            report = metrics.report(it + 1)
            if mesh.is_master:
                log_metrics(report)

        # save_checkpoint all-gathers the FSDP shards, so every rank must call it.
        if (it + 1) % config.steps_per_checkpoint == 0:
            if mesh.is_master:
                logging.info("saving checkpoint at step %d", it + 1)
            save_checkpoint(save_dir, it + 1, params, optimizer, data_state, mesh)

    if step_done > init_step:
        if mesh.is_master:
            logging.info("saving final checkpoint at step %d", step_done)
        save_checkpoint(save_dir, step_done, params, optimizer, data_state, mesh)


def build_parser():
    parser = argparse.ArgumentParser(description="Pretrain a language model.")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to an experiment config, or a bundled name under "
        "configs/experiments, e.g. qwen/4B",
    )
    parser.add_argument(
        "--stage",
        default=None,
        choices=("pre", "mid"),
        help="Which dolma corpus to train on. Default: pre",
    )
    parser.add_argument(
        "--source",
        default=None,
        choices=("hf", "s3"),
        help="Where to read the data from. Default: hf. Options: hf (Hugging Face), s3 (Dolma corpus in S3)",
    )
    parser.add_argument(
        "--fsdp-dim",
        type=int,
        default=None,
        help="Number of ranks to shard the model over. Overrides the experiment config",
    )
    parser.add_argument(
        "--save-dir",
        default="checkpoints",
        help="Where to write the model and checkpoints",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="Checkpoint directory to start from",
    )
    parser.add_argument(
        "--restore",
        default=None,
        choices=("all", "optimizer", "model"),
        help="How much of the checkpoint to keep: all resumes the run, optimizer "
        "keeps the moments and restarts the schedule and the data, model loads "
        "the weights alone. Default: all",
    )
    parser.add_argument(
        "--resume-from-step",
        type=int,
        default=None,
        help="The step --init-from holds. With restore all the loop counts on "
        "from it; with restore optimizer it is the number of updates to hold the "
        "schedule back by. Checked against the checkpoint",
    )
    parser.add_argument(
        "--experiment-name", default=None, help="Run name for wandb; omit to disable"
    )
    return parser


def cli():
    from mlx_lm.train.utils import config_path, load_config

    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    config = load_config(config_path(args.config))

    if args.fsdp_dim is not None:
        config.fsdp_dim = args.fsdp_dim

    if args.init_from is not None:
        with config.ignore_type():
            config.init_from = args.init_from
    if args.restore is not None:
        config.restore = args.restore
    if args.resume_from_step is not None:
        config.resume_from_step = args.resume_from_step

    if args.stage or args.source or config.get("dataset") is None:
        config.dataset = data.dolma(args.stage or "pre", args.source or "hf")

    init_wandb(config, args, os.environ.get("MLX_RANK", "0") == "0")
    main(config, args.save_dir)


if __name__ == "__main__":
    cli()
