# Copyright © 2024 Apple Inc.


import time
from dataclasses import dataclass, field
from pathlib import Path
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten
from tqdm import tqdm

from .callbacks import TrainingCallback
from .datasets import CacheDataset


def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


@dataclass
class TrainingArgs:
    batch_size: int = field(default=4, metadata={"help": "Minibatch size."})
    iters: int = field(default=100, metadata={"help": "Iterations to train for."})
    val_batches: int = field(
        default=25,
        metadata={
            "help": "Number of validation batches, -1 uses the entire validation set."
        },
    )
    steps_per_report: int = field(
        default=10,
        metadata={"help": "Number of training steps between loss reporting."},
    )
    steps_per_eval: int = field(
        default=200, metadata={"help": "Number of training steps between validations."}
    )
    steps_per_save: int = field(
        default=100, metadata={"help": "Save the model every number steps"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length."}
    )
    adapter_file: str = field(
        default="adapters.safetensors",
        metadata={"help": "Save/load path for the trained adapter weights."},
    )
    grad_checkpoint: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to reduce memory use."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Accumulate gradients over this many micro-steps "
                "before applying an optimizer update."
            )
        },
    )


def default_loss(model, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    ce = ce.astype(mx.float32).sum() / ntoks

    return ce, ntoks


def iterate_batches(
    dataset,
    batch_size,
    max_seq_length,
    train=False,
):
    # Sort by length:
    if isinstance(dataset, CacheDataset):
        len_fn = lambda idx: dataset.itemlen(idx)
    else:
        len_fn = lambda idx: len(dataset[idx][0])
    idx = sorted(range(len(dataset)), key=len_fn)
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {len(dataset)}."
        )

    # If running in distributed mode (N machines) then each one should skip N-1
    # samples
    offset = mx.distributed.init().rank()
    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    # Make the batches:
    batch_idx = [
        idx[i + offset : i + offset + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]

    while True:
        indices = np.random.permutation(len(batch_idx))
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]
            if len(batch[0]) == 2:
                batch, offsets = zip(*batch)
            else:
                offsets = [0] * len(batch)
            lengths = [len(x) for x in batch]
            if max(lengths) > max_seq_length:
                print(
                    f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
                    f"The longest sentence {max(lengths)} will be truncated to {max_seq_length}. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to one plus nearest multiple of pad_to or the maximum length
            pad_to = 32
            max_length_in_batch = 1 + pad_to * ((max(lengths) + pad_to - 1) // pad_to)
            max_length_in_batch = min(max_length_in_batch, max_seq_length)

            batch_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)

            for j in range(batch_size // step):
                truncated_length = min(lengths[j], max_seq_length)
                batch_arr[j, :truncated_length] = batch[j][:truncated_length]
                lengths[j] = (
                    truncated_length  # Update lengths to match truncated lengths
                )
            batch = mx.array(batch_arr)
            yield batch, mx.array(list(zip(offsets, lengths)))

        if not train:
            break


def evaluate(
    model,
    dataset,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches,
):
    model.eval()
    all_losses = mx.array(0.0)
    ntokens = mx.array(0)

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in tqdm(
        zip(
            index_iterator,
            iterate_batches(
                dataset=dataset,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
            ),
        ),
        desc="Calculating loss...",
        total=min(len(dataset) // batch_size, num_batches),
    ):
        losses, toks = loss(model, *batch)
        all_losses += losses * toks
        ntokens += toks
        mx.eval(all_losses, ntokens)

    all_losses = mx.distributed.all_sum(all_losses, stream=mx.cpu)
    ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)

    return (all_losses / ntokens).item()


def train(
    model,
    optimizer,
    train_dataset,
    val_dataset,
    args: TrainingArgs = TrainingArgs(),
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches,
    training_callback: TrainingCallback = None,
):
    if mx.metal.is_available():
        mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
    print(f"Starting training..., iters: {args.iters}")
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    # tree math
    def zeros_like_tree(tree):
        if isinstance(tree, dict):
            return {k: zeros_like_tree(v) for k, v in tree.items()}
        elif isinstance(tree, (list, tuple)):
            t = [zeros_like_tree(v) for v in tree]
            return type(tree)(t)
        else:
            return mx.zeros_like(tree)

    def tree_add(a, b):
        if isinstance(a, dict):
            return {k: tree_add(a[k], b[k]) for k in a}
        elif isinstance(a, (list, tuple)):
            t = [tree_add(x, y) for x, y in zip(a, b)]
            return type(a)(t)
        else:
            return a + b

    def tree_div(a, scalar):
        if isinstance(a, dict):
            return {k: tree_div(v, scalar) for k, v in a.items()}
        elif isinstance(a, (list, tuple)):
            t = [tree_div(v, scalar) for v in a]
            return type(a)(t)
        else:
            return a / scalar

    def tree_l2_and_absmax(tree):
        """Return (sum_of_squares, absmax) over all leaves in the tree as MLX arrays."""
        def _acc(t):
            if isinstance(t, dict):
                ss = mx.array(0.0, dtype=mx.float32)
                am = mx.array(0.0, dtype=mx.float32)
                for v in t.values():
                    ssv, amv = _acc(v)
                    ss = ss + ssv
                    am = mx.maximum(am, amv)
                return ss, am
            elif isinstance(t, (list, tuple)):
                ss = mx.array(0.0, dtype=mx.float32)
                am = mx.array(0.0, dtype=mx.float32)
                for v in t:
                    ssv, amv = _acc(v)
                    ss = ss + ssv
                    am = mx.maximum(am, amv)
                return ss, am
            else:
                x = t.astype(mx.float32)
                ss = mx.sum(x * x)
                am = mx.max(mx.abs(x))
                return ss, am
        return _acc(tree)

    # gradfn & accum
    loss_value_and_grad = nn.value_and_grad(model, loss)
    grad_accum = zeros_like_tree(model.trainable_parameters())
    micro_count = 0

    @mx.compile
    def micro_forward_backward(batch):
        # returns ((loss, ntoks), grad_tree)
        return loss_value_and_grad(model, *batch)

    def apply_optimizer_update():
        nonlocal grad_accum, micro_count
        # average over local micro-steps first
        grad_local_avg = tree_div(grad_accum, args.gradient_accumulation_steps)
        if rank == 0:
            ss_loc, am_loc = tree_l2_and_absmax(grad_local_avg)
            mx.eval(ss_loc, am_loc)
            print(
                f"[GRAD-DBG] UPDATE(local) GA={args.gradient_accumulation_steps} "
                f"local_l2={mx.sqrt(ss_loc).item():.3e} local_absmax={am_loc.item():.3e}",
                flush=True,
            )
        # average across workers (single all-reduce per update)
        grad_avg = average_gradients(grad_local_avg)
        if rank == 0:
            ss_w, am_w = tree_l2_and_absmax(grad_avg)
            mx.eval(ss_w, am_w)
            print(
                f"[GRAD-DBG] UPDATE(world) GA={args.gradient_accumulation_steps} "
                f"world_l2={mx.sqrt(ss_w).item():.3e} world_absmax={am_w.item():.3e}",
                flush=True,
            )
        optimizer.update(model, grad_avg)
        mx.eval(model.state, optimizer.state)
        grad_accum = zeros_like_tree(model.trainable_parameters())
        micro_count = 0

    # ---------- training state ----------
    model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0.0

    # Main training loop
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        tic = time.perf_counter()
        # Report validation loss if needed, the first validation loss is always measured before any training.
        if args.steps_per_eval is not None and (it == 1 or it % args.steps_per_eval == 0 or it == args.iters):
            tic = time.perf_counter()
            val_loss = evaluate(
                model=model,
                dataset=val_dataset,
                loss=loss,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=iterate_batches,
            )
            model.train()
            val_time = time.perf_counter() - tic
            if rank == 0:
                print(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {val_time:.3f}s",
                    flush=True,
                )

            if training_callback is not None:
                val_info = {
                    "iteration": it - 1,
                    "val_loss": val_loss,
                    "val_time": val_time,
                }
                training_callback.on_val_loss_report(val_info)

            tic = time.perf_counter()

        # ----- micro step: forward+backward, accumulate grads -----
        (lvalue, toks), grad = micro_forward_backward(batch)
        grad_accum = tree_add(grad_accum, grad)
        micro_count += 1
        if rank == 0 and micro_count == args.gradient_accumulation_steps:
            ss_g, am_g = tree_l2_and_absmax(grad)
            ss_a, am_a = tree_l2_and_absmax(grad_accum)
            mx.eval(ss_g, am_g, ss_a, am_a)
            grad_l2 = mx.sqrt(ss_g).item()
            accum_l2 = mx.sqrt(ss_a).item()
            grad_absmax = am_g.item()
            accum_absmax = am_a.item()
            print(
                f"[GRAD-DBG] it={it} micro={micro_count}/{args.gradient_accumulation_steps} "
                f"loss={float(lvalue.item()):.4f} toks={int(toks.item())} "
                f"grad_l2={grad_l2:.3e} grad_absmax={grad_absmax:.3e} "
                f"accum_l2={accum_l2:.3e} accum_absmax={accum_absmax:.3e}",
                flush=True,
            )

        # logging accumulators (loss/tokens) – keep at micro-step granularity
        losses += lvalue
        n_tokens += toks
        steps += 1
        mx.eval(losses, n_tokens)
        train_time += time.perf_counter() - tic

        # ----- apply optimizer update every GA steps -----
        if micro_count == args.gradient_accumulation_steps:
            apply_optimizer_update()

        # Report training loss if needed
        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * world_size
            n_tokens_world = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / train_time
            tokens_sec = float(n_tokens_world) / train_time
            trained_tokens += n_tokens_world
            peak_mem = mx.get_peak_memory() / 1e9
            if rank == 0:
                print(
                    f"Iter {it}: Train loss {train_loss:.3f}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Trained Tokens {trained_tokens}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            n_tokens = 0
            steps = 0
            train_time = 0.0

        # Save adapter weights
        if it % args.steps_per_save == 0 and rank == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}.",
                flush=True,
            )

    # Flush a partial accumulation at the very end (if iters not divisible by GA)
    if micro_count > 0:
        apply_optimizer_update()

    # Save final weights
    if rank == 0:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        print(f"Saved final weights to {args.adapter_file}.")
