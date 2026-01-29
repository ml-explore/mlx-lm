# Copyright © 2024 Apple Inc.


import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
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
    grad_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of steps to accumulate gradients before applying an optimizer update."
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


# -----------------------------------------------------------------------------
# Sequence classification / regression helpers
#
# These helpers are designed to reuse the existing training loop (`train` /
# `evaluate`) by passing:
#   - iterate_batches=iterate_sequence_batches
#   - loss=sequence_loss (or a thin wrapper choosing classification/regression)
#
# Expected dataset item formats:
#   1) (input_ids, label)
#   2) (input_ids, attention_mask, label)
#   3) {"input_ids": ..., "attention_mask": ..., "label": ...}
#   4) {"text": ..., "label": ...}   (requires providing a tokenizer)
# -----------------------------------------------------------------------------

ArrayLike1D = Union[Sequence[int], np.ndarray, mx.array]


def _as_1d_int_tokens(x: Any) -> Sequence[int]:
    if isinstance(x, mx.array):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        raise TypeError(f"Unsupported token container type: {type(x)}")
    return [int(t) for t in x]


def _get_sequence_item_fields(
    item: Any,
    *,
    tokenizer=None,
    text_key: str = "text",
    label_key: str = "label",
    input_ids_key: str = "input_ids",
    attention_mask_key: str = "attention_mask",
) -> Tuple[Sequence[int], Optional[Sequence[int]], Any]:
    """
    Normalize dataset item into (input_ids, attention_mask, label).
    """
    if isinstance(item, dict):
        if input_ids_key in item:
            input_ids = _as_1d_int_tokens(item[input_ids_key])
            attention_mask = (
                _as_1d_int_tokens(item[attention_mask_key])
                if attention_mask_key in item and item[attention_mask_key] is not None
                else None
            )
            label = item[label_key]
            return input_ids, attention_mask, label

        if text_key in item and label_key in item:
            if tokenizer is None:
                raise ValueError(
                    "Dataset item provides text but no tokenizer was provided."
                )
            enc = tokenizer(item[text_key], return_tensors="np")
            input_ids = _as_1d_int_tokens(enc[input_ids_key][0])
            attention_mask = (
                _as_1d_int_tokens(enc.get(attention_mask_key, None)[0])
                if attention_mask_key in enc
                else None
            )
            label = item[label_key]
            return input_ids, attention_mask, label

        raise ValueError(
            f"Unsupported dict item format. Expected keys: "
            f"('{input_ids_key}', '{label_key}') or ('{text_key}', '{label_key}')."
        )

    if isinstance(item, (tuple, list)):
        if len(item) == 2:
            input_ids, label = item
            return _as_1d_int_tokens(input_ids), None, label
        if len(item) == 3:
            input_ids, attention_mask, label = item
            return (
                _as_1d_int_tokens(input_ids),
                _as_1d_int_tokens(attention_mask) if attention_mask is not None else None,
                label,
            )
        raise ValueError(
            "Tuple/list item must be (input_ids, label) or (input_ids, attention_mask, label)."
        )

    raise TypeError(f"Unsupported dataset item type: {type(item)}")


def iterate_sequence_batches(
    dataset: Any,
    batch_size: int,
    max_seq_length: int,
    loop: bool = False,
    seed: Optional[int] = None,
    comm_group=None,
    *,
    pad_to: int = 32,
    pad_token_id: int = 0,
    tokenizer=None,
    text_key: str = "text",
    label_key: str = "label",
    input_ids_key: str = "input_ids",
    attention_mask_key: str = "attention_mask",
) -> Generator[Tuple[mx.array, mx.array, mx.array], None, None]:
    """
    Batch iterator for sequence classification/regression.
    Yields (input_ids, attention_mask, labels).
    """
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} examples but only has {len(dataset)}."
        )

    # Distributed behavior matches iterate_batches
    if comm_group is not None:
        offset = comm_group.rank()
        step = comm_group.size()
    else:
        offset = 0
        step = 1
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    # Sort indices by length to reduce padding waste
    def item_len(idx: int) -> int:
        item = dataset[idx]
        input_ids, _, _ = _get_sequence_item_fields(
            item,
            tokenizer=tokenizer,
            text_key=text_key,
            label_key=label_key,
            input_ids_key=input_ids_key,
            attention_mask_key=attention_mask_key,
        )
        return len(input_ids)

    idx = sorted(range(len(dataset)), key=item_len)

    batch_idx = [
        idx[i + offset : i + offset + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]

    if seed is not None:
        np.random.seed(seed)

    local_bsz = batch_size // step

    while True:
        order = np.random.permutation(len(batch_idx))
        for bi in order:
            items = [dataset[j] for j in batch_idx[bi]]

            input_seqs: list[list[int]] = []
            masks: list[Optional[list[int]]] = []
            labels: list[Any] = []
            max_len = 0

            for it in items:
                ids, am, lab = _get_sequence_item_fields(
                    it,
                    tokenizer=tokenizer,
                    text_key=text_key,
                    label_key=label_key,
                    input_ids_key=input_ids_key,
                    attention_mask_key=attention_mask_key,
                )
                if len(ids) > max_seq_length:
                    ids = ids[:max_seq_length]
                    if am is not None:
                        am = am[:max_seq_length]
                input_seqs.append(list(ids))
                masks.append(list(am) if am is not None else None)
                labels.append(lab)
                max_len = max(max_len, len(ids))

            max_len = max(1, max_len)
            padded_len = pad_to * ((max_len + pad_to - 1) // pad_to)
            padded_len = min(padded_len, max_seq_length)

            input_arr = np.full((local_bsz, padded_len), pad_token_id, dtype=np.int32)
            attn_arr = np.zeros((local_bsz, padded_len), dtype=np.int32)

            for i in range(local_bsz):
                ids = input_seqs[i]
                L = min(len(ids), padded_len)
                input_arr[i, :L] = np.asarray(ids[:L], dtype=np.int32)
                if masks[i] is None:
                    attn_arr[i, :L] = 1
                else:
                    m = masks[i]
                    mL = min(len(m), padded_len)
                    attn_arr[i, :mL] = np.asarray(m[:mL], dtype=np.int32)

            labels_arr = np.asarray(labels[:local_bsz])
            yield mx.array(input_arr), mx.array(attn_arr), mx.array(labels_arr)

        if not loop:
            break


def sequence_loss(
    model: nn.Module,
    input_ids: mx.array,
    attention_mask: mx.array,
    labels: mx.array,
    *,
    task: str = "auto",
    num_labels: Optional[int] = None,
    label_smoothing: float = 0.0,
    regression_loss: str = "mse",
) -> Tuple[mx.array, mx.array]:
    """
    Compute loss for sequence classification/regression.
    Returns (loss, n_examples).

    task:
      - "auto": infer regression if labels are float or num_labels==1
      - "classification"
      - "regression"
    """
    logits = model(input_ids, attention_mask=attention_mask)

    if labels.ndim > 1:
        labels = labels.reshape((-1,))

    t = (task or "auto").lower()
    if t == "auto":
        if num_labels == 1:
            t = "regression"
        elif mx.issubdtype(labels.dtype, mx.floating):
            t = "regression"
        else:
            t = "classification"

    n = mx.array(input_ids.shape[0])

    if t == "classification":
        if logits.ndim != 2:
            raise ValueError(
                f"Classification expects logits of shape (B, C), got {logits.shape}"
            )
        labels = labels.astype(mx.int32)

        if label_smoothing and label_smoothing > 0.0:
            c = logits.shape[-1]
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            nll = -mx.take_along_axis(logp, labels[:, None], axis=-1).squeeze(-1)
            u = -mx.mean(logp, axis=-1)
            loss_vec = (1.0 - label_smoothing) * nll + label_smoothing * u
            loss = loss_vec.astype(mx.float32).mean()
        else:
            loss_vec = nn.losses.cross_entropy(logits, labels)
            loss = loss_vec.astype(mx.float32).mean()

        return loss, n

    if t == "regression":
        if logits.ndim == 2 and logits.shape[-1] == 1:
            preds = logits.squeeze(-1)
        elif logits.ndim == 1:
            preds = logits
        else:
            raise ValueError(
                f"Regression expects logits of shape (B,) or (B,1), got {logits.shape}"
            )

        y = labels.astype(mx.float32)
        p = preds.astype(mx.float32)

        if regression_loss.lower() == "huber":
            d = p - y
            abs_d = mx.abs(d)
            quad = mx.minimum(abs_d, mx.array(1.0, dtype=abs_d.dtype))
            lin = abs_d - quad
            loss_vec = 0.5 * quad * quad + lin
            loss = loss_vec.mean()
        else:
            loss = mx.mean(mx.square(p - y))

        return loss, n

    raise ValueError(
        f"Unknown task '{task}'. Use 'classification', 'regression', or 'auto'."
    )


def iterate_batches(
    dataset,
    batch_size,
    max_seq_length,
    loop=False,
    seed=None,
    comm_group=None,
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
    if comm_group is not None:
        offset = comm_group.rank()
        step = comm_group.size()
    else:
        offset = 0
        step = 1
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    # Make the batches:
    batch_idx = [
        idx[i + offset : i + offset + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]
    if seed:
        np.random.seed(seed)
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

        if not loop:
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
                comm_group=mx.distributed.init(),
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

    loss_value_and_grad = nn.value_and_grad(model, loss)

    grad_accum_steps = args.grad_accumulation_steps
    if grad_accum_steps < 1:
        raise ValueError("grad_accumulation_steps must be at least 1")

    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, toks, grad

    model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0
    grad_accum = None

    # Main training loop
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            loop=True,
            comm_group=world,
        ),
    ):
        tic = time.perf_counter()
        # Report validation loss if needed, the first validation loss
        # is always measured before any training.
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
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

        lvalue, toks, grad_accum = step(
            batch,
            grad_accum,
            it % grad_accum_steps == 0,
        )

        losses += lvalue
        n_tokens += toks
        steps += 1
        mx.eval(state, losses, n_tokens, grad_accum)
        train_time += time.perf_counter() - tic

        # Report training loss if needed
        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * world_size
            n_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / train_time
            tokens_sec = float(n_tokens) / train_time
            trained_tokens += n_tokens
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
            train_time = 0

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
                f"{args.adapter_file} and {checkpoint}."
            )

    # Save final weights
    if rank == 0:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        print(f"Saved final weights to {args.adapter_file}.")
