# Copyright © 2024 Apple Inc.


import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
from tqdm import tqdm

from .callbacks import TrainingCallback
from .datasets import CacheDataset
from .losses import dpo_loss


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


def dpo_iterate_batches(
    dataset,
    batch_size,
    max_seq_length,
    train=False,
):
    """
    Iterate over batches for DPO training.
    Each batch contains chosen and rejected sequences.
    """
    # Sort by the maximum sequence length between chosen and rejected
    if isinstance(dataset, CacheDataset):
        len_fn = lambda idx: max(
            len(dataset._data.process(dataset._data[idx])["chosen_tokens"]),
            len(dataset._data.process(dataset._data[idx])["rejected_tokens"]),
        )
    else:
        len_fn = lambda idx: max(
            len(dataset.process(dataset[idx])["chosen_tokens"]),
            len(dataset.process(dataset[idx])["rejected_tokens"]),
        )

    idx = sorted(range(len(dataset)), key=len_fn)
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {len(dataset)}."
        )

    # Distributed training setup
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
            batch_data = [dataset[j] for j in batch_idx[i]]

            # Process each item in the batch
            # CacheDataset returns already processed data, raw dataset needs processing
            if isinstance(dataset, CacheDataset):
                processed_batch = batch_data  # Already processed by CacheDataset
            else:
                processed_batch = [dataset.process(item) for item in batch_data]

            chosen_sequences = [item["chosen_tokens"] for item in processed_batch]
            rejected_sequences = [item["rejected_tokens"] for item in processed_batch]
            prompt_lengths = [item["prompt_length"] for item in processed_batch]

            # Get maximum sequence lengths
            max_chosen_len = max(len(seq) for seq in chosen_sequences)
            max_rejected_len = max(len(seq) for seq in rejected_sequences)
            max_len = max(max_chosen_len, max_rejected_len)

            if max_len > max_seq_length:
                print(
                    f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
                    f"The longest sentence {max_len} will be truncated to {max_seq_length}. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to nearest multiple of 32 or the maximum length
            pad_to = 32
            max_length_in_batch = 1 + pad_to * ((max_len + pad_to - 1) // pad_to)
            max_length_in_batch = min(max_length_in_batch, max_seq_length)

            # Create padded arrays
            chosen_batch = np.zeros((batch_size // step, max_length_in_batch), np.int32)
            rejected_batch = np.zeros(
                (batch_size // step, max_length_in_batch), np.int32
            )

            chosen_lengths = []
            rejected_lengths = []

            for j in range(batch_size // step):
                # Pad chosen sequence
                chosen_seq = chosen_sequences[j]
                chosen_len = min(len(chosen_seq), max_seq_length)
                chosen_batch[j, :chosen_len] = chosen_seq[:chosen_len]
                chosen_lengths.append(chosen_len)

                # Pad rejected sequence
                rejected_seq = rejected_sequences[j]
                rejected_len = min(len(rejected_seq), max_seq_length)
                rejected_batch[j, :rejected_len] = rejected_seq[:rejected_len]
                rejected_lengths.append(rejected_len)

            yield {
                "chosen_tokens": mx.array(chosen_batch),
                "rejected_tokens": mx.array(rejected_batch),
                "chosen_lengths": mx.array(chosen_lengths),
                "rejected_lengths": mx.array(rejected_lengths),
                "prompt_lengths": mx.array(prompt_lengths),
            }

        if not train:
            break


def dpo_loss_fn(
    policy_model,
    reference_model,
    batch,
    beta: float = 0.1,
):
    """
    DPO loss function for training.

    Args:
        policy_model: The model being trained
        reference_model: The frozen reference model
        batch: Batch dictionary with chosen/rejected tokens and lengths
        beta: Temperature parameter for DPO

    Returns:
        loss: DPO loss value
        ntoks: Number of tokens processed
    """
    chosen_tokens = batch["chosen_tokens"]
    rejected_tokens = batch["rejected_tokens"]
    chosen_lengths = batch["chosen_lengths"]
    rejected_lengths = batch["rejected_lengths"]
    prompt_lengths = batch["prompt_lengths"]

    # Forward pass through policy model
    chosen_inputs = chosen_tokens[:, :-1]
    chosen_targets = chosen_tokens[:, 1:]
    policy_chosen_logits = policy_model(chosen_inputs)

    rejected_inputs = rejected_tokens[:, :-1]
    rejected_targets = rejected_tokens[:, 1:]
    policy_rejected_logits = policy_model(rejected_inputs)

    # Forward pass through reference model (no gradients needed since model is frozen)
    reference_chosen_logits = reference_model(chosen_inputs)
    reference_rejected_logits = reference_model(rejected_inputs)

    # Compute DPO loss
    loss = dpo_loss(
        policy_chosen_logits=policy_chosen_logits,
        policy_rejected_logits=policy_rejected_logits,
        reference_chosen_logits=reference_chosen_logits,
        reference_rejected_logits=reference_rejected_logits,
        chosen_labels=chosen_targets,
        rejected_labels=rejected_targets,
        beta=beta,
    )

    # Calculate total number of tokens for reporting
    ntoks = mx.sum(chosen_lengths) + mx.sum(rejected_lengths)

    return loss, ntoks


def dpo_evaluate(
    policy_model,
    reference_model,
    dataset,
    batch_size,
    num_batches,
    max_seq_length=2048,
    beta: float = 0.1,
):
    """Evaluate DPO model on validation set with loss and accuracy."""
    policy_model.eval()
    reference_model.eval()

    all_losses = mx.array(0.0)
    ntokens = mx.array(0)

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in tqdm(
        zip(
            index_iterator,
            dpo_iterate_batches(
                dataset=dataset,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
            ),
        ),
        desc="Evaluating DPO...",
        total=min(len(dataset) // batch_size, num_batches),
    ):
        losses, toks = dpo_loss_fn(policy_model, reference_model, batch, beta)
        all_losses += losses * toks
        ntokens += toks
        mx.eval(all_losses, ntokens)

    all_losses = mx.distributed.all_sum(all_losses, stream=mx.cpu)
    ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)
    loss = (all_losses / ntokens).item()

    # Compute accuracy
    accuracy, margin = compute_preference_accuracy(
        policy_model,
        reference_model,
        dataset,
        batch_size,
        num_batches,
        max_seq_length,
        beta,
    )

    return {"loss": loss, "accuracy": accuracy, "avg_margin": margin}


def train_dpo(
    policy_model,
    reference_model,
    optimizer,
    train_dataset,
    val_dataset,
    args: TrainingArgs = TrainingArgs(),
    beta: float = 0.1,
    training_callback: TrainingCallback = None,
):
    """
    Train a model using Direct Preference Optimization (DPO).

    Args:
        policy_model: The model being trained
        reference_model: The frozen reference model
        optimizer: Optimizer for the policy model
        train_dataset: Training dataset with preference pairs
        val_dataset: Validation dataset with preference pairs
        args: Training arguments
        beta: Temperature parameter for DPO
        training_callback: Optional callback for logging
    """
    if mx.metal.is_available():
        mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
    print(f"Starting DPO training..., iters: {args.iters}, beta: {beta}")

    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(policy_model.layers[0])

    # Freeze reference model completely
    reference_model.freeze()

    # Create loss function with reference model
    def loss_fn(batch):
        return dpo_loss_fn(policy_model, reference_model, batch, beta)

    state = [
        policy_model.state,
        optimizer.state,
        mx.random.state,
        reference_model.state,
    ]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch):
        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(batch)

        # All reduce the gradients if running in distributed mode
        grad = average_gradients(grad)

        # Model update
        optimizer.update(policy_model, grad)

        return lvalue, toks

    loss_value_and_grad = nn.value_and_grad(policy_model, loss_fn)

    policy_model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0

    # Main training loop
    for it, batch in zip(
        range(1, args.iters + 1),
        dpo_iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        tic = time.perf_counter()
        # Report validation loss if needed
        # Replace the validation reporting section with:
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            tic = time.perf_counter()
            val_results = dpo_evaluate(
                policy_model=policy_model,
                reference_model=reference_model,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                beta=beta,
            )
            policy_model.train()
            val_time = time.perf_counter() - tic

            if rank == 0:
                print(
                    f"Iter {it}: "
                    f"Val loss {val_results['loss']:.3f}, "
                    f"Val accuracy {val_results.get('accuracy', 0):.3f}, "
                    f"Val margin {val_results.get('avg_margin', 0):.3f}, "
                    f"Val took {val_time:.3f}s",
                    flush=True,
                )

            if training_callback is not None:
                val_info = {
                    "iteration": it - 1,
                    "val_loss": val_results["loss"],
                    "val_accuracy": val_results.get("accuracy", 0),
                    "val_avg_margin": val_results.get("avg_margin", 0),
                    "val_time": val_time,
                }
                training_callback.on_val_loss_report(val_info)

            tic = time.perf_counter()

        lvalue, toks = step(batch)
        losses += lvalue
        n_tokens += toks
        steps += 1
        mx.eval(state, losses, n_tokens)
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
            adapter_weights = dict(tree_flatten(policy_model.trainable_parameters()))
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
        adapter_weights = dict(tree_flatten(policy_model.trainable_parameters()))
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        print(f"Saved final DPO weights to {args.adapter_file}.")


def compute_preference_accuracy(
    policy_model,
    reference_model,
    dataset,
    batch_size,
    num_batches,
    max_seq_length=2048,
    beta: float = 0.1,
):
    """Compute preference accuracy - how often policy prefers chosen over rejected."""
    policy_model.eval()
    reference_model.eval()

    correct_preferences = mx.array(0.0)
    total_examples = mx.array(0)
    total_margin = mx.array(0.0)

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in zip(
        index_iterator,
        dpo_iterate_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        chosen_tokens = batch["chosen_tokens"]
        rejected_tokens = batch["rejected_tokens"]

        # Forward pass
        chosen_inputs = chosen_tokens[:, :-1]
        chosen_targets = chosen_tokens[:, 1:]
        policy_chosen_logits = policy_model(chosen_inputs)
        reference_chosen_logits = reference_model(chosen_inputs)

        rejected_inputs = rejected_tokens[:, :-1]
        rejected_targets = rejected_tokens[:, 1:]
        policy_rejected_logits = policy_model(rejected_inputs)
        reference_rejected_logits = reference_model(rejected_inputs)

        # Compute log probabilities
        from .losses import _log_prob_from_logits_and_labels

        policy_chosen_logprobs = _log_prob_from_logits_and_labels(
            policy_chosen_logits, chosen_targets
        )
        policy_rejected_logprobs = _log_prob_from_logits_and_labels(
            policy_rejected_logits, rejected_targets
        )
        reference_chosen_logprobs = _log_prob_from_logits_and_labels(
            reference_chosen_logits, chosen_targets
        )
        reference_rejected_logprobs = _log_prob_from_logits_and_labels(
            reference_rejected_logits, rejected_targets
        )

        # Compute rewards
        policy_chosen_rewards = beta * (
            policy_chosen_logprobs - reference_chosen_logprobs
        )
        policy_rejected_rewards = beta * (
            policy_rejected_logprobs - reference_rejected_logprobs
        )

        # Check preferences
        reward_margin = policy_chosen_rewards - policy_rejected_rewards
        correct = (reward_margin > 0).astype(mx.float32)

        correct_preferences += mx.sum(correct)
        total_examples += chosen_tokens.shape[0]
        total_margin += mx.sum(reward_margin)

        mx.eval(correct_preferences, total_examples, total_margin)

    # Aggregate across distributed processes
    correct_preferences = mx.distributed.all_sum(correct_preferences, stream=mx.cpu)
    total_examples = mx.distributed.all_sum(total_examples, stream=mx.cpu)
    total_margin = mx.distributed.all_sum(total_margin, stream=mx.cpu)

    accuracy = (correct_preferences / total_examples).item()
    avg_margin = (total_margin / total_examples).item()

    return accuracy, avg_margin
