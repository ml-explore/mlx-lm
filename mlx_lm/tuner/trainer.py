import time
from dataclasses import dataclass, field
from pathlib import Path
import os

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
        # Use default loss function (no gradient_accumulation_steps parameter)
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

    # Debug: Check initial model parameters - FIXED
    if rank == 0:
        initial_params = model.trainable_parameters()
        flat_params = tree_flatten(initial_params)
        param_sum = sum(mx.sum(param).item() for key, param in flat_params)  # Fixed line
        print(f"Initial parameter sum: {param_sum}")

    def tree_l2_norm(tree):
        flat_tree = tree_flatten(tree)
        if not flat_tree:
            return mx.array(0.0)
        # Fixed: Extract values from (key, value) tuples
        values = [param for key, param in flat_tree]
        squared_vals = [mx.sum(v.astype(mx.float32) ** 2) for v in values]
        return mx.sqrt(mx.sum(mx.stack(squared_vals)))

    # Create loss and gradient function
    loss_and_grad_fn = nn.value_and_grad(model, loss)
    
    # Initialize gradient accumulator
    accumulated_grads = None
    accumulation_count = 0

    def reset_accumulation():
        nonlocal accumulated_grads, accumulation_count
        accumulated_grads = None
        accumulation_count = 0

    def accumulate_gradients(new_grads):
        nonlocal accumulated_grads, accumulation_count
        if accumulated_grads is None:
            accumulated_grads = new_grads
        else:
            accumulated_grads = tree_map(lambda acc, new: acc + new, accumulated_grads, new_grads)
        accumulation_count += 1

    def apply_accumulated_gradients():
        nonlocal accumulated_grads, accumulation_count
        if accumulated_grads is None or accumulation_count == 0:
            print("WARNING: No gradients to apply!")
            return
        
        # Average the accumulated gradients
        final_grads = tree_map(lambda g: g / accumulation_count, accumulated_grads)
        
        # Debug gradient norms
        if rank == 0:
            grad_norm = tree_l2_norm(final_grads)
            mx.eval(grad_norm)
            print(f"Gradient L2 norm: {grad_norm.item():.6f}")
        
        # Average across workers if distributed
        if world_size > 1:
            final_grads = average_gradients(final_grads)
        
        # Store parameters before update for debugging - FIXED
        if rank == 0:
            old_params = model.trainable_parameters()
            flat_old_params = tree_flatten(old_params)
            old_param_sum = sum(mx.sum(param).item() for key, param in flat_old_params)  # Fixed line
        
        # Apply optimizer update
        optimizer.update(model, final_grads)
        mx.eval(model.parameters(), optimizer.state)
        
        # Debug parameter changes - FIXED
        if rank == 0:
            new_params = model.trainable_parameters()
            flat_new_params = tree_flatten(new_params)
            new_param_sum = sum(mx.sum(param).item() for key, param in flat_new_params)  # Fixed line
            param_change = abs(new_param_sum - old_param_sum)
            print(f"Parameter change: {param_change:.6f} (old: {old_param_sum:.6f}, new: {new_param_sum:.6f})")
            if param_change < 1e-8:
                print("WARNING: Very small parameter change detected!")
        
        reset_accumulation()

    # Training state
    model.train()
    total_loss = 0.0
    total_tokens = 0
    report_loss = 0.0
    report_tokens = 0
    report_steps = 0
    train_time = 0.0

    # Create batch iterator
    batch_iterator = iterate_batches(
        dataset=train_dataset,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        train=True,
    )

    # Main training loop
    for iteration in range(1, args.iters + 1):
        tic = time.perf_counter()
        
        try:
            batch = next(batch_iterator)
        except StopIteration:
            print("WARNING: Batch iterator exhausted, recreating...")
            batch_iterator = iterate_batches(
                dataset=train_dataset,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                train=True,
            )
            batch = next(batch_iterator)
        
        # Forward and backward pass
        (loss_value, n_tokens), grads = loss_and_grad_fn(model, *batch)
        
        # Debug: Check if loss and gradients are reasonable
        if rank == 0 and iteration <= 5:
            grad_norm = tree_l2_norm(grads)
            mx.eval(loss_value, n_tokens, grad_norm)
            print(f"Iter {iteration}: loss={loss_value.item():.4f}, "
                  f"tokens={n_tokens.item()}, grad_norm={grad_norm.item():.6f}")
        
        # Accumulate gradients
        accumulate_gradients(grads)
        
        # Accumulate loss and tokens for reporting
        report_loss += loss_value.item()
        report_tokens += n_tokens.item()
        report_steps += 1
        
        total_loss += loss_value.item()
        total_tokens += n_tokens.item()
        
        mx.eval(loss_value, n_tokens)
        train_time += time.perf_counter() - tic
        
        # Apply gradients when accumulation is complete
        if accumulation_count >= args.gradient_accumulation_steps:
            apply_accumulated_gradients()
        
        # Validation
        if (args.steps_per_eval is not None and 
            iteration % args.steps_per_eval == 0):
            
            # Make sure any pending gradients are applied
            if accumulation_count > 0:
                apply_accumulated_gradients()
            
            model.eval()
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
            
            if rank == 0:
                print(f"Iter {iteration}: Val loss {val_loss:.4f}")
            
            if training_callback is not None:
                training_callback.on_val_loss_report({
                    "iteration": iteration,
                    "val_loss": val_loss,
                })
        
        # Report training progress
        if iteration % args.steps_per_report == 0 or iteration == args.iters:
            avg_loss = report_loss / report_steps
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / train_time
            tokens_sec = report_tokens / train_time
            peak_mem = mx.get_peak_memory() / 1e9
            
            if rank == 0:
                print(
                    f"Iter {iteration}: Train loss {avg_loss:.4f}, "
                    f"LR {learning_rate:.3e}, "
                    f"It/sec {it_sec:.2f}, "
                    f"Tokens/sec {tokens_sec:.0f}, "
                    f"Peak mem {peak_mem:.1f} GB"
                )
            
            if training_callback is not None:
                training_callback.on_train_loss_report({
                    "iteration": iteration,
                    "train_loss": avg_loss,
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "peak_memory": peak_mem,
                })
            
            # Reset reporting counters
            report_loss = 0.0
            report_tokens = 0
            report_steps = 0
            train_time = 0.0
        
        # Save checkpoints
        if iteration % args.steps_per_save == 0 and rank == 0:
            # Make sure any pending gradients are applied
            if accumulation_count > 0:
                apply_accumulated_gradients()
            
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = Path(args.adapter_file).parent / f"{iteration:07d}_adapters.safetensors"
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(f"Iter {iteration}: Saved weights to {checkpoint}")

    # Apply any remaining accumulated gradients
    if accumulation_count > 0:
        apply_accumulated_gradients()

    # Save final weights
    if rank == 0:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        print(f"Saved final weights to {args.adapter_file}")