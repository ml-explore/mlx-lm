# Copyright © 2025 Apple Inc.

import argparse
import copy
import time
import types
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
import numpy as np
from mlx.utils import tree_map
from tqdm import tqdm

from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.losses import kl_div_loss
from mlx_lm.tuner.trainer import grad_checkpoint, iterate_batches
from mlx_lm.tuner.utils import print_trainable_parameters
from mlx_lm.utils import (
    load,
    load_tokenizer,
    pipeline_load,
    quantize_model,
    save,
)


def compute_dwq_targets(
    model,
    save_dir,
    train_data,
    valid_data,
    batch_size,
    max_seq_length,
    seed,
):
    rank = mx.distributed.init().rank()

    def _compute_targets(data, path, split):

        if rank == 0:
            path = path / split
            path.mkdir(parents=True, exist_ok=True)
        for i, (batch, _) in (
            pbar := tqdm(
                enumerate(iterate_batches(data, batch_size, max_seq_length, seed=seed)),
                total=len(data) // batch_size,
                desc=f"Computing targets for {split}",
                disable=rank != 0,
            )
        ):
            batch = batch[:, :-1]
            logits = model(batch)
            # Hack to make the last op pre-eval on the CPU to avoid even timeout
            logits = mx.stop_gradient(logits, stream=mx.cpu)
            mx.eval(logits)
            if rank == 0:
                idx = mx.argpartition(logits, kth=-1024, axis=-1)[..., -1024:]
                logits = mx.take_along_axis(logits, idx, axis=-1)

                file = path / f"{i:010d}.safetensors"
                mx.save_safetensors(file, {"logits": logits, "indices": idx})

    _compute_targets(valid_data, save_dir, "valid")
    _compute_targets(train_data, save_dir, "train")


def dwq_quantize(
    model,
    target_fn,
    opt,
    train_data,
    valid_data,
    batch_size,
    max_seq_length,
    seed,
    dtype: mx.Dtype = mx.bfloat16,
    gradient_checkpoint: bool = False,
    temperature: float = 2.0,
):
    group = mx.distributed.init()
    world_size = group.size()
    rank = group.rank()

    def rprint(*args, **kwargs):
        if rank == 0:
            tqdm.write(*args, **kwargs)

    def unfreeze(_, m):
        if (
            hasattr(m, "bits")
            and hasattr(m, "group_size")
            and m.mode == "affine"
            and m.bits < 8
        ):
            m.unfreeze(keys=["scales", "biases"], recurse=False)

    model.train()
    model.apply_to_modules(unfreeze)
    print_trainable_parameters(model)

    if gradient_checkpoint:
        grad_checkpoint(model.layers[0])

    scale = 1 / temperature

    def loss_fn(params, x, targets, lengths):
        model.update(tree_map(lambda x: x.astype(dtype), params))
        logits = model(x)
        if isinstance(targets, tuple):
            targets, ids = targets
            logits = mx.take_along_axis(logits, ids, axis=-1)
        losses = kl_div_loss(scale * logits, scale * targets)
        mask = mx.arange(1, 1 + targets.shape[1]) < lengths[:, 1:]
        ntoks = mask.sum()
        loss = (mask * losses).sum() / ntoks
        return loss, ntoks

    def step(inputs, targets, lengths, params):
        (loss, ntoks), grads = mx.value_and_grad(loss_fn)(
            params, inputs, targets, lengths
        )
        grads = nn.average_gradients(grads)
        params = opt.apply_gradients(grads, params)
        return loss, ntoks, params

    def validate(params, it):
        v_loss = 0.0
        v_tokens = 0
        for i, (batch, lengths) in tqdm(
            enumerate(
                iterate_batches(valid_data, batch_size, max_seq_length, seed=seed)
            ),
            total=len(valid_data) // batch_size,
            desc="Computing validation loss",
            leave=False,
        ):
            batch = batch[:, :-1]
            targets = target_fn(batch, i, split="valid")
            mx.eval(targets)
            loss, ntoks = loss_fn(params, batch, targets, lengths)
            mx.eval(loss, ntoks)
            loss = mx.distributed.all_sum(loss, stream=mx.cpu).item() / world_size
            ntoks = mx.distributed.all_sum(ntoks, stream=mx.cpu).item()
            v_tokens += ntoks
            v_loss += loss * ntoks
        loss = v_loss / v_tokens
        rprint(f"Validation: {it=}, {loss=:.3f}")
        return loss

    # Accumulate learned weights in higher precision
    params = tree_map(
        lambda x: x.astype(mx.float32),
        model.trainable_parameters(),
    )

    total_loss = 0.0
    total_tokens = 0
    tokens = 0

    tic = time.time()

    # Compute initial validation loss
    initial_valid_loss = valid_loss = validate(params, it=0)

    for it, (batch, lengths) in (
        pbar := tqdm(
            enumerate(
                iterate_batches(train_data, batch_size, max_seq_length, seed=seed)
            ),
            total=len(train_data) // batch_size,
        )
    ):
        batch = batch[:, :-1]
        targets = target_fn(batch, it, split="train")
        mx.eval(targets)
        loss, ntoks, params = step(batch, targets, lengths, params)
        mx.eval(loss, params)
        loss = mx.distributed.all_sum(loss, stream=mx.cpu).item() / world_size
        ntoks = mx.distributed.all_sum(ntoks, stream=mx.cpu).item()
        tokens += ntoks
        total_loss += loss * ntoks
        if rank == 0:
            pbar.set_description(desc=f"{loss=:.4f}")
            if (it + 1) % 20 == 0:
                toks_per_sec = tokens / (time.time() - tic)
                peak_memory_gb = mx.get_peak_memory() / 1e9
                avg_loss = total_loss / tokens
                total_tokens += tokens
                rprint(
                    f"{it=}, {avg_loss=:.4f}, {total_tokens=},"
                    f" {toks_per_sec=:.3f}, {peak_memory_gb=:.3f}",
                )
                tic = time.time()
                tokens = 0
                total_loss = 0
        if (it + 1) % 200 == 0:
            valid_loss = validate(params, it=it)

    valid_loss = validate(params, it=it)
    if initial_valid_loss < valid_loss:
        rprint(
            f"❌❌❌\n[WARNING] Final validation loss {valid_loss:.3f} is "
            f"worse than initial validation loss {initial_valid_loss:.3f}."
            " Model quality will likely be degraded.\n❌❌❌"
        )

    model.update(tree_map(lambda x: x.astype(dtype), params))


def load_data(
    tokenizer,
    data_path: str,
    num_samples: int,
    max_seq_length: int,
    num_valid_samples: int = 32,
):
    args = types.SimpleNamespace(
        hf_dataset={
            "path": data_path,
            "train_split": "train",
            "valid_split": "train[:1]",
        },
        train=True,
        test=False,
    )
    dataset = load_dataset(args, tokenizer)[0]
    perm = np.random.permutation(len(dataset))
    train_perm = perm[:num_samples].tolist()
    valid_perm = perm[num_samples : num_samples + num_valid_samples].tolist()

    def process(idx):
        tokens, offset = dataset.process(dataset[idx])
        return (tokens[:max_seq_length], offset)

    train = [process(i) for i in train_perm]
    valid = [process(i) for i in valid_perm]
    return train, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        help="A model to distill from for DWQ. If `quantized-model` is not"
        " given the student model will be this model quantized according"
        " to `bits` and `group-size`.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--quantized-model",
        type=str,
        default=None,
        help="An already quantized model (the student model) to improve with DWQ.",
    )
    parser.add_argument(
        "--mlx-path", default="mlx_model", help="Path to save the quantized model."
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Bits per weight for quantization.",
    )
    parser.add_argument(
        "--group-size", type=int, default=64, help="Group size for quantization."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2048,
        help="Number of samples to use for training.",
    )
    parser.add_argument("--max-seq-length", type=int, default=1025)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--data-path",
        type=str,
        default="allenai/tulu-3-sft-mixture",
        help="A Hugging Face dataset which is compatible with an mlx-lm dataset format.",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Use gradient checkpointing to reduce memory use.",
    )
    parser.add_argument(
        "--target-dir", type=str, default=None, help="Directory to save/load targets."
    )
    parser.add_argument(
        "--targets-only", action="store_true", help="Compute the targets and exit."
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipeline parallel instead of data parallel.",
    )

    args = parser.parse_args()

    group = mx.distributed.init()

    num_samples = args.num_samples
    if not args.pipeline and num_samples % group.size() > 0:
        num_samples += group.size() - num_samples % group.size()

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    if args.target_dir is not None:
        target_dir = Path(args.target_dir)
        has_targets = target_dir.exists()
    else:
        has_targets = False
        target_dir = None

    tokenizer = load_tokenizer(args.model)

    train_data, valid_data = load_data(
        tokenizer, args.data_path, args.num_samples, args.max_seq_length
    )

    # Load the base model if we need it
    if not has_targets or args.quantized_model is None:
        if args.pipeline and group.size() > 1:
            model, _, config = pipeline_load(args.model, return_config=True)
        else:
            model, _, config = load(args.model, return_config=True, lazy=True)
    else:
        model = None

    # Pre-compute the targets
    if not has_targets and target_dir is not None:
        compute_dwq_targets(
            model,
            target_dir,
            train_data,
            valid_data,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
        )
        has_targets = True

    if args.targets_only:
        exit(0)

    if has_targets:

        def target_fn(_, idx, split):
            targets = mx.load(target_dir / split / f"{idx:010d}.safetensors")
            return targets["logits"], targets["indices"]

    else:

        def target_fn(batch, idx, split):
            return model(batch)

    if args.quantized_model is not None:
        q_model, tokenizer, config = load(
            args.quantized_model,
            lazy=True,
            return_config=True,
        )
        if "quantization" not in config:
            raise ValueError("Quantized model must already be quantized.")
    else:
        q_model = copy.deepcopy(model)
        _, config = quantize_model(
            q_model,
            config,
            group_size=args.group_size,
            bits=args.bits,
        )

    # Delete the base model if it's not needed
    if has_targets and model is not None:
        del model

    if mx.metal.is_available():
        max_rec_size = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(max_rec_size)

    opt = optimizers.Adam(learning_rate=args.learning_rate, bias_correction=True)
    dwq_quantize(
        q_model,
        target_fn,
        opt,
        train_data,
        valid_data,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        gradient_checkpoint=args.grad_checkpoint,
    )
    save(
        args.mlx_path,
        args.model,
        q_model,
        tokenizer,
        config,
    )
