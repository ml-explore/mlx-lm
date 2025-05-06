# Copyright © 2025 Apple Inc.

import argparse
import copy
import glob
import shutil
import time
import types
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
import numpy as np
from mlx.utils import tree_flatten, tree_map

from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.trainer import iterate_batches
from mlx_lm.tuner.utils import print_trainable_parameters
from mlx_lm.utils import (
    create_model_card,
    fetch_from_hub,
    get_model_path,
    quantize_model,
    save_config,
    save_weights,
)


def dwq_quantize(
    model,
    q_model,
    opt,
    data,
    batch_size: int = 2,
    max_seq_length: int = 2048,
    temperature: float = 0.5,
    dtype: mx.Dtype = mx.bfloat16,
):
    group = mx.distributed.init()
    world_size = group.size()
    rank = group.rank()

    def unfreeze(_, m):
        if hasattr(m, "bits") and hasattr(m, "group_size"):
            m.unfreeze(keys=["scales", "biases"], recurse=False)

    q_model.apply_to_modules(unfreeze)
    print_trainable_parameters(q_model)

    def log_norm(x):
        x = x * (1 / temperature)
        return x - mx.logsumexp(x, axis=-1, keepdims=True)

    def loss_fn(params, x, targets, lengths):
        q_model.update(tree_map(lambda x: x.astype(dtype), params))
        logits = q_model(x).astype(mx.float32)
        losses = nn.losses.kl_div_loss(log_norm(logits), targets, reduction="none")
        mask = mx.arange(targets.shape[1]) < lengths[:, 1:]
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

    # Accumulate learned weights in higher precision
    params = tree_map(
        lambda x: x.astype(mx.float32),
        q_model.trainable_parameters(),
    )

    avg_loss = None
    tokens = 0
    tic = time.time()
    for it, (batch, lengths) in enumerate(
        iterate_batches(data, batch_size, max_seq_length)
    ):
        targets = log_norm(model(batch).astype(mx.float32))
        mx.eval(targets)
        loss, ntoks, params = step(batch, targets, lengths, params)
        mx.eval(loss, params)
        loss = mx.distributed.all_sum(loss, stream=mx.cpu).item() / world_size
        ntoks = mx.distributed.all_sum(ntoks, stream=mx.cpu).item()
        tokens += ntoks
        toks_per_sec = tokens / (time.time() - tic)
        avg_loss = 0.95 * (avg_loss or loss) + 0.05 * loss
        if rank == 0:
            print(
                f"{it=}, {loss=:.3f}, {avg_loss=:.4f}, {tokens=}, {toks_per_sec=:.3f}",
                flush=True,
            )
    q_model.update(tree_map(lambda x: x.astype(dtype), params))


def save_model(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    config,
    model_path: Path,
    mlx_path: str,
    hf_path: str,
):
    weights = dict(tree_flatten(model.parameters()))

    mlx_path = Path(mlx_path)
    save_weights(mlx_path, weights, donate_weights=True)

    py_files = glob.glob(str(model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, mlx_path)

    tokenizer.save_pretrained(mlx_path)

    save_config(config, config_path=mlx_path / "config.json")
    create_model_card(mlx_path, hf_path)


def load_data(tokenizer, data_path: str, num_samples: int):
    args = types.SimpleNamespace(
        hf_dataset={
            "path": data_path,
            "train_split": f"train",
            "valid_split": "train[:1]",
        },
        train=True,
        test=False,
    )
    dataset = load_dataset(args, tokenizer)[0]
    perm = np.random.permutation(len(dataset))[:num_samples].tolist()
    return [dataset.process(dataset[i]) for i in perm]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--quantized-model", default=None)
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
        default=1024,
        help="Number of samples to use for training.",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--data-path",
        type=str,
        default="allenai/tulu-3-sft-mixture",
        help="A Hugging Face dataset which is compatible with an mlx-lm dataset format.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature scaling for the loss.",
    )
    args = parser.parse_args()

    group = mx.distributed.init()

    num_samples = args.num_samples
    if num_samples % group.size() > 0:
        num_samples += group.size() - num_samples % group.size()

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    model_path = get_model_path(args.model, revision=None)
    model, config, tokenizer = fetch_from_hub(model_path, lazy=True)

    calibration_data = load_data(tokenizer, args.data_path, args.num_samples)

    if args.quantized_model is not None:
        q_model_path = get_model_path(args.quantized_model, revision=None)
        q_model, config, _ = fetch_from_hub(q_model_path, lazy=True)
    else:
        q_model = copy.deepcopy(model)
        _, config = quantize_model(
            q_model,
            config,
            q_group_size=args.group_size,
            q_bits=args.bits,
        )

    opt = optimizers.Adam(learning_rate=args.learning_rate, bias_correction=True)
    dwq_quantize(
        model,
        q_model,
        opt,
        calibration_data,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        temperature=args.temperature,
    )
    save_model(q_model, tokenizer, config, model_path, args.mlx_path, args.model)
