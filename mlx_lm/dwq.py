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
from mlx.utils import tree_flatten, tree_map

from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.trainer import iterate_batches
from mlx_lm.tuner.utils import print_trainable_parameters
from mlx_lm.utils import (
    fetch_from_hub,
    get_model_path,
    quantize_model,
    save,
)


class Catcher(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def __call__(self, *args, **kwargs):
        self.outputs = self.module(*args, **kwargs)
        return self.outputs


def dwq_quantize(
    model,
    q_model,
    opt,
    data,
    batch_size: int = 2,
    max_seq_length: int = 2048,
    temperature: float = 0.5,
    activation_layer_step: float = 0.25,
    activation_loss_weight: float = 1e-1,
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

    layer_id_step = int(activation_layer_step * len(model.layers))
    layer_ids = list(range(len(model.layers)))[layer_id_step::layer_id_step]

    for lid in layer_ids:
        model.layers[lid] = Catcher(model.layers[lid])
        q_model.layers[lid] = Catcher(q_model.layers[lid])

    def log_norm(x):
        if temperature != 1.0:
            x = x * (1 / temperature)
        return x - mx.logsumexp(x, axis=-1, keepdims=True)

    def forward(model, inputs):
        logprobs = log_norm(model(inputs).astype(mx.float32))
        extra_targets = [
            model.layers[lid].outputs.astype(mx.float32) for lid in layer_ids
        ]
        for lid in layer_ids:
            model.layers[lid].outputs = None
        return logprobs, extra_targets

    def loss_fn(params, x, targets, extra_targets, lengths):
        q_model.update(tree_map(lambda x: x.astype(dtype), params))
        logprobs, q_extra_targets = forward(q_model, x)
        losses = nn.losses.kl_div_loss(logprobs, targets, reduction="none")
        mask = mx.arange(targets.shape[1]) < lengths[:, 1:]
        ntoks = mask.sum()
        kl_loss = (mask * losses).sum() / ntoks
        act_loss = mx.stack(
            [
                (mask * (qe - e).abs().mean(axis=-1)).sum() / ntoks
                for qe, e in zip(q_extra_targets, extra_targets)
            ]
        )
        loss = kl_loss + activation_loss_weight * act_loss.mean()
        return loss, ntoks

    def step(inputs, targets, extra_targets, lengths, params):
        (loss, ntoks), grads = mx.value_and_grad(loss_fn)(
            params, inputs, targets, extra_targets, lengths
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
        batch = batch[:, :-1]
        targets, extra_targets = forward(model, batch)
        mx.eval(targets, extra_targets)
        loss, ntoks, params = step(batch, targets, extra_targets, lengths, params)
        mx.eval(loss, params)
        loss = mx.distributed.all_sum(loss, stream=mx.cpu).item() / world_size
        ntoks = mx.distributed.all_sum(ntoks, stream=mx.cpu).item()
        tokens += ntoks
        toks_per_sec = tokens / (time.time() - tic)
        avg_loss = 0.95 * (avg_loss or loss) + 0.05 * loss
        if rank == 0:
            peak_memory_gb = mx.get_peak_memory() / 1e9
            print(
                f"{it=}, {loss=:.3f}, {avg_loss=:.4f}, {tokens=},"
                f" {toks_per_sec=:.3f}, {peak_memory_gb=:.3f}",
                flush=True,
            )
    q_model.update(tree_map(lambda x: x.astype(dtype), params))
    for lid in layer_ids:
        q_model.layers[lid] = q_model.layers[lid].module


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
    parser.add_argument("--model", "-m", default="Qwen/Qwen3-4B")
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
    parser.add_argument("--max-seq-length", type=int, default=2049)
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
    save(
        args.mlx_path,
        model_path,
        dict(tree_flatten(q_model.parameters())),
        tokenizer,
        config,
        hf_repo=args.model,
    )
