# Copyright © 2025 Apple Inc.

import argparse
import copy
import glob
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
from datasets import load_dataset
from mlx.utils import tree_flatten, tree_map

from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tuner.utils import print_trainable_parameters
from mlx_lm.utils import (
    create_model_card,
    fetch_from_hub,
    get_model_path,
    quantize_model,
    save_config,
    save_weights,
)


def dist_split(x: mx.array, group: mx.distributed.Group):
    B = x.shape[0]
    N = group.size()
    assert B % N == 0
    r = group.rank()
    local_B = (B + N - 1) // N
    return x[r * local_B : (r + 1) * local_B]


def dwq_quantize(
    model,
    q_model,
    opt,
    inputs: mx.array,
    group_size: int = 64,
    bits: int = 3,
    batch_size: int = 2,
    dtype: mx.Dtype = mx.bfloat16,
):
    group = mx.distributed.init()
    q_model.freeze()
    q_model.unfreeze(keys=["scales", "biases"])
    print_trainable_parameters(q_model)

    def log_norm(x):
        return x - mx.logsumexp(x, axis=-1, keepdims=True)

    def loss_fn(params, x, targets):
        q_model.update(tree_map(lambda x: x.astype(dtype), params))
        logits = q_model(x).astype(mx.float32)
        return nn.losses.kl_div_loss(log_norm(logits), targets, reduction="mean")

    # TODO distributed support
    def step(inputs, targets, params):
        loss, grads = mx.value_and_grad(loss_fn)(params, inputs, targets)
        params = opt.apply_gradients(grads, params)
        return loss, params

    # Accumulate learned weights in higher precision
    params = tree_map(
        lambda x: x.astype(mx.float32),
        q_model.trainable_parameters(),
    )

    losses = []
    it = 1
    for i in range(0, inputs.shape[0], batch_size):
        batch = inputs[i : i + batch_size]
        targets = log_norm(model(batch).astype(mx.float32))
        mx.eval(targets)
        loss, params = step(batch, targets, params)
        mx.eval(loss, params)
        losses.append(loss.item())
        print(f"Iter: {it}, Loss: {loss.item():.3f}")
        if (it + 1) % 10 == 0:
            loss_avg = sum(losses) / len(losses)
            print(f"Average Loss {loss_avg:.3f}")
            losses = []
        it += 1
    q_model.update(tree_map(lambda x: x.astype(dtype), params))


def load_wikitext(
    tokenizer, num_samples: int = 32, sequence_length: int = 2048, split: str = "train"
) -> mx.array:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    texts = "\n\n".join(dataset["text"])
    tokens = tokenizer.encode(texts, return_tensors="mlx")[0]

    # Select random chunks
    starts = mx.random.randint(
        0, len(tokens) - sequence_length - 1, shape=(num_samples, 1)
    )
    return tokens[starts + mx.arange(sequence_length)]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", default="mlx-community/Qwen2.5-7B-Instruct-bf16"
    )
    parser.add_argument("--mlx-path", default="mlx_model")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=float, default=8)
    args = parser.parse_args()

    group = mx.distributed.init()

    num_samples = args.num_samples
    if group is not None and num_samples % group.size() > 0:
        num_samples += group.size() - num_samples % group.size()

    mx.random.seed(args.seed)

    model_path = get_model_path(args.model, revision=None)
    model, config, tokenizer = fetch_from_hub(model_path, lazy=True)

    calibration_data = load_wikitext(tokenizer, args.num_samples, args.sequence_length)

    if group is not None:
        calibration_data = dist_split(calibration_data, group)

    q_model = copy.deepcopy(model)
    tree_flatten(q_model.parameters())
    _, config = quantize_model(
        q_model,
        config,
        q_group_size=args.group_size,
        q_bits=args.bits,
    )

    opt = optimizers.SGD(learning_rate=args.learning_rate)
    dwq_quantize(
        model,
        q_model,
        opt,
        calibration_data,
        bits=args.bits,
        group_size=args.group_size,
        batch_size=args.batch_size,
    )
    save_model(q_model, tokenizer, config, model_path, args.mlx_path, args.model)


if __name__ == "__main__":
    main()
