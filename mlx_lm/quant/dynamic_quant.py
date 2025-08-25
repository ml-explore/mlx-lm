# Copyright © 2025 Apple Inc.

import argparse
import copy
import json
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from tqdm import tqdm

from mlx_lm.quant.utils import load_data
from mlx_lm.tuner.losses import kl_div_loss
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.utils import (
    compute_bits_per_weight,
    fetch_from_hub,
    get_model_path,
    load,
    quantize_model,
    save,
)


def eval_ppl(model, data, batch_size=8):
    all_loss = 0.0
    ntoks = 0
    for s in range(0, len(data), batch_size):
        batch = data[s : s + batch_size]
        logits = model(batch[:, :-1]).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, batch[:, 1:])
        all_loss += losses.sum().item()
        ntoks += losses.size
    ppl = math.exp(all_loss / ntoks)
    return ppl


def estimate_sensitivities(
    model,
    data,
    low_bits,
    low_group_size,
    high_bits,
    high_group_size,
    batch_size: int = 4,
    gradient_accum_dtype: mx.Dtype = mx.float32,
    gradient_checkpoint: bool = False,
):
    def qdq(w, bits, group_size):
        w, s, b = mx.quantize(w, bits=bits, group_size=group_size)
        return mx.dequantize(w, scales=s, biases=b, bits=bits, group_size=group_size)

    layers = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    layers = {k: l for k, l in layers if hasattr(l, "to_quantized")}
    q_model = copy.deepcopy(model)
    q_layers = copy.deepcopy(layers)
    for l in q_layers.values():
        l.weight = qdq(l.weight, low_bits, low_group_size)
        # Freeze everything but the quantizable weight
        l.freeze()
        l.unfreeze(keys=["weight"])
    q_model.freeze()
    q_model.update_modules(tree_unflatten(list(q_layers.items())))

    def loss_fn(batch, targets):
        return kl_div_loss(q_model(batch), targets).mean()

    if gradient_checkpoint:
        grad_checkpoint(q_model.layers[0])

    grad_accum = tree_map(
        lambda x: mx.zeros(x.shape, dtype=gradient_accum_dtype),
        q_model.trainable_parameters(),
    )
    for e, s in tqdm(
        enumerate(range(0, len(data), batch_size)),
        total=len(data) // batch_size,
        desc="Estimating sensitivities",
    ):
        batch = data[s : s + batch_size]
        targets = model(batch)
        mx.eval(targets)
        _, grads = nn.value_and_grad(q_model, loss_fn)(batch, targets)
        grad_accum = tree_map(lambda x, y: x + y, grad_accum, grads)
        del grads
        mx.eval(grad_accum)

    def compute_sensitivity(gradient, low_q_weight, original_weight):
        n_batches = (len(data) + batch_size - 1) // batch_size
        gradient = gradient / n_batches
        high_q_weight = qdq(original_weight, high_bits, high_group_size)
        param_size = original_weight.size / 1e6
        alignment = (gradient * (low_q_weight - high_q_weight)).sum()
        return alignment / param_size

    sensitivities = tree_map(
        compute_sensitivity,
        grad_accum,
        q_model.parameters(),
        model.parameters(),
    )
    mx.eval(sensitivities)

    sensitivities = [(k[:-7], s.item()) for k, s in tree_flatten(sensitivities)]

    return sensitivities


def estimate_threshold(
    model,
    sensitivities,
    target_bpw,
    low_bits,
    low_group_size,
    high_bits,
    high_group_size,
):
    def predicate(p, m, high_threshold):
        if not hasattr(m, "to_quantized"):
            return False
        if sensitivities[p] > high_threshold:
            return {"bits": high_bits, "group_size": high_group_size}
        return True

    # Binary search for the threshold
    sens_vals = list(sensitivities.values())
    min_threshold = min(sens_vals)
    max_threshold = max(sens_vals)
    tolerance = 1e-3 * (max_threshold - min_threshold)
    while (max_threshold - min_threshold) > tolerance:
        mid = (max_threshold + min_threshold) / 2
        class_predicate = lambda p, m: predicate(p, m, mid)
        q_model = copy.deepcopy(model)
        nn.quantize(
            q_model,
            group_size=low_group_size,
            bits=low_bits,
            class_predicate=class_predicate,
        )
        bpw = compute_bits_per_weight(q_model)
        if bpw > target_bpw:
            min_threshold = mid
        else:
            max_threshold = mid

    return (max_threshold + min_threshold) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="Qwen/Qwen3-0.6B-base")
    parser.add_argument(
        "--mlx-path", default="mlx_model", help="Path to save the model"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--sensitivities",
        type=str,
        default=None,
        help="Path to a pre-computed sensitivity JSON file.",
    )
    parser.add_argument(
        "--target-bpw", type=float, default=5.0, help="Target bits per weight."
    )
    parser.add_argument("--low-bits", type=int, default=4)
    parser.add_argument("--low-group-size", type=int, default=64)
    parser.add_argument("--high-bits", type=int, default=5)
    parser.add_argument("--high-group-size", type=int, default=64)
    parser.add_argument(
        "--report-ppl",
        action="store_true",
        help="Compute the perplexity of the base and quantized models.",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Use gradient checkpointing to reduce memory use.",
    )
    parser.add_argument(
        "--accumulation-dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="What type to use to accumulate the gradients for the sensitivities",
    )
    args = parser.parse_args()

    group = mx.distributed.init()

    if args.sensitivities is None:
        model, tokenizer = load(args.model)
        mx.random.seed(args.seed)
        data = load_data(tokenizer, num_samples=-1, sequence_length=512)

        sensitivities = estimate_sensitivities(
            model,
            data,
            args.low_bits,
            args.low_group_size,
            args.high_bits,
            args.high_group_size,
            gradient_accum_dtype=getattr(mx, args.accumulation_dtype),
            gradient_checkpoint=args.grad_checkpoint,
        )
        model_name = args.model.replace("/", "_")
        with open(f"{model_name}_sensitivities.json", "w") as fid:
            json.dump(sensitivities, fid)
    else:
        with open(args.sensitivities, "r") as fid:
            sensitivities = json.load(fid)

    sensitivities = dict(sensitivities)
    model_path, hf_repo = get_model_path(args.model, revision=None)
    model, config, tokenizer = fetch_from_hub(model_path, lazy=True)
    mx.random.seed(args.seed)
    data = load_data(tokenizer, num_samples=-1, sequence_length=512)

    if args.report_ppl:
        ppl = eval_ppl(model, data)
        print(f"Original PPL: {ppl:.3f}")

    threshold = estimate_threshold(
        model,
        sensitivities,
        target_bpw=args.target_bpw,
        low_bits=args.low_bits,
        low_group_size=args.low_group_size,
        high_bits=args.high_bits,
        high_group_size=args.high_group_size,
    )

    def quant_predicate(p, m, _):
        if not hasattr(m, "to_quantized"):
            return False
        if sensitivities[p] > threshold:
            return {"bits": args.high_bits, "group_size": args.high_group_size}
        return True

    model, config = quantize_model(
        model,
        config,
        group_size=args.low_group_size,
        bits=args.low_bits,
        quant_predicate=quant_predicate,
    )

    if args.report_ppl:
        ppl = eval_ppl(model, data)
        print(f"Quantized PPL: {ppl:.3f}")

    save(
        args.mlx_path,
        model_path,
        model,
        tokenizer,
        config,
        hf_repo=hf_repo,
    )
    print(f"Peak memory used: {mx.get_peak_memory() / 1000**3:.3f}GB")


if __name__ == "__main__":
    main()
