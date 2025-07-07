# Copyright © 2025 Apple Inc.

import argparse
import copy
import json
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_reduce, tree_unflatten
from tqdm import tqdm

from mlx_lm.quant.utils import load_data
from mlx_lm.tuner.losses import kl_div_loss
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import get_total_parameters
from mlx_lm.utils import (
    compute_bits_per_weight,
    fetch_from_hub,
    get_model_path,
    load,
    quantize_model,
    save,
)


def make_quant_predicate(config):
    def quant_predicate(p, m, _):
        if not hasattr(m, "to_quantized"):
            return False
        return config.get(p, True)

    return quant_predicate


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


def make_options(
    low_bits, low_group_size, high_bits, high_group_size, include_bpw=True
):
    options = []
    min_bpw = low_bits + 32 / low_group_size
    max_bpw = high_bits + 32 / high_group_size
    for b in range(low_bits, high_bits + 1):
        for g in [32, 64, 128]:
            cbpw = b + 32 / g
            if b == 7 or not (min_bpw <= cbpw <= max_bpw):
                continue
            options.append({"bits": b, "group_size": g, "bpw": cbpw})
    options.sort(key=lambda x: x["bpw"])
    if not include_bpw:
        for o in options:
            o.pop("bpw")

    return options


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

    options = make_options(low_bits, low_group_size, high_bits, high_group_size)
    current_bpw = options[0]["bpw"]

    def compute_sensitivity(gradient, low_q_weight, original_weight):
        n_batches = (len(data) + batch_size - 1) // batch_size
        gradient = gradient / n_batches
        scores = [{"loss_change": 0, "extra_bits": 0}]
        for opt in options[1:]:
            extra_bits = (opt["bpw"] - current_bpw) * original_weight.size
            other_weight = qdq(original_weight, opt["bits"], opt["group_size"])
            loss_change = (gradient * (low_q_weight - other_weight)).sum()
            scores.append({"loss_change": loss_change, "extra_bits": extra_bits})
        return scores

    sensitivities = tree_map(
        compute_sensitivity,
        grad_accum,
        q_model.parameters(),
        model.parameters(),
    )
    mx.eval(sensitivities)

    sensitivities = [
        (k.replace(".weight", ""), s.item() if isinstance(s, mx.array) else s)
        for k, s in tree_flatten(sensitivities)
    ]

    return sensitivities


def compute_bit_budget(model, target_bpw):
    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    model_params = get_total_parameters(model)

    return model_params * target_bpw - model_bytes * 8


def estimate_threshold(
    model,
    sensitivities,
    target_bpw,
    low_bits,
    low_group_size,
    high_bits,
    high_group_size,
):
    options = make_options(
        low_bits, low_group_size, high_bits, high_group_size, include_bpw=False
    )
    sensitivities = tree_flatten(
        tree_unflatten(list(sensitivities.items())),
        is_leaf=lambda x: isinstance(x, list) and "loss_change" in x[0],
    )

    q_model = copy.deepcopy(model)
    nn.quantize(q_model, group_size=low_group_size, bits=low_bits)
    budget = int(compute_bit_budget(q_model, target_bpw))
    benefit_map = {}

    def benefit(layer, option, budget):
        if (layer, option, budget) in benefit_map:
            return benefit_map[layer, option, budget]

        stack = [(layer, option, budget)]
        while stack:
            layer, option, budget = stack[-1]

            if budget <= 0 or layer < 0 or option < 0:
                benefit_map[layer, option, budget] = 0
                stack.pop()
                continue

            # We either not use this option
            prev_layer = layer if option > 0 else layer - 1
            prev_option = (option if option > 0 else len(options)) - 1
            if (prev_layer, prev_option, budget) not in benefit_map:
                stack.append((prev_layer, prev_option, budget))
                continue
            a = benefit_map[prev_layer, prev_option, budget]

            # Or we use it so we have less budget for before
            b = float("-inf")
            info = sensitivities[layer][1][option]
            prev_layer = layer - 1
            prev_option = len(options) - 1
            prev_budget = budget - info["extra_bits"]
            if (
                prev_layer,
                prev_option,
                prev_budget,
            ) not in benefit_map and prev_budget >= 0:
                stack.append((prev_layer, prev_option, prev_budget))
                continue
            if prev_budget >= 0:
                b = benefit_map[prev_layer, prev_option, prev_budget]
                b += info["loss_change"]

            benefit_map[layer, option, budget] = max(a, b)
            stack.pop()

        return benefit_map[layer, option, budget]

    def backtrack(layer, budget):
        selected = []
        while layer >= 0:
            prev_benefit = benefit(layer - 1, len(options) - 1, budget)
            option_benefits = [benefit(layer, i, budget) for i in range(len(options))]
            idx, v = max(enumerate(option_benefits), key=lambda x: x[1] - prev_benefit)
            info = sensitivities[layer][1][idx]
            if v != 0:
                budget -= info["extra_bits"]
                selected.append((layer, idx))
            layer -= 1
        return selected[::-1]

    selected = backtrack(len(sensitivities) - 1, budget)
    config = {sensitivities[l][0]: options[i] for l, i in selected}

    return config


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
    parser.add_argument("--low-group-size", type=int, default=128)
    parser.add_argument("--high-bits", type=int, default=5)
    parser.add_argument("--high-group-size", type=int, default=32)
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

    quant_config = estimate_threshold(
        model,
        sensitivities,
        target_bpw=args.target_bpw,
        low_bits=args.low_bits,
        low_group_size=args.low_group_size,
        high_bits=args.high_bits,
        high_group_size=args.high_group_size,
    )

    model, config = quantize_model(
        model,
        config,
        q_group_size=args.low_group_size,
        q_bits=args.low_bits,
        quant_predicate=make_quant_predicate(quant_config),
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
