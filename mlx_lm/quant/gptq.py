# Copyright © 2025 Apple Inc.

"""
Implements GPTQ

- https://arxiv.org/abs/2210.17323
- https://github.com/AutoGPTQ
"""

import argparse

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from tqdm import tqdm

from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
from mlx_lm.quant.utils import load_data
from mlx_lm.utils import (
    compute_bits_per_weight,
    load,
    save,
)


def quantize(w, bits, scales, biases):
    assert bits in {2, 4, 8}, f"Unsupported bits {bits}"
    el_per_int = 32 // bits
    n_bins = 2**bits - 1
    w = mx.unflatten(w, -1, (scales.shape[-1], -1))
    w = mx.clip(
        mx.round((w - biases[..., None]) / scales[..., None]), 0.0, n_bins
    ).astype(mx.uint32)
    shifts = mx.power(2, mx.arange(0, 32, bits, mx.uint32))
    w = mx.unflatten(w, -1, (-1, el_per_int))
    w = mx.sum(w * shifts, axis=-1)
    return w.flatten(-2, -1)


class Catcher(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.H = mx.array(0.0)

    def __call__(self, x, *args, **kwargs):
        xf = x.flatten(0, -2)
        self.H = self.H + xf.T @ xf
        return self.module(x, *args, **kwargs)


def gptq_quantize(
    model,
    data,
    bits,
    group_size,
    fallback_bits,
    fallback_group_size,
    batch_size=8,
):
    layers = []
    gptq_types = {nn.Linear, SwitchLinear}
    for k, l in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if type(l) in gptq_types:
            layers.append((k, Catcher(l)))
    model.update_modules(tree_unflatten(layers))

    # Evaluate the Hessians for all quantizable layers
    for e, s in tqdm(
        enumerate(range(0, len(data), batch_size)),
        total=len(data) // batch_size,
        desc="Computing Hessians",
    ):
        batch = data[s : s + batch_size]
        model(batch)
        mx.eval(layers)

    def compute_inverse_hessian(H):
        with mx.stream(mx.cpu):
            damp = 1e-2 * mx.mean(mx.diag(H))
            diag = mx.arange(H.shape[0])
            H[diag, diag] += damp
            H = mx.linalg.cholesky(H)
            H = mx.linalg.cholesky_inv(H)
            Hinv = mx.linalg.cholesky(H, upper=True)
            return Hinv

    @mx.compile
    def gptq_error(w, d, scales, biases):
        n_bins = 2**bits - 1
        q = mx.clip(mx.round((w - biases) / scales), 0.0, n_bins)
        q = scales * q + biases
        return (w - q) / d

    for lid, (key, l) in tqdm(
        enumerate(layers),
        total=len(layers),
        desc="Quantizing",
    ):
        Hinv = compute_inverse_hessian(l.H)
        del l.H
        mx.eval(Hinv)

        orig_type = l.module.weight.dtype
        W = l.module.weight.astype(mx.float32)

        all_scales = []
        all_biases = []
        for i in range(0, W.shape[-1], group_size):
            j = i + group_size
            Wl = W[..., i:j]
            err = mx.zeros_like(Wl)

            # Find scales and biases
            _, scales, biases = mx.quantize(Wl, bits=bits, group_size=group_size)

            all_scales.append(scales)
            all_biases.append(biases)
            for k in range(group_size):
                k += i
                w = W[..., k : k + 1]
                d = Hinv[k, k]

                e = gptq_error(w, d, scales, biases)

                W[..., k : k + j] -= e @ Hinv[k : k + 1, k : k + j]
                err[..., k : k + 1] = e
                mx.eval(err, W)

            W[..., j:] -= err @ Hinv[i:j, j:]

        # Quantize with the given scales and biases
        scales = mx.concatenate(all_scales, axis=-1)
        biases = mx.concatenate(all_biases, axis=-1)
        Wq = quantize(W, bits, scales, biases)
        layer = l.module.to_quantized(bits=bits, group_size=group_size)
        layer.weight = Wq
        layer.scales = scales
        layer.biases = biases
        layer.set_dtype(orig_type)
        mx.eval(layer)
        layers[lid] = (key, layer)

    model.update_modules(tree_unflatten(layers))

    layers = tree_flatten(
        model.leaf_modules(),
        is_leaf=nn.Module.is_module,
    )
    config = {"bits": bits, "group_size": group_size}
    fallback_config = {"bits": fallback_bits, "group_size": fallback_group_size}
    q_layers = []
    for e, (k, l) in enumerate(layers):
        if hasattr(l, "to_quantized"):
            config[k] = fallback_config
            q_layers.append((k, l.to_quantized(**fallback_config)))
    if len(q_layers) > 0:
        model.update_modules(tree_unflatten(q_layers))
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="Qwen/Qwen3-0.6B-base")
    parser.add_argument("--mlx-path", default="mlx_model")
    parser.add_argument(
        "--bits", type=int, default=4, help="Quantization bits for GPTQ layers"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size for GPTQ layers",
    )
    parser.add_argument(
        "--fallback-bits",
        type=int,
        default=6,
        help="Quantization bits for non-GPTQ layers",
    )
    parser.add_argument(
        "--fallback-group-size",
        type=int,
        default=64,
        help="Quantization group size for non-GPTQ layers",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples from the calibration dataset, use -1 for all.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512,
        help="Sequence length for the calibration data.",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    calibration_data = load_data(tokenizer, args.num_samples, args.sequence_length)

    model, config["quantization"] = gptq_quantize(
        model,
        calibration_data,
        args.bits,
        args.group_size,
        args.fallback_bits,
        args.fallback_group_size,
    )

    bpw = compute_bits_per_weight(model)
    print(f"Quantized model with {bpw:.3f} bits per weight.")

    save(
        args.mlx_path,
        args.model,
        model,
        tokenizer,
        config,
    )


if __name__ == "__main__":
    main()
