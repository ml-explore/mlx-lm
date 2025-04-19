# Learned quantization using AWQ

# References:
# https://arxiv.org/abs/2306.00978
# https://github.com/mit-han-lab/llm-awq

import argparse
import glob
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from datasets import load_dataset
from mlx.utils import tree_flatten, tree_map, tree_map_with_path
from tqdm import tqdm

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.switch_layers import SwitchLinear
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import (
    fetch_from_hub,
    get_model_path,
    save_config,
    save_weights,
)

SUPPORTED_MODEL_TYPES = {
    "llama",
    "llama4",
    "mistral",
    "qwen2",
    "gemma3",
    "gemma3_text",
    "gemma2",
}


def mse(x, y):
    return ((x - y).astype(mx.float32)) ** 2


def run_layer(
    layer: nn.Module,
    x: mx.array,
    indices: mx.array | None = None,
    batch_size: int = 32,
    **kwargs,
):
    y = []
    for i in range(0, x.shape[0], batch_size):
        if indices is not None:
            y.append(
                layer(x[i : i + batch_size], indices[i : i + batch_size], **kwargs)
            )
        else:
            y.append(layer(x[i : i + batch_size], **kwargs))
        mx.eval(y)
    y = mx.concatenate(y, axis=0)
    return y


def dist_split(x: mx.array, group: mx.distributed.Group):
    B = x.shape[0]
    N = group.size()
    assert B % N == 0
    r = group.rank()
    local_B = (B + N - 1) // N
    return x[r * local_B : (r + 1) * local_B]


def search_best_scale(
    layers: list[nn.Module],
    quantize_func: Callable,
    block: nn.Module | None = None,
    layer_kwargs: dict | None = None,
    n_grid: int = 20,
):
    group = mx.distributed.init()

    layer_kwargs = layer_kwargs or {}

    x = layers[0].input_feat
    if (indices := layers[0].indices) is not None:
        layer_kwargs["indices"] = indices

    block = block or layers[0]
    out = block(x, **layer_kwargs)

    x_max = x.abs().mean(axis=(0, 1))

    best_error = float("inf")
    best_scales = None

    weights = tree_flatten(block.parameters())

    # Search across different scaling ratios
    # and take the best loss.
    for ratio in range(n_grid):
        ratio = ratio / n_grid
        scales = mx.maximum(x_max**ratio, 1e-4).reshape(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        for layer in layers:
            layer.weight = quantize_func(layer.weight * scales) / scales

        out_q = run_layer(block, x, **layer_kwargs)
        loss = mse(out, out_q).sum()
        if group is not None:
            loss = mx.distributed.all_sum(loss) / group.size()
        loss /= out.size
        mx.eval(loss)
        if loss.item() < best_error:
            best_error = loss.item()
            best_scales = scales

        # reload the original weights
        block.load_weights(weights)

    best_scales = best_scales.reshape(-1)
    mx.eval(best_scales)
    return best_scales


def apply_scale(prev_op, layers, scales):
    # Fuse the scales into the previous op
    if isinstance(prev_op, (nn.Linear, SwitchLinear)):
        assert len(layers) == 1
        prev_op.weight = prev_op.weight / scales[:, mx.newaxis]
        if hasattr(prev_op, "bias"):
            prev_op.bias = prev_op.bias / scales
        layers[0].weight = layers[0].weight * scales[mx.newaxis]
    elif isinstance(prev_op, (nn.LayerNorm, nn.RMSNorm)):
        prev_op.weight = prev_op.weight / scales
        if hasattr(prev_op, "bias"):
            prev_op.bias = prev_op.bias / scales
        for layer in layers:
            layer.weight = layer.weight * scales
    elif prev_op.__class__.__name__ == "RMSNorm":  # For gemma models
        prev_op.weight = (1.0 + prev_op.weight) / scales - 1.0
        for layer in layers:
            layer.weight = layer.weight * scales
    else:
        raise NotImplementedError(f"Could not apply scale to prev_op: {prev_op}")

    for layer in layers:
        layer.input_feat = layer.input_feat / scales


@dataclass
class LayerConfig:
    prev_op: nn.Module
    layers: list[nn.Module]
    block: nn.Module | None = None
    layer_kwargs: dict | None = None


def scale_block(
    block: nn.Module, mask: mx.array, quantize_func: Callable, n_grid: int = 20
):
    # Layers which have the same inputs (within an elementwise multiplication)
    # can be scaled together.
    mlp_norm = getattr(
        block, "pre_feedforward_layernorm", block.post_attention_layernorm
    )
    config = [
        LayerConfig(
            block=block.self_attn,
            prev_op=block.input_layernorm,
            layers=[
                block.self_attn.q_proj,
                block.self_attn.k_proj,
                block.self_attn.v_proj,
            ],
            layer_kwargs={"mask": mask},
        )
    ]
    # Llama 4 MOE layer
    if hasattr(block, "feed_forward") and block.is_moe_layer:
        config += [
            LayerConfig(
                block=block.feed_forward,
                prev_op=block.post_attention_layernorm,
                layers=[
                    block.feed_forward.shared_expert.gate_proj,
                    block.feed_forward.shared_expert.up_proj,
                    block.feed_forward.experts.gate_proj,
                    block.feed_forward.experts.up_proj,
                    block.feed_forward.router,
                ],
            ),
            LayerConfig(
                prev_op=block.feed_forward.experts.up_proj,
                layers=[
                    block.feed_forward.experts.down_proj,
                ],
            ),
            LayerConfig(
                prev_op=block.feed_forward.shared_expert.up_proj,
                layers=[
                    block.feed_forward.shared_expert.down_proj,
                ],
            ),
        ]
    else:
        config += [
            LayerConfig(
                block=block.mlp,
                prev_op=mlp_norm,
                layers=[
                    block.mlp.gate_proj,
                    block.mlp.up_proj,
                ],
            ),
            LayerConfig(
                prev_op=block.mlp.up_proj,
                layers=[
                    block.mlp.down_proj,
                ],
            ),
        ]
    for conf in config:
        scales = search_best_scale(
            layers=conf.layers,
            block=conf.block,
            layer_kwargs=conf.layer_kwargs,
            quantize_func=quantize_func,
            n_grid=n_grid,
        )
        apply_scale(conf.prev_op, conf.layers, scales)


def search_best_clip(
    module: nn.Module,
    quantize_func: Callable,
    group_size: int,
    n_grid: int = 20,
    max_shrink: float = 0.5,
    subsample: int = 4,
    batch_size: int = 64,
):
    group = mx.distributed.init()

    x = module.input_feat
    w = module.weight

    x = x[:, ::subsample]
    x = x.reshape(*x.shape[:-1], -1, group_size)

    w_all = w
    w_max_all = []
    w_min_all = []

    # batch across W to save memory
    for b in range(0, w.shape[0], batch_size):
        w = w_all[b : b + batch_size]

        group_shape = (w.shape[0], w.shape[-1] // group_size)
        best_error = mx.full(group_shape, float("inf"))
        best_w_max = mx.zeros((*group_shape, 1), dtype=x.dtype)
        best_w_min = mx.zeros((*group_shape, 1), dtype=x.dtype)

        w_shape = w.shape

        w = w.reshape(*w.shape[:-1], -1, group_size)
        out = mx.einsum("btdg,odg->btod", x, w)

        # try a range of clips and pick the one with the smallest loss
        for i in range(int(max_shrink * n_grid)):
            p = 1 - i / n_grid
            w_max = p * w.max(axis=-1, keepdims=True)
            w_min = p * w.min(axis=-1, keepdims=True)
            w_m = mx.clip(w, w_min, w_max).reshape(w_shape)

            w_q = quantize_func(w_m)

            w_q = w_q.reshape(*w_q.shape[:-1], -1, group_size)
            out_q = mx.einsum("btdg,odg->btod", x, w_q)

            # Take the mean across the input batch
            loss = mse(out, out_q).sum(axis=(0, 1))
            if group is not None:
                loss = mx.distributed.all_sum(loss) / group.size()
            loss /= out.shape[0] * out.shape[1]
            best_indices = loss < best_error
            best_error = mx.where(best_indices, loss, best_error)
            best_w_max = mx.where(best_indices[..., mx.newaxis], w_max, best_w_max)
            best_w_min = mx.where(best_indices[..., mx.newaxis], w_min, best_w_min)
            mx.eval(best_w_max, best_w_min, best_error)

        w_max_all.append(best_w_max)
        w_min_all.append(best_w_min)

    best_w_max = mx.concatenate(w_max_all, axis=0)
    best_w_min = mx.concatenate(w_min_all, axis=0)

    w_r = w_all.reshape(*w_all.shape[:-1], -1, group_size)
    best_w = mx.clip(w_r, best_w_min, best_w_max)
    best_w = best_w.reshape(w_all.shape)

    mx.eval(best_w)
    return best_w


def clip_block(
    block: nn.Module, quantize_func: Callable, group_size: int, n_grid: int = 20
):

    def apply_clip(path, module):
        if (
            isinstance(module, nn.Linear)
            and "q_proj" not in path
            and "k_proj" not in path
        ):
            best_weight = search_best_clip(
                module,
                quantize_func=quantize_func,
                group_size=group_size,
                n_grid=n_grid,
            )
            module.weight = best_weight

    tree_map_with_path(apply_clip, block.leaf_modules(), is_leaf=nn.Module.is_module)


def awq_quantize(
    model,
    inputs: mx.array,
    group_size: int = 64,
    bits: int = 3,
    embed_group_size: int = 32,
    embed_bits: int = 4,
    n_grid: int = 20,
):
    group = mx.distributed.init()

    def quantize_func(w):
        wq = mx.quantize(w, bits=bits, group_size=group_size)
        return mx.dequantize(*wq, bits=bits, group_size=group_size)

    mask = create_attention_mask(inputs)

    model.model.embed_tokens = model.model.embed_tokens.to_quantized(
        group_size=embed_group_size, bits=embed_bits
    )
    inputs = model.model.embed_tokens(inputs)

    def capture(module):
        if not isinstance(module, (nn.Linear, SwitchLinear)):
            return module

        class Catcher(nn.Module):
            def __call__(self, x: mx.array, *args, **kwargs):
                # Store the input features on the original modules.
                if hasattr(module, "input_feat"):
                    module.input_feat = mx.concatenate([module.input_feat, x], axis=0)
                else:
                    module.input_feat = x

                # Also store the MOE indices if applicabale
                module.indices = None
                if isinstance(module, SwitchLinear):
                    indices = args[0]
                    if module.indices is not None:
                        module.indices = mx.concatenate(
                            [module.indices, indices], axis=0
                        )
                    else:
                        module.indices = indices

                return module(x, *args, **kwargs)

        return Catcher()

    for block in tqdm(model.layers):
        # Capture the input features for each of the layers in the transformer block
        orig_leaves = block.leaf_modules()
        capture_leaves = tree_map(capture, orig_leaves, is_leaf=nn.Module.is_module)
        block.update_modules(capture_leaves)
        outputs = run_layer(block, inputs, mask=mask)
        block.update_modules(orig_leaves)
        del capture_leaves

        # Quantize the block without AWQ to obtain a reference loss
        nn.quantize(block, group_size=group_size, bits=bits)
        outputs_q = run_layer(block, inputs, mask=mask)
        before_loss = mse(outputs, outputs_q).sum()
        if group is not None:
            before_loss = mx.distributed.all_sum(before_loss) / group.size()
        before_loss /= outputs.size
        block.update_modules(orig_leaves)
        del orig_leaves

        scale_block(
            block=block,
            mask=mask,
            quantize_func=quantize_func,
            n_grid=n_grid,
        )

        clip_block(
            block=block,
            quantize_func=quantize_func,
            group_size=group_size,
            n_grid=n_grid,
        )

        # Quantize the scaled and clipped block
        nn.quantize(block, group_size=group_size, bits=bits)
        outputs_q = run_layer(block, inputs, mask=mask)
        after_loss = mse(outputs, outputs_q).sum()
        if group is not None:
            after_loss = mx.distributed.all_sum(after_loss) / group.size()
        after_loss /= outputs.size
        tqdm.write(f"Loss reduction: {after_loss / before_loss}")

        inputs = outputs

        mx.eval(block)
        mx.clear_cache()

    if hasattr(model, "lm_head"):
        model.lm_head = model.lm_head.to_quantized(
            group_size=embed_group_size, bits=embed_bits
        )


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
    data = tokens[starts + mx.arange(sequence_length)]
    if tokenizer.bos_token_id:
        data = mx.concatenate(
            [mx.full((*data.shape[:1], 1), tokenizer.bos_token_id), data], axis=-1
        )
    return data


def save_model(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    config,
    model_path: Path,
    mlx_path: str,
):
    weights = dict(tree_flatten(model.parameters()))

    mlx_path = Path(mlx_path)
    save_weights(mlx_path, weights, donate_weights=True)

    py_files = glob.glob(str(model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, mlx_path)

    tokenizer.save_pretrained(mlx_path)

    # dummy
    config["quantization"] = {"group_size": 64, "bits": 4}

    def update_config(path, module):
        if hasattr(module, "bits"):
            config["quantization"][path] = {
                "group_size": module.group_size,
                "bits": module.bits,
            }
        else:
            config["quantization"][path] = False

    tree_map_with_path(update_config, model.leaf_modules(), is_leaf=nn.Module.is_module)

    save_config(config, config_path=mlx_path / "config.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", default="mlx-community/Qwen2.5-7B-Instruct-bf16"
    )
    parser.add_argument("--mlx-path", default="mlx_model")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--embed-bits", type=int, default=4)
    parser.add_argument("--embed-group-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--n-grid", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    group = mx.distributed.init()

    num_samples = args.num_samples
    if group is not None and num_samples % group.size() > 0:
        num_samples += group.size() - num_samples % group.size()

    mx.random.seed(args.seed)

    model_path = get_model_path(args.model, revision=None)
    model, config, tokenizer = fetch_from_hub(model_path, lazy=True)

    model_type = config["model_type"]
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise NotImplementedError(f"AWQ support for {config['model_type']} models NYI.")

    calibration_data = load_wikitext(tokenizer, args.num_samples, args.sequence_length)

    if group is not None:
        calibration_data = dist_split(calibration_data, group)

    # For Gemma3 vision/text models
    language_model = getattr(model, "language_model", model)
    awq_quantize(
        language_model,
        calibration_data,
        bits=args.bits,
        group_size=args.group_size,
        embed_bits=args.embed_bits,
        embed_group_size=args.embed_group_size,
        n_grid=args.n_grid,
    )

    save_model(model, tokenizer, config, model_path, args.mlx_path)


if __name__ == "__main__":
    main()
