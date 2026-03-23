# Copyright © 2025 Apple Inc.

import argparse
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict
from urllib import request

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_map_with_path
from tqdm import tqdm

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.switch_layers import SwitchLinear
from mlx_lm.quant.utils import load_data
from mlx_lm.utils import (
    load,
    save,
)


@dataclass
class ScaleConfig:
    prev: nn.Module
    layers: list[nn.Module]
    block: nn.Module | None = None
    kwargs: list = field(default_factory=list)
    use_config: Callable[[nn.Module], bool] | None = None


@dataclass
class AWQConfig:
    embed: str
    lm_head: str
    no_clip: list[str]
    scale_configs: list[ScaleConfig]
    lm_key: str | None = None


def update(cfg, **kwargs):
    cfg = copy.deepcopy(cfg)
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


llama_awq = AWQConfig(
    embed="embed_tokens",
    lm_head="lm_head",
    no_clip=["q_proj", "k_proj"],
    scale_configs=[
        ScaleConfig(
            block="self_attn",
            prev="input_layernorm",
            layers=["q_proj", "k_proj", "v_proj"],
            kwargs=["mask"],
        ),
        ScaleConfig(prev="mlp.up_proj", layers=["mlp.down_proj"]),
        ScaleConfig(
            block="mlp",
            prev="post_attention_layernorm",
            layers=["gate_proj", "up_proj"],
        ),
    ],
)

gemma3_text_awq = AWQConfig(
    embed="embed_tokens",
    lm_head="lm_head",
    no_clip=["q_proj", "k_proj"],
    scale_configs=[
        ScaleConfig(
            block="self_attn",
            prev="input_layernorm",
            layers=["q_proj", "k_proj", "v_proj"],
            kwargs=["mask"],
        ),
        ScaleConfig(prev="mlp.up_proj", layers=["mlp.down_proj"]),
        ScaleConfig(
            block="mlp",
            prev="pre_feedforward_layernorm",
            layers=["gate_proj", "up_proj"],
        ),
    ],
)

gemma3_awq = update(gemma3_text_awq, lm_key="language_model")

deepseek_v2_awq = AWQConfig(
    embed="embed_tokens",
    lm_head="lm_head",
    no_clip=["q_proj", "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"],
    scale_configs=[
        ScaleConfig(
            block="self_attn",
            prev="input_layernorm",
            layers=["q_proj", "kv_a_proj_with_mqa"],
            kwargs=["mask"],
        ),
        ScaleConfig(
            prev="self_attn.kv_a_layernorm",
            layers=["self_attn.kv_b_proj"],
        ),
        ScaleConfig(
            prev="mlp.up_proj",
            layers=["mlp.down_proj"],
            use_config=lambda block: not "switch_mlp" in block.mlp,
        ),
        ScaleConfig(
            prev="mlp.shared_experts.up_proj",
            layers=["mlp.shared_experts.down_proj"],
            use_config=lambda block: "switch_mlp" in block.mlp,
        ),
        ScaleConfig(
            prev="mlp.switch_mlp.up_proj",
            layers=["mlp.switch_mlp.down_proj"],
            use_config=lambda block: "switch_mlp" in block.mlp,
            kwargs=["indices"],
        ),
        ScaleConfig(
            block="mlp",
            prev="post_attention_layernorm",
            layers=["gate_proj", "up_proj"],
            use_config=lambda block: not "switch_mlp" in block.mlp,
        ),
        ScaleConfig(
            block="mlp",
            prev="post_attention_layernorm",
            layers=[
                "switch_mlp.gate_proj",
                "switch_mlp.up_proj",
                "shared_experts.gate_proj",
                "shared_experts.up_proj",
                "gate",  # not quantized, just scaled
            ],
            use_config=lambda block: "switch_mlp" in block.mlp,
        ),
    ],
)

AWQ_MODEL_CONFIGS = {
    "llama": llama_awq,
    "mistral": llama_awq,
    "qwen2": llama_awq,
    "qwen3": llama_awq,
    "gemma3_text": gemma3_text_awq,
    "gemma3": update(gemma3_text_awq, lm_key="language_model"),
    "deepseek_v2": deepseek_v2_awq,
}


def mse(x, y):
    return ((x - y).astype(mx.float32)) ** 2


def submodule_from_key(module, key):
    keys = key.split(".")
    for k in keys:
        module = module[k]
    return module


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
    N = group.size()
    if N == 1:
        return x
    B = x.shape[0]
    assert B % N == 0
    r = group.rank()
    local_B = (B + N - 1) // N
    return x[r * local_B : (r + 1) * local_B]


def search_best_scale(
    layers: list[nn.Module],
    quantize_func: Callable,
    block: nn.Module | None,
    layer_kwargs: dict,
    n_grid: int,
):
    group = mx.distributed.init()

    layer_kwargs = layer_kwargs or {}

    x = layers[0].input_feat

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
            if isinstance(layer, (nn.Linear, SwitchLinear)):
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
        layers[0].weight = layers[0].weight * scales
    elif isinstance(prev_op, (nn.LayerNorm, nn.RMSNorm)):
        prev_op.weight = prev_op.weight / scales
        if hasattr(prev_op, "bias"):
            prev_op.bias = prev_op.bias / scales
        for layer in layers:
            layer.weight = layer.weight * scales
    elif prev_op.__class__.__name__ == "RMSNorm":  # For gemma models
        dt = prev_op.weight.dtype
        prev_op.weight = (
            (1.0 + prev_op.weight.astype(mx.float32)) / scales - 1.0
        ).astype(dt)
        for layer in layers:
            layer.weight = layer.weight * scales
    else:
        raise NotImplementedError(f"Could not apply scale to prev_op: {prev_op}")

    for layer in layers:
        if hasattr(layer, "input_feat"):
            layer.input_feat = layer.input_feat / scales


def scale_block(
    block: nn.Module,
    configs: list[ScaleConfig],
    quantize_func: Callable,
    layer_kwargs: dict,
    n_grid: int,
):
    for conf in configs:
        if conf.use_config is not None and not conf.use_config(block):
            continue
        if conf.block is not None:
            local_block = block[conf.block]
            layers = [submodule_from_key(local_block, l) for l in conf.layers]
        else:
            local_block = None
            layers = [submodule_from_key(block, l) for l in conf.layers]
        local_kwargs = {k: layer_kwargs[k] for k in conf.kwargs if k in layer_kwargs}
        for k in conf.kwargs:
            if hasattr(layers[0], k):
                local_kwargs[k] = getattr(layers[0], k)

        scales = search_best_scale(
            layers=layers,
            block=local_block,
            layer_kwargs=local_kwargs,
            quantize_func=quantize_func,
            n_grid=n_grid,
        )
        apply_scale(submodule_from_key(block, conf.prev), layers, scales)


def search_best_clip(
    module: nn.Module,
    quantize_func: Callable,
    group_size: int,
    n_grid: int,
    max_shrink: float = 0.5,
    batch_size: int = 64,
    n_frames: int = 512,
):
    group = mx.distributed.init()

    # subsample the input features
    x = module.input_feat.flatten(0, 1)
    stride = (x.shape[0] + n_frames - 1) // n_frames
    x = x[::stride]

    w = module.weight
    x = x.reshape(x.shape[0], -1, group_size)

    w_init_shape = w.shape
    w_all = mx.flatten(w, 0, w.ndim - 2)
    w_max_all = []

    # batch across W to save memory
    for b in range(0, w_all.shape[0], batch_size):
        w = w_all[b : b + batch_size]

        group_shape = (w.shape[0], w.shape[-1] // group_size)
        best_error = mx.full(group_shape, float("inf"))
        best_w_max = mx.zeros((*group_shape, 1), dtype=x.dtype)

        w_shape = w.shape

        w = w.reshape(*w.shape[:-1], -1, group_size)
        out = mx.einsum("bdg,odg->bod", x, w)
        init_max = w.abs().max(axis=-1, keepdims=True)

        # try a range of clips and pick the one with the smallest loss
        for i in range(int(max_shrink * n_grid)):
            p = 1 - i / n_grid
            w_max = p * init_max
            w_m = mx.clip(w, -w_max, w_max).reshape(w_shape)

            w_q = quantize_func(w_m)

            w_q = w_q.reshape(*w_q.shape[:-1], -1, group_size)
            out_q = mx.einsum("bdg,odg->bod", x, w_q)

            # Take the mean across the input batch
            loss = mse(out, out_q).sum(axis=0)
            if group is not None:
                loss = mx.distributed.all_sum(loss) / group.size()
            loss /= out.shape[0]
            best_indices = loss < best_error
            best_error = mx.where(best_indices, loss, best_error)
            best_w_max = mx.where(best_indices[..., mx.newaxis], w_max, best_w_max)
            mx.eval(best_w_max, best_error)

        w_max_all.append(best_w_max)

    best_w_max = mx.concatenate(w_max_all, axis=0)

    w_r = w_all.reshape(*w_all.shape[:-1], -1, group_size)
    best_w = mx.clip(w_r, -best_w_max, best_w_max)
    best_w = best_w.reshape(w_init_shape)

    mx.eval(best_w)
    return best_w


def clip_block(
    block: nn.Module,
    no_clip_keys: list[str],
    quantize_func: Callable,
    group_size: int,
    n_grid: int = 20,
):
    def apply_clip(path, module):
        if isinstance(module, (nn.Linear, SwitchLinear)) and all(
            k not in path for k in no_clip_keys
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
    awq_config: AWQConfig,
    group_size: int = 64,
    bits: int = 3,
    embed_group_size: int = 32,
    embed_bits: int = 4,
    n_grid: int = 20,
):
    if awq_config.lm_key is not None:
        model = model[awq_config.lm_key]

    group = mx.distributed.init()

    def quantize_func(w):
        wq = mx.quantize(w, bits=bits, group_size=group_size)
        return mx.dequantize(*wq, bits=bits, group_size=group_size)

    mask = create_attention_mask(inputs)

    embed_key = awq_config.embed
    model.model[embed_key] = model.model[embed_key].to_quantized(
        group_size=embed_group_size, bits=embed_bits
    )
    inputs = model.model[embed_key](inputs)

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
                if isinstance(module, SwitchLinear):
                    indices = args[0]
                    if hasattr(module, "indices"):
                        module.indices = mx.concatenate(
                            [module.indices, indices], axis=0
                        )
                    else:
                        module.indices = indices

                return module(x, *args, **kwargs)

        return Catcher()

    for e, block in enumerate(tqdm(model.layers)):
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
        orig_params = block.parameters()

        scale_block(
            block=block,
            configs=awq_config.scale_configs,
            quantize_func=quantize_func,
            n_grid=n_grid,
            layer_kwargs={"mask": mask},
        )

        clip_block(
            block=block,
            no_clip_keys=awq_config.no_clip,
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
        if after_loss > before_loss:
            # Reload original weights and quantize
            block.update_modules(orig_leaves)
            block.update(orig_params)
            nn.quantize(block, group_size=group_size, bits=bits)
            tqdm.write("Loss is not reduced, falling back to original weights.")

        inputs = outputs

        mx.eval(block)
        mx.clear_cache()

    if (lm_head := awq_config.lm_head) in model:
        model[lm_head] = model[lm_head].to_quantized(
            group_size=embed_group_size, bits=embed_bits
        )


def update_config(
    model: nn.Module,
    config: Dict[str, Any],
):
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
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", default="mlx-community/Qwen2.5-7B-Instruct-bf16"
    )
    parser.add_argument("--mlx-path", default="mlx_model")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--embed-bits", type=int, default=4)
    parser.add_argument("--embed-group-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    group = mx.distributed.init()

    num_samples = args.num_samples
    if group is not None and num_samples % group.size() > 0:
        num_samples += group.size() - num_samples % group.size()

    mx.random.seed(args.seed)

    model, tokenizer, config = load(args.model, lazy=True, return_config=True)

    model_type = config["model_type"]
    if (awq_config := AWQ_MODEL_CONFIGS.get(model_type, None)) is None:
        raise NotImplementedError(f"AWQ support for {model_type} models NYI.")

    calibration_data = load_data(tokenizer, args.num_samples, args.sequence_length)

    calibration_data = dist_split(calibration_data, group)

    awq_quantize(
        model,
        calibration_data,
        awq_config,
        bits=args.bits,
        group_size=args.group_size,
        embed_bits=args.embed_bits,
        embed_group_size=args.embed_group_size,
        n_grid=args.n_grid,
    )

    config = update_config(model, config)
    save(
        args.mlx_path,
        args.model,
        model,
        tokenizer,
        config,
    )
