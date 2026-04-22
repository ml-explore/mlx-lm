"""Convert a model to Residual Precision Streaming format.

Takes a source model (FP16 or quantized) and produces:
1. Base model: 2-bit quantized (standard mlx-lm format)
2. Residuals: per-layer 2-bit quantized deltas (base → original)
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import RPSConfig


def _dequantize_layer_weight(layer_module) -> Optional[mx.array]:
    """Dequantize a QuantizedLinear layer's weight to float16."""
    if not isinstance(layer_module, nn.QuantizedLinear):
        return None
    w = mx.dequantize(
        layer_module.weight,
        layer_module.scales,
        layer_module.biases,
        group_size=layer_module.group_size,
        bits=layer_module.bits,
    )
    return w


def compute_residuals(
    original_weight: mx.array,
    base_bits: int = 2,
    base_group_size: int = 64,
    residual_bits: int = 2,
    residual_group_size: int = 64,
):
    """Compute quantized residual between original and base-quantized weight.

    Returns:
        (base_q, base_scales, base_biases,
         res_q, res_scales, res_biases,
         mse_base, mse_with_residual)
    """
    w = original_weight.astype(mx.float16)

    # Base quantization
    base_q, base_s, base_b = mx.quantize(
        w, group_size=base_group_size, bits=base_bits
    )
    w_base = mx.dequantize(base_q, base_s, base_b,
                            group_size=base_group_size, bits=base_bits)

    # Residual = original - base approximation
    residual = w - w_base

    # Quantize the residual
    res_q, res_s, res_b = mx.quantize(
        residual, group_size=residual_group_size, bits=residual_bits
    )
    w_residual = mx.dequantize(res_q, res_s, res_b,
                                group_size=residual_group_size, bits=residual_bits)

    # Measure improvement
    mse_base = mx.mean((w - w_base) ** 2).item()
    mse_combined = mx.mean((w - w_base - w_residual) ** 2).item()

    return (
        base_q, base_s, base_b,
        res_q, res_s, res_b,
        mse_base, mse_combined,
    )


def convert(
    model_path: str,
    base_output: str,
    residual_output: str,
    config: Optional[RPSConfig] = None,
    verbose: bool = True,
):
    """Convert a model to RPS format.

    Args:
        model_path: HuggingFace model ID or local path
        base_output: Directory for base 2-bit model
        residual_output: Directory for residual files
        config: RPS configuration (defaults used if None)
        verbose: Print progress
    """
    from mlx_lm import load

    if config is None:
        config = RPSConfig()

    if verbose:
        print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)

    base_dir = Path(base_output)
    res_dir = Path(residual_output)
    base_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    total_mse_base = 0
    total_mse_combined = 0
    n_weights = 0

    num_layers = len(model.layers)
    for layer_idx in range(num_layers):
        layer = model.layers[layer_idx]
        layer_residuals = {}

        # Find all quantized linear projections
        targets = []
        for attr in ("self_attn", "attention", "attn"):
            attn = getattr(layer, attr, None)
            if attn is not None:
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    m = getattr(attn, proj, None)
                    if m is not None:
                        targets.append((f"{attr}.{proj}", m))
                break
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                m = getattr(mlp, proj, None)
                if m is not None:
                    targets.append((f"mlp.{proj}", m))

        if verbose:
            print(f"\nLayer {layer_idx}/{num_layers}:")

        for name, module in targets:
            if not isinstance(module, nn.QuantizedLinear):
                continue

            # Get original weight (dequantize from source quantization)
            w_original = _dequantize_layer_weight(module)
            if w_original is None:
                continue
            mx.eval(w_original)

            # Compute base + residual
            (base_q, base_s, base_b,
             res_q, res_s, res_b,
             mse_base, mse_combined) = compute_residuals(
                w_original,
                base_bits=config.base_bits,
                base_group_size=config.base_group_size,
                residual_bits=config.residual_bits,
                residual_group_size=config.residual_group_size,
            )
            mx.eval(base_q, base_s, base_b, res_q, res_s, res_b)

            # Update the model's base weights in-place
            module.weight = base_q
            module.scales = base_s
            module.biases = base_b
            module.bits = config.base_bits
            module.group_size = config.base_group_size

            # Store residuals
            layer_residuals[f"{name}.weight"] = res_q
            layer_residuals[f"{name}.scales"] = res_s
            layer_residuals[f"{name}.biases"] = res_b

            improvement = (1 - mse_combined / max(mse_base, 1e-10)) * 100
            total_mse_base += mse_base
            total_mse_combined += mse_combined
            n_weights += 1

            if verbose:
                print(f"  {name}: MSE {mse_base:.6f} → {mse_combined:.6f} "
                      f"({improvement:.0f}% reduction)")

            del w_original

        # Save layer residuals
        if layer_residuals:
            res_path = res_dir / f"residual-layer-{layer_idx:02d}.safetensors"
            mx.save_safetensors(str(res_path), layer_residuals)

        mx.eval()

    # Save base model
    if verbose:
        print(f"\nSaving base model to {base_dir}...")

    # Copy tokenizer and config files
    from huggingface_hub import snapshot_download
    src = Path(model_path) if Path(model_path).exists() else Path(
        snapshot_download(model_path)
    )
    for fname in ("tokenizer.json", "tokenizer_config.json", "config.json",
                   "special_tokens_map.json", "chat_template.jinja",
                   "tokenizer.model"):
        src_file = src / fname
        if src_file.exists():
            shutil.copy2(src_file, base_dir / fname)

    # Save quantized weights
    weights = dict(nn.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(base_dir / "model.safetensors"), weights)

    # Update config with quantization info
    config_path = base_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
        model_config["quantization"] = {
            "group_size": config.base_group_size,
            "bits": config.base_bits,
        }
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)

    # Save RPS config
    rps_meta = {
        "base_bits": config.base_bits,
        "base_group_size": config.base_group_size,
        "residual_bits": config.residual_bits,
        "residual_group_size": config.residual_group_size,
        "tier2_keys": config.tier2_keys,
        "num_layers": num_layers,
    }
    with open(res_dir / "rps_config.json", "w") as f:
        json.dump(rps_meta, f, indent=2)

    if verbose and n_weights > 0:
        overall = (1 - total_mse_combined / max(total_mse_base, 1e-10)) * 100
        print(f"\nDone! Overall MSE reduction: {overall:.0f}%")
        print(f"Base model: {base_dir}")
        print(f"Residuals: {res_dir}")
