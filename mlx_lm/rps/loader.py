"""Load models with Residual Precision Streaming."""

import json
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import RPSConfig
from .linear import RPSLinear


def _replace_with_rps(model: nn.Module) -> int:
    """Replace all QuantizedLinear layers with RPSLinear. Returns count."""
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            # Navigate to parent and replace
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], RPSLinear(module))
            count += 1
    return count


def _attach_tier2_residuals(
    model: nn.Module,
    residual_dir: Path,
    config: RPSConfig,
    verbose: bool = True,
) -> int:
    """Load and attach Tier 2 (in-RAM) residuals to RPSLinear layers."""
    attached = 0
    num_layers = config.get("num_layers", len(model.layers)) if isinstance(config, dict) else len(model.layers)
    tier2_keys = config["tier2_keys"] if isinstance(config, dict) else config.tier2_keys
    r_bits = config["residual_bits"] if isinstance(config, dict) else config.residual_bits
    r_gs = config["residual_group_size"] if isinstance(config, dict) else config.residual_group_size

    for layer_idx in range(num_layers):
        res_file = residual_dir / f"residual-layer-{layer_idx:02d}.safetensors"
        if not res_file.exists():
            continue

        layer_data = mx.load(str(res_file))
        layer = model.layers[layer_idx]

        for key in tier2_keys:
            # Find the RPSLinear module matching this key
            for attr in ("self_attn", "attention", "attn"):
                attn = getattr(layer, attr, None)
                if attn is not None:
                    proj = getattr(attn, key, None)
                    if proj is not None and isinstance(proj, RPSLinear):
                        prefix = f"{attr}.{key}"
                        r_w = layer_data.get(f"{prefix}.weight")
                        r_s = layer_data.get(f"{prefix}.scales")
                        r_b = layer_data.get(f"{prefix}.biases")
                        if r_w is not None:
                            mx.eval(r_w, r_s, r_b)
                            proj.attach_residual(r_w, r_s, r_b, r_bits, r_gs)
                            attached += 1
                    break

            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                proj = getattr(mlp, key, None)
                if proj is not None and isinstance(proj, RPSLinear):
                    prefix = f"mlp.{key}"
                    r_w = layer_data.get(f"{prefix}.weight")
                    r_s = layer_data.get(f"{prefix}.scales")
                    r_b = layer_data.get(f"{prefix}.biases")
                    if r_w is not None:
                        mx.eval(r_w, r_s, r_b)
                        proj.attach_residual(r_w, r_s, r_b, r_bits, r_gs)
                        attached += 1

    if verbose:
        print(f"Attached {attached} Tier 2 residuals ({tier2_keys})")
    return attached


def load_rps(
    base_path: str,
    residual_path: str,
    tier: int = 2,
    verbose: bool = True,
) -> Tuple[nn.Module, any]:
    """Load a model with Residual Precision Streaming.

    Args:
        base_path: Path to base 2-bit model
        residual_path: Path to residual directory
        tier: 1 = base only, 2 = base + critical residuals in RAM
        verbose: Print progress

    Returns:
        (model, tokenizer) tuple
    """
    from mlx_lm import load

    if verbose:
        print(f"Loading base model: {base_path}")
    model, tokenizer = load(base_path)

    # Replace QuantizedLinear → RPSLinear
    count = _replace_with_rps(model)
    if verbose:
        print(f"Replaced {count} layers with RPSLinear")

    if tier >= 2:
        res_dir = Path(residual_path)
        config_path = res_dir / "rps_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = RPSConfig().__dict__

        _attach_tier2_residuals(model, res_dir, config, verbose)

    return model, tokenizer
