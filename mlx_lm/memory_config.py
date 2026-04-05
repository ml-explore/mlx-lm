"""
Memory-aware inference auto-configuration for Apple Silicon.

Detects available unified memory and configures optimal KV cache settings
to maximize context length without OOM. Designed for 32GB machines where
memory is the primary constraint.

Usage:
    from mlx_lm.memory_config import auto_configure

    config = auto_configure(model)
    # config.kv_bits, config.max_kv_size, config.estimated_max_context, etc.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class InferenceConfig:
    """Recommended inference settings based on available memory."""

    kv_bits: Optional[int] = None  # None (FP16), 4, or 8
    kv_group_size: int = 64
    max_kv_size: Optional[int] = None  # None = unlimited, or rotating cache size
    quantized_kv_start: int = 0  # layers to keep FP16 before quantizing
    estimated_max_context: int = 0  # max tokens before OOM
    headroom_gb: float = 0.0  # estimated free memory at max context
    warnings: List[str] = field(default_factory=list)


def _model_bytes(model: nn.Module) -> int:
    """Estimate model weight size in bytes."""
    total = 0
    for _, v in nn.utils.tree_flatten(model.parameters()):
        if isinstance(v, mx.array):
            total += v.nbytes
    return total


def _kv_bytes_per_token(model: nn.Module, bits: Optional[int] = None) -> float:
    """Estimate KV cache bytes per token across all layers.

    For FP16: 2 * num_kv_heads * head_dim * 2 bytes * num_layers
    For quantized: adjusted by bits/16 ratio plus scale/bias overhead
    """
    args = getattr(model, "args", None)
    if args is None:
        # Fallback: try to infer from first layer
        layer = model.layers[0]
        for attr in ("self_attn", "attention", "attn"):
            attn = getattr(layer, attr, None)
            if attn is not None:
                break
        else:
            return 0

        num_kv_heads = getattr(attn, "n_kv_heads", getattr(attn, "num_kv_heads", 8))
        head_dim = getattr(attn, "head_dim", 128)
    else:
        num_kv_heads = getattr(args, "num_key_value_heads", getattr(args, "num_kv_heads", 8))
        head_dim = getattr(args, "head_dim", None)
        if head_dim is None:
            hidden = getattr(args, "hidden_size", 4096)
            num_heads = getattr(args, "num_attention_heads", 32)
            head_dim = hidden // num_heads

    num_layers = len(model.layers)

    # 2 for keys + values, 2 bytes per float16 element
    fp16_per_token = 2 * num_kv_heads * head_dim * 2 * num_layers

    if bits is None:
        return fp16_per_token

    # Quantized: data is packed, plus per-group scales and biases
    group_size = 64
    el_per_int = 32 // bits  # elements packed per uint32
    data_bytes = num_kv_heads * head_dim / el_per_int * 4  # uint32 = 4 bytes
    groups = num_kv_heads * head_dim / group_size
    scale_bias_bytes = groups * 2 * 2  # scales + biases, float16 each

    quant_per_layer = 2 * (data_bytes + scale_bias_bytes)  # keys + values
    return quant_per_layer * num_layers


def auto_configure(
    model: nn.Module,
    target_context: Optional[int] = None,
    memory_budget_fraction: float = 0.85,
) -> InferenceConfig:
    """
    Automatically configure inference settings based on available memory.

    Args:
        model: Loaded MLX model
        target_context: Desired context length (None = maximize)
        memory_budget_fraction: Fraction of recommended working set to use

    Returns:
        InferenceConfig with recommended settings
    """
    info = mx.device_info()
    total_memory = info.get("memory_size", 32 * 1024**3)
    recommended = info.get("max_recommended_working_set_size", total_memory * 0.75)
    budget = int(recommended * memory_budget_fraction)

    model_size = _model_bytes(model)
    active = mx.get_active_memory()
    overhead = 512 * 1024 * 1024  # 512MB buffer for activations, framework, etc.

    available_for_kv = budget - model_size - overhead
    if available_for_kv < 0:
        available_for_kv = 0

    config = InferenceConfig()
    total_gb = total_memory / 1024**3
    model_gb = model_size / 1024**3
    avail_gb = available_for_kv / 1024**3

    # Try configurations from least to most aggressive
    candidates = [
        (None, "FP16"),
        (8, "8-bit"),
        (4, "4-bit"),
    ]

    # Compute max context for each option
    options = []
    for bits, label in candidates:
        bpt = _kv_bytes_per_token(model, bits)
        if bpt > 0:
            max_ctx = int(available_for_kv / bpt)
            options.append((bits, label, max_ctx, bpt))

    if not options:
        config.warnings.append(
            f"Model ({model_gb:.1f}GB) may not fit in memory ({total_gb:.0f}GB)"
        )
        config.kv_bits = 4
        config.estimated_max_context = 0
        return config

    if target_context is not None:
        # Find least aggressive option that meets target
        chosen = None
        for bits, label, max_ctx, bpt in options:
            if max_ctx >= target_context:
                chosen = (bits, max_ctx, bpt)
                break

        if chosen is not None:
            config.kv_bits = chosen[0]
            config.estimated_max_context = target_context
            config.headroom_gb = (available_for_kv - target_context * chosen[2]) / 1024**3
        else:
            # Can't meet target — use most aggressive and report actual max
            bits, label, max_ctx, bpt = options[-1]
            config.kv_bits = bits
            config.estimated_max_context = max_ctx
            config.headroom_gb = 0
            config.warnings.append(
                f"Cannot fit {target_context:,} tokens. "
                f"Max with {label} KV: {max_ctx:,} tokens."
            )
    else:
        # No target: pick least aggressive option with >= 16K context
        config.kv_bits = options[-1][0]  # default to most aggressive
        config.estimated_max_context = options[-1][2]

        for bits, label, max_ctx, bpt in options:
            if max_ctx >= 16384:
                config.kv_bits = bits
                config.estimated_max_context = max_ctx
                break

    # If even 4-bit can't handle a reasonable context, suggest rotating cache
    if config.estimated_max_context < 4096:
        config.max_kv_size = max(config.estimated_max_context, 2048)
        config.warnings.append(
            f"Limited to {config.max_kv_size} token context window "
            f"(rotating cache). Model uses {model_gb:.1f}GB of {total_gb:.0f}GB."
        )

    # Warn if tight
    if config.estimated_max_context < 8192:
        config.warnings.append(
            f"Memory constrained: ~{config.estimated_max_context:,} max tokens. "
            f"Consider a smaller model for long-context tasks."
        )

    return config


def describe_config(config: InferenceConfig) -> str:
    """Human-readable description of the configuration."""
    lines = []
    if config.kv_bits is None:
        lines.append("KV cache: FP16 (no quantization)")
    else:
        lines.append(f"KV cache: {config.kv_bits}-bit quantized (group_size={config.kv_group_size})")

    lines.append(f"Estimated max context: {config.estimated_max_context:,} tokens")

    if config.max_kv_size is not None:
        lines.append(f"Rotating cache: {config.max_kv_size:,} tokens (oldest context dropped)")

    for w in config.warnings:
        lines.append(f"Warning: {w}")

    return "\n".join(lines)
