# Copyright © 2024 Apple Inc.
import json
import re
import types
from pathlib import Path
from typing import Dict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

from ..models.switch_layers import QuantizedSwitchLinear, SwitchLinear
from ..utils import get_total_parameters
from .dora import DoRAEmbedding, DoRALinear
from .lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear


def build_schedule(schedule_config: Dict):
    """
    Build a learning rate schedule from the given config.
    """
    schedule_fn = getattr(opt.schedulers, schedule_config["name"])
    arguments = schedule_config["arguments"]
    initial_lr = arguments[0]
    bound_schedule_fn = schedule_fn(*arguments)
    if warmup_steps := schedule_config.get("warmup", 0):
        warmup_init = schedule_config.get("warmup_init", 0.0)
        warmup_fn = opt.schedulers.linear_schedule(
            warmup_init, initial_lr, warmup_steps
        )
        return opt.schedulers.join_schedules(
            [warmup_fn, bound_schedule_fn], [warmup_steps + 1]
        )
    else:
        return bound_schedule_fn


def linear_to_lora_layers(
    model: nn.Module,
    num_layers: int,
    config: Dict,
    use_dora: bool = False,
):
    """
    Convert some of the models linear layers to lora layers.

    Args:
        model (nn.Module): The neural network model.
        num_layers (int): The number of blocks to convert to lora layers
        starting from the last layer.
        config (dict): More configuration parameters for LoRA, including the
          rank, scale, and optional layer keys.
        use_dora (bool): If True, uses DoRA instead of LoRA.
          Default: ``False``
    """

    def to_lora(layer):
        if not use_dora and hasattr(layer, "to_lora"):
            return layer.to_lora(
                r=config["rank"],
                scale=config["scale"],
                dropout=config["dropout"],
            )

        if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
            LoRALayer = DoRALinear if use_dora else LoRALinear
        elif isinstance(layer, (SwitchLinear, QuantizedSwitchLinear)):
            if use_dora:
                raise ValueError(f"{type(layer).__name__} doesn't support DoRA yet.")
            LoRALayer = LoRASwitchLinear
        elif isinstance(layer, (nn.Embedding, nn.QuantizedEmbedding)):
            LoRALayer = DoRAEmbedding if use_dora else LoRAEmbedding
        else:
            raise ValueError(
                f"Can't convert layer of type {type(layer).__name__} to LoRA"
            )

        return LoRALayer.from_base(
            layer,
            r=config["rank"],
            scale=config["scale"],
            dropout=config["dropout"],
        )

    if (keys := config.get("keys", None)) is None:
        keys = set()

        def get_keys_for_lora(p, m):
            types = (
                nn.Linear,
                nn.QuantizedLinear,
                SwitchLinear,
                QuantizedSwitchLinear,
                nn.Embedding,
                nn.QuantizedEmbedding,
            )
            if hasattr(m, "to_lora") or isinstance(m, types):
                keys.add(p)

        for l in model.layers:
            l.apply_to_modules(get_keys_for_lora)

    for l in model.layers[-max(num_layers, 0) :]:
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in keys]
        if lora_layers:
            l.update_modules(tree_unflatten(lora_layers))

    lora_modules = [(k, to_lora(m)) for k, m in model.named_modules() if k in keys]
    if lora_modules:
        model.update_modules(tree_unflatten(lora_modules))


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _is_peft_config(raw_config: dict) -> bool:
    """Return True if the config dict is in PEFT/Hugging Face format."""
    return "peft_type" in raw_config


def _convert_peft_config(
    peft_config: dict, adapter_path: Path
) -> types.SimpleNamespace:
    """
    Convert a PEFT adapter_config dict into an MLX-compatible config namespace.

    Args:
        peft_config: Raw dict loaded from PEFT's adapter_config.json.
        adapter_path: Path to the adapter directory (used to inspect weight keys).

    Returns:
        types.SimpleNamespace with fields: fine_tune_type, num_layers, lora_parameters.

    Raises:
        ValueError: If peft_type is not "LORA", or required fields are missing.
        FileNotFoundError: If adapter_model.safetensors is not present.
    """
    peft_type = peft_config.get("peft_type", "").upper()
    if peft_type != "LORA":
        raise ValueError(
            f"Only PEFT peft_type='LORA' is supported, got '{peft_type}'. "
            "DoRA and AdaLoRA are not currently supported via this path."
        )

    weight_file = adapter_path / "adapter_model.safetensors"
    if not weight_file.exists():
        raise FileNotFoundError(
            f"Expected PEFT weight file not found: {weight_file}"
        )

    raw_weights = mx.load(str(weight_file))
    layer_indices = [
        int(m.group(1))
        for key in raw_weights.keys()
        if (m := _LAYER_RE.search(key))
    ]
    if not layer_indices:
        raise ValueError(
            "Could not determine num_layers: no layer index found in weight keys. "
            f"Sample keys: {list(raw_weights.keys())[:5]}"
        )
    num_layers = max(layer_indices) + 1

    r = peft_config.get("r")
    if r is None:
        raise ValueError("PEFT config missing required field 'r' (rank).")
    lora_alpha = peft_config.get("lora_alpha", r)
    dropout = peft_config.get("lora_dropout", 0.0)

    return types.SimpleNamespace(
        fine_tune_type="lora",
        num_layers=num_layers,
        lora_parameters={"rank": r, "scale": lora_alpha / r, "dropout": dropout},
    )


def _remap_peft_weights(weights: dict) -> dict:
    """
    Remap PEFT weight keys and shapes to MLX LoRA conventions.

    PEFT stores lora_A as (rank, in_features) and lora_B as (out_features, rank).
    MLX expects lora_a as (in_features, rank) and lora_b as (rank, out_features).
    Both matrices require transposing.

    Args:
        weights: Dict mapping PEFT weight key strings to mx.array values.

    Returns:
        Dict mapping MLX weight key strings to correctly-shaped mx.array values.
        Keys that are not recognizable LoRA adapter weights are silently dropped.
    """
    remapped = {}
    for key, value in weights.items():
        if not key.startswith("base_model.model."):
            continue
        stripped = key[len("base_model.model."):]

        if stripped.endswith(".lora_A.weight"):
            mlx_key = stripped[: -len(".lora_A.weight")] + ".lora_a"
            remapped[mlx_key] = mx.transpose(value)  # (rank, in) → (in, rank)
        elif stripped.endswith(".lora_B.weight"):
            mlx_key = stripped[: -len(".lora_B.weight")] + ".lora_b"
            remapped[mlx_key] = mx.transpose(value)  # (out, rank) → (rank, out)
        # Other keys (base model weights, scaling vectors, etc.) are silently skipped.

    return remapped


def load_adapters(model: nn.Module, adapter_path: str) -> nn.Module:
    """
    Load fine-tuned adapters / layers, supporting both mlx-lm native format
    and PEFT/Unsloth LoRA adapter format.

    For native mlx-lm adapters, the adapter directory must contain:
        - adapter_config.json  (with fine_tune_type, num_layers, lora_parameters)
        - adapters.safetensors

    For PEFT/Unsloth adapters, the adapter directory must contain:
        - adapter_config.json  (with peft_type, r, lora_alpha, ...)
        - adapter_model.safetensors

    Args:
        model (nn.Module): The neural network model.
        adapter_path (str): Path to the adapter directory.

    Returns:
        nn.Module: The updated model with LoRA layers applied and weights loaded.
    """
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"The adapter path does not exist: {adapter_path}")

    with open(adapter_path / "adapter_config.json", "r") as fid:
        raw_config = json.load(fid)

    if _is_peft_config(raw_config):
        # --- PEFT / Unsloth path ---
        config = _convert_peft_config(raw_config, adapter_path)
        linear_to_lora_layers(
            model,
            config.num_layers,
            config.lora_parameters,
            use_dora=False,
        )
        raw_weights = mx.load(str(adapter_path / "adapter_model.safetensors"))
        remapped = _remap_peft_weights(raw_weights)
        if not remapped:
            raise ValueError(
                "No LoRA weights found after remapping PEFT adapter. "
                "Ensure adapter_model.safetensors contains lora_A/lora_B weights."
            )
        model.load_weights(list(remapped.items()), strict=False)
    else:
        # --- Native mlx-lm path (unchanged) ---
        config = types.SimpleNamespace(**raw_config)
        fine_tune_type = getattr(config, "fine_tune_type", "lora")
        if fine_tune_type != "full":
            linear_to_lora_layers(
                model,
                config.num_layers,
                config.lora_parameters,
                use_dora=(fine_tune_type == "dora"),
            )
        model.load_weights(str(adapter_path / "adapters.safetensors"), strict=False)

    return model


def remove_lora_layers(model: nn.Module) -> nn.Module:
    """
    Remove the LoRA layers from the model.

    Args:
        model (nn.Module): The model with LoRA layers.

    Returns:
        nn.Module: The model without LoRA layers.
    """
    reset_layers = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            reset_layers.append((name, module.linear))
    if len(reset_layers) > 0:
        model.update_modules(tree_unflatten(reset_layers))
    return model


def print_trainable_parameters(model):
    total_p = get_total_parameters(model) / 1e6
    trainable_p = (
        sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 1e6
    )
    print(
        f"Trainable parameters: {(trainable_p * 100 / total_p):.3f}% "
        f"({trainable_p:.3f}M/{total_p:.3f}M)"
    )
