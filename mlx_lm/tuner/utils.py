# Copyright © 2024 Apple Inc.
import json
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


def load_adapters(model: nn.Module, adapter_path: str) -> nn.Module:
    """
    Load any fine-tuned adapters / layers.

    Args:
        model (nn.Module): The neural network model.
        adapter_path (str): Path to the adapter configuration file.

    Returns:
        nn.Module: The updated model with LoRA layers applied.
    """
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"The adapter path does not exist: {adapter_path}")
    with open(adapter_path / "adapter_config.json", "r") as fid:
        config = types.SimpleNamespace(**json.load(fid))
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
