# Copyright © 2024 Apple Inc.
import json
import types
from math import ceil
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


def build_schedule(
    schedule_config: Dict,
    learning_rate: float = None,
    iters: int = None,
    grad_accumulation_steps: int = 1,
):
    """
    Build a learning rate schedule from the given config.

    Args:
        schedule_config (dict): The configuration for the learning rate schedule.
        learning_rate (float): The initial learning rate.
        iters (int): The total number of iterations.
        grad_accumulation_steps (int): The number of steps to accumulate gradients before applying an optimizer update.
    """
    # Determine scheduler name. If missing and warmup present, treat as warmup-only constant.
    name = schedule_config.get("name")
    arguments = schedule_config.get("arguments", [])
    initial_lr = arguments[0] if len(arguments) > 0 else learning_rate

    # Convert to update units to account for gradient accumulation
    convert_to_update_units = lambda x: max(
        1, ceil(x / max(1, grad_accumulation_steps))
    )
    warmup_config = schedule_config.get("warmup")  # raw number of steps
    if warmup_config is not None and warmup_config > 0:
        warmup_steps = convert_to_update_units(warmup_config)
    else:
        warmup_steps = 0

    # If no lr was supplied anywhere and no explicit initial lr was provided, error early
    if initial_lr is None and len(arguments) == 0:
        raise KeyError("learning_rate")

    # Handle missing name:
    if not name:
        if warmup_steps > 0:
            name = "constant"
        else:
            raise KeyError("name")

    if name == "constant":
        warmup_init = schedule_config.get("warmup_init", 0.0)
        if warmup_steps > 0:
            return opt.schedulers.linear_schedule(warmup_init, initial_lr, warmup_steps)
        return opt.schedulers.linear_schedule(initial_lr, initial_lr, 1)

    # Set sane defaults for arguments of known schedulers
    if name == "linear_schedule":
        # params: start, end, steps
        start, end, steps = initial_lr, 0.0, convert_to_update_units(iters or 1)
        if len(arguments) >= 1:
            start = arguments[0]
        if len(arguments) >= 2:
            end = arguments[1]
        if len(arguments) >= 3:
            steps = convert_to_update_units(arguments[2])
        if len(arguments) > 3:
            raise ValueError(f"Unsupported number of arguments for scheduler: {name}")
        arguments = [start, end, steps]
    elif name == "cosine_decay":
        # params: initial_lr, decay_steps, end_lr
        init, decay_steps, end_lr = (
            initial_lr,
            convert_to_update_units(iters or 1),
            0.0,
        )
        if len(arguments) >= 1:
            init = arguments[0]
        if len(arguments) >= 2:
            decay_steps = convert_to_update_units(arguments[1])
        if len(arguments) >= 3:
            end_lr = arguments[2]
        if len(arguments) > 3:
            raise ValueError(f"Unsupported number of arguments for scheduler: {name}")
        arguments = [init, decay_steps, end_lr]
    elif name == "exponential_decay":
        # params: initial_lr, decay_rate
        init, decay_rate = initial_lr, 0.999
        if len(arguments) >= 1:
            init = arguments[0]
        if len(arguments) >= 2:
            decay_rate = arguments[1]
        if len(arguments) > 2:
            raise ValueError(f"Unsupported number of arguments for scheduler: {name}")
        arguments = [init, decay_rate]
    elif name == "step_decay":
        # params: initial_lr, decay_rate, step_size
        init, factor, step_size = (
            initial_lr,
            0.5,
            max(1, convert_to_update_units((iters or 1) // 10)),
        )
        if len(arguments) >= 1:
            init = arguments[0]
        if len(arguments) >= 2:
            factor = arguments[1]
        if len(arguments) >= 3:
            step_size = convert_to_update_units(arguments[2])
        if len(arguments) > 3:
            raise ValueError(f"Unsupported number of arguments for scheduler: {name}")
        arguments = [init, factor, step_size]
    else:
        raise ValueError(f"Unsupported scheduler: {name}")

    schedule_fn = getattr(opt.schedulers, name)
    bound_schedule_fn = schedule_fn(*arguments)
    if warmup_steps > 0:
        warmup_init = schedule_config.get("warmup_init", 0.0)
        warmup_fn = opt.schedulers.linear_schedule(
            warmup_init, initial_lr, warmup_steps
        )
        return opt.schedulers.join_schedules(
            [warmup_fn, bound_schedule_fn], [warmup_steps]
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
