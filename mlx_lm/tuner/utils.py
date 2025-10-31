# Copyright © 2024 Apple Inc.
import json
import types
import warnings
from math import ceil
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_map_with_path, tree_unflatten

from ..models.switch_layers import QuantizedSwitchLinear, SwitchLinear
from .dora import DoRAEmbedding, DoRALinear
from .lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear


def build_schedule_config(
    learning_rate: float,
    iters: int,
    lr_schedule: Optional[str] = None,
    lr_steps: Optional[int] = None,
    lr_end: Optional[float] = None,
    lr_decay_rate: Optional[float] = None,
    lr_step_size: Optional[int] = None,
    warmup_steps: Optional[int] = None,
    warmup_init: float = 0.0,
    grad_accumulation_steps: int = 1,
) -> Dict:
    """
    Build the scheduler configuration dict for ``build_schedule`` from explicit parameters.

    Args:
        learning_rate: Initial learning rate (init parameter for the scheduler).
        iters: Total training iterations, used as a default for `lr_steps` when applicable.
        lr_schedule: One of {`linear`, `cosine`, `exponential`, `step`}, or None for constant.
        lr_steps: Number of steps for linear/cosine schedulers (defaults to `iters` when None).
        lr_end: End learning rate for linear/cosine (linear defaults to 0.0 when None; cosine end optional).
        lr_decay_rate: Decay rate for exponential/step (defaults to 0.999 for exponential, 0.5 for step when None).
        lr_step_size: Number of steps for step decay (defaults to `max(1, schedule_updates // 10)` when None).
        warmup_steps: Optional warmup steps (>= 0) to prepend.
        warmup_init: Initial learning rate during warmup (defaults to 0.0).
        grad_accumulation_steps: Number of steps to accumulate before each optimizer update.
    Returns:
        Dict with keys: `name`, `arguments`, `warmup`, `warmup_init` suitable for `build_schedule`.
    """
    initial_lr = learning_rate

    # Convert to update units to account for gradient accumulation
    convert_to_update_units = lambda x: max(
        1, ceil(x / max(1, grad_accumulation_steps))
    )

    # number of times the optimizer's update function will be called during warmup
    warmup_updates = (
        None if warmup_steps in (None, 0) else convert_to_update_units(warmup_steps)
    )

    if lr_schedule is None:
        return {
            "name": "constant",
            "arguments": [initial_lr],
            "warmup": warmup_updates,
            "warmup_init": warmup_init,
        }

    # map of simplified lr_schedule names to MLX scheduler names
    schedule_name_map = {
        "linear": "linear_schedule",
        "cosine": "cosine_decay",
        "exponential": "exponential_decay",
        "step": "step_decay",
    }
    # list of allowed MLX scheduler names. Important for YAML config files.
    allowed_mlx_names = {
        "linear_schedule",
        "cosine_decay",
        "exponential_decay",
        "step_decay",
    }
    if lr_schedule in schedule_name_map:
        name = schedule_name_map[lr_schedule]
    elif lr_schedule in allowed_mlx_names:
        # lr_schedule is already an MLX scheduler name
        name = lr_schedule
    else:
        warnings.warn(
            f"Unknown value for --lr_schedule: '{lr_schedule}', defaulting to constant learning rate.",
            RuntimeWarning,
        )
        return {
            "name": "constant",
            "arguments": [initial_lr],
            "warmup": warmup_updates,
            "warmup_init": warmup_init,
        }

    if lr_steps is not None:
        # use lr_steps to calculate the number of main schedule updates
        schedule_updates = convert_to_update_units(lr_steps)
    else:
        # use iters to calculate the number of main schedule updates
        schedule_updates = convert_to_update_units(iters)

    if warmup_updates:
        # number main schedule updates = number of main schedule updates - number of warmup updates
        schedule_updates = max(1, schedule_updates - warmup_updates)

    if name == "linear_schedule":
        # linear_schedule(init: float, end: float, steps: int)
        end = lr_end if lr_end is not None else 0.0
        arguments = [initial_lr, end, schedule_updates]
    elif name == "cosine_decay":
        # cosine_decay(init: float, decay_steps: int[, end: float])
        end = lr_end if lr_end is not None else 0.0
        arguments = [initial_lr, schedule_updates, end]
    elif name == "exponential_decay":
        # exponential_decay(init: float, decay_rate: float)
        decay_rate = lr_decay_rate if lr_decay_rate is not None else 0.999
        arguments = [initial_lr, decay_rate]
    elif name == "step_decay":
        # step_decay(init: float, decay_rate: float, decay_every_n_steps: int)
        decay_rate = lr_decay_rate if lr_decay_rate is not None else 0.5
        decay_every_n_steps = (
            convert_to_update_units(lr_step_size)
            if lr_step_size is not None
            else max(1, schedule_updates // 10)
        )
        arguments = [initial_lr, decay_rate, decay_every_n_steps]
    else:
        arguments = [initial_lr]

    return {
        "name": name,
        "arguments": arguments,
        "warmup": warmup_updates,
        "warmup_init": warmup_init,
    }


def build_schedule(schedule_config: Dict):
    """
    Build and return a step-indexed learning-rate function from a config dict.

    See: https://ml-explore.github.io/mlx/build/html/python/optimizers/schedulers.html for more details.

    schedule_config keys:
    - `name` (str): MLX scheduler under `mlx.optimizers.schedulers`
      (e.g., `linear_schedule`, `cosine_decay`, `exponential_decay`, `step_decay`)
      or `constant` for a fixed LR.
    - `arguments` (list): Positional args for the chosen scheduler. For `constant`, use `[init_lr]`.
    - `warmup` (int, optional): Warmup steps (>= 0). If > 0, a linear warmup from `warmup_init` to
      the initial LR is prepended.
    - `warmup_init` (float, optional): Initial LR during warmup. Defaults to 0.0.

    Returns:
        Callable[[int], float]: A function mapping `step` (int) -> learning rate (float).

    Raises:
        AttributeError: If `name` is not a valid MLX scheduler (except `constant`).
    """
    name = schedule_config["name"]
    arguments = schedule_config["arguments"]
    initial_lr = arguments[0]

    if name == "constant":
        warmup_steps = schedule_config.get("warmup", 0) or 0
        warmup_init = schedule_config.get("warmup_init", 0.0)
        if warmup_steps > 0:
            return opt.schedulers.linear_schedule(warmup_init, initial_lr, warmup_steps)
        return opt.schedulers.linear_schedule(initial_lr, initial_lr, 1)

    schedule_fn = getattr(opt.schedulers, name)
    bound_schedule_fn = schedule_fn(*arguments)
    if warmup_steps := schedule_config.get("warmup", 0):
        warmup_init = schedule_config.get("warmup_init", 0.0)
        warmup_fn = opt.schedulers.linear_schedule(
            warmup_init, initial_lr, warmup_steps
        )
        return opt.schedulers.join_schedules(
            [warmup_fn, bound_schedule_fn], [warmup_steps]
        )
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


def dequantize(model: nn.Module) -> nn.Module:
    """
    Dequantize the quantized linear layers in the model.

    Args:
        model (nn.Module): The model with quantized linear layers.

    Returns:
        nn.Module: The model with dequantized layers.
    """
    dequantize_layers = []
    for name, module in model.named_modules():
        bias = "bias" in module
        if isinstance(module, nn.QuantizedLinear):
            cls = nn.Linear
            kwargs = {"bias": bias}
        elif isinstance(module, nn.QuantizedEmbedding):
            kwargs = {}
            cls = nn.Embedding
        elif isinstance(module, QuantizedSwitchLinear):
            kwargs = {"bias": bias}
            cls = SwitchLinear
        else:
            continue
        weight = mx.dequantize(
            module.weight,
            module.scales,
            module.biases,
            module.group_size,
            module.bits,
        )
        args = weight.shape[::-1]
        m = cls(*args, **kwargs)
        if bias:
            m.bias = module.bias
        m.weight = weight
        dequantize_layers.append((name, m))

    if len(dequantize_layers) > 0:
        model.update_modules(tree_unflatten(dequantize_layers))
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


def get_total_parameters(model):
    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )

    def nparams(m):
        if hasattr(m, "bits"):
            n = 0 if not hasattr(m, "bias") else m.bias.size
            return n + m.weight.size * 32 // m.bits
        return sum(v.size for _, v in tree_flatten(m.parameters()))

    return sum(nparams(m) for _, m in leaf_modules)


def print_trainable_parameters(model):
    total_p = get_total_parameters(model) / 1e6
    trainable_p = (
        sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 1e6
    )
    print(
        f"Trainable parameters: {(trainable_p * 100 / total_p):.3f}% "
        f"({trainable_p:.3f}M/{total_p:.3f}M)"
    )
