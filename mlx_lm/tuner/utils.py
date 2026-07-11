# Copyright © 2024 Apple Inc.
import json
import shutil
import tempfile
import types
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

from ..cli_ui import rprint
from ..models.switch_layers import QuantizedSwitchLinear, SwitchLinear
from ..utils import get_total_parameters
from .dora import DoRAEmbedding, DoRALinear
from .lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear, QuantizedLoRALinear

QUANTIZED_LORA_FORMAT = "mlx_lm.quantized_lora"
QUANTIZED_LORA_VERSION = 1


def _parse_quantized_lora_layers(quantization: Dict) -> Dict[str, Dict[str, int]]:
    try:
        group_size = int(quantization["group_size"])
        rank_group_size = int(quantization["rank_group_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Quantized LoRA adapter has invalid group sizes") from error
    if group_size not in (32, 64, 128):
        raise ValueError(f"Unsupported LoRA group size: {group_size}")
    if rank_group_size not in (32, 64, 128):
        raise ValueError(f"Unsupported LoRA rank group size: {rank_group_size}")

    layers = quantization.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError("Quantized LoRA adapter has no layer metadata")

    required = {
        "bits",
        "input_dims",
        "output_dims",
        "rank",
        "padded_input_dims",
        "padded_rank",
    }
    parsed = {}
    for name, metadata in layers.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"Quantized LoRA layer '{name}' has invalid metadata")
        missing = sorted(required - set(metadata))
        if missing:
            raise ValueError(
                f"Quantized LoRA layer '{name}' is missing metadata: "
                + ", ".join(missing)
            )
        values = {key: int(metadata[key]) for key in required}
        if any(value <= 0 for value in values.values()):
            raise ValueError(f"Quantized LoRA layer '{name}' metadata must be positive")
        if values["padded_input_dims"] < values["input_dims"]:
            raise ValueError(f"Quantized LoRA layer '{name}' has invalid input padding")
        if values["padded_rank"] < values["rank"]:
            raise ValueError(f"Quantized LoRA layer '{name}' has invalid rank padding")
        if values["bits"] not in (2, 3, 4, 5, 6, 8):
            raise ValueError(f"Quantized LoRA layer '{name}' has unsupported bit width")
        expected_input = (
            (values["input_dims"] + group_size - 1) // group_size * group_size
        )
        expected_rank = (
            (values["rank"] + rank_group_size - 1) // rank_group_size * rank_group_size
        )
        if values["padded_input_dims"] != expected_input:
            raise ValueError(
                f"Quantized LoRA layer '{name}' input padding does not match "
                "the group size"
            )
        if values["padded_rank"] != expected_rank:
            raise ValueError(
                f"Quantized LoRA layer '{name}' rank padding does not match "
                "the rank group size"
            )
        parsed[name] = values
    return parsed


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


def quantize_lora_layers(
    model: nn.Module,
    group_size: int = 64,
    bits: int = 8,
    rank_group_size: int = 32,
    mode: str = "affine",
    layer_bits: Optional[Dict[str, int]] = None,
):
    """Replace all LoRA linear layers with quantized inference layers.

    LoRA embedding and switch-linear layers are left unchanged.
    """
    lora_layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }
    if layer_bits is None:
        selected_layers = {name: bits for name in lora_layers}
    else:
        missing = sorted(set(layer_bits) - set(lora_layers))
        if missing:
            raise ValueError(
                "Quantized LoRA metadata references missing linear layers: "
                + ", ".join(missing)
            )
        selected_layers = layer_bits

    replacements = []
    for name, layer_bits_value in selected_layers.items():
        replacements.append(
            (
                name,
                lora_layers[name].to_quantized(
                    group_size=group_size,
                    bits=layer_bits_value,
                    rank_group_size=rank_group_size,
                    mode=mode,
                ),
            )
        )
    if replacements:
        model.update_modules(tree_unflatten(replacements))
    return model


def select_lora_layer_bits(
    model: nn.Module,
    layer_inputs: Dict[str, mx.array],
    candidate_bits: Sequence[int] = (4, 5, 6, 8),
    group_size: int = 64,
    rank_group_size: int = 32,
    mode: str = "affine",
    max_relative_l2: float = 0.01,
    min_cosine: float = 0.9999,
    min_memory_reduction: float = 0.0,
) -> Tuple[Dict[str, int], Dict[str, Dict]]:
    """Select the lowest acceptable precision from real layer activations.

    Layers that do not satisfy both error thresholds remain unquantized and
    are omitted from the returned bit mapping.
    """
    candidate_bits = tuple(sorted(set(candidate_bits)))
    if not candidate_bits:
        raise ValueError("candidate_bits must not be empty")

    lora_layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }
    missing = sorted(set(layer_inputs) - set(lora_layers))
    if missing:
        raise ValueError(
            "Calibration inputs reference missing LoRA linear layers: "
            + ", ".join(missing)
        )

    selected = {}
    report = {}
    for name, x in layer_inputs.items():
        layer = lora_layers[name]
        if x.shape[-1] != layer.lora_a.shape[0]:
            raise ValueError(
                f"Calibration input for {name} has dimension {x.shape[-1]}, "
                f"expected {layer.lora_a.shape[0]}"
            )

        reference = ((x @ layer.lora_a) @ layer.lora_b).astype(mx.float32)
        mx.eval(reference)
        reference_norm = float(mx.linalg.norm(reference))
        float_bytes = layer.lora_a.nbytes + layer.lora_b.nbytes
        candidates = {}
        selected_bits = None
        for bits in candidate_bits:
            quantized = layer.to_quantized(
                group_size=group_size,
                bits=bits,
                rank_group_size=rank_group_size,
                mode=mode,
            )
            candidate = quantized.lora_delta(x).astype(mx.float32)
            difference = candidate - reference
            mx.eval(candidate, difference)
            candidate_norm = float(mx.linalg.norm(candidate))
            difference_norm = float(mx.linalg.norm(difference))
            quantized_bytes = sum(
                value.nbytes
                for parameter_name, value in tree_flatten(quantized.parameters())
                if parameter_name.startswith("lora_")
            )
            memory_reduction = 1 - quantized_bytes / float_bytes
            if reference_norm == 0.0:
                relative_l2 = 0.0 if difference_norm == 0.0 else float("inf")
                cosine = 1.0 if candidate_norm == 0.0 else 0.0
            else:
                relative_l2 = difference_norm / reference_norm
                cosine = float(
                    mx.sum(reference * candidate)
                    / (mx.linalg.norm(reference) * mx.linalg.norm(candidate) + 1e-12)
                )
            candidates[str(bits)] = {
                "relative_l2": relative_l2,
                "cosine": cosine,
                "adapter_bytes": quantized_bytes,
                "memory_reduction": memory_reduction,
            }
            if (
                selected_bits is None
                and relative_l2 <= max_relative_l2
                and cosine >= min_cosine
                and memory_reduction >= min_memory_reduction
            ):
                selected_bits = bits

        if selected_bits is not None:
            selected[name] = selected_bits
        report[name] = {
            "selected_bits": selected_bits,
            "float_adapter_bytes": float_bytes,
            "candidates": candidates,
        }
    return selected, report


def save_quantized_adapter(
    model: nn.Module,
    source_adapter_path: str,
    output_path: str,
) -> Path:
    """Save a model's quantized LoRA layers as a directly loadable adapter."""
    source_adapter_path = Path(source_adapter_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")

    config_path = source_adapter_path / "adapter_config.json"
    with open(config_path, "r", encoding="utf-8") as fid:
        config = json.load(fid)
    if config.get("fine_tune_type", "lora") != "lora":
        raise ValueError("Only LoRA adapters can be quantized")
    if "adapter_quantization" in config:
        raise ValueError("The source adapter is already quantized")

    quantized_layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, QuantizedLoRALinear)
    }
    if not quantized_layers:
        raise ValueError("The model has no quantized LoRA linear layers")

    settings = {
        (module.group_size, module.rank_group_size, module.mode)
        for module in quantized_layers.values()
    }
    if len(settings) != 1:
        raise ValueError(
            "All persisted LoRA layers must use the same group sizes and mode"
        )
    group_size, rank_group_size, mode = settings.pop()
    if mode != "affine":
        raise ValueError("Persisted quantized LoRA currently supports affine mode")

    adapter_weights = {
        name: value
        for name, value in tree_flatten(model.parameters())
        if any(part.startswith("lora_") for part in name.split("."))
    }
    if not adapter_weights:
        raise ValueError("No LoRA adapter parameters were found")

    config["adapter_quantization"] = {
        "format": QUANTIZED_LORA_FORMAT,
        "version": QUANTIZED_LORA_VERSION,
        "group_size": group_size,
        "rank_group_size": rank_group_size,
        "mode": mode,
        "layers": {
            name: {
                "bits": module.bits,
                "input_dims": module.input_dims,
                "output_dims": module.output_dims,
                "rank": module.rank,
                "padded_input_dims": module.padded_input_dims,
                "padded_rank": module.padded_rank,
            }
            for name, module in sorted(quantized_layers.items())
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    )
    try:
        mx.save_safetensors(
            str(temporary_path / "adapters.safetensors"),
            adapter_weights,
        )
        (temporary_path / "adapter_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise
    return output_path


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
    with open(adapter_path / "adapter_config.json", "r", encoding="utf-8") as fid:
        config_dict = json.load(fid)
    config = types.SimpleNamespace(**config_dict)
    fine_tune_type = getattr(config, "fine_tune_type", "lora")
    if fine_tune_type != "full":
        linear_to_lora_layers(
            model,
            config.num_layers,
            config.lora_parameters,
            use_dora=(fine_tune_type == "dora"),
        )
    quantization = config_dict.get("adapter_quantization")
    adapter_file = adapter_path / "adapters.safetensors"
    if quantization is not None:
        if quantization.get("format") != QUANTIZED_LORA_FORMAT:
            raise ValueError("Unsupported quantized LoRA adapter format")
        if quantization.get("version") != QUANTIZED_LORA_VERSION:
            raise ValueError("Unsupported quantized LoRA adapter version")
        if quantization.get("mode") != "affine":
            raise ValueError("Persisted quantized LoRA currently supports affine mode")
        layers = _parse_quantized_lora_layers(quantization)
        layer_bits = {name: metadata["bits"] for name, metadata in layers.items()}
        quantize_lora_layers(
            model,
            group_size=int(quantization["group_size"]),
            rank_group_size=int(quantization["rank_group_size"]),
            mode=quantization["mode"],
            layer_bits=layer_bits,
        )
        model_layers = dict(model.named_modules())
        for name, metadata in layers.items():
            layer = model_layers[name]
            actual = {
                "bits": layer.bits,
                "input_dims": layer.input_dims,
                "output_dims": layer.output_dims,
                "rank": layer.rank,
                "padded_input_dims": layer.padded_input_dims,
                "padded_rank": layer.padded_rank,
            }
            if actual != metadata:
                raise ValueError(
                    f"Quantized LoRA layer '{name}' metadata does not match "
                    "the base model"
                )
        adapter_weights = mx.load(str(adapter_file))
        suffixes = (
            "lora_a",
            "lora_a_scales",
            "lora_a_biases",
            "lora_b",
            "lora_b_scales",
            "lora_b_biases",
        )
        missing_weights = [
            f"{name}.{suffix}"
            for name in layer_bits
            for suffix in suffixes
            if f"{name}.{suffix}" not in adapter_weights
        ]
        if missing_weights:
            raise ValueError(
                "Quantized LoRA adapter is missing weights: "
                + ", ".join(missing_weights)
            )
        model.load_weights(list(adapter_weights.items()), strict=False)
    else:
        model.load_weights(str(adapter_file), strict=False)
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
        if isinstance(module, (LoRALinear, QuantizedLoRALinear)):
            reset_layers.append((name, module.linear))
    if len(reset_layers) > 0:
        model.update_modules(tree_unflatten(reset_layers))
    return model


def print_trainable_parameters(model):
    total_p = get_total_parameters(model) / 1e6
    trainable_p = (
        sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 1e6
    )
    rprint(
        f"Trainable parameters: {(trainable_p * 100 / total_p):.3f}% "
        f"({trainable_p:.3f}M/{total_p:.3f}M)"
    )
