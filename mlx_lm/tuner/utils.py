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

    keys = config.get("keys", None)
    if keys is not None:
        keys = set(keys)
    elif model.model_type in {
        "mistral",
        "mistral3",
        "llama",
        "lfm2",
        "phi",
        "mixtral",
        "nemotron",
        "stablelm",
        "hunyuan",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "phimoe",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "granite",
        "helium",
        "starcoder2",
        "cohere",
        "cohere2",
        "minicpm",
        "minicpm3",
        "minicpm4",
        "deepseek",
        "olmo2",
        "olmoe",
        "internlm3",
        "glm4",
        "mimo",
        "ernie4_5",
        "dots1",
        "smollm3",
        "exaone4",
        "hunyuan_v1_dense",
        "gpt_oss",
        "ernie4_5_moe",
        "granitemoe",
        "longcat_flash",
        "seed_oss",
        "apertus",
    }:
        keys = {"self_attn.q_proj", "self_attn.v_proj"}
        if model.model_type in ["mixtral", "phimoe"]:
            keys.add("block_sparse_moe.gate")
        if model.model_type == "qwen2_moe":
            keys.add("mlp.gate")
            keys.add("mlp.shared_expert_gate")
        if model.model_type in ["olmoe", "qwen3_moe", "dots1"]:
            keys.add("mlp.gate")
        if model.model_type in ["longcat_flash"]:
            keys.add("mlp.router.classifier")
    elif model.model_type == "gpt_bigcode":
        keys = {"attn.c_attn"}
    elif model.model_type == "gpt2":
        keys = {"attn.c_attn"}
    elif model.model_type == "gpt_neox":
        keys = {"attention.query_key_value"}
    elif model.model_type == "olmo":
        keys = {"att_proj"}
    elif model.model_type == "openelm":
        keys = {"attn.qkv_proj"}
    elif model.model_type == "phi3":
        keys = {"self_attn.qkv_proj"}
    elif model.model_type == "phi-msft":
        keys = {"mixer.Wqkv", "moe.gate"}
    elif model.model_type == "dbrx":
        keys = {"norm_attn_norm.attn.Wqkv", "ffn.router.layer"}
    elif model.model_type == "internlm2":
        keys = {"attention.wqkv", "attention.wo"}
    elif model.model_type in {
        "deepseek_v2",
        "deepseek_v3",
        "longcat_flash",
        "minicpm3",
    }:
        keys = {
            "self_attn.q_proj",
            "self_attn.q_a_proj",
            "self_attn.q_b_proj",
            "self_attn.kv_a_proj_with_mqa",
            "self_attn.kv_b_proj",
        }
    elif model.model_type == "mamba":
        keys = {
            "mixer.in_proj",
            "mixer.x_proj",
            "mixer.dt_proj",
            "mixer.out_proj",
        }
    elif model.model_type == "exaone":
        keys = {"attn.attention.q_proj", "attn.attention.v_proj"}
    elif model.model_type == "bailing_moe":
        keys = {"attention.query_key_value", "attention.dense"}
    elif model.model_type == "nemotron_h":
        keys.add("mixer.in_proj")
        keys.add("mixer.out_proj")
        keys.add("mixer.q_proj")
        keys.add("mixer.k_proj")
        keys.add("mixer.v_proj")
        keys.add("mixer.o_proj")
    else:
        raise ValueError(f"Lora does not support {model.model_type}")

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
