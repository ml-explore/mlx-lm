# Copyright © 2024 Apple Inc.

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from mlx_lm.plugins import lookup_name

__all__ = [
    "MODEL_ARCHITECTURE_REMAPPING",
    "MODEL_REMAPPING",
    "package_name",
]

MODEL_REMAPPING: Mapping[str, str] = MappingProxyType(
    {
        "mistral": "llama",
        "llava": "mistral3",
        "pixtral": "mistral3",
        "phi_msft": "phixtral",
        "falcon_mamba": "mamba",
        "joyai_llm_flash": "deepseek_v3",
        "kimi_k2": "deepseek_v3",
        "qwen2_5_vl": "qwen2_vl",
        "minimax_m2": "minimax",
        "iquestcoder": "llama",
        "xverse": "llama",
        "gemma4_unified": "gemma4",  # encoder-free multimodal variant; vision/audio weights stripped by sanitize()
        "deepseek_v41_text": "deepseek_v41",
        "qwen3_5_moe": "qwen3_5",
        "qwen3_vl_moe": "qwen3_vl",
        "solar_open": "glm4_moe",
    }
)

MODEL_ARCHITECTURE_REMAPPING: Mapping[str, str] = MappingProxyType(
    {
        # Ling-3.x reuses Ling-2.6's model type.
        "BailingMoeV3ForCausalLM": "bailing_moe_v3",
    }
)


def package_name(config: dict[str, Any]) -> str:
    """The name of the package, built-in or plug-in, that runs a config's model."""
    for architecture in _architectures(config):
        if architecture in MODEL_ARCHITECTURE_REMAPPING:
            return MODEL_ARCHITECTURE_REMAPPING[architecture]
    model_type = lookup_name(config["model_type"])
    return MODEL_REMAPPING.get(model_type, model_type)


def _architectures(config: dict[str, Any]) -> tuple[str, ...]:
    """The Hugging Face architectures of a configuration."""
    architectures = config.get("architectures") or ()
    if isinstance(architectures, str):
        return (architectures,)
    return tuple(architectures)
