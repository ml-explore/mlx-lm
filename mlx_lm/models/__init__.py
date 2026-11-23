# Copyright © 2024 Apple Inc.

from typing import Any

from mlx_lm.plugins import lookup_name

__all__ = [
    "MODEL_ARCHITECTURE_REMAPPING",
    "MODEL_REMAPPING",
    "package_name",
]

# TODO: Make these tables read-only. Code outside mlx-lm still changes them.
MODEL_ARCHITECTURE_REMAPPING: dict[tuple[str, str], str] = {
    # Ling-3.x reuses Ling-2.6's model type.
    ("bailing_hybrid", "BailingMoeV3ForCausalLM"): "bailing_moe_v3",
}

MODEL_REMAPPING: dict[str, str] = {
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


def package_name(config: dict[str, Any]) -> str:
    """The lookup name of the package that runs a configuration's model."""
    model_type = lookup_name(config["model_type"])
    architectures = config.get("architectures") or ()
    if isinstance(architectures, str):
        architectures = (architectures,)
    for architecture in architectures:
        if (model_type, architecture) in MODEL_ARCHITECTURE_REMAPPING:
            return MODEL_ARCHITECTURE_REMAPPING[model_type, architecture]
    return MODEL_REMAPPING.get(model_type, model_type)
