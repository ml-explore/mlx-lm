# Copyright © 2026 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models import llama, ministral3, mistral4
from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
    "language_model_package",
]

# Mistral's defaults, which transformers leaves out of the text_config it saves.
_MISTRAL_DEFAULTS = {
    "vocab_size": 32000,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 4096 * 32,
    "rms_norm_eps": 1e-6,
}


def language_model_package(model_type):
    """The package of the language model: Ministral 3's, Mistral 4's, or Llama's,
    which runs the Mistral and Llama language models."""
    if model_type == "ministral3":
        return ministral3
    if model_type == "mistral4":
        return mistral4
    return llama


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    tie_word_embeddings: bool = False

    def __post_init__(self):
        # Hugging Face's checkpoints keep tie_word_embeddings at the top level.
        text_config = {
            "tie_word_embeddings": self.tie_word_embeddings,
            **self.text_config,
        }
        if text_config.get("model_type") == "mistral":
            text_config = {**_MISTRAL_DEFAULTS, **text_config}
        package = language_model_package(text_config.get("model_type"))
        self.text = package.ModelArgs.from_dict(text_config)
