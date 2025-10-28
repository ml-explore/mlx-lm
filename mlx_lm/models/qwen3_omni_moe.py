# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from . import qwen3_moe
from .base import BaseModelArgs
from .qwen3_moe import create_attention_mask


@dataclass
class ModelArgs(BaseModelArgs):
    """
    Arguments for Qwen3-Omni-MoE model.

    Qwen3-Omni-MoE is a multimodal model that contains text, audio, and vision
    components. This implementation only supports text generation.

    Attributes:
        model_type (str): The model type identifier
        thinker_config (dict): Configuration for the text language model component
    """

    model_type: str
    thinker_config: dict


class Model(nn.Module):
    """
    Qwen3-Omni-MoE text language model wrapper for MLX.

    This model wraps the Qwen3-MoE text model and handles weight loading from
    the full Qwen3-Omni-MoE checkpoint, which includes multimodal components
    (audio_tower, talker, code2wav) that are filtered out during conversion.

    The model supports input_embeddings for multimodal use cases where embeddings
    are pre-computed externally (e.g., audio embeddings from a separate audio encoder).

    Example:
        >>> # Load model
        >>> model, tokenizer = load("pherber3/Qwen3-Omni-30B-A3B-Instruct-4bit-mlx")

        >>> # Text-only generation
        >>> output = model(input_ids, cache=None)

        >>> # With pre-computed embeddings (e.g., audio)
        >>> output = model(input_ids, cache=None, input_embeddings=audio_embeddings)
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        # Nested to match Qwen3-Omni structure & allow inclusion of other modalities downstream
        self.language_model = qwen3_moe.Model(
            qwen3_moe.ModelArgs.from_dict(args.thinker_config["text_config"])
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        """
        Forward pass through the model.

        Args:
            inputs: Input token IDs of shape (seq_len,) or (batch, seq_len)
            cache: Optional KV cache for efficient generation
            input_embeddings: Optional pre-computed embeddings of shape (seq_len, hidden_dim).
                            When provided, bypasses the embedding layer and directly feeds
                            these embeddings to the transformer layers. Useful for multimodal
                            inputs where embeddings are computed externally.

        Returns:
            Logits of shape (seq_len, vocab_size) or (batch, seq_len, vocab_size)
        """
        if input_embeddings is not None:
            # Multimodal path: use pre-computed embeddings
            h = input_embeddings

            # Initialize cache if needed
            if cache is None:
                cache = [None] * len(self.language_model.model.layers)

            # Create attention mask
            mask = create_attention_mask(h, cache[0])

            # Forward through transformer layers
            for layer, c in zip(self.language_model.model.layers, cache):
                h = layer(h, mask, c)

            # Apply final norm and language model head
            h = self.language_model.model.norm(h)
            return self.language_model.lm_head(h)
        else:
            # Standard text-only path
            return self.language_model(inputs, cache=cache)

    def sanitize(self, weights):
        """
        Sanitize weights from HuggingFace checkpoint to MLX format.

        This handles two scenarios:
        1. HuggingFace format: Weights have 'thinker.*' prefix and include multimodal
           components (audio_tower, talker, etc.) that need to be filtered out
        2. MLX format: Weights already converted and sanitized

        Args:
            weights: Dictionary of model weights

        Returns:
            Dictionary of sanitized weights with 'language_model.*' prefix
        """
        # Check if already MLX-converted with language_model prefix
        has_language_model_prefix = any(
            k.startswith("language_model.") for k in weights.keys()
        )
        if has_language_model_prefix:
            # Already in final format, no sanitization needed
            return weights

        # Check if this is HuggingFace format (has thinker.* prefix)
        has_thinker_prefix = any(k.startswith("thinker.") for k in weights.keys())

        if has_thinker_prefix:
            # HuggingFace format: Strip thinker.* prefix and filter out non-text components
            sanitized = {}
            for key, value in weights.items():
                if key.startswith("thinker.lm_head.") or key.startswith(
                    "thinker.model."
                ):
                    # Keep text model weights, remove thinker prefix
                    new_key = key.replace("thinker.", "")
                    sanitized[new_key] = value
                elif key.startswith(
                    (
                        "talker.",  # TTS generation component
                        "code_predictor.",  # Audio code prediction
                        "code2wav.",  # Audio waveform generation
                        "thinker.visual.",  # Vision encoder
                        "thinker.audio_tower.",  # Audio encoder
                    )
                ):
                    # Skip multimodal components not needed for text-only MLX model
                    continue
                else:
                    # Keep other weights as-is
                    sanitized[key] = value

            # Call parent qwen3_moe sanitize to handle MoE expert consolidation
            sanitized = self.language_model.sanitize(sanitized)
        else:
            # MLX format: Already converted, pass through
            sanitized = weights

        # Add language_model.* prefix to match wrapper structure
        final_weights = {}
        for key, value in sanitized.items():
            if key.startswith("language_model."):
                # Already has prefix (e.g., from previous MLX conversion)
                final_weights[key] = value
            else:
                # Add prefix
                final_weights[f"language_model.{key}"] = value

        return final_weights

    @property
    def quant_predicate(self):
        """Predicate for determining which layers to quantize."""
        return self.language_model.quant_predicate

    @property
    def layers(self):
        """Access to transformer layers for cache management."""
        return self.language_model.model.layers
