# Copyright © 2025 Apple Inc.

"""SmolLM3 model – thin wrapper re-using Llama blocks with optional *NoPE*."""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from . import llama


@dataclass
class ModelArgs(llama.ModelArgs):
    """Extends the standard Llama arguments with SmolLM-3 specifics.

    The main additional feature is the *NoPE* pattern: every *n*-th transformer
    layer can opt-out of rotary position embeddings.  The interval (or an
    explicit list) is parsed from the Hugging Face configuration so that
    checkpoints load without further changes.
    """

    # We keep model_type positional to satisfy dataclass ordering rules and
    # assign the canonical value inside ``__post_init__`` if it wasn't passed.
    model_type: str

    # --- NoPE (No-Rotary-Pos-Embedding) configuration -------------------
    no_rope_layer_interval: int = 4              # every 4th layer by default
    no_rope_layers: Optional[list[int]] = None   # explicit pattern (0/1 per layer)

    def __post_init__(self):
        super().__post_init__()

        self.model_type = self.model_type or "smollm3"

        if self.no_rope_layers is None:
            self.no_rope_layers = [
                int((i + 1) % self.no_rope_layer_interval != 0)
                for i in range(self.num_hidden_layers)
            ]
        elif len(self.no_rope_layers) != self.num_hidden_layers:
            raise ValueError("`no_rope_layers` length mismatch")


class NoPE(nn.Module):
    """No-op RoPE used to disable rotary embeddings in selected layers."""

    def __call__(self, x, offset: int = 0):  # noqa: D401, D403
        return x


class Model(nn.Module):
    """Wrapper around Llama that respects *NoPE* layers in SmolLM-3."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type: str = args.model_type

        # Build underlying language model directly (no extra 'model' nesting)
        self.model = llama.LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        # ------------------------------------------------------------------
        # Patch rotary embeddings for layers that opt-out (NoPE)
        # ------------------------------------------------------------------
        identity_rope = NoPE()
        for idx, use_rope in enumerate(args.no_rope_layers):
            if not use_rope:
                # Navigate to the attention module inside the layer and swap
                # the `rope` object with an identity mapping.
                try:
                    attn = self.model.layers[idx].self_attn
                    attn.rope = identity_rope
                except (AttributeError, IndexError):
                    # Should never happen unless the internal structure of
                    # `llama.Model` changes.
                    raise RuntimeError(
                        f"Unable to patch NoPE for layer {idx}: layout mismatch"
                    )

    # ------------------------------------------------------------------
    # Forward pass simply delegates to the underlying model
    # ------------------------------------------------------------------
    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        # Forward identical to llama.Model wrapper
        out = self.model(inputs, mask, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    # ------------------------------------------------------------------
    # Convenience proxies
    # ------------------------------------------------------------------
    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights: dict):
        """Remove parameters that are unused in the MLX implementation."""
        # Drop pre-computed rotary frequencies that exist in Transformers ckpts
        weights = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
        # If embeddings are tied, the checkpoint still contains a separate lm_head
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights 