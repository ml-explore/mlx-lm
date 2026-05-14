# Copyright © 2026 Apple Inc.
"""
Gemma 4 MTP drafter (assistant) model.

This is the speculative-decoding companion released alongside Gemma 4. The
drafter has NO key/value projections of its own — at each layer it cross-
attends to the target model's K/V via `shared_kv_states`. See the spec at
core/life/docs/superpowers/specs/2026-05-13-mlx-lm-gemma4-assistant.md
(broomva workspace) or the HF reference at
transformers/models/gemma4_assistant/modeling_gemma4_assistant.py.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4_assistant"
    backbone_hidden_size: int = 2560
    num_centroids: int = 2048
    centroid_intermediate_top_k: int = 32
    use_ordered_embeddings: bool = True
    tie_word_embeddings: bool = True
    text_config: Dict[str, Any] = field(default_factory=dict)
    vocab_size: int = 262144  # echoed at top level for convenience

    def __post_init__(self):
        # Mirror gemma4.ModelArgs: vocab_size flows down into text_config
        if "vocab_size" not in self.text_config:
            self.text_config["vocab_size"] = self.vocab_size
        # Defaults for fields the HF config sometimes elides
        self.text_config.setdefault("num_attention_heads", 4)
        self.text_config.setdefault("num_key_value_heads", 2)
        self.text_config.setdefault("head_dim", 256)


class Model(nn.Module):
    """Top-level Gemma 4 assistant model. Filled in by subsequent tasks."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        # Submodules added in Tasks 3–7

    def __call__(self, *a, **kw):
        raise NotImplementedError("Forward pass added in Task 7")
