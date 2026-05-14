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


class MaskedEmbedder(nn.Module):
    """Centroid-clustered logit head.

    Vocab is split into `num_centroids` clusters of `vocab_size_per_centroid`
    tokens each. For each (batch, position) we:
      1. Score all centroids via `self.centroids` (a tiny Linear).
      2. Pick the top-K clusters.
      3. Compute logits only for the tokens in those clusters via a gather
         + matmul against `lm_head_weight` (which is tied with embed_tokens).
      4. Scatter those logits back into a (V,) tensor, with non-selected
         positions filled with `min(selected_logits) - 1.0` so they never win.

    The `token_ordering` buffer maps cluster-ordered positions back to the
    canonical token id space.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        text_config = config.text_config
        self.hidden_size = text_config["hidden_size"]
        self.vocab_size = text_config["vocab_size"]
        self.num_centroids = config.num_centroids
        self.top_k = config.centroid_intermediate_top_k
        assert self.vocab_size % self.num_centroids == 0, (
            f"vocab_size {self.vocab_size} not divisible by "
            f"num_centroids {self.num_centroids}"
        )
        self.vocab_size_per_centroid = self.vocab_size // self.num_centroids

        self.centroids = nn.Linear(self.hidden_size, self.num_centroids, bias=False)
        # token_ordering is a buffer (not trained). Stored as int32 for MLX gather.
        self.token_ordering = mx.zeros((self.vocab_size,), dtype=mx.int32)

    def __call__(self, hidden_states: mx.array, lm_head_weight: mx.array) -> mx.array:
        B, L = hidden_states.shape[:2]
        V = self.vocab_size
        V_pc = self.vocab_size_per_centroid

        # 1. Score centroids: (B, L, num_centroids)
        centroid_logits = self.centroids(hidden_states)

        # 2. Top-K clusters by score: (B, L, top_k)
        # argpartition gives unsorted top-k; that's fine here.
        top_k_indices = mx.argpartition(
            -centroid_logits, kth=self.top_k - 1, axis=-1
        )[..., : self.top_k]

        # 3. canonical_positions_per_cluster: (num_centroids, V_pc)
        canonical = self.token_ordering.reshape(self.num_centroids, V_pc)

        # 4. Gather the V_pc canonical token ids for each of the top_k clusters:
        # selected_canonical: (B, L, top_k, V_pc)
        selected_canonical = canonical[top_k_indices]

        # 5. Gather rows of lm_head_weight at those canonical positions.
        # lm_head_weight: (V, H). Flat index → reshape.
        flat = selected_canonical.reshape(-1)  # (B*L*top_k*V_pc,)
        selected_emb = lm_head_weight[flat].reshape(
            B, L, self.top_k * V_pc, self.hidden_size
        )

        # 6. Dot product: (B, L, 1, H) @ (B, L, H, top_k*V_pc) → (B, L, top_k*V_pc)
        h_exp = mx.expand_dims(hidden_states, -2)               # (B, L, 1, H)
        selected_logits = (h_exp @ selected_emb.swapaxes(-1, -2)).squeeze(-2)

        # 7. Scatter into full-vocab output, with floor-1 mask for non-selected.
        mask_value = mx.min(selected_logits).item() - 1.0
        output = mx.full(
            (B, L, V), mask_value, dtype=hidden_states.dtype
        )
        scatter_idx = selected_canonical.reshape(B, L, -1)      # (B, L, top_k*V_pc)
        return mx.put_along_axis(output, scatter_idx, selected_logits, axis=-1)


class Model(nn.Module):
    """Top-level Gemma 4 assistant model. Filled in by subsequent tasks."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        # Submodules added in Tasks 3–7

    def __call__(self, *a, **kw):
        raise NotImplementedError("Forward pass added in Task 7")
