# Copyright © 2026 Apple Inc.

from mlx_lm.models.longcat_flash import checkpoint as longcat_flash_checkpoint

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    weights = longcat_flash_checkpoint.sanitize(model, weights)
    if "model.embed_tokens.weight" in weights:
        weights["model.ngram_embeddings.word_embeddings.weight"] = weights.pop(
            "model.embed_tokens.weight"
        )
    return weights
