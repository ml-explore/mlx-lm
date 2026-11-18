# Copyright © 2025 Apple Inc.

__all__ = [
    "sanitize",
]


def sanitize(model, weights: dict):
    weights = {
        k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
    }
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    return weights
