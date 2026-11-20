# Copyright © 2026 Apple Inc.

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    # Some MLX conversions rename the embedding to embed_tokens while the
    # Hugging Face checkpoint ships it as model.embedding.weight.
    if (
        "model.embed_tokens.weight" in weights
        and "model.embedding.weight" not in weights
    ):
        weights["model.embedding.weight"] = weights.pop("model.embed_tokens.weight")
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    # Drop the serialized RoPE inverse-frequency buffer some conversions ship.
    return {
        name: value
        for name, value in weights.items()
        if "rotary_emb.inv_freq" not in name
    }
