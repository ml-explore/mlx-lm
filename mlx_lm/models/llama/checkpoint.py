# Copyright © 2023 Apple Inc.


__all__ = [
    "sanitize",
    "sanitize_language_model",
]


def sanitize_language_model(language_model, weights, prefix):
    """Sanitize the language model's weights, named from prefix: "model." in the
    text model, "model.language_model." in a multimodal one."""
    # Remove unused precomputed rotary freqs
    return {
        k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
    }


def sanitize(model, weights):
    weights = sanitize_language_model(model.model, weights, prefix="model.")
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    return weights
