# Copyright © 2026 Apple Inc.


__all__ = [
    "sanitize",
    "sanitize_text_model",
]


def sanitize_text_model(text_model, weights):
    has_unsanitized_conv1d = any(
        "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
    )
    weights = {k: v for k, v in weights.items() if "mtp." not in k}

    if text_model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    norm_keys = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )
    for k, v in weights.items():
        if "conv1d.weight" in k and v.shape[-1] != 1:
            weights[k] = v.moveaxis(2, 1)
        if has_unsanitized_conv1d and any(k.endswith(sfx) for sfx in norm_keys):
            if v.ndim == 1:
                weights[k] = v + 1.0
    return weights


def sanitize(model, weights):
    sanitized = {}
    for key, value in weights.items():
        if key.startswith("vision_tower") or key.startswith("model.visual"):
            continue
        if key.startswith("model.visual"):
            continue
        if key.startswith("model.language_model"):
            key = key.replace("model.language_model", "language_model.model")
        elif key.startswith("language_model."):
            pass
        else:
            key = "language_model." + key
        sanitized[key] = value
    return model.language_model.sanitize(sanitized)
