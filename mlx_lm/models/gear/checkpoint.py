# Copyright © 2026 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    sanitized_weights = {}
    for name, param in weights.items():
        if "rotary_emb.inv_freq" in name or name.endswith("embed_scale"):
            continue
        if model.args.tie_word_embeddings and name == "lm_head.weight":
            continue
        if name.endswith(("key_conv.weight", "value_conv.weight")):
            if len(param.shape) == 3 and param.shape[1] == 1:
                param = param.transpose(0, 2, 1)
        sanitized_weights[name] = param
    return sanitized_weights
