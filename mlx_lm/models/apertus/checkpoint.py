# Copyright © 2023 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    for k, v in weights.items():
        if k.endswith("alpha_p") or k.endswith("alpha_n"):
            weights[k] = v.squeeze()
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    return weights
