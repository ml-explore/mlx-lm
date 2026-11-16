# Copyright © 2023 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    return {
        k: v
        for k, v in weights.items()
        if "self_attn.rotary_emb.inv_freq" not in k
        and not k.startswith("model.mtp_layers.")
    }
