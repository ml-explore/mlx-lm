# Copyright © 2023 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    # Remove unused precomputed rotary freqs
    return {
        k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
    }
