# Copyright © 2023 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    new_weights = {}

    for w_key, w_value in weights.items():
        # Created through register_buffer in Pytorch, not needed here.
        ignore_suffixes = [
            ".attention.bias",
            ".attention.masked_bias",
            ".attention.rotary_emb.inv_freq",
        ]

        skip_weight = False
        for ignored_suffix in ignore_suffixes:
            if w_key.endswith(ignored_suffix):
                skip_weight = True
                break

        if skip_weight:
            continue

        if not w_key.startswith("model."):
            w_key = f"model.{w_key}"

        w_key = w_key.replace(".gpt_neox.layers.", ".h.")
        w_key = w_key.replace(".gpt_neox.", ".")

        new_weights[w_key] = w_value

    return new_weights
