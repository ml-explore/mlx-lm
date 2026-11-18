# Copyright © 2025 Apple Inc.

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    # Check if needs sanitization
    c1d = weights["model.layers.0.mamba.conv1d.weight"]
    if c1d.shape[-1] <= c1d.shape[1]:
        return weights

    sanitized_weights = {}
    args = model.args

    for name, param in weights.items():
        # Fold-in multipliers
        if name.endswith("embed_tokens.weight"):
            param *= args.embedding_multiplier
        elif name.endswith("lm_head.weight"):
            param *= args.lm_head_multiplier
        elif name.endswith("q_proj.weight") or name.endswith("k_proj.weight"):
            param *= args.attention_in_multiplier
        elif name.endswith("key_proj.weight"):
            param *= args.attention_in_multiplier * args.key_multiplier
        elif name.endswith("o_proj.weight"):
            param *= args.attention_out_multiplier
        elif name.endswith("out_proj.weight"):
            param *= args.ssm_out_multiplier
        elif name.endswith("gate_proj.weight"):
            param *= args.mlp_multipliers[0]
        elif name.endswith("down_proj.weight"):
            param *= args.mlp_multipliers[1]
        elif name.endswith("in_proj.weight"):
            param *= (
                args.ssm_in_multiplier
                * model.model._mup_vector.astype(param.dtype)[:, None]
            )
        elif "conv1d.weight" in name:
            param = param.transpose(0, 2, 1)
        sanitized_weights[name] = param
    return sanitized_weights
