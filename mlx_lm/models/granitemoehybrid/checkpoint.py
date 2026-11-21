# Copyright © 2025 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    # Handle conv1d weights
    for k, v in weights.items():
        if "conv1d.weight" in k and v.shape[-1] != 1:
            weights[k] = v.moveaxis(2, 1)

    # Handle MoE weight transformation to SwitchGLU format (only for MoE models)
    if (
        model.args.use_moe
        and "model.layers.0.block_sparse_moe.input_linear.weight" in weights
    ):
        for l in range(model.args.num_hidden_layers):
            prefix = f"model.layers.{l}.block_sparse_moe"

            input_weight = weights.pop(f"{prefix}.input_linear.weight")
            _, expert_hidden, _ = input_weight.shape

            # Split into gate and up projections (each half of expert_hidden)
            gate_proj = input_weight[:, : expert_hidden // 2, :]
            up_proj = input_weight[:, expert_hidden // 2 :, :]
            weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_proj
            weights[f"{prefix}.switch_mlp.up_proj.weight"] = up_proj

            weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                f"{prefix}.output_linear.weight"
            )

    # Handle dense MLP weight transformation (for dense models)
    elif (
        not model.args.use_moe
        and "model.layers.0.shared_mlp.input_linear.weight" in weights
    ):
        for l in range(model.args.num_hidden_layers):
            prefix = f"model.layers.{l}.shared_mlp"

            # Transform shared_mlp weights to standard mlp weights
            input_weight = weights.pop(f"{prefix}.input_linear.weight")
            # Split into gate and up projections (each half)
            gate_proj, up_proj = mx.split(input_weight, 2, axis=0)
            weights[f"model.layers.{l}.mlp.gate_proj.weight"] = gate_proj
            weights[f"model.layers.{l}.mlp.up_proj.weight"] = up_proj

            weights[f"model.layers.{l}.mlp.down_proj.weight"] = weights.pop(
                f"{prefix}.output_linear.weight"
            )

    # Some checkpoints ship a redundant lm_head.weight even though
    # embeddings are tied and this model never instantiates lm_head.
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    return weights
