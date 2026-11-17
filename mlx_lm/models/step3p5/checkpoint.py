# Copyright © 2026 Apple Inc.

__all__ = [
    "sanitize",
]

# Older MLX conversions put the routed and shared experts under mlp.
_MLX_NAMES = (
    (".mlp.switch_mlp.", ".moe."),
    (".mlp.gate.gate.", ".moe.gate."),
    (".mlp.gate.router_bias", ".moe.router_bias"),
    (".mlp.share_expert.", ".share_expert."),
)


def _upgrade_mlx_layout(weights):
    """Older MLX conversions: the checkpoint's names, and the norms stored as
    1 + weight."""
    upgraded = {}
    for k, v in weights.items():
        for old, new in _MLX_NAMES:
            if old in k:
                k = k.replace(old, new)
                break
        if k.endswith(".weight") and "norm" in k:
            v = v - 1
        upgraded[k] = v
    return upgraded


def sanitize(model, weights):
    if any(".mlp.switch_mlp." in k for k in weights):
        weights = _upgrade_mlx_layout(weights)

    # The multi-token prediction layers follow the model's; they are not used.
    sanitized = {}
    for k, v in weights.items():
        if ".mtp" in k:
            continue
        parts = k.split(".")
        if k.startswith("model.layers.") and parts[2].isdigit():
            if int(parts[2]) >= model.args.num_hidden_layers:
                continue
        sanitized[k] = v
    return sanitized
