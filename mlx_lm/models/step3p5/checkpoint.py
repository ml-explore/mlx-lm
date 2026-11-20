# Copyright © 2026 Apple Inc.

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    remappings = [
        (".moe.gate_proj.", ".mlp.switch_mlp.gate_proj."),
        (".moe.up_proj.", ".mlp.switch_mlp.up_proj."),
        (".moe.down_proj.", ".mlp.switch_mlp.down_proj."),
        (".moe.gate.", ".mlp.gate.gate."),
        (".moe.router_bias", ".mlp.gate.router_bias"),
        (".share_expert.", ".mlp.share_expert."),
    ]

    is_vanilla = any(
        src in k and dst not in k for k in weights for src, dst in remappings
    )

    new_weights = {}
    for k, v in weights.items():
        if ".mtp" in k:
            continue
        if "model.layers." in k:
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                if int(parts[2]) >= model.args.num_hidden_layers:
                    continue

        for src, dst in remappings:
            if src in k and dst not in k:
                k = k.replace(src, dst)
                break

        if is_vanilla and k.endswith(".weight") and "norm" in k:
            v = v + 1

        new_weights[k] = v

    return new_weights
