# Copyright © 2026 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def _is_mtp_weight(key: str, num_hidden_layers: int) -> bool:
    """The multi-token-prediction module is stored as the layer after the last."""
    prefix = "model.layers."
    if not key.startswith(prefix):
        return False
    layer_index = key[len(prefix) :].split(".", 1)[0]
    return layer_index.isdigit() and int(layer_index) >= num_hidden_layers


def sanitize(model, weights):
    args = model.args
    if args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    # Generation does not use the MTP module (it drafts token t+2 in training).
    weights = {
        k: v
        for k, v in weights.items()
        if not _is_mtp_weight(k, args.num_hidden_layers)
    }

    mla = args.mla_config
    for l in range(args.num_hidden_layers):
        prefix = f"model.layers.{l}"

        # The routed experts are stacked, with gate and up in one tensor.
        gate_up = weights.pop(f"{prefix}.mlp.experts.gate_up_proj", None)
        if gate_up is not None:
            gate, up = mx.split(gate_up, 2, axis=1)
            weights[f"{prefix}.mlp.switch_mlp.gate_proj.weight"] = gate
            weights[f"{prefix}.mlp.switch_mlp.up_proj.weight"] = up
        down = weights.pop(f"{prefix}.mlp.experts.down_proj", None)
        if down is not None:
            weights[f"{prefix}.mlp.switch_mlp.down_proj.weight"] = down

        # Split kv_b_proj into the per-head key and value up-projections.
        kv_b = weights.pop(f"{prefix}.self_attn.kv_b_proj.weight", None)
        if kv_b is not None:
            nope = mla["qk_nope_head_dim"]
            kv_b = kv_b.reshape(mla["num_heads"], nope + mla["v_head_dim"], -1)
            weights[f"{prefix}.self_attn.embed_q.weight"] = mx.contiguous(
                kv_b[:, :nope, :].swapaxes(-1, -2)
            )
            weights[f"{prefix}.self_attn.unembed_out.weight"] = mx.contiguous(
                kv_b[:, nope:, :]
            )

    # Depthwise convolutions: [channels, 1, kernel] -> [channels, kernel, 1].
    for k, v in weights.items():
        if "conv1d.weight" in k and v.shape[-1] != 1:
            weights[k] = v.moveaxis(2, 1)
    return weights
