# Copyright © 2026 Apple Inc.

import re

import mlx.core as mx

__all__ = [
    "sanitize",
]

# Norms stored as `w` for a `1 + w` scale; `linear_attn.norm` is not one.
_FOLD_ONE = (
    "q_layernorm.weight",
    "k_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "hc_norm.weight",
    "norm_key.weight",
    "norm_query.weight",
    "norm_conv.weight",
)

_NGRAM_CONSTANTS = (
    "ple_embedding.layer_multipliers",
    "ple_embedding.ngram_heads_vocab_sizes",
    "ple_embedding.ngram_heads_offsets",
)

_NGRAM_SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)\.")


def _merge_linears(
    weights: dict[str, mx.array], parts: list[str], *, target: str
) -> None:
    """Concatenate the layers `parts` into `target` along the output axis,
    with their scales and biases when quantized."""
    for name in ("weight", "scales", "biases"):
        keys = [f"{p}.{name}" for p in parts]
        if all(k in weights for k in keys):
            weights[f"{target}.{name}"] = mx.concatenate(
                [weights.pop(k) for k in keys], axis=-2
            )


def _split_rows(shards: list[mx.array], sizes: list[int]) -> list[mx.array]:
    """Split a table stored as equal row shards into tables of `sizes` rows."""
    rows = shards[0].shape[0]
    tables, start = [], 0
    for size in sizes:
        end = start + size
        first, last = start // rows, (end - 1) // rows
        pieces = [
            s[max(start - i * rows, 0) : min(end - i * rows, rows)]
            for i, s in enumerate(shards[first : last + 1], first)
        ]
        tables.append(mx.concatenate(pieces))
        start = end
    return tables


def sanitize(model, weights: dict[str, mx.array]) -> dict[str, mx.array]:
    # HF and mlx-vlm (flat layout, `shards.{i}` n-gram names) store these norms
    # as `w` for a `1 + w` scale; other checkpoints already hold `1 + w`.
    fold = any(
        k.startswith("model.language_model.")
        or (k.startswith("language_model.") and ".ngram_embedding.shards." in k)
        for k in weights
    )

    out = {}
    for k, v in weights.items():
        # Not in this text-only port: the MTP head and the vision tower.
        if k.startswith(("mtp.", "model.mtp.")):
            continue
        if k.startswith(("model.visual.", "visual.", "vision_tower.")):
            continue

        # Prefixes: HF `model.language_model.`, mlx-vlm `language_model.`.
        if k.startswith("model.language_model."):
            k = "model." + k[len("model.language_model.") :]
        elif k.startswith("language_model."):
            k = k[len("language_model.") :]

        # The n-gram hash constants: check them against the config, then drop.
        if k.endswith(_NGRAM_CONSTANTS):
            model._check_ngram_constant(k, v)
            continue

        # Rename the n-gram shards to `shards.{i}`, as mlx-vlm saves them.
        k = _NGRAM_SHARD_RE.sub(r".ngram_embedding.shards.\1.", k)

        # The experts stack as (E, out, in); gate_up_proj holds gate, then up.
        if k.endswith("mlp.experts.gate_up_proj"):
            base = k[: -len("experts.gate_up_proj")]
            mid = v.shape[-2] // 2
            out[base + "switch_mlp.gate_proj.weight"] = v[..., :mid, :]
            out[base + "switch_mlp.up_proj.weight"] = v[..., mid:, :]
            continue
        if k.endswith("mlp.experts.down_proj"):
            base = k[: -len("experts.down_proj")]
            out[base + "switch_mlp.down_proj.weight"] = v
            continue

        # torch (C, 1, K) to mlx (C, K, 1); a converted weight has shape[1] > 1.
        if k.endswith("conv1d.weight") and v.ndim == 3 and v.shape[1] == 1:
            v = v.transpose(0, 2, 1)
        if fold and k.endswith(_FOLD_ONE):
            v = 1.0 + v
        out[k] = v

    # Merge the checkpoint's block inject into the mix down.
    for k in [k for k in out if k.endswith("block_inject_weight.weight")]:
        p = k[: -len("block_inject_weight.weight")]
        down = p + "input_mix_weight_down"
        _merge_linears(out, [down, p + "block_inject_weight"], target=down)

    # The checkpoint has equal row shards; the model keeps one table per head.
    for k in [k for k in out if k.endswith("ngram_embedding.shards.0.weight")]:
        p = k[: -len("shards.0.weight")]
        n = sum(q.startswith(p + "shards.") and q.endswith(".weight") for q in out)
        layer = int(k.split(".layers.")[1].split(".")[0])
        sizes = model.model.layers[layer].ple.ple_embedding.head_vocab_sizes
        # A quantized table splits its scales and biases by the same rows.
        for name in ("weight", "scales", "biases"):
            if f"{p}shards.0.{name}" in out:
                shards = [out.pop(f"{p}shards.{i}.{name}") for i in range(n)]
                for h, table in enumerate(_split_rows(shards, sizes)):
                    out[f"{p}{h}.{name}"] = table

    return out
