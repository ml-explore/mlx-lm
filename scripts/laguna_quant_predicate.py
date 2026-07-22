"""Custom per-module quantization predicate for Laguna's requested mixed
variants: attention/router/embeddings/output at one bit width, with routed
experts at a lower one (2, 4, or 6). The three uniform variants
(4bit/6bit/8bit-everything) need no custom predicate — use
`mlx_lm.convert(..., quantize=True, q_bits=4|6|8)` directly.

Usage:
    from mlx_lm.convert import convert
    from laguna_quant_predicate import build_laguna_quant_predicate

    # 8-bit attention, 4-bit experts
    convert(
        hf_path="poolside/Laguna-S-2.1",
        mlx_path="Laguna-S-2.1-8bit-Att-4bit-Ex",
        quantize=True,
        q_group_size=64,
        quant_predicate=build_laguna_quant_predicate(expert_bits=4),
        upload_repo="poolside/Laguna-S-2.1-8bit-Att-4bit-Ex-mlx",
    )

    # 4-bit attention, 2-bit experts (routed experts hold ~116B of Laguna
    # S-2.1's ~117.6B stored parameters, so this is the variant that
    # actually shrinks a 118B checkpoint to fit a 64 GB Mac)
    convert(
        hf_path="poolside/Laguna-S-2.1",
        mlx_path="Laguna-S-2.1-4bit-Att-2bit-Ex",
        quantize=True,
        q_group_size=64,
        quant_predicate=build_laguna_quant_predicate(
            expert_bits=2, attention_bits=4
        ),
        upload_repo="poolside/Laguna-S-2.1-4bit-Att-2bit-Ex-mlx",
    )
"""

from typing import Callable, Union

import mlx.nn as nn


def build_laguna_quant_predicate(
    expert_bits: int, group_size: int = 64, attention_bits: int = 8
) -> Callable[[str, nn.Module], Union[bool, dict]]:
    if expert_bits not in (2, 4, 6):
        raise ValueError(
            f"Laguna's requested variants use 2, 4, or 6-bit experts, got {expert_bits}"
        )

    def predicate(path: str, module: nn.Module) -> Union[bool, dict]:
        # Routed experts (SwitchGLU's three projections) get the lower bit
        # width poolside requested; everything else quantizable (attention
        # projections, the router's `weight` matmul, embeddings, lm_head,
        # the shared expert) stays at attention_bits. The router's
        # `e_score_correction_bias` is excluded from casting entirely via
        # `Model.cast_predicate`, and `mx.quantize` only touches weight
        # matrices, not that bias vector, so it needs no special-casing here.
        if "switch_mlp" in path:
            return {"group_size": group_size, "bits": expert_bits, "mode": "affine"}
        return {"group_size": group_size, "bits": attention_bits, "mode": "affine"}

    return predicate
