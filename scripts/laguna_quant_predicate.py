"""Custom per-module quantization predicate for Laguna's requested mixed
variants: attention/embeddings/output at one bit width, with routed experts
at a lower one (2, 4, or 6). The three uniform variants
(4bit/6bit/8bit-everything) need no custom predicate — use
`mlx_lm.convert(..., quantize=True, q_bits=4|6|8)` directly.

Note on the router: `Router.weight` (`mlx_lm/models/laguna.py`) is a raw
array, not an `nn.Linear`, so it has no `to_quantized`.
`mlx_lm.utils.quantize_model`'s wrapper checks `hasattr(module,
"to_quantized")` before ever calling a custom `quant_predicate`, so the
router (and the model's RMSNorm layers) are always left at full precision
regardless of what a predicate here returns for their paths -- there is no
way to quantize the router via this mechanism today, and no need to
special-case it above. Concretely, this means a "protect the router, shrink
everything else" recipe needs no custom predicate at all: plain
`mlx_lm.convert(..., quantize=True, q_bits=6)` already leaves the router at
full precision (stricter than 8-bit) while quantizing attention and routed
experts uniformly to 6-bit.

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
        # projections, embeddings, lm_head, the shared expert) stays at
        # attention_bits. The router never reaches this predicate at all --
        # see the module docstring -- so it needs no special-casing here.
        if "switch_mlp" in path:
            return {"group_size": group_size, "bits": expert_bits, "mode": "affine"}
        return {"group_size": group_size, "bits": attention_bits, "mode": "affine"}

    return predicate
