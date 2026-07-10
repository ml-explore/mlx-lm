# Copyright © 2026 Apple Inc.

# GLM-4.5V / GLM-4.6V (`glm4v_moe` / Glm4vMoeForConditionalGeneration), text-only.
# The text tower (`text_config`, model_type glm4v_moe_text) is architecturally
# GLM-4.5/4.6 MoE (glm4_moe): same attention (biased qkv, partial rotary 0.5,
# rotate-half/non-traditional RoPE) and the same sigmoid + e_score_correction_bias
# `noaux_tc` router with a shared expert and first_k_dense_replace.
#
# GLM-V uses 3D m-rope (mrope_section over temporal/height/width position grids)
# for vision, but for text-only decoding it degenerates exactly to standard 1D
# RoPE: transformers' Glm4vMoeTextModel builds position_ids as a plain arange
# expanded identically over the 3 grids, and apply_mrope's chunk[i % 3] selection
# then reproduces the 1D frequencies verbatim (see modeling_glm4v_moe.py,
# Glm4vMoeTextRotaryEmbedding.apply_mrope and the position_ids construction in
# Glm4vMoeTextModel.forward). So we reuse glm4_moe wholesale; this module only
# flattens text_config onto glm4_moe's ModelArgs and strips the vision tower.

from dataclasses import dataclass, field
from typing import Any, Dict

from . import glm4_moe
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "glm4v_moe"
    text_config: Dict[str, Any] = field(default_factory=dict)
    tie_word_embeddings: bool = False

    def flat_text_args(self) -> glm4_moe.ModelArgs:
        cfg = dict(self.text_config)
        rope_params = cfg.pop("rope_parameters", None) or {}

        rope_type = rope_params.get("rope_type", rope_params.get("type", "default"))
        if rope_type not in (None, "default"):
            raise NotImplementedError(
                f"glm4v_moe text tower: unsupported rope_type {rope_type!r}; "
                "only 'default' (plain RoPE / degenerate text m-rope) is supported."
            )

        cfg.setdefault("rope_theta", rope_params.get("rope_theta", 10000.0))
        prf = cfg.get(
            "partial_rotary_factor", rope_params.get("partial_rotary_factor", 1.0)
        )
        rope_prf = rope_params.get("partial_rotary_factor", prf)
        if prf != rope_prf:
            raise ValueError(
                "glm4v_moe text tower: partial_rotary_factor mismatch between "
                f"text_config ({prf}) and rope_parameters ({rope_prf})."
            )
        cfg["partial_rotary_factor"] = prf
        cfg.setdefault("rope_scaling", None)
        cfg.setdefault("tie_word_embeddings", self.tie_word_embeddings)
        cfg.setdefault("use_qk_norm", False)
        return glm4_moe.ModelArgs.from_dict(cfg)


class Model(glm4_moe.Model):
    def __init__(self, args: ModelArgs):
        super().__init__(args.flat_text_args())
        self.model_type = args.model_type

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            # Vision tower + projector (unused in text-only decoding).
            if k.startswith(
                ("vision_tower.", "visual.", "model.visual.", "model.vision_tower.")
            ):
                continue
            # mlx-vlm-style nesting: language_model.{model,lm_head}.* -> {model,lm_head}.*
            if k.startswith("language_model."):
                sanitized[k[len("language_model.") :]] = v
            # transformers>=5 nesting: model.language_model.* -> model.*
            elif k.startswith("model.language_model."):
                sanitized["model." + k[len("model.language_model.") :]] = v
            else:
                sanitized[k] = v
        return super().sanitize(sanitized)
