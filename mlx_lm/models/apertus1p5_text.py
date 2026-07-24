# Copyright © 2026 mlx-uag lab.
#
# Apertus 1.5 text backbone.
#
# Apertus 1.5 (swiss-ai/Apertus-v1.5-8B) is an early-fusion, discrete-token
# multimodal model (image + audio + text -> text). Its language backbone is a
# plain Apertus decoder (xIELU MLP, qk-norm attention, llama3 RoPE), with two
# differences from `apertus.py` that this module handles:
#
#   1. Split vocabulary. The input embedding spans the *extended* vocabulary
#      (`vocab_size` = 266752: text + visual + audio code tokens), while the LM
#      head is physically pruned to the text-only prefix
#      (`output_vocab_size` = 131072). Image/audio code ids are input-only and
#      never generated, so the pruned head is exactly the generatable range.
#
#   2. Checkpoint layout. Text weights live under `model.language_model.*`
#      (the multimodal wrapper's submodule) and the head is a top-level
#      `lm_head.weight`. `rope_parameters` is a single nested dict rather than
#      the legacy `rope_theta` + `rope_scaling` pair.
#
# The trunk (embedding, decoder layers, final norm) is reused verbatim from
# `apertus.py`; only the outer `Model` (head width + weight remap) differs.

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .apertus import ApertusModel
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    post_norm: bool
    qk_norm: bool
    mlp_bias: bool = False
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    output_vocab_size: Optional[int] = None
    # Apertus 1.5 configs carry a single `rope_parameters` dict (the field the
    # `apertus.py` trunk expects, `rope_theta`/`rope_scaling`, is derived below).
    rope_parameters: Optional[Dict[str, Any]] = None
    rope_theta: float = 4000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        rp = self.rope_parameters
        if rp is not None:
            # `rope_theta` lives inside the dict; the remaining keys are the
            # llama3 scaling config consumed by `initialize_rope`.
            self.rope_theta = rp.get("rope_theta", self.rope_theta)
            scaling = {k: v for k, v in rp.items() if k != "rope_theta"}
            # `initialize_rope` dispatches on `rope_scaling["rope_type"]`.
            scaling.setdefault("rope_type", scaling.pop("type", "llama3"))
            self.rope_scaling = scaling


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        # Trunk embeds over the *extended* vocabulary (`vocab_size`).
        self.model = ApertusModel(args)
        head_dim_out = args.output_vocab_size or args.vocab_size
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, head_dim_out, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            # A pruned head cannot be tied; tie only applies to full-width heads.
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            # Strip the multimodal-wrapper prefix so keys match this module.
            if k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model.") :]
            # xIELU parameters are stored as (1,)-shaped tensors.
            if any(
                k.endswith(s)
                for s in ("alpha_p", "alpha_n", ".beta", ".eps")
            ):
                v = v.squeeze()
            out[k] = v
        if self.args.tie_word_embeddings:
            out.pop("lm_head.weight", None)
        return out

    @property
    def layers(self):
        return self.model.layers
