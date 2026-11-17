# Copyright © 2025 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.gemma4 import checkpoint
from mlx_lm.models.gemma4.config import ModelArgs
from mlx_lm.models.gemma4_text.model import (
    Gemma4TextModel,
    logit_softcap,
    make_cache,
    moe_quant_predicate,
)

__all__ = [
    "Gemma4Model",
    "Model",
]


class Gemma4Model(nn.Module):
    """The language model, named as in the checkpoint."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = Gemma4TextModel(args.text)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Gemma4Model(args)
        if not args.text.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.text.hidden_size, args.text.vocab_size, bias=False
            )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        per_layer_inputs: Optional[mx.array] = None,
    ):
        language_model = self.model.language_model
        out = language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            per_layer_inputs=per_layer_inputs,
        )
        if self.args.text.tie_word_embeddings:
            out = language_model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        softcap = self.args.text.final_logit_softcapping
        if softcap is not None:
            out = logit_softcap(softcap, out)
        return out

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.language_model.layers

    @property
    def quant_predicate(self):
        return moe_quant_predicate

    def make_cache(self):
        return make_cache(self.args.text)
