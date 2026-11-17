# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.qwen3.model import Qwen3Model
from mlx_lm.models.qwen3.model import shard_layers as shard_qwen3_layers
from mlx_lm.models.qwen3_moe.config import ModelArgs as Qwen3MoeArgs
from mlx_lm.models.qwen3_moe.model import Qwen3MoeModel, moe_quant_predicate
from mlx_lm.models.qwen3_moe.model import shard_layers as shard_qwen3_moe_layers
from mlx_lm.models.qwen3_vl import checkpoint
from mlx_lm.models.qwen3_vl.config import ModelArgs

__all__ = [
    "Model",
    "Qwen3VLModel",
]


class Qwen3VLModel(nn.Module):
    """The language model, named as in the checkpoint."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        if isinstance(args.text, Qwen3MoeArgs):
            self.language_model = Qwen3MoeModel(args.text)
        else:
            self.language_model = Qwen3Model(args.text)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3VLModel(args)
        if not args.text.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.text.hidden_size, args.text.vocab_size, bias=False
            )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        language_model = self.model.language_model
        out = language_model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.text.tie_word_embeddings:
            return language_model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def quant_predicate(self):
        if isinstance(self.args.text, Qwen3MoeArgs):
            return moe_quant_predicate
        return None

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        if isinstance(self.args.text, Qwen3MoeArgs):
            shard_qwen3_moe_layers(self.layers, group)
        else:
            shard_qwen3_layers(self.layers, group)

    @property
    def layers(self):
        return self.model.language_model.layers
