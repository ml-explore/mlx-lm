# Copyright © 2026 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import llama, ministral3, mistral4
from mlx_lm.models.mistral3 import checkpoint
from mlx_lm.models.mistral3.config import ModelArgs

__all__ = [
    "Model",
]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        if args.text_config.get("model_type") == "ministral3":
            self.language_model = ministral3.Model(
                ministral3.ModelArgs.from_dict(args.text_config)
            )
        elif args.text_config.get("model_type") == "mistral4":
            self.language_model = mistral4.Model(
                mistral4.ModelArgs.from_dict(args.text_config)
            )
        else:
            self.language_model = llama.Model(
                llama.ModelArgs.from_dict(args.text_config)
            )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    def shard(self, group: Optional[mx.distributed.Group] = None):
        self.language_model.shard(group)

    @property
    def model(self):
        return self.language_model.model

    @property
    def layers(self):
        # Only the layers of this pipeline stage, so caches match them.
        return self.language_model.layers
