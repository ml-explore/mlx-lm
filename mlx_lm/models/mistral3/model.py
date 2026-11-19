# Copyright © 2026 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.distributed import PipelineMixin, gather_last_axis
from mlx_lm.models import llama, ministral3, mistral4
from mlx_lm.models.mistral3 import checkpoint
from mlx_lm.models.mistral3.config import ModelArgs, language_model_package

__all__ = [
    "Mistral3Model",
    "Model",
]


class Mistral3Model(nn.Module):
    """Puts the language model at model.language_model, as the checkpoint names it."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        if args.text.model_type == "ministral3":
            self.language_model = ministral3.model.LanguageModel(args.text)
        elif args.text.model_type == "mistral4":
            self.language_model = mistral4.model.Mistral4Model(args.text)
        else:
            self.language_model = llama.model.LlamaModel(args.text)

    # mlx_lm.utils pipelines model.model with these.
    @property
    def layers(self):
        return self.language_model.layers

    @property
    def start_idx(self):
        return self.language_model.start_idx

    @property
    def end_idx(self):
        return self.language_model.end_idx

    def pipeline(self, group, split=None):
        if not isinstance(self.language_model, PipelineMixin):
            raise ValueError("Llama's language model does not support pipelining.")
        self.language_model.pipeline(group, split)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Mistral3Model(args)
        if not args.text.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.text.hidden_size, args.text.vocab_size, bias=False
            )
        self.sharding_group = None

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        language_model = self.model.language_model
        out = language_model(inputs, cache, input_embeddings)
        if self.args.text.tie_word_embeddings:
            return language_model.embed_tokens.as_linear(out)
        out = self.lm_head(out)
        if self.sharding_group is not None:
            out = gather_last_axis(out, self.sharding_group)
        return out

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        package = language_model_package(self.args.text.model_type)
        package.model.shard_layers(self.model.language_model.layers, group)
        # Only Ministral 3 shards its output layer.
        if package is ministral3 and not self.args.text.tie_word_embeddings:
            vocab = self.args.text.vocab_size
            self.lm_head = ministral3.model.shard_lm_head(self.lm_head, vocab, group)
            self.sharding_group = group

    @property
    def layers(self):
        # Only the layers of this pipeline stage, so caches match them.
        language_model = self.model.language_model
        if isinstance(language_model, PipelineMixin):
            return language_model.pipeline_layers
        return language_model.layers
