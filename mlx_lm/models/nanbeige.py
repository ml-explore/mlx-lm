# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from . import llama
from .cache import KVCache


@dataclass
class ModelArgs(llama.ModelArgs):
    num_loops: int = 1
    loop_loss_weights: Optional[List[float]] = None
    skip_loop_final_norm: bool = False

    @classmethod
    def from_dict(cls, params):
        unsupported = {
            "qk_layernorm": params.get("qk_layernorm"),
            "emb_neighbor_num": params.get("emb_neighbor_num") is not None,
            "insert_ngram_layer_idx": bool(params.get("insert_ngram_layer_idx")),
            "ngram_insert_all_layers": params.get("ngram_insert_all_layers"),
            "enable_double_loop_split": params.get("enable_double_loop_split"),
            "loop_middle_layers": bool(params.get("loop_middle_layers")),
            "loop_share_kv": params.get("loop_share_kv"),
            "enable_hyper_connection": params.get("enable_hyper_connection"),
            "enable_mhc": params.get("enable_mhc"),
            "enable_h_res_identity": params.get("enable_h_res_identity"),
            "enable_depth_attention": params.get("enable_depth_attention"),
        }
        enabled = [name for name, value in unsupported.items() if value]
        if enabled:
            raise ValueError(
                "Unsupported Nanbeige architecture options: " + ", ".join(enabled)
            )
        return super().from_dict(params)

    def __post_init__(self):
        super().__post_init__()
        if self.num_loops != 2:
            raise ValueError(f"Only num_loops=2 is supported, got {self.num_loops}.")
        if self.skip_loop_final_norm:
            raise ValueError("skip_loop_final_norm=True is not supported.")
        if self.loop_loss_weights:
            raise ValueError("loop_loss_weights is only supported when empty.")
        if any(layer_type != "full_attention" for layer_type in self.layer_types):
            raise ValueError("Only full-attention Nanbeige layers are supported.")


class NanbeigeModel(llama.LlamaModel):
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        h = input_embeddings
        num_layers = len(self.layers)
        expected_caches = self.args.num_loops * num_layers

        if cache is None:
            cache = [None] * expected_caches
        elif len(cache) != expected_caches:
            raise ValueError(
                f"Nanbeige requires {expected_caches} cache entries, got {len(cache)}."
            )

        for loop_idx in range(self.args.num_loops):
            loop_cache = cache[loop_idx * num_layers : (loop_idx + 1) * num_layers]
            h = super().__call__(inputs, cache=loop_cache, input_embeddings=h)

        return h


class Model(llama.Model):
    def __init__(self, args: ModelArgs):
        nn.Module.__init__(self)
        self.args = args
        self.model_type = args.model_type
        self.model = NanbeigeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def make_cache(self):
        return [KVCache() for _ in range(self.args.num_loops * len(self.layers))]
