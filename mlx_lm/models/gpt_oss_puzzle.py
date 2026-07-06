# Copyright © 2026 Apple Inc.

"""NVIDIA gpt-oss-puzzle: a NAS-pruned ("Puzzle") gpt-oss variant whose
layers are heterogeneous — per-layer expert count (64 or 128) and per-layer
attention window (128, 8192, or full) chosen by neural architecture search.

The layer internals are identical to stock gpt-oss (mxfp4 experts, attention
sinks, YaRN rope, SwiGLU), so this module reuses gpt_oss's blocks and mxfp4
weight sanitizer; it only reads `block_configs` to build each layer with its
own expert count + window, and to size the per-layer KV cache.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask
from .cache import KVCache, RotatingKVCache
from .gpt_oss import Model as GptOssModel
from .gpt_oss import TransformerBlock


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gpt_oss_puzzle"
    num_hidden_layers: int = 36
    num_experts_per_tok: int = 4
    vocab_size: int = 201088
    rms_norm_eps: float = 1e-05
    hidden_size: int = 2880
    intermediate_size: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    rope_theta: int = 150000
    rope_scaling: Any = None
    # YaRN params live here rather than under `rope_scaling` so the HF
    # tokenizer's AutoConfig (which crashes standardizing a yarn
    # rope_scaling in transformers 5.x) can load the config; the model
    # copies it back to rope_scaling at construction.
    yarn_rope_scaling: Any = None
    # Per-layer [{num_local_experts, sliding_window}, ...]; sliding_window
    # None => full attention. Uniform fields below are fallbacks only.
    block_configs: Optional[List[dict]] = None
    num_local_experts: int = 128
    sliding_window: int = 128


def _layer_config(args: ModelArgs, num_local_experts: int) -> ModelArgs:
    """A shallow per-layer view of args with this layer's expert count, so the
    reused gpt_oss blocks pick up the right SwitchGLU size."""
    lc = copy.copy(args)
    lc.num_local_experts = num_local_experts
    return lc


class PuzzleModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        if not args.block_configs:
            raise ValueError("gpt_oss_puzzle requires 'block_configs' in config")
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        # per-layer window: int (sliding) or None (full attention)
        self.windows = [b.get("sliding_window") for b in args.block_configs]
        self.layers = [
            TransformerBlock(_layer_config(args, b["num_local_experts"]))
            for b in args.block_configs
        ]
        # first layer index carrying each distinct window, for mask reuse
        self._mask_ref = {}
        for i, w in enumerate(self.windows):
            self._mask_ref.setdefault(w, i)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            x = input_embeddings
        else:
            x = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        # One mask per distinct window (all caches share the decode offset).
        masks = {}
        for w, ref in self._mask_ref.items():
            if w is None:
                masks[w] = create_attention_mask(x, cache[ref])
            else:
                masks[w] = create_attention_mask(x, cache[ref], window_size=w)

        for layer, c, w in zip(self.layers, cache, self.windows):
            x = layer(x, masks[w], c)
        return self.norm(x)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Restore YaRN (parked under yarn_rope_scaling to keep the HF
        # tokenizer loadable) before the layers build their rope.
        if args.rope_scaling is None and args.yarn_rope_scaling is not None:
            args.rope_scaling = args.yarn_rope_scaling
        self.args = args
        self.model_type = args.model_type
        self.model = PuzzleModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        return self.lm_head(self.model(inputs, cache))

    def sanitize(self, weights):
        # Drop the fp8-KV-cache calibration scales (k_scale/v_scale): they
        # are serving-time quantization constants for vLLM/TensorRT and are
        # unused by the bf16 KV path here — NVIDIA's own modeling ignores
        # them (_keys_to_ignore_on_load_unexpected). The mxfp4 expert weight
        # layout is otherwise identical to stock gpt-oss.
        weights = {
            k: v
            for k, v in weights.items()
            if not (k.endswith(".k_scale") or k.endswith(".v_scale"))
        }
        return GptOssModel.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for w in self.model.windows:
            if w is None:
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=w))
        return caches
