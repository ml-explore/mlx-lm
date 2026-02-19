# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask
from .cache import ArraysCache, CacheList, KVCache
from .longcat_flash import LongcatFlashDecoderLayer
from .longcat_flash import Model as LongcatFlashLM


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    ffn_hidden_size: int
    moe_topk: int
    expert_ffn_hidden_size: int
    n_routed_experts: int
    zero_expert_num: int
    num_layers: int
    vocab_size: int
    max_position_embeddings: int
    num_attention_heads: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    routed_scaling_factor: float
    rms_norm_eps: float
    rope_theta: float
    mla_scale_q_lora: bool
    mla_scale_kv_lora: bool
    attention_bias: bool = False
    zero_expert_type: str = "identity"
    ngram_vocab_size_ratio: int = 78
    emb_neighbor_num: int = 4
    emb_split_num: int = 4
    norm_topk_prob: bool = False
    router_bias: bool = False
    rope_scaling: Optional[Dict] = None


class NgramEmbedding(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.m = args.ngram_vocab_size_ratio * args.vocab_size
        self.k = args.emb_split_num
        self.n = args.emb_neighbor_num

        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)

        num_embedders = self.k * (self.n - 1)
        emb_dim = args.hidden_size // num_embedders

        self.embedders = []
        self.post_projs = []
        for i in range(num_embedders):
            emb_vocab_size = int(self.m + i * 2 + 1)
            self.embedders.append(nn.Embedding(emb_vocab_size, emb_dim))
            self.post_projs.append(nn.Linear(emb_dim, args.hidden_size, bias=False))
        self._compute_vocab_mods()

    def _compute_vocab_mods(self):
        vocab_mods = {}
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                mods = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * self.vocab_size) % emb_vocab_dim
                    mods.append(power_mod)
                vocab_mods[(i, j)] = mods
        self._vocab_mods = vocab_mods

    def _shift_right(self, x: mx.array, n: int) -> mx.array:
        if n <= 0:
            return x
        batch_size, seq_len = x.shape
        if seq_len <= n:
            return mx.zeros_like(x)
        return mx.concatenate(
            [mx.zeros((batch_size, n), dtype=x.dtype), x[..., :-n]], axis=-1
        )

    def _get_ngram_ids(
        self,
        input_ids: mx.array,
        shifted_ids: Dict[int, mx.array],
        vocab_mods: List[int],
        ngram: int,
    ) -> mx.array:
        ngram_ids = input_ids
        for k in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
        return ngram_ids

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        seq_len = input_ids.shape[-1]

        input_ids = input_ids.astype(mx.int64)
        if cache is not None:
            context = cache[0]
            if context is None:
                context = input_ids
            else:
                context = mx.concatenate([context, input_ids], axis=-1)
            cache[0] = context[..., max(0, context.shape[-1] - self.n + 1) :]
        else:
            context = input_ids

        x = self.word_embeddings(input_ids)
        vocab_mods = self._vocab_mods

        shifted_ids = {}
        for i in range(2, self.n + 1):
            shifted_ids[i] = self._shift_right(context, i - 1)

        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                ngram_ids = self._get_ngram_ids(
                    context, shifted_ids, vocab_mods[(i, j)], ngram=i
                )
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
                x_ngram = self.embedders[index](new_ids)
                x_proj = self.post_projs[index](x_ngram)
                x = x + x_proj

        return x / (1 + self.k * (self.n - 1))


class LongcatFlashNgramModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_layers = args.num_layers
        self.ngram_embeddings = NgramEmbedding(args)
        self.layers = [LongcatFlashDecoderLayer(args) for _ in range(args.num_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if cache is None:
            cache = [None] + [(None, None)] * self.num_layers

        h = self.ngram_embeddings(input_ids, cache=cache[0])

        mask = create_attention_mask(h, cache[1][0], return_array=True)

        for layer, c in zip(self.layers, cache[1:]):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LongcatFlashNgramModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        return LongcatFlashLM.quant_predicate.fget(self)

    @property
    def cast_predicate(self):
        return LongcatFlashLM.cast_predicate.fget(self)

    def sanitize(self, weights):
        weights = LongcatFlashLM.sanitize(self, weights)
        if "model.embed_tokens.weight" in weights:
            weights["model.ngram_embeddings.word_embeddings.weight"] = weights.pop(
                "model.embed_tokens.weight"
            )
        return weights

    def make_cache(self):
        return [ArraysCache(size=1)] + [
            CacheList(KVCache(), KVCache()) for _ in self.model.layers
        ]

    def shard(self, group: Optional[mx.distributed.Group] = None):
        LongcatFlashLM.shard(self, group)
