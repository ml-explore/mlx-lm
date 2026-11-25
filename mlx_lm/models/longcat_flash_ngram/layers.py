# Copyright © 2026 Apple Inc.

from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.longcat_flash_ngram.config import ModelArgs

__all__ = [
    "NgramEmbedding",
]


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

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        seq_len = input_ids.shape[-1]

        input_ids = input_ids.astype(mx.int64)
        if cache is not None:
            n_keep = self.n - 1
            history = cache[0]
            if history is None:
                history = mx.zeros((input_ids.shape[0], n_keep), dtype=input_ids.dtype)
            elif history.shape[-1] < n_keep:
                history = mx.pad(history, [(0, 0), (n_keep - history.shape[-1], 0)])
            context = mx.concatenate([history, input_ids], axis=-1)
            if cache.lengths is None:
                cache[0] = context[:, seq_len:]
            else:
                # Skip right padding when saving each row's last real tokens.
                ends = mx.clip(cache.lengths, 0, seq_len)
                positions = ends[:, None] + mx.arange(n_keep)
                cache[0] = mx.take_along_axis(context, positions, axis=1)
                cache.advance(seq_len)
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
