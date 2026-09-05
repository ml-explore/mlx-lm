"""Bounded paged-cache admission for the existing batch scheduler."""

import copy

import mlx.core as mx

from .generate import BatchGenerator
from .models.paged_cache import QwenHybridPagedKVCacheManager


class PagedBatchGenerator(BatchGenerator):
    def __init__(
        self, model, *, capacity_pages, page_size=256, paged_attention=False, **kwargs
    ):
        self._admissions = {}
        if kwargs.get("max_kv_size") is not None:
            raise ValueError("Paged KV does not support a rotating max_kv_size")
        self.cache_manager = QwenHybridPagedKVCacheManager(
            model, capacity_pages=capacity_pages, page_size=page_size
        )
        if paged_attention:
            from eco_paged_attention import _paged_attention

            text_model = getattr(model, "language_model", model)
            for layer, pool in zip(text_model.layers, self.cache_manager._pools):
                if pool is None:
                    continue
                if (
                    pool.key_head_dim != 256
                    or layer.self_attn.num_attention_heads
                    not in (
                        6 * pool.num_kv_heads,
                        8 * pool.num_kv_heads,
                    )
                ):
                    raise ValueError("ECO paged attention requires D256 and GQA 6 or 8")
                pool.attention = _paged_attention
        super().__init__(model, **kwargs)
        with mx.stream(self.stream):
            self.cache_manager.materialize()

    def insert_segments(
        self,
        segments,
        max_tokens=None,
        caches=None,
        all_tokens=None,
        samplers=None,
        logits_processors=None,
        stop_matchers=None,
    ):
        count = len(segments)
        limits = max_tokens or [self.max_tokens] * count
        caches = caches or [None] * count
        if len(limits) != count or len(caches) != count:
            raise ValueError("caches and max_tokens must match the number of sequences")
        prepared = []
        try:
            for seq, limit, cache in zip(segments, limits, caches):
                cache = (
                    self.cache_manager.make_cache()
                    if cache is None
                    else copy.deepcopy(cache)
                )
                owner = object()
                lengths = [len(s) for s in seq if s]
                if lengths and lengths[-1] > 1:
                    lengths[-1] -= 1
                    lengths.append(1)
                self.cache_manager.admit_segments(owner, lengths, limit, cache)
                prepared.append((owner, cache))
            uids = super().insert_segments(
                segments,
                limits,
                [c for _, c in prepared],
                all_tokens,
                samplers,
                logits_processors,
                stop_matchers,
            )
        except Exception:
            for owner, cache in prepared:
                self.cache_manager.rollback_admission(owner, cache)
            raise
        self._admissions.update(zip(uids, (owner for owner, _ in prepared)))
        return uids

    def _release_admissions(self, uids):
        for uid in uids:
            owner = self._admissions.pop(uid, None)
            if owner is not None:
                self.cache_manager.release_admission(owner)

    def _next(self):
        try:
            prompts, responses = super()._next()
        except Exception:
            self.remove(list(self._admissions))
            raise
        self._release_admissions(
            r.uid for r in responses if r.finish_reason is not None
        )
        return prompts, responses

    def remove(self, uids, return_prompt_caches=False):
        uids = list(uids)
        mx.synchronize(self.stream)
        result = super().remove(uids, return_prompt_caches)
        self._release_admissions(uids)
        return result

    def close(self):
        if hasattr(self, "_stream"):
            self.remove(list(self._admissions))
            super().close()
