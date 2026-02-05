import json
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Type

from outlines.backends.outlines_core import OutlinesCoreBackend, OutlinesCoreLogitsProcessor
from outlines_core import Index, outlines_core
from pydantic import BaseModel


class StructuredProcessorCache:
    """LRU cache of compiled Outlines schema indices; used by _make_structured_processor to reuse processors."""

    @dataclass(frozen=True)
    class TokenizerFingerprint:
        tokenizer_id: int
        eos_token_id: int
        vocab_size: int

    @dataclass
    class TokenizerCacheEntry:
        schema_cache: OrderedDict
        outlines_vocab: Any

    def __init__(self, max_size: int = 5):
        """Initialize cache storage; used by module-level _STRUCTURED_PROCESSOR_CACHE."""
        self.max_size = max_size
        self._lock = Lock()
        # Tokenizer fingerprint -> TokenizerCacheEntry
        # Note: mlx_lm.server only loads one model at a time, so this per-tokenizer
        # cache might be unnecessary, but we keep it for safety and future changes.
        # and it will save you time even if you swap models, if those models
        # share tokenizer. So... myeah.
        self._per_tokenizer: Dict[
            StructuredProcessorCache.TokenizerFingerprint,
            StructuredProcessorCache.TokenizerCacheEntry,
        ] = {}
        self._tensor_library_name = "mlx"

    def _ensure_entry(self, tokenizer) -> "StructuredProcessorCache.TokenizerCacheEntry":
        """Create or fetch per-tokenizer cache state; used by get_processor."""
        fp = StructuredProcessorCache.TokenizerFingerprint(
            id(tokenizer), tokenizer.eos_token_id, tokenizer.vocab_size
        )
        entry = self._per_tokenizer.get(fp)
        if entry is not None:
            return entry

        vocabulary = tokenizer.get_vocab()
        eos_token_id = tokenizer.eos_token_id
        eos_token = tokenizer.eos_token
        token_to_str = lambda token: tokenizer.convert_tokens_to_string([token])
        outlines_vocab = OutlinesCoreBackend.create_outlines_core_vocabulary(
            vocabulary, eos_token_id, eos_token, token_to_str
        )
        entry = StructuredProcessorCache.TokenizerCacheEntry(
            schema_cache=OrderedDict(),
            outlines_vocab=outlines_vocab,
        )
        self._per_tokenizer[fp] = entry
        return entry

    def _normalize_schema(self, schema) -> str:
        """Normalize schema to a JSON string key; used by get_processor for cache lookup."""
        if isinstance(schema, str):
            try:
                json.loads(schema)
                return schema
            except Exception:
                return schema
        return json.dumps(schema)

    def get_processor(self, schema, tokenizer):
        """Return a cached OutlinesCoreLogitsProcessor for a schema; used by _make_structured_processor."""
        if schema is None:
            return None
        schema_key = self._normalize_schema(schema)
        with self._lock:
            entry = self._ensure_entry(tokenizer)
            cache = entry.schema_cache
            vocab = entry.outlines_vocab
            if schema_key in cache:
                index = cache.pop(schema_key)
                cache[schema_key] = index
            else:
                regex = outlines_core.json_schema.build_regex_from_schema(schema_key)
                index = Index(regex, vocab)
                cache[schema_key] = index
                if len(cache) > self.max_size:
                    cache.popitem(last=False)
            return OutlinesCoreLogitsProcessor(index, self._tensor_library_name)

    def get_processor_from_model(self, model_cls: Type[BaseModel], tokenizer):
        """Return a cached processor for a Pydantic v2 model class."""
        schema = model_cls.model_json_schema()
        return self.get_processor(schema, tokenizer)

    # I'm going to add this here, because it feels like it belongs here more than at the module level
    def _make_structured_processor(self, schema, tokenizer):
        """Build a per-request Outlines processor; used by LocalResponseGenerator."""
        processor = self.get_processor(schema, tokenizer)
        if processor is None:
            return None
        return processor
