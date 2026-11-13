# Copyright © 2026 Apple Inc.

import dataclasses
import sys
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_lm import sample_utils, tokenizer_utils
from mlx_lm.generate import BatchGenerator, GenerationBatch, TextStateMachine
from mlx_lm.server_v2.engine import protocol

__all__ = ["Decoding", "Generation"]


@dataclasses.dataclass(eq=False, kw_only=True)
class Generation:
    """A request being generated: its prefill, its counters and its text."""

    handle: int
    request: protocol.Request
    # The token limit: the requested one, or the default within the context.
    max_tokens: int

    detokenizer: tokenizer_utils.StreamingDetokenizer
    # Splits the text into content, reasoning and tool calls.
    text_reader: TextStateMachine.Reader | None
    # Finds the stop strings in the content; None without them.
    stop_reader: TextStateMachine.Reader | None
    eos_token_ids: frozenset[int]

    # Set when the prefill starts, and cleared when the decode batch takes it.
    prompt_cache: list[Any] | None = None
    num_prefilled: int = 0
    num_cached: int = 0

    num_generated: int = 0

    @property
    def is_prefilled(self) -> bool:
        """Whether the cache holds the prompt; its last token goes to the decode."""
        end = len(self.request.tokens) - 1
        return self.num_prefilled == end

    @property
    def prefilled_tokens(self) -> list[int]:
        """The prompt tokens that the prompt cache holds."""
        return list(self.request.tokens[: self.num_prefilled])

    @property
    def num_computed(self) -> int:
        """The prompt tokens prefilled here, not taken from the cache store."""
        return self.num_prefilled - self.num_cached

    @property
    def usage(self) -> protocol.Usage:
        """The token counts so far."""
        return protocol.Usage(
            prompt_tokens=len(self.request.tokens),
            cached_tokens=self.num_cached,
            completion_tokens=self.num_generated,
        )

    @property
    def stop_string(self) -> str | None:
        """The stop string that the text met; None if it met none."""
        return None if self.stop_reader is None else self.stop_reader.stop_word

    def read_token(
        self, token: int
    ) -> tuple[TextStateMachine.Delta, protocol.FinishReason | None]:
        """Count and read a new token: its parsed text, and the reason if it ends here.

        Advances the detokenizer and the readers, so read each token once.
        """
        self.num_generated += 1
        reason = self._finish_reason(token)
        final = reason is not None

        delta = self._read_text(token, final=final)
        if self.stop_reader is None:
            return delta, reason

        delta = self._cut_at_stop(delta, final=final)
        if self.stop_reader.stopped:
            return delta, protocol.FinishReason.STOP_STRING
        return delta, reason

    def _finish_reason(self, token: int) -> protocol.FinishReason | None:
        """Why this token ends the request: end of sequence or the token limit."""
        if token in self.eos_token_ids:
            return protocol.FinishReason.EOS
        if self.num_generated >= self.max_tokens:
            return protocol.FinishReason.LENGTH
        return None

    def _read_text(self, token: int, *, final: bool) -> TextStateMachine.Delta:
        """The text of a token, split into content, reasoning and tool calls."""
        # The text of an end-of-sequence token is not in the response.
        if token not in self.eos_token_ids:
            self.detokenizer.add_token(token)
        if final:
            self.detokenizer.finalize()
        text = self.detokenizer.last_segment

        if self.text_reader is None:
            return TextStateMachine.Delta(content=text)
        return self.text_reader.finish(text) if final else self.text_reader.feed(text)

    def _cut_at_stop(
        self, delta: TextStateMachine.Delta, *, final: bool
    ) -> TextStateMachine.Delta:
        """The delta without the content from a stop string on."""
        stop = self.stop_reader
        cut = stop.finish(delta.content) if final else stop.feed(delta.content)
        # A tool call that closes in this delta is complete: keep it on a stop too.
        return dataclasses.replace(delta, content=cut.content)


class Decoding:
    """The requests in decode, one row each.

    The requests decide when they end: the batch never ends a row by itself.
    """

    def __init__(self, model: nn.Module, *, max_rows: int):
        self._generator = BatchGenerator(
            model,
            # No stop tokens and no limit: only drop ends a row.
            max_tokens=sys.maxsize,
            completion_batch_size=max_rows,
            prefill_batch_size=max_rows,
            stream=mx.default_stream(mx.default_device()),
        )

        self._max_rows = max_rows
        # The requests by row id, and the rows of dropped requests.
        self._generations: dict[int, Generation] = {}
        self._uids: dict[Generation, int] = {}
        self._dropped: list[int] = []

    def __contains__(self, generation: Generation) -> bool:
        return generation in self._uids

    @property
    def is_full(self) -> bool:
        """Whether no row is free."""
        return len(self._generations) >= self._max_rows

    @property
    def is_empty(self) -> bool:
        """Whether there is no row, also none left to remove."""
        return not (self._generations or self._dropped)

    def add(
        self,
        generation: Generation,
        *,
        sampler: sample_utils.Sampler,
        logits_processors: list[sample_utils.LogitsProcessor],
    ) -> None:
        """Give a prefilled request a row; the row takes its prompt cache."""
        request = generation.request
        last_token = list(request.tokens[-1:])
        earlier_tokens = list(request.tokens[:-1])
        (uid,) = self._generator.insert(
            [last_token],
            caches=[generation.prompt_cache],
            all_tokens=[earlier_tokens],
            samplers=[sampler],
            logits_processors=[logits_processors],
        )

        # A reference here would keep the memory of the row.
        generation.prompt_cache = None
        self._generations[uid] = generation
        self._uids[generation] = uid

    def drop(self, generation: Generation) -> None:
        """Take a request out of the rows; ``remove_dropped`` removes its row."""
        uid = self._uids.pop(generation)
        del self._generations[uid]
        self._dropped.append(uid)

    def remove_dropped(self) -> list[tuple[list[Any], list[int]]]:
        """Remove the rows of the dropped requests; return their caches and tokens."""
        if not self._dropped:
            return []

        # In one call: each removal gathers the whole batch.
        caches = self._generator.remove(self._dropped, return_prompt_caches=True)
        self._dropped = []
        for prompt_cache, _ in caches.values():
            # A lazy row holds the whole batch cache in memory.
            mx.eval([c.state for c in prompt_cache])

        return list(caches.values())

    def decode(
        self,
    ) -> list[tuple[Generation, int, tuple[protocol.TokenLogprob, ...]]]:
        """Decode one token for each request, with its log probabilities if asked."""
        if not self._generations:
            return []

        _, responses = self._generator.next()
        generations = [self._generations[resp.uid] for resp in responses]
        tokens = [resp.token for resp in responses]
        logprobs = _read_logprobs(generations, responses)

        return list(zip(generations, tokens, logprobs, strict=True))

    def close(self) -> None:
        """Close the generator, on the thread that made it: its stream is there."""
        self._generator.close()


def _read_logprobs(
    generations: list[Generation],
    responses: list[GenerationBatch.Response],
) -> list[tuple[protocol.TokenLogprob, ...]]:
    """The log probabilities of each row's token: empty if its request does not ask.

    Reads the rows that ask in one evaluation.
    """
    arrays = {}
    for i, (generation, resp) in enumerate(zip(generations, responses, strict=True)):
        num_top = generation.request.params.logprobs
        if num_top is not None:
            top = mx.zeros(0, dtype=mx.uint32)
            if num_top > 0:
                # The most likely first.
                top = mx.argsort(-resp.logprobs)[:num_top]
            arrays[i] = (resp.logprobs[resp.token], top, resp.logprobs[top])

    mx.eval(list(arrays.values()))

    logprobs = [()] * len(responses)
    for i, (logprob, top, top_logprobs) in arrays.items():
        pairs = zip(top.tolist(), top_logprobs.tolist(), strict=True)
        token_logprob = protocol.TokenLogprob(
            token=responses[i].token, logprob=logprob.item(), top=tuple(pairs)
        )
        logprobs[i] = (token_logprob,)

    return logprobs
