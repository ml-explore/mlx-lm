from itertools import chain, groupby
from typing import List, Any, Iterable, Tuple, Optional, Callable, Generator

from mlx import core as mx, nn as nn
from mlx_lm.generate import SearchStrategy, generation_stream

from mlx_lm.models import cache


class _BeamState:
    def __init__(
        self,
        prompt_cache: List[Any],
        next_token_seq: mx.array,
        score: mx.array,
        all_tokens: mx.array,
        prompt_length: int,
    ):
        self._prompt_cache = prompt_cache
        self._next_token_seq = next_token_seq
        self._score = score
        self._all_tokens = all_tokens
        self._prompt_length = prompt_length

    def fork(
        self, new_data: Iterable[Tuple[mx.array, mx.array, mx.array]]
    ) -> Tuple["_BeamState"]:
        # As an optimisation, we only create a new beam state if we need to fork more
        # than one beam. Otherwise, we can just update the state of the current beam.
        return tuple(
            self._fork_beam(data, create_new=(i == 0))
            for i, data in enumerate(new_data)
        )

    @property
    def prompt_cache(self) -> List[Any]:
        return self._prompt_cache

    @property
    def next_token_seq(self) -> mx.array:
        return self._next_token_seq

    @property
    def total_score(self) -> mx.array:
        return self._score

    @property
    def all_tokens(self) -> mx.array:
        return self._all_tokens

    @property
    def output_tokens(self) -> mx.array:
        return self._all_tokens[self._prompt_length :]

    @property
    def sequence_length(self) -> int:
        return self._all_tokens.size - self._prompt_length

    def _fork_beam(
        self, data: Tuple[mx.array, mx.array, mx.array], create_new: bool
    ) -> "_BeamState":
        _, token, score = data
        token = mx.expand_dims(token, axis=0)
        new_score = self._score + score
        new_tokens = mx.concat([self._all_tokens, token])
        # We need to copy the prompt cache when creating a new beam state.
        if create_new:
            return _BeamState(
                prompt_cache=cache.copy_prompt_cache(self._prompt_cache),
                next_token_seq=token,
                score=new_score,
                all_tokens=new_tokens,
                prompt_length=self._prompt_length,
            )
        else:
            self._next_token_seq = token
            self._score = new_score
            self._all_tokens = new_tokens
            return self


class BeamSearch(SearchStrategy):
    def __init__(
        self,
        model: nn.Module,
        max_sequence_length: int,
        max_steps: int,
        beam_width: int,
        logits_processors: Optional[
            List[Callable[[mx.array, mx.array], mx.array]]
        ] = None,
    ):
        self._model = model
        self._logits_processors = logits_processors
        self._max_sequence_length = max_sequence_length
        self._max_steps = max_steps
        self._beam_width = beam_width

    def generate(
        self,
        y: mx.array,
        prompt_cache: List[Any],
        quantize_cache_fn: Callable[[Any], None],
        total_prompt_tokens: int,
        prompt_progress_callback: Callable[[int, int], None],
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        initial_beam = _BeamState(
            prompt_cache=prompt_cache,
            next_token_seq=y,
            score=mx.array(0.0),
            all_tokens=y,
            prompt_length=y.size,
        )
        beams: Tuple[_BeamState, ...] = (initial_beam,)

        steps = 0
        while any(beam.sequence_length < self._max_sequence_length for beam in beams):
            beam_logprobs = mx.array(
                tuple(self._step(beam, quantize_cache_fn) for beam in beams)
            )
            beam_current_scores = mx.array(tuple((beam.total_score,) for beam in beams))
            new_scores = mx.flatten(
                mx.array(beam_logprobs) + mx.array(beam_current_scores)
            )
            top_score_indices = mx.argpartition(-new_scores, kth=self._beam_width - 1)[
                : self._beam_width
            ]
            top_beams = top_score_indices // beam_logprobs.shape[1]
            top_tokens = top_score_indices % beam_logprobs.shape[1]
            top_scores = new_scores[top_score_indices]
            beams = tuple(
                chain.from_iterable(
                    beams[beam_index.item()].fork(groups)
                    for beam_index, groups in groupby(
                        zip(top_beams, top_tokens, top_scores), key=lambda x: x[0]
                    )
                )
            )
            if steps >= self._max_steps:
                break
            if steps % 256 == 0:
                mx.clear_cache()
            steps += 1

        # TODO: Stream the results when all beams convergence
        best_beam = max(beams, key=lambda x: x.total_score)
        for token in best_beam.output_tokens:
            # TODO: (Optionally) Return the logprobs here
            yield token.item(), mx.array([])

    def _step(
        self,
        beam: _BeamState,
        quantize_cache_fn: Callable[[Any], None],
    ) -> mx.array:
        with mx.stream(generation_stream):
            logits = self._model(beam.next_token_seq[None], cache=beam.prompt_cache)
            logits = logits[:, -1, :]
            if self._logits_processors:
                for processor in self._logits_processors:
                    logits = processor(beam.all_tokens, logits)
            quantize_cache_fn(beam.prompt_cache)
            logprobs = (logits - mx.logsumexp(logits, keepdims=True)).squeeze(0)
            mx.async_eval(logprobs)
            return logprobs
