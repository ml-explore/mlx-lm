# Copyright © 2026 Apple Inc.

from contextlib import contextmanager, nullcontext
import unittest
from unittest.mock import patch
from typing import List, Optional

import mlx.core as mx

from mlx_lm.generate import (
    GenerationForward,
    GenerationForwardPhase,
    generate_step,
    speculative_generate_step,
    stream_generate,
)


class SyntheticCache:
    """Minimal trimmable cache for generation plumbing tests."""

    def __init__(self):
        self.offset = 0
        self.state = mx.array(0)

    def is_trimmable(self):
        return True

    def trim(self, num_to_trim):
        self.offset = max(self.offset - num_to_trim, 0)


class SyntheticModel:
    """A small model double that constructs MLX logits without loading weights."""

    def __init__(self):
        self.layers = [object()]

    def make_cache(self):
        return [SyntheticCache()]

    def __call__(self, input_tokens, *, cache, input_embeddings=None):
        del input_embeddings
        cache[0].offset += input_tokens.shape[1]
        cache[0].state = mx.array(cache[0].offset)
        return mx.zeros((input_tokens.shape[0], input_tokens.shape[1], 8))


class FailingModel(SyntheticModel):
    """A cache-compatible synthetic model whose Python forward always fails."""

    def __call__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("forward failure")


class ForwardContextRecorder:
    """Records forward scope lifetime without retaining active contexts."""

    def __init__(self):
        self.events: List[GenerationForward] = []
        self.exits: List[Optional[BaseException]] = []
        self.active = 0

    @contextmanager
    def context(self, forward):
        self.events.append(forward)
        self.active += 1
        error = None
        try:
            yield
        except BaseException as exc:
            error = exc
            raise
        finally:
            self.exits.append(error)
            self.active -= 1


class FakeDetokenizer:
    def __init__(self):
        self.last_segment = ""

    def add_token(self, token):
        self.last_segment = str(token)

    def finalize(self):
        pass


class FakeTokenizerWrapper:
    """The small stream_generate surface needed by this callback test."""

    def __init__(self):
        self.detokenizer = FakeDetokenizer()
        self.eos_token_ids = set()


class TestGenerationForwardContext(unittest.TestCase):
    def test_generate_step_context_preserves_default_parity_and_metadata(self):
        prompt = mx.array([1, 2, 3, 4])
        baseline = list(
            generate_step(
                prompt,
                SyntheticModel(),
                max_tokens=2,
                prefill_step_size=1,
            )
        )
        model = SyntheticModel()
        recorder = ForwardContextRecorder()
        observed = list(
            generate_step(
                prompt,
                model,
                max_tokens=2,
                prefill_step_size=1,
                model_forward_context=recorder.context,
            )
        )

        self.assertEqual(
            [token for token, _ in observed], [token for token, _ in baseline]
        )
        for (_, observed_logprobs), (_, baseline_logprobs) in zip(observed, baseline):
            self.assertTrue(mx.allclose(observed_logprobs, baseline_logprobs))

        self.assertEqual(recorder.active, 0)
        self.assertEqual(len(recorder.events), len(recorder.exits))
        self.assertEqual(
            [event.phase for event in recorder.events],
            [
                GenerationForwardPhase.PREFILL,
                GenerationForwardPhase.PREFILL,
                GenerationForwardPhase.PREFILL,
                GenerationForwardPhase.PREFILL,
                GenerationForwardPhase.DECODE,
                GenerationForwardPhase.DECODE,
            ],
        )
        for event in recorder.events:
            self.assertIs(event.model, model)
            self.assertEqual(event.input_tokens.ndim, 2)
            self.assertEqual(event.input_tokens.shape[0], 1)
            self.assertIsNone(event.input_embeddings)

    def test_supplied_populated_cache_classifies_final_prompt_step_as_decode(self):
        model = SyntheticModel()
        prompt_cache = model.make_cache()
        model(mx.array([[1, 2]]), cache=prompt_cache)
        recorder = ForwardContextRecorder()

        list(
            generate_step(
                mx.array([3]),
                model,
                max_tokens=1,
                prompt_cache=prompt_cache,
                model_forward_context=recorder.context,
            )
        )

        self.assertEqual(
            [event.phase for event in recorder.events],
            [GenerationForwardPhase.DECODE, GenerationForwardPhase.DECODE],
        )

    def test_generate_step_context_reports_input_embeddings(self):
        prompt = mx.array([1, 2, 3])
        input_embeddings = mx.zeros((3, 4))
        recorder = ForwardContextRecorder()

        list(
            generate_step(
                prompt,
                SyntheticModel(),
                max_tokens=1,
                prefill_step_size=1,
                input_embeddings=input_embeddings,
                model_forward_context=recorder.context,
            )
        )

        embedded_events = [
            event for event in recorder.events if event.input_embeddings is not None
        ]
        self.assertTrue(embedded_events)
        for event in embedded_events:
            self.assertEqual(
                event.input_embeddings.shape[0], event.input_tokens.shape[0]
            )
            self.assertEqual(
                event.input_embeddings.shape[1], event.input_tokens.shape[1]
            )

    def test_generate_step_context_unwinds_on_forward_error(self):
        recorder = ForwardContextRecorder()
        model = FailingModel()

        with self.assertRaisesRegex(RuntimeError, "forward failure"):
            list(
                generate_step(
                    mx.array([1, 2]),
                    model,
                    max_tokens=1,
                    model_forward_context=recorder.context,
                )
            )

        self.assertEqual(recorder.active, 0)
        self.assertEqual(len(recorder.events), 1)
        self.assertIs(recorder.events[0].model, model)
        self.assertIsInstance(recorder.exits[0], RuntimeError)

    def test_generate_step_context_is_inactive_while_generator_is_yielded(self):
        recorder = ForwardContextRecorder()
        generator = generate_step(
            mx.array([1, 2, 3]),
            SyntheticModel(),
            max_tokens=2,
            prefill_step_size=1,
            model_forward_context=recorder.context,
        )

        next(generator)
        self.assertEqual(recorder.active, 0)
        generator.close()
        self.assertEqual(recorder.active, 0)
        self.assertEqual(len(recorder.events), len(recorder.exits))

    def test_external_speculative_context_reports_target_and_draft(self):
        target = SyntheticModel()
        draft = SyntheticModel()
        recorder = ForwardContextRecorder()

        list(
            speculative_generate_step(
                mx.array([1, 2, 3, 4]),
                target,
                draft,
                max_tokens=1,
                num_draft_tokens=1,
                prefill_step_size=1,
                model_forward_context=recorder.context,
            )
        )

        target_events = [event for event in recorder.events if event.model is target]
        draft_events = [event for event in recorder.events if event.model is draft]
        self.assertTrue(target_events)
        self.assertTrue(draft_events)
        self.assertIn(
            GenerationForwardPhase.PREFILL,
            [event.phase for event in target_events],
        )
        self.assertIn(
            GenerationForwardPhase.PREFILL,
            [event.phase for event in draft_events],
        )
        self.assertIn(
            GenerationForwardPhase.DRAFT,
            [event.phase for event in draft_events],
        )
        self.assertIn(
            GenerationForwardPhase.VERIFY,
            [event.phase for event in target_events],
        )
        self.assertEqual(
            [event.phase for event in recorder.events],
            [
                GenerationForwardPhase.PREFILL,
                GenerationForwardPhase.PREFILL,
                GenerationForwardPhase.PREFILL,
                GenerationForwardPhase.PREFILL,
                GenerationForwardPhase.PREFILL,
                GenerationForwardPhase.PREFILL,
                GenerationForwardPhase.DRAFT,
                GenerationForwardPhase.VERIFY,
            ],
        )
        self.assertEqual(len({id(event.cache) for event in target_events}), 1)
        self.assertEqual(len({id(event.cache) for event in draft_events}), 1)
        self.assertNotEqual(id(target_events[0].cache), id(draft_events[0].cache))
        for event in recorder.events:
            self.assertEqual(event.input_tokens.ndim, 2)
            self.assertEqual(event.input_tokens.shape[0], 1)
            self.assertIsNone(event.input_embeddings)
        self.assertEqual(recorder.active, 0)

    def test_external_speculative_without_drafts_reports_prefill_then_decode(self):
        target = SyntheticModel()
        draft = SyntheticModel()
        recorder = ForwardContextRecorder()

        list(
            speculative_generate_step(
                mx.array([1]),
                target,
                draft,
                max_tokens=2,
                num_draft_tokens=0,
                model_forward_context=recorder.context,
            )
        )

        target_phases = [
            event.phase for event in recorder.events if event.model is target
        ]
        self.assertEqual(
            target_phases,
            [GenerationForwardPhase.PREFILL, GenerationForwardPhase.DECODE],
        )
        self.assertNotIn(GenerationForwardPhase.VERIFY, target_phases)
        self.assertEqual(recorder.active, 0)

    def test_stream_generate_forwards_model_forward_context(self):
        recorder = ForwardContextRecorder()
        tokenizer = FakeTokenizerWrapper()

        with patch("mlx_lm.generate.TokenizerWrapper", FakeTokenizerWrapper), patch(
            "mlx_lm.generate.wired_limit", lambda *_args, **_kwargs: nullcontext()
        ):
            responses = list(
                stream_generate(
                    SyntheticModel(),
                    tokenizer,
                    mx.array([1, 2, 3]),
                    max_tokens=1,
                    model_forward_context=recorder.context,
                )
            )

        self.assertEqual(len(responses), 1)
        self.assertTrue(recorder.events)
        self.assertEqual(recorder.active, 0)
