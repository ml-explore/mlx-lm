# Copyright © 2024 Apple Inc.

import unittest
from typing import List

import mlx.core as mx

from mlx_lm.generate import (
    BatchGenerator,
    GenerationCancelled,
    GenerationResponse,
    generate,
    stream_generate,
)
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load


class TestGenerate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        cls.model, cls.tokenizer = load(cls.HF_MODEL_PATH)
        cls.model.set_dtype(mx.float32)

    def _chat_prompt_tokens(self, content: str) -> List[int]:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
        )

    def test_generate(self):
        # Simple test that generation runs
        text = generate(
            self.model, self.tokenizer, "hello", max_tokens=5, verbose=False
        )

    def test_generate_with_logit_bias(self):
        logit_bias = {0: 2000.0, 1: -20.0}
        text = generate(
            self.model,
            self.tokenizer,
            "hello",
            max_tokens=5,
            logits_processors=make_logits_processors(logit_bias),
            verbose=False,
        )
        self.assertEqual(text, "!!!!!")

    def test_generate_propagates_generationcancelled(self):
        def should_cancel() -> bool:
            return True

        with self.assertRaises(GenerationCancelled):
            generate(
                self.model,
                self.tokenizer,
                "hello",
                max_tokens=5,
                verbose=False,
                should_cancel=should_cancel,
            )

    def test_stream_generate_max_tokens(self):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a story about Einstein"}],
            tokenize=True,
            add_generation_prompt=True,
        )

        tokens = []
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=4,
        ):
            tokens.append(response.token)
        self.assertEqual(len(tokens), 4)

    def test_stream_generate_cancel_immediate(self):
        prompt = self._chat_prompt_tokens("hello")

        def should_cancel() -> bool:
            return True

        gen = stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=5,
            should_cancel=should_cancel,
        )
        with self.assertRaises(GenerationCancelled):
            next(gen)

    def test_stream_generate_cancel_during_prefill(self):
        prompt = self._chat_prompt_tokens("Please cancel during prefill. " * 16)

        progress: List[tuple[int, int]] = []
        cancelled = False

        def progress_callback(processed: int, total: int) -> None:
            nonlocal cancelled
            progress.append((processed, total))
            if processed > 0:
                cancelled = True

        def should_cancel() -> bool:
            return cancelled

        gen = stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=5,
            prefill_step_size=4,
            prompt_progress_callback=progress_callback,
            should_cancel=should_cancel,
        )
        with self.assertRaises(GenerationCancelled):
            next(gen)

        self.assertGreaterEqual(len(progress), 2)
        self.assertEqual(progress[0][0], 0)
        self.assertGreater(progress[0][1], 0)
        self.assertTrue(any(p > 0 for p, _ in progress))

    def test_stream_generate_cancel_before_first_step(self):
        prompt = self._chat_prompt_tokens("Please cancel before first-step. " * 16)

        ready = False
        calls_after_ready = 0

        def progress_callback(processed: int, total: int) -> None:
            nonlocal ready
            if processed == total - 1:
                ready = True

        def should_cancel() -> bool:
            nonlocal calls_after_ready
            if not ready:
                return False
            calls_after_ready += 1
            return calls_after_ready >= 2

        gen = stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=5,
            prefill_step_size=2048,
            prompt_progress_callback=progress_callback,
            should_cancel=should_cancel,
        )
        with self.assertRaises(GenerationCancelled):
            next(gen)

        self.assertTrue(ready)
        self.assertEqual(calls_after_ready, 2)

    def test_generate_with_processor(self):
        init_toks = self.tokenizer.encode("hello")

        all_toks = None

        def logits_processor(toks, logits):
            nonlocal all_toks
            all_toks = toks
            return logits

        generate(
            self.model,
            self.tokenizer,
            "hello",
            max_tokens=5,
            verbose=False,
            logits_processors=[logits_processor],
        )
        self.assertEqual(len(all_toks), len(init_toks) + 5)

    def test_stream_generate_speculative(self):
        # Use same model as draft model, this is not a speed test
        draft_model = self.model

        results: List[GenerationResponse] = []
        drafted: List[bool] = []

        # make a determinate sampler
        sampler = make_sampler(temp=0.0)
        messages = [{"role": "user", "content": "hello"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        for generation_result in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            draft_model=draft_model,
            num_draft_tokens=2,
            sampler=sampler,
        ):
            drafted.append(generation_result.from_draft)
            results.append(generation_result)

        self.assertEqual(len(results), 5)
        # since num_draft_tokens is 2 and draft model is the same, the
        # first 2 generations should be drafts, the third should come
        # from the target model, and last two should be drafts
        self.assertEqual(drafted, [True, True, False, True, True])

    def test_stream_generate_speculative_cancel_immediate(self):
        # Use same model as draft model to exercise the speculative path
        draft_model = self.model
        sampler = make_sampler(temp=0.0)
        prompt = self._chat_prompt_tokens("hello")

        def should_cancel() -> bool:
            return True

        gen = stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            draft_model=draft_model,
            num_draft_tokens=2,
            sampler=sampler,
            should_cancel=should_cancel,
        )
        with self.assertRaises(GenerationCancelled):
            next(gen)

    def test_stream_generate_speculative_cancel_during_prefill(self):
        draft_model = self.model
        sampler = make_sampler(temp=0.0)
        prompt = self._chat_prompt_tokens(
            "Please cancel during speculative prefill. " * 16
        )

        progress: List[tuple[int, int]] = []
        cancelled = False

        def progress_callback(processed: int, total: int) -> None:
            nonlocal cancelled
            progress.append((processed, total))
            if processed > 0:
                cancelled = True

        def should_cancel() -> bool:
            return cancelled

        gen = stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            draft_model=draft_model,
            num_draft_tokens=2,
            sampler=sampler,
            prefill_step_size=4,
            prompt_progress_callback=progress_callback,
            should_cancel=should_cancel,
        )
        with self.assertRaises(GenerationCancelled):
            next(gen)

        self.assertGreaterEqual(len(progress), 2)
        self.assertEqual(progress[0][0], 0)
        self.assertGreater(progress[0][1], 0)
        self.assertTrue(any(p > 0 for p, _ in progress))

    def test_stream_generate_speculative_cancel_before_first_step(self):
        draft_model = self.model
        sampler = make_sampler(temp=0.0)
        prompt = self._chat_prompt_tokens(
            "Please cancel before speculative first-step. " * 16
        )

        ready = False
        calls_after_ready = 0

        def progress_callback(processed: int, total: int) -> None:
            nonlocal ready
            if processed == total - 1:
                ready = True

        def should_cancel() -> bool:
            nonlocal calls_after_ready
            if not ready:
                return False
            calls_after_ready += 1
            return calls_after_ready >= 2

        gen = stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            draft_model=draft_model,
            num_draft_tokens=2,
            sampler=sampler,
            prefill_step_size=2048,
            prompt_progress_callback=progress_callback,
            should_cancel=should_cancel,
        )
        with self.assertRaises(GenerationCancelled):
            next(gen)

        self.assertTrue(ready)
        self.assertEqual(calls_after_ready, 2)

    def test_stream_generate_speculative_prompt_progress_callback(self):
        draft_model = self.model
        sampler = make_sampler(temp=0.0)
        prompt = self._chat_prompt_tokens("Track speculative prompt progress. " * 16)

        progress: List[tuple[int, int]] = []

        def progress_callback(processed: int, total: int) -> None:
            progress.append((processed, total))

        for _ in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=4,
            draft_model=draft_model,
            num_draft_tokens=2,
            sampler=sampler,
            prefill_step_size=4,
            prompt_progress_callback=progress_callback,
        ):
            pass

        self.assertGreaterEqual(len(progress), 2)
        first_processed, first_total = progress[0]
        self.assertEqual(first_processed, 0)
        self.assertGreater(first_total, 0)
        self.assertTrue(any(p == t and t > 0 for p, t in progress))

        prev = -1
        for processed, total in progress:
            self.assertGreaterEqual(processed, prev)
            self.assertLessEqual(processed, total)
            prev = processed

    def test_stream_generate_input_embeddings(self):
        sampler = make_sampler(temp=0.0)  # determinate sampler

        # get prompt embeddings
        messages = [{"role": "user", "content": "Say 'TEST' and nothing else"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        prompt_embeddings = self.model.model.embed_tokens(prompt)

        response = ""
        for generation_result in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            sampler=sampler,
            input_embeddings=prompt_embeddings,
        ):
            response += generation_result.text

        self.assertEqual("TEST", response)

    def test_stream_generate_input_embeddings_prefill(self):
        sampler = make_sampler(temp=0.0)  # determinate sampler

        # get prompt embeddings
        messages = [{"role": "user", "content": "Say 'TEST' and nothing else"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        prompt_embeddings = self.model.model.embed_tokens(prompt)

        # setup prompt progress callback to track batched prefill
        num_prompt_processing_callbacks = 0

        def progress_callback(processed: int, total: int) -> None:
            nonlocal num_prompt_processing_callbacks
            num_prompt_processing_callbacks += 1

        # generate
        prefill_step_size = 5
        response = ""
        for generation_result in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            sampler=sampler,
            input_embeddings=prompt_embeddings,
            prefill_step_size=prefill_step_size,
            prompt_progress_callback=progress_callback,
        ):
            response += generation_result.text

        self.assertEqual("TEST", response)
        num_embeddings = prompt_embeddings.shape[0]
        self.assertTrue(
            num_embeddings / prefill_step_size < num_prompt_processing_callbacks
        )

    def test_batch_matches_single(self):

        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        gen = BatchGenerator(
            self.model, stop_tokens=self.tokenizer.eos_token_ids, max_tokens=1
        )
        uids = gen.insert(prompts)
        batch_responses = {r.uid: r for r in gen.next()}

        # Do a test for each prompt the logits are close
        for e, prompt in enumerate(prompts):

            for response in stream_generate(
                self.model, self.tokenizer, prompt, max_tokens=1
            ):
                blp = batch_responses[uids[e]].logprobs
                lp = response.logprobs
                self.assertTrue(mx.allclose(blp, lp))
                break

    def test_many_batches(self):

        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            max_tokens=1,
            prefill_batch_size=2,
            prefill_step_size=8,
            completion_batch_size=3,
        )
        uids = gen.insert(prompts)
        batch_responses = {}
        not_in = True
        iters = 0
        while responses := gen.next():
            for r in responses:
                not_in &= r.uid not in batch_responses
                batch_responses[r.uid] = r
            iters += 1
        # only one token per prompt means only one response per prompt
        self.assertTrue(not_in)

        # completion batch size is too small for a single iteration
        self.assertTrue(iters > 1)

        # Do a test for each prompt the logits are close
        for e, prompt in enumerate(prompts):

            for response in stream_generate(
                self.model, self.tokenizer, prompt, max_tokens=1
            ):
                blp = batch_responses[uids[e]].logprobs
                lp = response.logprobs
                self.assertTrue(mx.allclose(blp, lp))
                break

    def test_batch_unique_max_toks(self):
        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            prefill_batch_size=2,
            prefill_step_size=8,
            completion_batch_size=3,
        )
        num_toks = [2, 3, 4, 5]
        uids = gen.insert(prompts, max_tokens=num_toks)
        batch_responses = {uid: [] for uid in uids}
        while responses := gen.next():
            for r in responses:
                batch_responses[r.uid].append(r.token)

        # Do a test for each prompt the logits are close
        for e, prompt in enumerate(prompts):

            tokens = []
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=num_toks[e],
            ):
                tokens.append(response.token)

            batch_tokens = batch_responses[uids[e]]
            self.assertEqual(tokens, batch_tokens)

    def test_batch_sliding_window(self):
        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        self.model.make_cache = lambda: [
            RotatingKVCache(max_size=4) for _ in self.model.layers
        ]
        batch_gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            max_tokens=10,
            prefill_batch_size=1,
            prefill_step_size=8,
            completion_batch_size=2,
        )
        uids = batch_gen.insert(prompts)
        batch_responses = {uid: [] for uid in uids}
        while responses := batch_gen.next():
            for r in responses:
                batch_responses[r.uid].append(r.logprobs)

        for e, uid in enumerate(uids):
            for i, response in enumerate(
                stream_generate(
                    self.model,
                    self.tokenizer,
                    prompts[e],
                    max_tokens=10,
                )
            ):
                batch_logprobs = batch_responses[uid][i]
                logprobs = response.logprobs
                self.assertTrue(
                    mx.allclose(batch_logprobs, logprobs, rtol=1e-4, atol=1e-4)
                )

        del self.model.make_cache

    def test_batch_continued_generation(self):
        for rotating in [False, True]:
            if rotating:
                self.model.make_cache = lambda: [
                    RotatingKVCache(max_size=4) for _ in self.model.layers
                ]

            # Make the prompts
            prompts_a = [
                "Write a story about Einstein",
                "Hi",
                "What time is it?",
                "How tall is Mt Everest?",
            ]
            prompts_a = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for p in prompts_a
            ]
            prompts_b = [
                "Another one",
                "sup?",
                "And how about the date?",
                "Mt Olympus?",
            ]
            prompts_b = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for p in prompts_b
            ]

            # Generate once
            batch_gen = BatchGenerator(
                self.model,
                stop_tokens=self.tokenizer.eos_token_ids,
                max_tokens=10,
                prefill_batch_size=1,
                prefill_step_size=8,
                completion_batch_size=2,
            )
            uids = batch_gen.insert(prompts_a)
            caches = {uid: None for uid in uids}
            while responses := batch_gen.next():
                for r in responses:
                    if r.finish_reason is not None:
                        caches[r.uid] = r.prompt_cache
            caches = [caches[uid] for uid in uids]

            # Generate the 2nd time
            uids = batch_gen.insert(prompts_b, caches=caches)
            batch_responses = {uid: [] for uid in uids}
            while responses := batch_gen.next():
                for r in responses:
                    batch_responses[r.uid].append(r.logprobs)

            for e, uid in enumerate(uids):
                for i, response in enumerate(
                    stream_generate(
                        self.model,
                        self.tokenizer,
                        prompts_b[e],
                        max_tokens=10,
                        prompt_cache=caches[e],
                    )
                ):
                    batch_logprobs = batch_responses[uid][i]
                    logprobs = response.logprobs
                    self.assertTrue(
                        mx.allclose(batch_logprobs, logprobs, rtol=1e-4, atol=1e-4)
                    )

            if rotating:
                del self.model.make_cache


if __name__ == "__main__":
    unittest.main()
