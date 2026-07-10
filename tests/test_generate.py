# Copyright © 2024 Apple Inc.

import random
import unittest
from typing import List

import mlx.core as mx

from mlx_lm.batch_admission import LinearStateCost, StateBudget
from mlx_lm.generate import (
    BatchGenerator,
    GenerationResponse,
    StopSequenceMatcher,
    batch_generate,
    generate,
    generate_step,
    stream_generate,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load


class TestGenerate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        cls.model, cls.tokenizer = load(cls.HF_MODEL_PATH)
        cls.model.set_dtype(mx.float32)

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
        batch_responses = {r.uid: r for r in gen.next_generated()}

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
        while responses := gen.next_generated():
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
        while responses := gen.next_generated():
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
        while responses := batch_gen.next_generated():
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

    def test_batch_generate_with_logits_processors(self):
        """Test that batch_generate with logits_processors produces correct results."""
        logit_bias = {0: 2000.0, 1: -2000.0}
        processors = make_logits_processors(logit_bias)

        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            logits_processors=processors,
        )
        prompt = self.tokenizer.encode("hello")
        uids = batch_gen.insert([prompt])
        response = batch_gen.next_generated()[0]
        logprobs = response.logprobs
        self.assertEqual(logprobs[0].item(), 0.0)
        self.assertEqual(logprobs.argmin().item(), 1)

        del batch_gen

        logit_bias = {0: 2000.0}
        processors = make_logits_processors(logit_bias)
        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            logits_processors=processors,
        )

        (uid0,) = batch_gen.insert([prompt])

        logit_bias = {1: 2000.0}
        processors = make_logits_processors(logit_bias)
        (uid1,) = batch_gen.insert([prompt], logits_processors=[processors])

        logit_bias = {2: 2000.0}
        processors = make_logits_processors(logit_bias)
        (uid2,) = batch_gen.insert([prompt], logits_processors=[processors])

        responses = batch_gen.next_generated()
        responses = {response.uid: response for response in responses}
        self.assertEqual(responses[uid0].logprobs[0].item(), 0.0)
        self.assertEqual(responses[uid1].logprobs[1].item(), 0.0)
        self.assertEqual(responses[uid2].logprobs[2].item(), 0.0)

    def test_batch_generate_processor_tokens_match_prompt_on_first_step(self):
        prompt = self.tokenizer.encode("hello")
        seen = []

        def processor(tokens, logits):
            seen.append(tokens)
            return logits

        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            logits_processors=[processor],
        )
        batch_gen.insert([prompt])
        batch_gen.next_generated()

        self.assertTrue(hasattr(seen[0], "shape"))
        self.assertEqual(seen[0].tolist(), prompt)

    def test_batch_generate_function_with_logits_processors(self):
        """Test that batch_generate function with logits_processors produces correct results."""
        logit_bias = {0: 2000.0, 1: -2000.0}
        processors = make_logits_processors(logit_bias)

        prompts = [self.tokenizer.encode("hello")]
        response = batch_generate(
            self.model,
            self.tokenizer,
            prompts,
            max_tokens=1,
            logits_processors=processors,
        )
        self.assertEqual(len(response.texts), 1)
        generated_token = self.tokenizer.encode(response.texts[0])[0]
        self.assertEqual(generated_token, 0)

    def test_batch_generate_with_samplers(self):
        """Test that batch_generate with logits_processors produces correct results."""
        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            sampler=lambda _: mx.array([1]),
        )
        prompt = self.tokenizer.encode("hello")
        uids = batch_gen.insert([prompt])
        response = batch_gen.next_generated()[0]
        self.assertEqual(response.token, 1)

        del batch_gen

        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            sampler=lambda _: mx.array([1]),
        )

        (uid0,) = batch_gen.insert([prompt])
        uid1, uid2 = batch_gen.insert(
            [prompt, prompt],
            samplers=[lambda _: mx.array([2]), lambda _: mx.array([3])],
        )

        responses = batch_gen.next_generated()
        responses = {response.uid: response for response in responses}
        self.assertEqual(responses[uid0].token, 1)
        self.assertEqual(responses[uid1].token, 2)
        self.assertEqual(responses[uid2].token, 3)

    def test_kv_budget_none_is_current_behavior(self):
        """kv_budget_bytes=None must leave admission purely count-based."""
        gen = BatchGenerator(self.model, max_tokens=4)
        self.assertIsNone(gen.kv_budget_bytes)
        # _budget_admissible must be the identity when budgeting is off
        self.assertEqual(gen._budget_admissible(5), 5)

    def test_batch_generator_rejects_peak_only_state_policy(self):
        """AR engine cannot infer resident state for a peak-only projector."""

        def quadratic_prefill_state(state):
            return 100 * state.projected_units**2

        with self.assertRaisesRegex(ValueError, "requires LinearStateCost"):
            BatchGenerator(
                self.model,
                max_tokens=0,
                state_budget=StateBudget(6_000, quadratic_prefill_state),
            )

        class _NonlinearSubclass(LinearStateCost):
            def __call__(self, state):
                return quadratic_prefill_state(state)

        with self.assertRaisesRegex(ValueError, "requires LinearStateCost"):
            BatchGenerator(
                self.model,
                state_budget=StateBudget(6_000, _NonlinearSubclass(0, 1)),
            )

        with self.assertRaises(ValueError):
            BatchGenerator(
                self.model,
                kv_budget_bytes=1_000,
                kv_cost=(0, 1),
                state_budget=StateBudget(1_000, quadratic_prefill_state),
            )

    def test_kv_budget_requires_kv_cost(self):
        with self.assertRaises(ValueError):
            BatchGenerator(self.model, kv_budget_bytes=1 << 30)

    def test_kv_budget_limits_admission(self):
        """Only as many rows admit as the projected bytes allow."""
        prompt = self.tokenizer.encode("hello world")
        per_tok = 1000.0
        # Each row projects to ~(len(prompt)+4)*1000 bytes; budget fits 2 rows
        row = (len(prompt) + 4) * per_tok
        gen = BatchGenerator(
            self.model,
            max_tokens=4,
            kv_budget_bytes=int(2.5 * row),
            kv_cost=(0.0, per_tok, 1),
        )
        gen.insert([prompt] * 4)
        self.assertEqual(gen._budget_admissible(4), 2)

    def test_kv_budget_counts_fixed_row_state(self):
        """Hybrid-style fixed per-row bytes gate admission too."""
        prompt = self.tokenizer.encode("hello")
        fixed = 10_000.0
        gen = BatchGenerator(
            self.model,
            max_tokens=4,
            kv_budget_bytes=int(2.5 * fixed),
            kv_cost=(fixed, 0.0, 1),
        )
        gen.insert([prompt] * 4)
        self.assertEqual(gen._budget_admissible(4), 2)

    def test_kv_budget_caps_projection_at_max_kv_size(self):
        """max_kv_size bounds the projected per-row growth."""
        prompt = self.tokenizer.encode("hello world " * 20)
        per_tok = 1000.0
        gen = BatchGenerator(
            self.model,
            max_tokens=10_000,
            max_kv_size=8,
            kv_budget_bytes=int(2.5 * 8 * per_tok),
            kv_cost=(0.0, per_tok, 1),
        )
        gen.insert([prompt] * 4)
        # Uncapped projection would admit 0; capped admits 2
        self.assertEqual(gen._budget_admissible(4), 2)

    def test_state_budget_caps_active_continued_history(self):
        """A primed prefill row's capped growth starts after its history."""

        class _FakeCache:
            nbytes = 10_000

        class _StubPromptBatch:
            uids = [123]
            max_tokens = [0]
            prompt_cache = [_FakeCache()]

            def __len__(self):
                return 1

        gen = BatchGenerator(
            self.model,
            max_tokens=0,
            max_kv_size=16,
            kv_budget_bytes=17_000,
            kv_cost=(0.0, 1_000.0, 1),
        )
        gen._prompt_batch = _StubPromptBatch()
        # 10 cached history + 10 new prompt tokens, none processed yet.
        # Only 6 more token slots can materialize before the size-16 cap.
        gen._currently_processing = [[[list(range(10))], 0, 10, False, 10]]
        gen.insert([[1]])
        # Global-cohort semantics: the active row's final extent caps at
        # max_kv_size=16 (its 10 history + 10 new would exceed it), and the
        # 1-unit candidate shares the cohort width: 2 rows x capped-16 x
        # 1000 = 32K. The cap is what keeps this finite — uncapped it
        # would be 2 x 20 x 1000 = 40K.
        self.assertEqual(gen._budget_admissible(1), 0)
        gen.kv_budget_bytes = 32_000
        self.assertEqual(gen._budget_admissible(1), 1)

    def test_kv_budget_oversized_single_request_liveness(self):
        """A request alone over budget still admits when nothing is active."""
        prompt = self.tokenizer.encode("hello world " * 50)
        gen = BatchGenerator(
            self.model,
            max_tokens=64,
            kv_budget_bytes=10,  # absurdly small
            kv_cost=(0.0, 1000.0, 1),
        )
        gen.insert([prompt])
        self.assertEqual(gen._budget_admissible(1), 1)

    def test_kv_budget_continued_generation_full_projection(self):
        """A primed row still costs its full final length minus existing
        bytes; history tokens are part of the projection (codex repro:
        26KB projection must NOT fit a 20KB budget when a row is active)."""

        class _FakeCache:
            def __init__(self, nbytes):
                self.nbytes = nbytes

        class _StubGenBatch:
            """Minimal active-batch stand-in: one row, no remaining growth."""

            uids = [999]
            tokens = [[1] * 5]
            max_tokens = [5]
            _num_tokens = [5]
            prompt_cache = []

            def __len__(self):
                return 1

        prompt = list(range(11))  # 11 new tokens
        gen = BatchGenerator(
            self.model,
            max_tokens=5,
            kv_budget_bytes=20_000,
            kv_cost=(0.0, 1000.0, 1),
        )
        gen.insert(
            [prompt],
            caches=[[_FakeCache(10_000)]],
            all_tokens=[list(range(10))],  # 10 history tokens in the cache
        )
        gen._generation_batch = _StubGenBatch()
        # Full projection = (10 + 11 + 5) * 1000 = 26000; credit 10000 →
        # need 16000; committed already holds the 10000 live bytes →
        # 26000 > 20000 must reject, and liveness must NOT fire (row active).
        self.assertEqual(gen._budget_admissible(1), 0)

    def test_kv_budget_no_credit_without_history(self):
        """Cache bytes with no all_tokens history grant no byte credit."""

        class _FakeCache:
            def __init__(self, nbytes):
                self.nbytes = nbytes

        class _StubGenBatch:
            uids = [999]
            tokens = [[1] * 5]
            max_tokens = [5]
            _num_tokens = [5]
            prompt_cache = []

            def __len__(self):
                return 1

        prompt = list(range(10))
        gen = BatchGenerator(
            self.model,
            max_tokens=0,
            kv_budget_bytes=18_000,
            kv_cost=(0.0, 1000.0, 1),
        )
        gen.insert([prompt], caches=[[_FakeCache(9_000)]])
        gen._generation_batch = _StubGenBatch()  # suppress liveness escape
        # Global cohort: stub (5-unit done row) and the 10-unit candidate
        # share one allocation at the cohort-max width: 2 x 10 x 1000 =
        # 20000. The candidate's 9000 unverifiable supplied-cache bytes
        # are ADDED on top (never credited, never absorbed by the live
        # floor): committed = 29000. 18000 rejects; 29000 admits.
        self.assertEqual(gen._budget_admissible(1), 0)
        gen.kv_budget_bytes = 29_000
        self.assertEqual(gen._budget_admissible(1), 1)

    def test_kv_budget_remove_releases_headroom(self):
        """H4: removing a queued row frees its committed bytes (mixed
        cached/uncached queue)."""

        class _FakeCache:
            def __init__(self, nbytes):
                self.nbytes = nbytes

        expensive = list(range(10))
        cheap = list(range(2))
        gen = BatchGenerator(
            self.model,
            max_tokens=0,
            kv_budget_bytes=13_000,
            kv_cost=(0.0, 1000.0, 1),
        )

        class _StubGenBatch:
            uids = [999]
            tokens = [[1] * 5]
            max_tokens = [5]
            _num_tokens = [5]
            prompt_cache = []

            def __len__(self):
                return 1

        (uid_primed,) = gen.insert(
            [expensive],
            caches=[[_FakeCache(10_000)]],
            all_tokens=[list(range(10))],
        )
        gen.insert([cheap, cheap])
        # Suppress the liveness escape: pretend one row is generating
        empty_gen_batch = gen._generation_batch
        gen._generation_batch = _StubGenBatch()
        # Primed row projects 20 units x 1000 shared with the stub →
        # merge 2 x 20000 = 40000 >> 13000: rejected (its 10000 live
        # bytes also appear in floors, never as credit)
        self.assertEqual(gen._budget_admissible(3), 0)
        gen.remove([uid_primed])
        gen._generation_batch = empty_gen_batch
        # Primed row's live bytes and width are gone entirely: the two
        # cheap rows share a 2-unit width: 2 x 2000 = 4000 <= 13000
        self.assertEqual(gen._budget_admissible(2), 2)

    def test_kv_budget_rejects_nonfinite(self):
        for bad_budget, bad_cost in (
            (float("inf"), (0.0, 1.0)),
            (1 << 30, (float("nan"), 1.0)),
            (1 << 30, (0.0, float("inf"))),
        ):
            with self.assertRaises(ValueError):
                BatchGenerator(
                    self.model,
                    kv_budget_bytes=bad_budget,
                    kv_cost=bad_cost,
                )

    def test_budget_requires_allocation_step(self):
        """Fail closed: growing state without a validated step is rejected
        at construction — both the kv_cost 2-tuple path and a direct
        LinearStateCost without allocation geometry."""
        from mlx_lm.batch_admission import LinearStateCost, StateBudget

        with self.assertRaises(ValueError):
            BatchGenerator(
                self.model,
                kv_budget_bytes=1 << 30,
                kv_cost=(0.0, 1000.0),  # 2-tuple: no step
            )
        with self.assertRaises(ValueError):
            BatchGenerator(
                self.model,
                kv_budget_bytes=1 << 30,
                kv_cost=(0.0, 1000.0, None),  # 3-tuple None-step bypass
            )
        with self.assertRaises(ValueError):
            BatchGenerator(
                self.model,
                state_budget=StateBudget(
                    1 << 30, LinearStateCost(0.0, 1000.0)  # growing, no step
                ),
            )
        # Fixed-only state may omit the step
        gen = BatchGenerator(
            self.model,
            state_budget=StateBudget(1 << 30, LinearStateCost(1000.0, 0.0)),
        )
        self.assertIsNotNone(gen.state_budget)
        del gen

    def test_unselected_resident_caches_add_not_max(self):
        """Reviewer P1 regression: resident bytes of still-unselected queued
        rows are simultaneous with projected growth of the selected prefix —
        they must ADD to committed, never fold into a max. Exact repro:
        budget 1000; selected uncached 1-unit row projects 256 (stepped);
        unselected queued supplied cache holds 900 resident bytes;
        max(256, 900) = 900 would wrongly admit; 256 + 900 = 1156 must
        reject."""

        class _FakeCache:
            def __init__(self, nbytes):
                self.nbytes = nbytes

        class _StubGenBatch:  # suppress liveness escape
            uids = [999]
            tokens = [[1]]
            max_tokens = [1]
            _num_tokens = [1]
            prompt_cache = []

            def __len__(self):
                return 1

        gen = BatchGenerator(
            self.model,
            max_tokens=0,
            kv_budget_bytes=1_300,  # stub projects 256 too: see arithmetic
            kv_cost=(0.0, 1.0, 256),
        )
        gen.insert([[1]])  # selected: 1 unit -> stepped 256 bytes
        gen.insert([[1]], caches=[[_FakeCache(900)]])  # unselected resident
        gen._generation_batch = _StubGenBatch()
        # cohort(selected+stub) = 2 rows x 256 = 512; + unselected 900
        # = 1412 > 1300 must reject. A max() formulation would compute
        # max(512, 900) = 900 <= 1300 and wrongly admit.
        self.assertEqual(gen._budget_admissible(1), 0)
        gen.kv_budget_bytes = 1_500
        self.assertEqual(gen._budget_admissible(1), 1)
        del gen

    def test_live_floor_never_absorbs_unverified_bytes(self):
        """Codex order-of-operations regression: the live floor applies to
        the base projection only. Active admitted live 1000, base global
        projection below 1000, selected unverified cache 900: committed
        must be at least 1900 — max(base + 900, 1000) would lose the
        unverified charge into the floor."""

        class _FakeCache:
            def __init__(self, nbytes):
                self.nbytes = nbytes

        class _StubGenBatch:
            uids = [999]
            tokens = [[1]]
            max_tokens = [1]
            _num_tokens = [1]
            prompt_cache = [_FakeCache(1000)]  # admitted live = 1000

            def __len__(self):
                return 1

        gen = BatchGenerator(
            self.model,
            max_tokens=0,
            kv_budget_bytes=1_800,
            kv_cost=(0.0, 1.0, 1),
        )
        # Selected candidate: 1 unit projected (base cohort tiny), with a
        # 900-byte unverifiable supplied cache (no history)
        gen.insert([[1]], caches=[[_FakeCache(900)]])
        gen._generation_batch = _StubGenBatch()
        state = gen._candidate_admission_state(gen._unprocessed_sequences[0])
        committed = gen._cohort_committed([state])
        self.assertGreaterEqual(committed, 1_900)
        # And the budget decision agrees: 1800 rejects, 1900 admits
        self.assertEqual(gen._budget_admissible(1), 0)
        gen.kv_budget_bytes = 1_900
        self.assertEqual(gen._budget_admissible(1), 1)
        del gen

    def test_w3_shape_active_cache_never_exceeds_budget(self):
        """Reviewer test 2 — the exact W3 smoke shape as a regression:
        0.5 GiB budget, eight 512-prompt/16-gen rows, REAL measured cost.
        The live admitted cache must never exceed the budget at any
        scheduler step (the original defect admitted 0.70 GB into 0.5 GiB).
        """
        from mlx_lm.server import _measure_kv_cost

        fixed, per_tok, step = _measure_kv_cost(self.model)
        budget = int(0.5 * (1 << 30))
        gen = BatchGenerator(
            self.model,
            max_tokens=16,
            prefill_batch_size=8,
            kv_budget_bytes=budget,
            kv_cost=(fixed, per_tok, step),
        )
        prompt = [(i % 100) + 1 for i in range(512)]
        uids = gen.insert([prompt] * 8)
        done = set()
        max_live = 0
        for _ in range(400):
            _, responses = gen.next()
            live = gen.prompt_cache_nbytes
            max_live = max(max_live, live)
            self.assertLessEqual(live, budget)
            for r in responses:
                if r.finish_reason is not None:
                    done.add(r.uid)
            if len(done) == len(uids):
                break
        self.assertEqual(len(done), len(uids))
        self.assertGreater(max_live, 0)
        del gen

    def test_heterogeneous_cohort_projection_covers_actual(self):
        """Reviewer test 5 — widely different prompt lengths and unequal
        max_tokens: the committed projection at admission must cover the
        ACTUAL merged BatchKVCache bytes reached during processing."""
        from mlx_lm.server import _measure_kv_cost

        fixed, per_tok, step = _measure_kv_cost(self.model)
        gen = BatchGenerator(
            self.model,
            prefill_batch_size=4,
            kv_budget_bytes=1 << 33,  # generous: measuring, not gating
            kv_cost=(fixed, per_tok, step),
        )
        short = [(i % 100) + 1 for i in range(8)]
        longer = [(i % 100) + 1 for i in range(300)]
        uids = gen.insert([short, longer], max_tokens=[4, 40])
        states = [
            gen._candidate_admission_state(seq) for seq in gen._unprocessed_sequences
        ]
        projected = gen._cohort_committed(states)
        done = set()
        max_live = 0
        for _ in range(200):
            _, responses = gen.next()
            max_live = max(max_live, gen.prompt_cache_nbytes)
            for r in responses:
                if r.finish_reason is not None:
                    done.add(r.uid)
            if len(done) == len(uids):
                break
        self.assertEqual(len(done), len(uids))
        self.assertGreaterEqual(projected, max_live)
        del gen

    def test_continued_and_removal_at_capacity_boundary(self):
        """Reviewer test 6 — continued generation crossing an allocation
        boundary, then removal: projection covers actual across the
        boundary, and removing a row releases real headroom."""
        from mlx_lm.server import _measure_kv_cost

        fixed, per_tok, step = _measure_kv_cost(self.model)
        gen = BatchGenerator(
            self.model,
            prefill_batch_size=4,
            kv_budget_bytes=1 << 33,
            kv_cost=(fixed, per_tok, step),
        )
        # Prompt just below the boundary; generation crosses it
        prompt = [(i % 100) + 1 for i in range(step - 4)]
        uid_a, uid_b = gen.insert([prompt, prompt], max_tokens=[12, 12])
        crossed = False
        done = set()
        for _ in range(200):
            _, responses = gen.next()
            live = gen.prompt_cache_nbytes
            states = [
                gen._candidate_admission_state(seq)
                for seq in gen._unprocessed_sequences
            ]
            projected = gen._cohort_committed(states)
            self.assertGreaterEqual(projected, live)
            if live > 2 * (step - 4) * per_tok:
                crossed = True  # allocation stepped past the boundary
            for r in responses:
                if r.finish_reason is not None:
                    done.add(r.uid)
            if len(done) == 2:
                break
        self.assertTrue(crossed)
        before = gen.prompt_cache_nbytes
        # rows completed -> removed by the engine; live must have released
        self.assertLessEqual(before, 2 * per_tok * step)
        del gen

    def test_kv_budget_e2e_generation_completes(self):
        """Budgeted end-to-end run finishes all requests (queued, not lost)."""
        prompt = self.tokenizer.encode("hello world")
        per_tok = 1000.0
        row = (len(prompt) + 4) * per_tok
        gen = BatchGenerator(
            self.model,
            max_tokens=4,
            # Generous REAL-scale budget: stale-width floors read actual
            # cache nbytes (~114 KB/token on this model), so the budget
            # must be sized to reality, not the synthetic per-token cost
            kv_budget_bytes=int(100e6),
            kv_cost=(0.0, per_tok, 1),
        )
        uids = gen.insert([prompt] * 4)
        done = set()
        for _ in range(200):
            for r in gen.next_generated():
                if r.finish_reason is not None:
                    done.add(r.uid)
            if len(done) == len(uids):
                break
        self.assertEqual(len(done), len(uids))

    def test_batch_generate_with_stop_matchers(self):
        """Test that batch_generate with per-sequence stop_matchers stops on different tokens."""
        batch_gen = BatchGenerator(
            self.model,
            max_tokens=10,
        )
        prompt = self.tokenizer.encode("hello")

        sm_0 = StopSequenceMatcher([[0]])
        sm_1 = StopSequenceMatcher([[1]])
        sm_2 = StopSequenceMatcher([[2]])

        processor_0 = make_logits_processors({0: 2000.0})
        processor_1 = make_logits_processors({1: 2000.0})
        processor_2 = make_logits_processors({2: 2000.0})

        uid0, uid1, uid2 = batch_gen.insert(
            [prompt, prompt, prompt],
            logits_processors=[processor_0, processor_1, processor_2],
            stop_matchers=[sm_0, sm_1, sm_2],
        )

        responses = batch_gen.next_generated()
        responses = {response.uid: response for response in responses}

        self.assertEqual(responses[uid0].token, 0)
        self.assertEqual(responses[uid1].token, 1)
        self.assertEqual(responses[uid2].token, 2)
        self.assertEqual(responses[uid0].finish_reason, "stop")
        self.assertEqual(responses[uid1].finish_reason, "stop")
        self.assertEqual(responses[uid2].finish_reason, "stop")

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
                prefill_batch_size=4,
                prefill_step_size=8,
                completion_batch_size=2,
            )
            uids = batch_gen.insert(prompts_a)
            caches = {uid: None for uid in uids}
            while responses := batch_gen.next_generated():
                for r in responses:
                    if r.finish_reason is not None:
                        caches[r.uid] = r.prompt_cache
            caches = [caches[uid] for uid in uids]

            # Generate the 2nd time
            uids = batch_gen.insert(prompts_b, caches=caches)
            batch_responses = {uid: [] for uid in uids}
            while responses := batch_gen.next_generated():
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

    def _continued_generation_test_helper(self, model):
        def rand_prompt(n):
            return [random.randint(0, 1000) for _ in range(n)]

        # Make the prompts
        prompts_a = [
            rand_prompt(5),
            rand_prompt(3),
            rand_prompt(8),
            rand_prompt(1),
        ]
        prompts_b = [
            rand_prompt(2),
            rand_prompt(7),
            rand_prompt(4),
            rand_prompt(6),
        ]

        # Generate once
        batch_gen = BatchGenerator(
            model,
            stop_tokens={},
            max_tokens=10,
            prefill_batch_size=4,
            prefill_step_size=32,
            completion_batch_size=2,
        )

        uids = batch_gen.insert(prompts_a)
        caches = {uid: None for uid in uids}
        while responses := batch_gen.next_generated():
            for r in responses:
                if r.finish_reason is not None:
                    caches[r.uid] = r.prompt_cache

        caches = [caches[uid] for uid in uids]

        # Generate the 2nd time
        uids = batch_gen.insert(prompts_b, caches=caches)
        batch_responses = {uid: [] for uid in uids}
        while responses := batch_gen.next_generated():
            for r in responses:
                batch_responses[r.uid].append(r.logprobs)

        for e, uid in enumerate(uids):
            for i, (_, logprobs) in enumerate(
                generate_step(
                    mx.array(prompts_b[e]),
                    model,
                    max_tokens=10,
                    prompt_cache=caches[e],
                )
            ):
                batch_logprobs = batch_responses[uid][i]
                self.assertTrue(
                    mx.allclose(batch_logprobs, logprobs, rtol=1e-4, atol=1e-4)
                )

    def test_batch_continued_generation_ssm(self):
        from mlx_lm.models import mamba2

        random.seed(0)
        mx.random.seed(4)

        # Make a small SSM model
        args = mamba2.ModelArgs(
            model_type="mamba2",
            num_heads=8,
            head_dim=16,
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=128,
            state_size=32,
            num_hidden_layers=4,
            layer_norm_epsilon=1e-4,
            conv_kernel=3,
            n_groups=4,
            use_bias=False,
            use_conv_bias=False,
            tie_word_embeddings=True,
            time_step_limit=(0.01, 10),
            time_step_rank="auto",
        )
        model = mamba2.Model(args)
        self._continued_generation_test_helper(model)

    def test_batch_continued_generation_gated_delta(self):
        from mlx_lm.models import qwen3_next

        random.seed(0)
        mx.random.seed(4)
        args = qwen3_next.ModelArgs(
            model_type="qwen3_next",
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=128,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=1000,
            linear_num_value_heads=4,
            linear_num_key_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=3,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            shared_expert_intermediate_size=128,
            mlp_only_layers=[0],
            moe_intermediate_size=128,
            rms_norm_eps=1e-5,
            head_dim=64,
            rope_theta=1000.0,
            partial_rotary_factor=0.5,
            max_position_embeddings=1000,
        )
        model = qwen3_next.Model(args)
        self._continued_generation_test_helper(model)

    def test_extend_cache_with_empty(self):
        from mlx_lm.generate import _extend_cache
        from mlx_lm.models.cache import make_prompt_cache

        cache_a = make_prompt_cache(self.model)

        prompt = mx.array([[1, 2, 3]])
        self.model(prompt, cache=cache_a)
        mx.eval([c.state for c in cache_a])

        result = _extend_cache(cache_a, [])
        self.assertEqual(len(result), len(cache_a))
        for c in result:
            self.assertGreater(c.offset, 0)

        result = _extend_cache([], cache_a)
        self.assertEqual(len(result), len(cache_a))
        for c in result:
            self.assertGreater(c.offset, 0)

    def test_remove_prompt_batch_updates_currently_processing(self):
        prompt_a = self.tokenizer.encode("Write a long story about a cat")
        prompt_b = self.tokenizer.encode("Write a long story about a dog")

        gen = BatchGenerator(
            self.model,
            max_tokens=5,
            prefill_batch_size=2,
            prefill_step_size=4,
            completion_batch_size=4,
        )
        uid_a, uid_b = gen.insert([prompt_a, prompt_b])

        gen.next()

        found = gen._find_uids([uid_a, uid_b])
        for uid in [uid_a, uid_b]:
            self.assertIn(uid, found)
            self.assertEqual(found[uid][0], 1)

        gen.remove([uid_a])

        self.assertEqual(len(gen._currently_processing), len(gen._prompt_batch))

        found = gen._find_uids([uid_b])
        self.assertIn(uid_b, found)

        while responses := gen.next_generated():
            if all(r.finish_reason is not None for r in responses):
                break

    def test_batch_max_kv_size_creates_rotating_cache(self):
        max_kv_size = 256
        gen = BatchGenerator(
            self.model,
            max_tokens=1,
            max_kv_size=max_kv_size,
        )

        prompt = self.tokenizer.encode("Write a long story about a cat")
        gen.insert([prompt])

        for r in gen.next_generated():
            if r.finish_reason is not None:
                for cache in r.prompt_cache:
                    self.assertIsInstance(cache, RotatingKVCache)
                    self.assertEqual(cache.max_size, max_kv_size)

    def test_batch_max_kv_size_limits_cache_growth(self):
        max_kv_size = 5
        gen = BatchGenerator(
            self.model,
            max_tokens=10,
            max_kv_size=max_kv_size,
            prefill_batch_size=1,
            prefill_step_size=128,
            completion_batch_size=1,
        )

        prompt = self.tokenizer.encode("Write a long story about a cat")
        gen.insert([prompt])

        for r in gen.next_generated():
            if r.finish_reason is not None:
                for cache in r.prompt_cache:
                    self.assertLessEqual(cache.keys.shape[2], max_kv_size)

    def test_batch_max_kv_size_none_creates_regular_cache(self):
        gen = BatchGenerator(
            self.model,
            max_tokens=1,
            max_kv_size=None,
        )

        prompt = self.tokenizer.encode("Write a long story about a cat")
        gen.insert([prompt])

        for r in gen.next_generated():
            if r.finish_reason is not None:
                for cache in r.prompt_cache:
                    self.assertIsInstance(cache, KVCache)

    def test_batch_generate_return_logprobs(self):
        """Test that batch_generate returns per-token logprobs when requested."""
        prompts = [
            self.tokenizer.encode("hello"),
            self.tokenizer.encode("write a poem"),
        ]
        max_tokens = 5
        response = batch_generate(
            self.model,
            self.tokenizer,
            prompts,
            max_tokens=max_tokens,
            return_logprobs=True,
            return_token_ids=True,
        )

        # Check that logprobs and token_ids are returned
        self.assertIsNotNone(response.logprobs)
        self.assertIsNotNone(response.token_ids)
        self.assertEqual(len(response.logprobs), len(prompts))
        self.assertEqual(len(response.token_ids), len(prompts))

        for i in range(len(prompts)):
            # token_ids and logprobs should have same length
            self.assertEqual(len(response.token_ids[i]), len(response.logprobs[i]))
            # logprobs should be non-positive (log-probabilities)
            for lp in response.logprobs[i]:
                self.assertLessEqual(lp, 0.0)

    def test_batch_generate_no_logprobs_by_default(self):
        """Test that batch_generate does not return logprobs by default."""
        prompts = [self.tokenizer.encode("hello")]
        response = batch_generate(
            self.model,
            self.tokenizer,
            prompts,
            max_tokens=3,
        )
        self.assertIsNone(response.logprobs)
        self.assertIsNone(response.token_ids)


if __name__ == "__main__":
    unittest.main()
