# Copyright © 2024 Apple Inc.

import random
import unittest
from typing import List

import mlx.core as mx

from mlx_lm.generate import (
    BatchGenerator,
    GenerationResponse,
    PendingSequence,
    PromptProcessingBatch,
    SequencePolicy,
    StopSequences,
    _merge_caches,
    batch_generate,
    generate,
    generate_step,
    stream_generate,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
from mlx_lm.sample_utils import greedy_sampler, make_logits_processors, make_sampler
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
        finished = {}
        while responses := gen.next_generated():
            for r in responses:
                batch_responses[r.uid].append(r.token)
                if r.finish_reason is not None:
                    finished[r.uid] = r

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

            # The ledger matches the cache: the prompt, then exactly the tokens
            # generated for it.
            self.assertEqual(finished[uids[e]].all_tokens, list(prompt) + batch_tokens)

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

    def test_batch_generate_with_stop_matchers(self):
        """Test that batch_generate with per-sequence stop_sequences stops on different tokens."""
        batch_gen = BatchGenerator(
            self.model,
            max_tokens=10,
        )
        prompt = self.tokenizer.encode("hello")

        ss_0 = StopSequences([[0]])
        ss_1 = StopSequences([[1]])
        ss_2 = StopSequences([[2]])

        processor_0 = make_logits_processors({0: 2000.0})
        processor_1 = make_logits_processors({1: 2000.0})
        processor_2 = make_logits_processors({2: 2000.0})

        uid0, uid1, uid2 = batch_gen.insert(
            [prompt, prompt, prompt],
            logits_processors=[processor_0, processor_1, processor_2],
            stop_sequences=[ss_0, ss_1, ss_2],
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

    def test_remove_from_prompt_batch(self):
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

        # The removed sequence is gone from the batch, and the survivor's rows
        # in the batched caches line up with its sequence again.
        self.assertEqual(gen._prompt_batch.uids, [uid_b])
        self.assertEqual(len(gen._prompt_batch.sequences), len(gen._prompt_batch))

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

    def test_batch_rejects_caches_with_keep_tokens(self):
        """BatchRotatingKVCache cannot hold keep tokens, so merging must fail
        rather than silently discard the attention sinks."""
        caches = [
            make_prompt_cache(self.model, max_kv_size=32),
            make_prompt_cache(self.model, max_kv_size=32),
        ]
        self.assertEqual(caches[0][0].keep, 4)

        prompt = self.tokenizer.encode("Write a long story about a cat")
        with self.assertRaises(ValueError) as ctx:
            batch_generate(
                self.model,
                self.tokenizer,
                [prompt, prompt],
                max_tokens=2,
                prompt_caches=caches,
            )
        self.assertIn("keep", str(ctx.exception))

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

    def test_batch_stats_empty_window(self):
        """A window that did no work reports zero, not ZeroDivisionError."""
        batch_gen = BatchGenerator(self.model)
        with batch_gen.stats() as stats:
            pass

        self.assertEqual(stats.prompt_tokens, 0)
        self.assertEqual(stats.prompt_tps, 0.0)
        self.assertEqual(stats.generation_tokens, 0)
        self.assertEqual(stats.generation_tps, 0.0)

    def test_batch_stats_windows_nest(self):
        """Windows snapshot and diff, so an inner one cannot rob an outer one."""
        batch_gen = BatchGenerator(self.model, max_tokens=2)
        prompt = self.tokenizer.encode("hello there friend")

        batch_gen.insert([prompt])
        with batch_gen.stats() as outer:
            for _ in range(4):
                batch_gen.next()
            batch_gen.insert([prompt])
            with batch_gen.stats() as inner:
                for _ in range(4):
                    batch_gen.next()

        self.assertGreater(inner.prompt_tokens, 0)
        self.assertGreater(inner.generation_tokens, 0)
        self.assertGreaterEqual(outer.prompt_tokens, inner.prompt_tokens)
        self.assertGreaterEqual(outer.generation_tokens, inner.generation_tokens)
        # The outer window saw both prompts, the inner one only the second.
        self.assertGreater(outer.prompt_tokens, inner.prompt_tokens)

    def _pending(self, uid, prompt, tokens=None):
        """A (sequence, cache) pair for building a prompt batch directly."""
        sequence = PendingSequence.create(
            policy=SequencePolicy(
                uid=uid,
                sampler=greedy_sampler,
                stop_sequences=StopSequences(),
            ),
            segments=[prompt],
            tokens=tokens,
        )
        return sequence, make_prompt_cache(self.model)

    def _prompt_batch(self, *pending):
        sequences, caches = zip(*pending)
        return PromptProcessingBatch(
            model=self.model,
            sequences=list(sequences),
            prompt_cache=_merge_caches(list(caches)),
        )

    def test_prompt_batch_split_ready_does_not_mutate(self):
        """The halves own their caches and sequences: filtering or advancing one
        must not be visible through the other, nor through the batch split."""

        def rows(batch):
            """Batch width of the first merged cache layer."""
            c = batch.prompt_cache[0]
            return 0 if c.keys is None else c.keys.shape[0]

        def steps(batch):
            return batch.prompt_cache[0]._idx

        def cursors(batch):
            return [s.cursor for s in batch.sequences]

        def tokens(batch):
            return [s.tokens for s in batch.sequences]

        batch = self._prompt_batch(
            self._pending(10, [1, 2, 3]),
            self._pending(11, list(range(20))),
            self._pending(12, [7, 8, 9]),
        )
        # One prefill step: the two short prompts finish, the long one does not,
        # so it crosses the split with a non-zero cursor to preserve.
        batch.prompt([s.take_chunk(8)[0] for s in batch.sequences])
        self.assertEqual(cursors(batch), [2, 8, 2])
        self.assertEqual(
            [s.ready_to_decode for s in batch.sequences], [True, False, True]
        )
        before = steps(batch)

        selected, remaining = batch.split_ready()

        # Prefill progress survives the copy. Resetting a cursor here would
        # silently re-prefill a prompt into a cache that already holds it.
        self.assertEqual(cursors(selected), [2, 2])
        self.assertEqual(cursors(remaining), [8])
        self.assertEqual(cursors(batch), [2, 8, 2])

        # The original is untouched, and each half took its own rows.
        self.assertEqual(batch.uids, [10, 11, 12])
        self.assertEqual((rows(batch), steps(batch)), (3, before))
        self.assertEqual(selected.uids, [10, 12])
        self.assertEqual(remaining.uids, [11])
        self.assertEqual(rows(selected), 2)
        self.assertEqual(rows(remaining), 1)
        self.assertEqual(tokens(selected), [[1, 2], [7, 8]])
        self.assertEqual(tokens(remaining), [list(range(8))])

        # Each half owns its caches: prefilling one leaves the others alone.
        # Each keeps its own step count -- ``filter`` trims the left padding
        # that became uniform once the ragged rows were separated.
        selected_steps = steps(selected)
        remaining.prompt([remaining.sequences[0].take_chunk(4)[0]])
        self.assertEqual(cursors(remaining), [12])
        self.assertGreater(steps(remaining), before)
        self.assertEqual(steps(selected), selected_steps)
        self.assertEqual((rows(batch), steps(batch)), (3, before))
        self.assertEqual(cursors(batch), [2, 8, 2])
        self.assertEqual(tokens(batch), [[1, 2], list(range(8)), [7, 8]])

    def _prompt_stream(self, segments, prefill_step_size, max_ticks=64):
        """One ``(progress, end_of_segment, end_of_prompt)`` per prompt response,
        until the prompt is consumed."""
        gen = BatchGenerator(
            self.model,
            max_tokens=1,
            prefill_batch_size=1,
            prefill_step_size=prefill_step_size,
            completion_batch_size=1,
        )
        gen.insert_segments([segments])

        stream = []
        for _ in range(max_ticks):
            prompt_responses, _ = gen.next()
            for r in prompt_responses:
                stream.append((r.progress, r.end_of_segment, r.end_of_prompt))
            if any(r.end_of_prompt for r in prompt_responses):
                break
        else:
            self.fail("the prompt never finished processing")
        return stream

    def test_prompt_progress_stream_is_exact(self):
        """A wrong ``progress`` or misplaced ``end_of_segment`` raises nothing and
        changes no token, but the server checkpoints its cache on these signals.
        Totals include the reserved final token, which is never prefilled."""
        # (segments, prefill_step_size) -> the exact stream.
        cases = {
            # A single segment, chunked evenly. 6 prefilled + 1 reserved.
            ("single", 3): (
                [[1, 2, 3, 4, 5, 6, 7]],
                [
                    ((3, 7), False, False),
                    ((6, 7), True, False),
                    ((7, 7), True, True),
                ],
            ),
            # The same prompt in one chunk: step size exceeds the segment.
            ("single", 16): (
                [[1, 2, 3, 4, 5, 6, 7]],
                [
                    ((6, 7), True, False),
                    ((7, 7), True, True),
                ],
            ),
            # Multiple segments: chunks never cross a boundary, so a short
            # trailing piece is its own tick.
            ("multi", 3): (
                [[1, 2, 3, 4], [5, 6, 7, 8]],
                [
                    ((3, 8), False, False),
                    ((4, 8), True, False),
                    ((7, 8), True, False),
                    ((8, 8), True, True),
                ],
            ),
            ("multi", 16): (
                [[1, 2, 3, 4], [5, 6, 7, 8]],
                [
                    ((4, 8), True, False),
                    ((7, 8), True, False),
                    ((8, 8), True, True),
                ],
            ),
            # A prompt of one token: nothing to prefill, it is the reserved
            # token, so the sequence promotes on the first tick.
            ("minimal", 3): ([[1]], [((1, 1), True, True)]),
            ("minimal", 16): ([[1]], [((1, 1), True, True)]),
        }

        for (shape, step_size), (segments, expected) in cases.items():
            with self.subTest(shape=shape, prefill_step_size=step_size):
                self.assertEqual(self._prompt_stream(segments, step_size), expected)


if __name__ == "__main__":
    unittest.main()
