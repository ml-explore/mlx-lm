# Copyright © 2024 Apple Inc.

import unittest
from unittest.mock import patch, MagicMock
from typing import List

import mlx.core as mx

from mlx_lm.generate import (
    BatchGenerator,
    GenerationResponse,
    generate,
    stream_generate,
    batch_generate
)
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
        draft_model, _ = load(self.HF_MODEL_PATH)

        results: List[GenerationResponse] = []
        drafted: List[bool] = []

        # make a determinate sampler
        sampler = make_sampler(temp=0.0)
        messages = [{"role": "user", "content": "hello"}]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
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
            messages, add_generation_prompt=True
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
            messages, add_generation_prompt=True
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


    @patch('mlx_lm.generate.BatchGenerator')
    def test_batch_generate_with_return_tokens(self, mock_batch_generator_class):
        """
        Verify that when return_tokens=True, the BatchResponse object
        contains the correct list of generated token IDs, excluding the stop token.
        """
        # --- Arrange ---
        mock_generator_instance = mock_batch_generator_class.return_value
        mock_generator_instance.insert.return_value = [0, 1]
        
        mock_generator_instance.next.side_effect = [
            [
                MagicMock(uid=0, token=101, finish_reason=None),
                MagicMock(uid=1, token=201, finish_reason=None)
            ],
            [
                MagicMock(uid=0, token=102, finish_reason="stop"), # This token will be excluded
                MagicMock(uid=1, token=202, finish_reason="length") # This one will be included
            ],
            []
        ]
        mock_generator_instance.stats.return_value = MagicMock()
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.side_effect = ["text 1", "text 2"]
        
        prompts = [[1], [2]]

        # --- Act ---
        response = batch_generate(
            self.model, mock_tokenizer, prompts, return_tokens=True
        )

        # --- Assert ---
        self.assertIsNotNone(response.tokens)
        self.assertEqual(len(response.tokens), 2)
        # The test now correctly expects only the non-stop tokens
        self.assertEqual(response.tokens[0], [101]) 
        self.assertEqual(response.tokens[1], [201, 202])
        self.assertIsNone(response.logprobs)

    @patch('mlx_lm.generate.BatchGenerator')
    def test_batch_generate_with_return_logprobs(self, mock_batch_generator_class):
        """
        Verify that when return_logprobs=True, the BatchResponse object
        contains the correct log probabilities, excluding the stop token.
        """
        # --- Arrange ---
        logprob1 = mx.array([-0.1, -2.3])
        logprob2 = mx.array([-0.5, -1.5])
        logprob3 = mx.array([-0.2, -2.0]) # This one will be excluded
        logprob4 = mx.array([-0.8, -1.2])

        mock_generator_instance = mock_batch_generator_class.return_value
        mock_generator_instance.insert.return_value = [0, 1]
        
        mock_generator_instance.next.side_effect = [
            [
                MagicMock(uid=0, token=101, logprobs=logprob1, finish_reason=None),
                MagicMock(uid=1, token=201, logprobs=logprob2, finish_reason=None)
            ],
            [
                MagicMock(uid=0, token=102, logprobs=logprob3, finish_reason="stop"),
                MagicMock(uid=1, token=202, logprobs=logprob4, finish_reason="length")
            ],
            []
        ]
        mock_generator_instance.stats.return_value = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.side_effect = ["text 1", "text 2"]

        prompts = [[1], [2]]

        # --- Act ---
        response = batch_generate(
            self.model, mock_tokenizer, prompts, return_logprobs=True
        )

        # --- Assert ---
        self.assertIsNotNone(response.logprobs)
        self.assertEqual(len(response.logprobs), 2)
        # The test now correctly expects only the logprobs of the non-stop tokens
        self.assertTrue(mx.allclose(response.logprobs[0], mx.stack([logprob1])))
        self.assertTrue(mx.allclose(response.logprobs[1], mx.stack([logprob2, logprob4])))
        self.assertIsNone(response.tokens)

if __name__ == "__main__":
    unittest.main()
