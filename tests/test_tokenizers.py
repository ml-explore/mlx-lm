# Copyright © 2024 Apple Inc.

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download

from mlx_lm.tokenizer_utils import (
    BPEStreamingDetokenizer,
    NaiveStreamingDetokenizer,
    SPMStreamingDetokenizer,
)
from mlx_lm.utils import load_tokenizer


class TestTokenizers(unittest.TestCase):

    def check_tokenizer(self, tokenizer):
        def check(tokens):
            expected_text = tokenizer.decode(tokens)
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            text = ""
            for e, t in enumerate(tokens):
                detokenizer.add_token(t)
                seg = detokenizer.last_segment
                text += seg
                self.assertEqual(detokenizer.tokens, tokens[: e + 1])
            detokenizer.finalize()
            text += detokenizer.last_segment
            self.assertEqual(text, expected_text)

        tokens = tokenizer.encode("こんにちは！私の名前はAI")
        check(tokens)

        tokens = tokenizer.encode("⊕ ⊻ ∧ ¬")
        check(tokens)

        tokens = tokenizer.encode("a ,b")
        check(tokens)

        tokens = tokenizer.encode('{"why_its_funny" :"a_joke_explainer" ,"rating":3.5}')
        check(tokens)

        tokens = tokenizer.encode("3 3")
        check(tokens)

        tokens = tokenizer.encode("import 'package:flutter/material.dart';")
        check(tokens)

        tokens = tokenizer.encode("hello\nworld")
        check(tokens)

    def test_tokenizers(self):
        tokenizer_repos = [
            ("mlx-community/Qwen1.5-0.5B-Chat-4bit", BPEStreamingDetokenizer),
            ("mlx-community/Mistral-7B-v0.2-4bit", SPMStreamingDetokenizer),
            ("mlx-community/Phi-3.5-mini-instruct-4bit", SPMStreamingDetokenizer),
            ("mlx-community/Mistral-7B-Instruct-v0.3", SPMStreamingDetokenizer),
            ("mlx-community/Llama-3.2-1B-Instruct-4bit", BPEStreamingDetokenizer),
            ("mlx-community/Falcon3-7B-Instruct-4bit", BPEStreamingDetokenizer),
        ]
        for tokenizer_repo, expected_detokenizer in tokenizer_repos:
            with self.subTest(tokenizer=tokenizer_repo):
                tokenizer = load_tokenizer(tokenizer_repo)
                tokenizer.decode([0, 1, 2])
                self.assertTrue(isinstance(tokenizer.detokenizer, expected_detokenizer))
                self.check_tokenizer(tokenizer)

        # Try one with a naive detokenizer
        tokenizer = load_tokenizer("mlx-community/Llama-3.2-1B-Instruct-4bit")
        tokenizer._detokenizer = NaiveStreamingDetokenizer(tokenizer)
        self.check_tokenizer(tokenizer)

    def test_special_tokens(self):
        tokenizer_repo = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
        tokenizer = load_tokenizer(tokenizer_repo)

        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        detokenizer.add_token(tokenizer.eos_token_id)
        detokenizer.finalize()

        self.assertEqual(detokenizer.last_segment, tokenizer.eos_token)

    def test_tool_calling(self):
        tokenizer_repo = "mlx-community/Qwen3-4B-4bit"
        tokenizer = load_tokenizer(tokenizer_repo)
        self.assertTrue(tokenizer.has_tool_calling)
        self.assertEqual(tokenizer.tool_call_start, "<tool_call>")
        self.assertEqual(tokenizer.tool_call_end, "</tool_call>")

        tokenizer_repo = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        tokenizer = load_tokenizer(tokenizer_repo)
        self.assertFalse(tokenizer.has_tool_calling)

    def test_thinking(self):
        tokenizer_repo = "mlx-community/Qwen3-4B-4bit"
        tokenizer = load_tokenizer(tokenizer_repo)
        self.assertTrue(tokenizer.has_thinking)
        self.assertEqual(tokenizer.think_start, "<think>")
        self.assertEqual(tokenizer.think_end, "</think>")

        tokenizer_repo = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        tokenizer = load_tokenizer(tokenizer_repo)
        self.assertFalse(tokenizer.has_thinking)
        self.assertIsNone(tokenizer.think_start)
        self.assertIsNone(tokenizer.think_end)
        self.assertIsNone(tokenizer.think_start_id)
        self.assertIsNone(tokenizer.think_end_id)

    def test_plamo3_tokenizer_jsonl(self):
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            vocab = [
                ["<|plamo:unk|>", 0.0, "UNKNOWN"],
                ["<|plamo:bos|>", 0.0, "CONTROL"],
                ["<|plamo:eos|>", 0.0, "CONTROL"],
                ["<|plamo:pad|>", 0.0, "CONTROL"],
                ["hello", 10.0, "NORMAL"],
                ["world", 10.0, "NORMAL"],
                [" ", 1.0, "NORMAL"],
            ]
            vocab.extend([[f"<0x{i:02X}>", -100.0, "BYTE"] for i in range(256)])
            with open(model_path / "tokenizer.jsonl", "w", encoding="utf-8") as f:
                for row in vocab:
                    print(json.dumps(row, ensure_ascii=False), file=f)

            tokenizer_config = {
                "tokenizer_class": "Plamo3Tokenizer",
                "unk_token": "<|plamo:unk|>",
                "bos_token": "<|plamo:bos|>",
                "eos_token": "<|plamo:eos|>",
                "pad_token": "<|plamo:pad|>",
                "add_bos_token": True,
                "add_eos_token": False,
                "clean_up_tokenization_spaces": False,
            }
            with open(model_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
                json.dump(tokenizer_config, f)

            tokenizer = load_tokenizer(model_path)
            token_ids = tokenizer.encode("hello😀", add_special_tokens=False)

            self.assertEqual(tokenizer.encode("hello", add_special_tokens=True)[0], 1)
            self.assertEqual(tokenizer.decode(token_ids), "hello😀")
            self.assertEqual(
                tokenizer.decode(
                    tokenizer.encode("hello world"), skip_special_tokens=True
                ),
                "hello world",
            )


if __name__ == "__main__":
    unittest.main()
