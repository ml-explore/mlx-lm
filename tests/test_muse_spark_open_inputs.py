# Copyright © 2026 Apple Inc.

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mlx import nn
from tokenizers import Tokenizer, decoders, models

from mlx_lm import tokenizer_utils
from mlx_lm.chat_templates.muse_spark_open import RESPONSE_FORMAT, apply_chat_template
from mlx_lm.generate import TextStateMachine
from mlx_lm.processors import load_processor, muse_spark_open

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


class TestMuseSparkOpenProcessor(unittest.TestCase):
    def test_preprocess_image(self):
        processor = muse_spark_open.Processor.from_config(
            {"vision_config": {"navit_config": {"max_num_image_tokens": 16}}}
        )
        # (height, width), pixel shape, number of tokens
        test_cases = [
            # A multiple of 28, no resize.
            ((56, 84), (3, 56, 84), 6),
            # Resized to the nearest grid with the same aspect ratio.
            ((100, 50), (3, 112, 56), 8),
            # Resized down to max_num_image_tokens.
            ((280, 280), (3, 112, 112), 16),
        ]
        for size, shape, num_tokens in test_cases:
            with self.subTest(size=size):
                image = np.zeros((*size, 3), dtype=np.uint8)
                pixels, n = processor.preprocess_image(image)
                self.assertEqual(pixels.shape, shape)
                self.assertEqual(n, num_tokens)

    def test_call(self):
        processor = muse_spark_open.Processor()
        images = [
            np.zeros((56, 84, 3), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.uint8),
        ]
        text, pixel_values = processor("A <image> and B <image>.", images=images)
        expected = (
            "A <|image_start|>" + "<|patch|>" * 6 + "<|image_end|>"
            " and B <|image_start|><|patch|><|image_end|>."
        )
        self.assertEqual(text, expected)
        self.assertEqual([p.shape for p in pixel_values], [(3, 56, 84), (3, 28, 28)])

    def test_call_errors(self):
        processor = muse_spark_open.Processor()
        image = np.zeros((28, 28, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "for 1 images"):
            processor("<image> <image>", images=[image])
        with self.assertRaisesRegex(ValueError, "for 1 images"):
            processor("No image", images=[image])
        with self.assertRaisesRegex(ValueError, "already has"):
            processor("<image><|patch|>", images=[image])

    def test_load_processor(self):
        model = nn.Module()
        self.assertIsNone(load_processor(model, {"model_type": "muse_spark_open"}))
        self.assertIsNone(load_processor(model, {"vision_config": {}}))
        config = {
            "vision_config": {"patch_size": 16},
            "processor_type": "muse_spark_open",
        }
        processor = load_processor(model, config)
        self.assertEqual(processor.patch_size, 16)


class TestMuseSparkOpenChatTemplate(unittest.TestCase):
    def test_user(self):
        prompt = apply_chat_template(
            [{"role": "user", "content": "Hi"}], add_generation_prompt=True
        )
        expected = (
            "<|begin_of_text|><|start|>system<|message|>"
            '# Valid recipients: "self", None, "user".<|eot|>'
            "<|start|>user<|message|>Hi<|eot|><|start|>assistant"
        )
        self.assertEqual(prompt, expected)

    def test_system(self):
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ]
        prompt = apply_chat_template(messages, add_generation_prompt=True)
        expected = (
            "<|begin_of_text|><|start|>system<|message|>Be brief.\n\n"
            '# Valid recipients: "self", None, "user".<|eot|>'
            "<|start|>user<|message|>Hi<|eot|><|start|>assistant"
        )
        self.assertEqual(prompt, expected)

    def test_tools(self):
        prompt = apply_chat_template(
            [{"role": "user", "content": "Weather in Paris?"}],
            tools=TOOLS,
            add_generation_prompt=True,
        )
        schema = (
            '{"name": "get_weather", "description": "Get the weather.", '
            '"parameters": {"type": "object", "properties": {"city": '
            '{"type": "string"}}, "required": ["city"]}}'
        )
        self.assertIn("// Function schemas\n" + schema + "\n", prompt)
        self.assertTrue(
            prompt.endswith(
                '# Valid recipients: "self", None, '
                '"[func_name] (e.g., get_weather)", "user".<|eot|>'
                "<|start|>user<|message|>Weather in Paris?<|eot|><|start|>assistant"
            )
        )

    def test_tool_call_and_result(self):
        arguments = '{"city": "Paris", "days": 3, "metric": true, "tags": ["a"]}'
        messages = [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": arguments},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 20}'},
        ]
        prompt = apply_chat_template(messages, add_generation_prompt=True)
        expected = (
            "<|begin_of_text|><|start|>system<|message|>"
            '# Valid recipients: "self", None, "user".<|eot|>'
            "<|start|>user<|message|>Weather in Paris?<|eot|>"
            "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n"
            '<atem:invoke name="get_weather">\n'
            '<atem:parameter name="city">Paris</atem:parameter>'
            '<atem:parameter name="days">3</atem:parameter>'
            '<atem:parameter name="metric">true</atem:parameter>'
            '<atem:parameter name="tags">["a"]</atem:parameter>\n'
            "</atem:invoke>\n</atem:function_calls><|eot|>"
            '<|start|>tool get_weather<|message|>{"temp": 20}<|eot|>'
            "<|start|>assistant"
        )
        self.assertEqual(prompt, expected)

    def test_reasoning(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "reasoning": "Greet back."},
        ]
        prompt = apply_chat_template(messages)
        expected = (
            "<|begin_of_text|><|start|>system<|message|>"
            '# Valid recipients: "self", None, "user".<|eot|>'
            "<|start|>user<|message|>Hi<|eot|>"
            "<|start|>assistant to=self<|message|>Greet back.<|eom|>"
            "<|start|>assistant to=user<|message|>Hello!<|eot|>"
        )
        self.assertEqual(prompt, expected)


class TestMuseSparkOpenResponseFormat(unittest.TestCase):
    def test_recipients(self):
        text = (
            " to=self<|message|>Plan.<|eom|><|start|>assistant"
            " to=commentary<|message|>Checking.<|eom|><|start|>assistant"
            " to=user<|message|>Done."
        )
        sm = TextStateMachine(RESPONSE_FORMAT["transitions"])
        state = sm.make_state(RESPONSE_FORMAT["initial_state"])
        state, segments = TextStateMachine.split(state, text)
        _, rest, last = TextStateMachine.flush(state)
        out = {}
        for s, t in segments + [(last, rest)]:
            out[s] = out.get(s, "") + t
        self.assertEqual(out["reasoning"], "Plan.")
        self.assertEqual(out["normal"], "Checking.Done.")

    def test_jinja_template(self):
        # A checkpoint with its own template still gets the response format.
        with tempfile.TemporaryDirectory() as path:
            path = Path(path)
            tokenizer = Tokenizer(models.BPE())
            tokenizer.decoder = decoders.ByteLevel()
            tokenizer.save(str(path / "tokenizer.json"))
            config = {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "chat_template": "{{ messages[0]['content'] }}",
            }
            (path / "tokenizer_config.json").write_text(json.dumps(config))
            tokenizer = tokenizer_utils.load(
                path, defaults={"chat_template_type": "muse_spark_open"}
            )
            messages = [{"role": "user", "content": "Hi"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        self.assertEqual(prompt, "Hi")
        self.assertEqual(tokenizer.response_initial_state, "header")
        self.assertEqual(tokenizer.response_transitions, RESPONSE_FORMAT["transitions"])


if __name__ == "__main__":
    unittest.main()
