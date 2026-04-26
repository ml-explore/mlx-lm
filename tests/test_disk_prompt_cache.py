# Copyright © 2026 Apple Inc.

import json
import os
import tempfile
import unittest
from pathlib import Path

from mlx_lm.disk_prompt_cache import compute_model_id, hash_tokens


class TestHashTokens(unittest.TestCase):

    def test_basic_deterministic(self):
        h1 = hash_tokens([1, 2, 3])
        h2 = hash_tokens([1, 2, 3])
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)
        self.assertTrue(all(c in "0123456789abcdef" for c in h1))

    def test_order_matters(self):
        self.assertNotEqual(hash_tokens([1, 2, 3]), hash_tokens([3, 2, 1]))

    def test_empty(self):
        h = hash_tokens([])
        self.assertEqual(len(h), 16)

    def test_large(self):
        h = hash_tokens(list(range(50_000)))
        self.assertEqual(len(h), 16)


class TestComputeModelId(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.tmpdir.name) / "model"
        self.model_path.mkdir()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write(self, name, content):
        (self.model_path / name).write_text(content)

    def test_deterministic(self):
        self._write("model.safetensors.index.json", '{"weight_map": {}}')
        self._write("tokenizer.json", '{"model": "test"}')
        a = compute_model_id(self.model_path)
        b = compute_model_id(self.model_path)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 16)

    def test_changes_with_weights(self):
        self._write("model.safetensors.index.json", '{"weight_map": {}}')
        self._write("tokenizer.json", '{"model": "test"}')
        a = compute_model_id(self.model_path)
        self._write("model.safetensors.index.json", '{"weight_map": {"x": "y"}}')
        b = compute_model_id(self.model_path)
        self.assertNotEqual(a, b)

    def test_changes_with_tokenizer(self):
        self._write("model.safetensors.index.json", '{"weight_map": {}}')
        self._write("tokenizer.json", '{"v": 1}')
        a = compute_model_id(self.model_path)
        self._write("tokenizer.json", '{"v": 2}')
        b = compute_model_id(self.model_path)
        self.assertNotEqual(a, b)

    def test_missing_optional_files_handled(self):
        # Only weights index present; tokenizer and chat template missing.
        # Should still produce a valid id, not raise.
        self._write("model.safetensors.index.json", '{"weight_map": {}}')
        h = compute_model_id(self.model_path)
        self.assertEqual(len(h), 16)


if __name__ == "__main__":
    unittest.main()
