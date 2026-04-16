import io
import json
import math
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

from mlx_lm import kld


def fake_load(path_or_hf_repo, *args, **kwargs):
    if path_or_hf_repo == "baseline-model":
        return FakeModel(vocab_size=4), FakeTokenizer()
    if path_or_hf_repo == "candidate-model":
        return FakeModel(vocab_size=4, shift=0.2), FakeTokenizer()
    raise AssertionError(path_or_hf_repo)


def run_main(argv):
    stdout = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
        kld.main(argv)
    return stdout.getvalue()


class FakeTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {f"tok_{i}": i for i in range(4)}
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        return dict(self.vocab)


class FakeModel:
    def __init__(self, vocab_size=4, shift=0.0):
        self.vocab_size = vocab_size
        self.shift = shift

    def __call__(self, inputs):
        tokens = np.array(inputs)
        vocab = np.arange(self.vocab_size, dtype=np.float32)
        logits = -((vocab - (tokens[..., None] % self.vocab_size)) ** 2)
        logits = logits + self.shift * vocab
        return mx.array(logits.astype(np.float32))


class TestKld(unittest.TestCase):
    def test_build_cache_writes_manifest(self):
        tokens = mx.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=mx.int32)

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "baseline-cache"
            model_dir = Path(tmp) / "baseline-model"
            model_dir.mkdir()
            save_file(
                {"weight": np.zeros((1,), dtype=np.float32)},
                str(model_dir / "model.safetensors"),
                metadata={"format": "mlx"},
            )

            with (
                patch("mlx_lm.kld.derive_cache_dir", return_value=cache_dir),
                patch("mlx_lm.kld.load", side_effect=fake_load) as mock_load,
                patch("mlx_lm.kld.load_eval_tokens", return_value=tokens),
                patch("mlx_lm.kld._download", return_value=model_dir),
            ):
                summary = json.loads(
                    run_main(
                        [
                            "--model",
                            "candidate-model",
                            "--baseline-model",
                            "baseline-model",
                            "--top-k",
                            "1",
                            "--sequence-length",
                            "4",
                            "--num-samples",
                            "2",
                            "--batch-size",
                            "1",
                        ]
                    )
                )

            self.assertEqual(mock_load.call_count, 2)
            self.assertEqual(summary["baseline_model"], "baseline-model")
            self.assertEqual(summary["top_k"], 1)
            self.assertEqual(summary["baseline_cache"], str(cache_dir.resolve()))

            manifest = kld.load_manifest(cache_dir.resolve())
            self.assertEqual(manifest["top_k"], 1)
            self.assertEqual(manifest["baseline_model"], "baseline-model")
            self.assertEqual(manifest["vocab_hash"], kld._get_vocab_hash(FakeTokenizer()))

    def test_existing_cache_reuses_manifest(self):
        tokens = mx.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=mx.int32)

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "baseline-cache"
            model_dir = Path(tmp) / "baseline-model"
            model_dir.mkdir()
            save_file(
                {"weight": np.zeros((1,), dtype=np.float32)},
                str(model_dir / "model.safetensors"),
                metadata={"format": "mlx"},
            )

            with (
                patch("mlx_lm.kld.derive_cache_dir", return_value=cache_dir),
                patch("mlx_lm.kld.load", side_effect=fake_load),
                patch("mlx_lm.kld.load_eval_tokens", return_value=tokens),
                patch("mlx_lm.kld._download", return_value=model_dir),
            ):
                run_main(
                    [
                        "--model",
                        "candidate-model",
                        "--baseline-model",
                        "baseline-model",
                        "--top-k",
                        "1",
                        "--sequence-length",
                        "4",
                        "--num-samples",
                        "2",
                        "--batch-size",
                        "1",
                    ]
                )

            with (
                patch("mlx_lm.kld.load", side_effect=fake_load) as mock_load,
                patch("mlx_lm.kld.load_eval_tokens") as mock_load_eval_tokens,
            ):
                summary = json.loads(
                    run_main(
                        [
                            "--model",
                            "candidate-model",
                            "--baseline-cache",
                            str(cache_dir),
                            "--top-k",
                            "9",
                        ]
                    )
                )

            mock_load_eval_tokens.assert_not_called()
            self.assertEqual(mock_load.call_count, 1)
            self.assertEqual(mock_load.call_args.args[0], "candidate-model")
            self.assertEqual(summary["top_k"], 1)
            self.assertEqual(summary["baseline_cache"], str(cache_dir.resolve()))

    def test_missing_baseline_cache_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "missing-cache"
            with self.assertRaisesRegex(ValueError, "Baseline cache does not exist"):
                kld.main(
                    [
                        "--model",
                        "candidate-model",
                        "--baseline-cache",
                        str(cache_dir),
                    ]
                )

    def test_kl_from_cached_batch_matches_bucketed_math(self):
        model_logprobs = mx.array(
            np.log(
                np.array([[[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]]], dtype=np.float32)
            ),
            dtype=mx.float32,
        )
        cached = {
            "indices": mx.array([[[0], [2]]], dtype=mx.int32),
            "logprobs": mx.array(
                [[[math.log(0.6)], [math.log(0.7)]]], dtype=mx.float32
            ),
            "tail_mass": mx.array([[0.4, 0.3]], dtype=mx.float32),
        }

        kl = kld.kl_from_cached_batch(model_logprobs, cached)
        expected = np.array(
            [
                [
                    0.6 * (math.log(0.6) - math.log(0.5))
                    + 0.4 * (math.log(0.4) - math.log(0.5)),
                    0.0,
                ]
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(np.array(kl), expected, rtol=1e-6, atol=1e-6)

    def test_candidate_tokenizer_hash_mismatch_raises(self):
        manifest = {
            "baseline_model": "baseline-model",
            "top_k": 1,
            "data_path": "dataset",
            "sequence_length": 3,
            "num_samples": 1,
            "seed": 123,
            "batch_size": 1,
            "vocab_size": 3,
            "vocab_hash": kld._get_vocab_hash(FakeTokenizer({"a": 0, "b": 1, "c": 2})),
        }
        args = types.SimpleNamespace(
            model="candidate-model",
            trust_remote_code=False,
        )
        candidate_tokenizer = FakeTokenizer({"a": 0, "c": 1, "b": 2})

        with patch(
            "mlx_lm.kld.load",
            return_value=(FakeModel(vocab_size=3), candidate_tokenizer),
        ):
            with self.assertRaisesRegex(ValueError, "token IDs do not match"):
                kld.load_candidate_model(args, manifest)


if __name__ == "__main__":
    unittest.main()
