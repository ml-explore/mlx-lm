# Copyright © 2026 Valliammai Valliappan

import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx_lm.benchmark as bm


def fake_metrics(prompt_tps, generation_tps, memory):
    """Create fake LLM performance measurements."""
    return SimpleNamespace(
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        peak_memory=memory,
    )


class TestBenchmarkAPI(unittest.TestCase):

    def replace(self, name, value):
        """Temporarily replace a real dependency with a fake."""
        patcher = patch.object(bm, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def setUp(self):
        """Prepare fake model outputs before each test."""

        # One warmup, followed by two measured trials.
        self.responses = iter(
            [
                fake_metrics(50, 10, 0.10),
                fake_metrics(100, 20, 0.20),
                fake_metrics(200, 40, 0.30),
            ]
        )

        # Fake Apple MLX runtime.
        fake_group = SimpleNamespace(
            rank=lambda: 0,
            size=lambda: 1,
        )

        fake_mx = SimpleNamespace(
            distributed=SimpleNamespace(
                init=lambda: fake_group,
            ),
            random=SimpleNamespace(
                seed=lambda value: None,
                randint=lambda low, high, shape: SimpleNamespace(
                    tolist=lambda: [[1] * shape[1] for _ in range(shape[0])]
                ),
            ),
        )

        # Fake model and tokenizer.
        fake_tokenizer = SimpleNamespace(_eos_token_ids={1})

        self.mock_load = MagicMock(
            return_value=(
                object(),
                fake_tokenizer,
                {"vocab_size": 100},
            )
        )

        # Fake single-prompt inference.
        self.mock_stream = MagicMock(
            side_effect=lambda *args, **kwargs: iter([next(self.responses)])
        )

        # Fake batch inference.
        self.mock_batch = MagicMock(
            return_value=SimpleNamespace(stats=fake_metrics(300, 60, 0.40))
        )

        self.replace("mx", fake_mx)
        self.replace("load", self.mock_load)
        self.replace("stream_generate", self.mock_stream)
        self.replace("batch_generate", self.mock_batch)

    def test_results_and_averages(self):
        """Check returned objects and average measurements."""

        result = bm.run_benchmark(
            model="fake-model",
            prompt_tokens=16,
            generation_tokens=8,
            num_trials=2,
        )

        self.assertIsInstance(result, bm.BenchmarkResult)
        self.assertEqual(result.model, "fake-model")
        self.assertEqual(result.prompt_tokens, 16)
        self.assertEqual(result.generation_tokens, 8)

        self.assertEqual(len(result.trials), 2)
        self.assertIsInstance(result.trials[0], bm.TrialMetrics)

        self.assertAlmostEqual(result.prompt_tps, 150)
        self.assertAlmostEqual(result.generation_tps, 30)
        self.assertAlmostEqual(result.peak_memory, 0.25)

        # One warmup + two measured runs.
        self.assertEqual(self.mock_stream.call_count, 3)
        self.assertEqual(
            self.mock_stream.call_args.kwargs["max_tokens"],
            8,
        )

    def test_invalid_inputs(self):
        """Invalid values should raise ValueError."""

        invalid_cases = [
            {"num_trials": 0},
            {"prompt_tokens": 0},
            {"generation_tokens": -1},
            {"batch_size": 0},
            {"prefill_step_size": 0},
            {"delay": -1},
        ]

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    bm.run_benchmark(**kwargs)

        # Invalid inputs must fail before loading a model.
        self.mock_load.assert_not_called()

    def test_silent_mode(self):
        """verbose=False should not print benchmark progress."""

        output = StringIO()

        with redirect_stdout(output):
            bm.run_benchmark(
                model="fake-model",
                num_trials=2,
                verbose=False,
            )

        self.assertEqual(output.getvalue(), "")

    def test_verbose_output(self):
        """verbose=True should print trial and average metrics."""

        output = StringIO()

        with redirect_stdout(output):
            bm.run_benchmark(
                model="fake-model",
                num_trials=2,
                verbose=True,
            )

        printed = output.getvalue()

        self.assertIn("Running warmup", printed)
        self.assertIn("Trial 1:", printed)
        self.assertIn("Trial 2:", printed)
        self.assertIn("Averages:", printed)

    def test_cli_arguments(self):
        """The existing CLI must call our new Python function."""

        argv = [
            "mlx_lm.benchmark",
            "--model",
            "fake-model",
            "-p",
            "16",
            "-g",
            "8",
            "-n",
            "2",
        ]

        with patch.object(bm, "run_benchmark") as mock_run:
            with patch.object(sys, "argv", argv):
                bm.main()

        mock_run.assert_called_once()

        passed = mock_run.call_args.kwargs

        self.assertEqual(passed["model"], "fake-model")
        self.assertEqual(passed["prompt_tokens"], 16)
        self.assertEqual(passed["generation_tokens"], 8)
        self.assertEqual(passed["num_trials"], 2)
        self.assertTrue(passed["verbose"])

    def test_batch_mode(self):
        """Batch size greater than one uses batch_generate."""

        result = bm.run_benchmark(
            model="fake-model",
            prompt_tokens=16,
            generation_tokens=8,
            batch_size=2,
            num_trials=1,
        )

        self.assertEqual(result.batch_size, 2)
        self.assertAlmostEqual(result.generation_tps, 60)

        # One warmup + one measured trial.
        self.assertEqual(self.mock_batch.call_count, 2)
        self.mock_stream.assert_not_called()

        # Verify that two prompts were supplied.
        prompts = self.mock_batch.call_args.args[2]
        self.assertEqual(len(prompts), 2)


if __name__ == "__main__":
    unittest.main()
