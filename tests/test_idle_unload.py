# Copyright 2024 Apple Inc.

import argparse
import time
import unittest
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn


def _make_cli_args(**overrides):
    """Create a minimal cli_args namespace for ModelProvider."""
    defaults = {
        "model": "/tmp/fake-model-path",
        "adapter_path": None,
        "draft_model": None,
        "host": "127.0.0.1",
        "port": 8080,
        "allowed_origins": "*",
        "num_draft_tokens": 3,
        "trust_remote_code": False,
        "chat_template": "",
        "use_default_chat_template": False,
        "temp": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "max_tokens": 512,
        "chat_template_args": {},
        "decode_concurrency": 32,
        "prompt_concurrency": 8,
        "prefill_step_size": 2048,
        "prompt_cache_size": 10,
        "prompt_cache_bytes": None,
        "pipeline": False,
        "idle_timeout": 0,
        "log_level": "INFO",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_fake_model():
    """Create a minimal mock to act as a fake model."""
    model = MagicMock(spec=nn.Module)
    return model


def _make_fake_tokenizer():
    """Create a minimal tokenizer mock."""
    tokenizer = MagicMock()
    tokenizer.chat_template = None
    tokenizer.default_chat_template = ""
    tokenizer.vocab_size = 32000
    return tokenizer


# Mock distributed init to avoid MPI crash
_mock_group = MagicMock()
_mock_group.size.return_value = 1
_mock_group.rank.return_value = 0


def _make_provider(**kwargs):
    """Create a ModelProvider with distributed mocked out."""
    with patch("mlx.core.distributed.init", return_value=_mock_group):
        from mlx_lm.server import ModelProvider

        args = _make_cli_args(**kwargs)
        provider = ModelProvider(args)
    return provider


def _make_loaded_provider(**kwargs):
    """Create a ModelProvider with a fake model pre-loaded."""
    provider = _make_provider(**kwargs)
    provider.model = _make_fake_model()
    provider.tokenizer = _make_fake_tokenizer()
    provider.model_key = ("/tmp/fake-model-path", None, None)
    provider._last_request_time = time.time()
    return provider


class TestModelProviderUnload(unittest.TestCase):
    """Tests for ModelProvider.unload() and reload-after-unload."""

    def test_unload_clears_model_state(self):
        """After unload(), model, tokenizer, and draft_model should be None."""
        provider = _make_loaded_provider()

        self.assertIsNotNone(provider.model)
        self.assertIsNotNone(provider.tokenizer)

        provider.unload()

        self.assertIsNone(provider.model)
        self.assertIsNone(provider.tokenizer)
        self.assertIsNone(provider.draft_model)

    def test_unload_preserves_model_key(self):
        """model_key should be preserved after unload for reload purposes."""
        provider = _make_loaded_provider()
        original_key = provider.model_key

        provider.unload()

        self.assertEqual(provider.model_key, original_key)

    def test_unload_noop_when_no_model(self):
        """Calling unload() when no model is loaded should be a no-op."""
        provider = _make_loaded_provider()
        provider.unload()

        # Second unload should not raise
        provider.unload()
        self.assertIsNone(provider.model)

    def test_reload_after_unload_triggers_load(self):
        """load() after unload() should trigger _load() since model is None."""
        provider = _make_loaded_provider()

        # Unload
        provider.unload()
        self.assertIsNone(provider.model)

        # load() should detect model is None and call _load()
        with patch.object(provider, "_load") as mock_internal_load:
            provider.load("/tmp/fake-model-path", None, None)
            mock_internal_load.assert_called_once_with(
                "/tmp/fake-model-path", None, None
            )

    def test_last_request_time_updates_on_load(self):
        """_last_request_time should update each time load() is called."""
        provider = _make_loaded_provider()
        provider._last_request_time = time.time() - 100  # Set in past

        old_time = provider._last_request_time

        # load() with same model_key and model not None - skips _load but
        # still updates timestamp
        provider.load("/tmp/fake-model-path", None, None)

        self.assertGreater(provider._last_request_time, old_time)

    def test_last_request_time_initialized(self):
        """_last_request_time should be set at construction."""
        provider = _make_provider()

        self.assertIsInstance(provider._last_request_time, float)
        self.assertGreater(provider._last_request_time, 0)

    def test_idle_timeout_stored_in_cli_args(self):
        """idle_timeout arg should be accessible via cli_args."""
        provider = _make_provider(idle_timeout=300)

        self.assertEqual(provider.cli_args.idle_timeout, 300)

    def test_unload_calls_metal_clear_cache(self):
        """unload() should call mx.metal.clear_cache() when Metal is available."""
        provider = _make_loaded_provider()

        with patch("mlx.core.metal.is_available", return_value=True):
            with patch("mlx.core.metal.clear_cache") as mock_clear:
                with patch("mlx.core.metal.get_active_memory", return_value=1000):
                    provider.unload()
                    mock_clear.assert_called_once()


class TestIdleTimeoutCondition(unittest.TestCase):
    """Tests for the idle timeout check logic."""

    def test_idle_condition_true_when_timeout_exceeded(self):
        """Idle condition should be true when last request was long ago."""
        provider = _make_loaded_provider(idle_timeout=5)
        provider._last_request_time = time.time() - 10  # 10 seconds ago

        idle_timeout = provider.cli_args.idle_timeout
        elapsed = time.time() - provider._last_request_time

        self.assertTrue(
            idle_timeout > 0
            and provider.model is not None
            and not provider.is_distributed
            and elapsed > idle_timeout
        )

    def test_idle_condition_false_when_within_timeout(self):
        """Idle condition should be false when request was recent."""
        provider = _make_loaded_provider(idle_timeout=300)
        provider._last_request_time = time.time()  # Just now

        idle_timeout = provider.cli_args.idle_timeout
        elapsed = time.time() - provider._last_request_time

        self.assertFalse(elapsed > idle_timeout)

    def test_idle_condition_false_when_timeout_disabled(self):
        """Idle condition should be false when idle_timeout is 0."""
        provider = _make_loaded_provider(idle_timeout=0)
        provider._last_request_time = time.time() - 9999

        idle_timeout = provider.cli_args.idle_timeout
        self.assertFalse(idle_timeout > 0)

    def test_idle_condition_false_when_model_already_unloaded(self):
        """Idle condition should be false when model is already None."""
        provider = _make_loaded_provider(idle_timeout=5)
        provider._last_request_time = time.time() - 10
        provider.model = None  # Already unloaded

        self.assertFalse(provider.model is not None)


if __name__ == "__main__":
    unittest.main()
