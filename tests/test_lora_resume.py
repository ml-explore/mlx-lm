import os
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from safetensors.numpy import safe_open

from mlx_lm.tuner.datasets import CacheDataset
from mlx_lm.tuner.trainer import TrainingArgs, train

# ---------------------
# Mock Components
# ---------------------


class MockModel(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = [nn.Linear(hidden_size, hidden_size) for _ in range(2)]
        self.out = nn.Linear(hidden_size, vocab_size)

    def __call__(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)

    def freeze(self):
        pass

    def unfreeze(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass


class DummyDataset:
    def __getitem__(self, idx):
        return [i % 100 for i in range(32)]

    def __len__(self):
        return 1000


class DummyTokenizer:
    def __call__(self, texts):
        return [[i % 100 for i, _ in enumerate(text.split())] for text in texts]


# ---------------------
# Training Runner
# ---------------------


def run_training(iters, adapter_file, resume_from=None):
    model = MockModel()
    dataset = DummyDataset()
    train_set = CacheDataset(dataset)
    val_set = CacheDataset(dataset)
    tokenizer = DummyTokenizer()
    optimizer = optim.Adam(learning_rate=1e-4)

    args = TrainingArgs(
        iters=iters,
        batch_size=16,
        val_batches=2,
        steps_per_report=5,
        steps_per_save=5,
        adapter_file=adapter_file,
        max_seq_length=64,
    )

    if resume_from:
        model.load_weights(resume_from, strict=False)

    train(model, tokenizer, optimizer, train_set, val_set, args=args)


# ---------------------
# Test Case
# ---------------------


@pytest.mark.order(1)
def test_adapter_resume_and_metadata(tmp_path):
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Train for 5 iters
    adapter_file = adapter_dir / "adapters.safetensors"
    run_training(iters=5, adapter_file=adapter_file)

    assert (adapter_dir / "0000005_adapters.safetensors").exists()

    # Step 2: Resume for 5 more iters (should end at 10)
    resume_file = adapter_dir / "0000005_adapters.safetensors"
    run_training(iters=10, adapter_file=adapter_file, resume_from=resume_file)

    final_file = adapter_dir / "0000010_adapters.safetensors"
    assert final_file.exists()

    # Step 3: Check metadata
    with safe_open(str(final_file), framework="numpy") as f:
        metadata = f.metadata()
        assert "step" in metadata
        assert metadata["step"] == "10"
