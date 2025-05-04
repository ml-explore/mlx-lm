import tempfile
import unittest
from pathlib import Path

import mlx.nn as nn
import mlx.optimizers as optim
from safetensors.numpy import safe_open

from mlx_lm.tuner.datasets import TextDataset
from mlx_lm.tuner.trainer import TrainingArgs, load_checkpoint, save_checkpoint, train

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

    def load_weights(self, weights_file, strict=True):
        pass


class DummyTokenizer:
    def __init__(self):
        self.eos_token_id = 3
        self.vocab_size = 128

    def encode(self, text):
        return [1, 2, 3] * 10

    def decode(self, tokens):
        return "dummy text"


class DummyDataset(TextDataset):
    def __init__(self, num_samples=100):
        self._data = [{"text": f"dummy text {i}"} for i in range(num_samples)]
        self.tokenizer = DummyTokenizer()
        self.text_key = "text"

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def process(self, item):
        return self.tokenizer.encode(item[self.text_key])


# ---------------------
# Test Case
# ---------------------


class TestLoraResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        cls.adapter_dir = Path(cls.test_dir) / "adapters"
        cls.adapter_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def setUp(self):
        # Create fresh model and datasets for each test
        self.model = MockModel()
        self.dataset = DummyDataset()
        self.train_set = self.dataset
        self.val_set = self.dataset
        self.tokenizer = DummyTokenizer()
        self.optimizer = optim.Adam(learning_rate=1e-4)

    def get_training_args(self, iters, adapter_file):
        return TrainingArgs(
            iters=iters,
            batch_size=16,
            val_batches=2,
            steps_per_report=5,
            steps_per_save=5,
            adapter_file=adapter_file,
            max_seq_length=64,
        )

    def test_initial_training(self):
        """Test the initial training phase without resuming."""
        adapter_file = self.adapter_dir / "adapters.safetensors"
        args = self.get_training_args(iters=5, adapter_file=adapter_file)

        train(
            self.model,
            self.tokenizer,
            self.optimizer,
            self.train_set,
            self.val_set,
            args=args,
        )

        # Verify checkpoint files were created
        self.assertTrue((self.adapter_dir / "0000005_adapters.safetensors").exists())
        self.assertTrue(adapter_file.exists())

    def test_resume_training(self):
        """Test resuming training from a checkpoint."""
        adapter_file = self.adapter_dir / "adapters.safetensors"

        # First training phase
        args = self.get_training_args(iters=5, adapter_file=adapter_file)
        train(
            self.model,
            self.tokenizer,
            self.optimizer,
            self.train_set,
            self.val_set,
            args=args,
        )

        # Resume training
        resume_file = self.adapter_dir / "0000005_adapters.safetensors"
        start_iteration = load_checkpoint(self.model, self.optimizer, resume_file)

        args = self.get_training_args(iters=10, adapter_file=adapter_file)
        train(
            self.model,
            self.tokenizer,
            self.optimizer,
            self.train_set,
            self.val_set,
            args=args,
            start_step=start_iteration,
        )

        # Verify final checkpoint exists
        final_file = self.adapter_dir / "0000010_adapters.safetensors"
        self.assertTrue(final_file.exists())

    def test_checkpoint_metadata(self):
        """Test that checkpoint metadata is correctly saved and loaded."""
        adapter_file = self.adapter_dir / "adapters.safetensors"
        args = self.get_training_args(iters=5, adapter_file=adapter_file)

        train(
            self.model,
            self.tokenizer,
            self.optimizer,
            self.train_set,
            self.val_set,
            args=args,
        )

        checkpoint_file = self.adapter_dir / "0000005_adapters.safetensors"
        with safe_open(str(checkpoint_file), framework="numpy") as f:
            metadata = f.metadata()

            # Verify required metadata fields
            self.assertIn("iteration", metadata)
            self.assertEqual(metadata["iteration"], "5")
            self.assertIn("optimizer", metadata)

            # Verify optimizer state
            optimizer_config = metadata["optimizer"]
            self.assertIn("learning_rate", optimizer_config)
            self.assertIn("step", optimizer_config)
            self.assertIn("m", optimizer_config)
            self.assertIn("v", optimizer_config)

    def test_save_checkpoint(self):
        """Test that save_checkpoint correctly saves model weights and metadata."""
        # Create a test checkpoint
        checkpoint_file = self.adapter_dir / "test_checkpoint.safetensors"

        # Save checkpoint
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_file=checkpoint_file,
            iteration=1,
            trained_tokens=100,
            loss=0.5,
        )

        # Verify checkpoint was created
        self.assertTrue(checkpoint_file.exists())

        # Verify checkpoint contents
        with safe_open(str(checkpoint_file), framework="numpy") as f:
            metadata = f.metadata()

            # Check metadata fields
            self.assertIn("iteration", metadata)
            self.assertEqual(metadata["iteration"], "1")
            self.assertIn("trained_tokens", metadata)
            self.assertEqual(metadata["trained_tokens"], "100")
            self.assertIn("loss", metadata)
            self.assertEqual(metadata["loss"], "0.500000")
            self.assertIn("optimizer", metadata)

            # Verify optimizer state
            optimizer_config = metadata["optimizer"]
            self.assertIn("learning_rate", optimizer_config)
            self.assertIn("step", optimizer_config)
            self.assertIn("m", optimizer_config)
            self.assertIn("v", optimizer_config)


if __name__ == "__main__":
    unittest.main()
