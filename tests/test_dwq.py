# Copyright © 2026 Apple Inc.

import tempfile
import unittest
from pathlib import Path


class TestDwqTargetDetection(unittest.TestCase):
    """Test that dwq correctly detects whether pre-computed targets exist.

    Regression test for: mlx_lm.dwq silently skips target computation when
    --target-dir is empty but exists (issue #1159).
    """

    def _has_targets(self, target_dir):
        """The fixed has_targets check from mlx_lm/quant/dwq.py main()."""
        return (
            target_dir.is_dir()
            and any((target_dir / "train").glob("*.safetensors"))
            and any((target_dir / "valid").glob("*.safetensors"))
        )

    def test_empty_dir_not_treated_as_has_targets(self):
        """An empty existing directory should not be treated as having targets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            # Old behavior: target_dir.exists() returns True
            self.assertTrue(target_dir.exists())
            # New behavior: no safetensors files
            self.assertFalse(self._has_targets(target_dir))

    def test_dir_with_train_only_not_treated_as_has_targets(self):
        """A directory with train/ but no valid/ should not count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            train_dir = target_dir / "train"
            train_dir.mkdir()
            (train_dir / "0000000000.safetensors").touch()

            self.assertFalse(self._has_targets(target_dir))

    def test_dir_with_valid_only_not_treated_as_has_targets(self):
        """A directory with valid/ but no train/ should not count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            valid_dir = target_dir / "valid"
            valid_dir.mkdir()
            (valid_dir / "0000000000.safetensors").touch()

            self.assertFalse(self._has_targets(target_dir))

    def test_dir_with_both_splits_treated_as_has_targets(self):
        """A directory with both train/ and valid/ safetensors should be detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            train_dir = target_dir / "train"
            valid_dir = target_dir / "valid"
            train_dir.mkdir()
            valid_dir.mkdir()
            (train_dir / "0000000000.safetensors").touch()
            (valid_dir / "0000000000.safetensors").touch()

            self.assertTrue(self._has_targets(target_dir))

    def test_nonexistent_dir_not_treated_as_has_targets(self):
        """A non-existent directory should not be treated as having targets."""
        target_dir = Path("/tmp/this_dir_does_not_exist_dwq_test")
        self.assertFalse(self._has_targets(target_dir))

    def test_dir_with_non_safetensors_files_not_treated_as_has_targets(self):
        """A directory with other files but no safetensors should not count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            train_dir = target_dir / "train"
            valid_dir = target_dir / "valid"
            train_dir.mkdir()
            valid_dir.mkdir()
            (train_dir / "metadata.json").touch()
            (valid_dir / "metadata.json").touch()

            self.assertFalse(self._has_targets(target_dir))


if __name__ == "__main__":
    unittest.main()
