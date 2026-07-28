import tempfile
import unittest
from pathlib import Path

from mlx_lm.share import DirectoryEntry, get_files


class TestShare(unittest.TestCase):
    def test_get_files_with_directory_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "target"
            target.mkdir()
            (target / "weights.safetensors").touch()
            (root / "alias").symlink_to("target", target_is_directory=True)

            _, entries = get_files(root)
            entries = {entry.path: entry for entry in entries}

            self.assertEqual(entries["alias"].entry_type, "symlink")
            self.assertEqual(entries["alias"].dst, "target")
            self.assertIsNone(entries["target"].dst)


if __name__ == "__main__":
    unittest.main()
