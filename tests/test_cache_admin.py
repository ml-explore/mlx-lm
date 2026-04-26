# Copyright © 2026 Apple Inc.

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _run(*args, capture=True):
    cmd = [sys.executable, "-m", "mlx_lm.cache_admin", *args]
    return subprocess.run(cmd, capture_output=capture, text=True)


class TestCacheAdmin(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)
        # Make a valid empty disk-cache layout
        (self.dir / "format-version").write_text("1\n")
        (self.dir / "models").mkdir()
        m = self.dir / "models" / "abc1234567890def"
        (m / "entries").mkdir(parents=True)
        (m / "info.json").write_text(
            json.dumps(
                {
                    "model_id": "abc1234567890def",
                    "model_path_hint": "/some/path",
                    "mlx_lm_version": "0.31.3",
                    "format_version": 1,
                    "created_at": 0,
                }
            )
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_stats_empty(self):
        result = _run("stats", str(self.dir))
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("abc1234567890def", result.stdout)
        self.assertIn("0 entries", result.stdout)

    def test_list_empty_model(self):
        result = _run("list", str(self.dir), "--model", "abc1234567890def")
        self.assertEqual(result.returncode, 0)

    def test_stats_missing_format_version(self):
        bad = Path(self.tmpdir.name) / "bad"
        bad.mkdir()
        result = _run("stats", str(bad))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("format-version", result.stderr.lower())

    def test_remove_model(self):
        result = _run("remove", str(self.dir), "--model", "abc1234567890def", "--yes")
        self.assertEqual(result.returncode, 0)
        self.assertFalse((self.dir / "models" / "abc1234567890def").exists())

    def test_remove_nonexistent_model(self):
        result = _run("remove", str(self.dir), "--model", "nonexistent", "--yes")
        self.assertNotEqual(result.returncode, 0)

    def test_verify_empty(self):
        result = _run("verify", str(self.dir))
        self.assertEqual(result.returncode, 0)
        self.assertIn("OK", result.stdout)

    def test_prune_removes_old(self):
        # Drop a fake .safetensors with a very old mtime
        m = self.dir / "models" / "abc1234567890def" / "entries"
        old = m / "deadbeefdeadbeef.safetensors"
        old.write_bytes(b"x")
        import os

        os.utime(str(old), (0, 0))  # epoch 0 mtime
        result = _run("prune", str(self.dir), "--older-than", "1d", "--yes")
        self.assertEqual(result.returncode, 0)
        self.assertFalse(old.exists())


if __name__ == "__main__":
    unittest.main()
