# Copyright © 2026 Apple Inc.
"""End-to-end integration tests for disk-backed prompt cache.

These tests spawn ``mlx_lm.server`` as a subprocess and exercise the full
HTTP path. They use a tiny model so first-time startup is ~5-10 seconds
(plus model download on first run).

Run explicitly:
    pytest tests/test_disk_prompt_cache_e2e.py -v

Skipped when MLX_LM_E2E_SKIP=1 is set in the environment.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_port(port: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _post_chat(port: int, content: str, timeout: float = 120.0) -> dict:
    import urllib.request

    body = json.dumps(
        {
            "model": HF_MODEL_PATH,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 5,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read())


@unittest.skipIf(
    os.environ.get("MLX_LM_E2E_SKIP") == "1",
    "skipping e2e disk-prompt-cache tests (MLX_LM_E2E_SKIP=1)",
)
class TestDiskPromptCacheE2E(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.disk_dir = Path(self.tmpdir.name) / "diskcache"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _start_server(self, port: int) -> subprocess.Popen:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlx_lm.server",
                "--model",
                HF_MODEL_PATH,
                "--port",
                str(port),
                "--prompt-cache-disk-dir",
                str(self.disk_dir),
                "--prompt-cache-disk-bytes",
                "100MB",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not _wait_for_port(port, timeout=120):
            proc.terminate()
            proc.wait(timeout=10)
            self.fail(f"Server did not bind to port {port} in 120s")
        return proc

    def _stop_server(self, proc: subprocess.Popen, sig: int = signal.SIGTERM) -> None:
        proc.send_signal(sig)
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def test_request_populates_disk(self):
        port = _free_port()
        proc = self._start_server(port)
        try:
            _post_chat(port, "hello")
            self._stop_server(proc, signal.SIGTERM)
            entries = list(self.disk_dir.glob("models/*/entries/*.safetensors"))
            entries = [p for p in entries if not p.name.endswith(".tmp.safetensors")]
            self.assertGreaterEqual(
                len(entries),
                1,
                msg=f"expected at least 1 entry, got {len(entries)}",
            )
        finally:
            if proc.poll() is None:
                self._stop_server(proc, signal.SIGKILL)

    def test_restart_serves_from_disk(self):
        port = _free_port()
        # Run 1: populate
        proc = self._start_server(port)
        try:
            _post_chat(port, "what is 2+2?")
        finally:
            self._stop_server(proc, signal.SIGTERM)
        entries_before = sorted(self.disk_dir.glob("models/*/entries/*.safetensors"))
        entries_before = [
            p for p in entries_before if not p.name.endswith(".tmp.safetensors")
        ]
        self.assertGreaterEqual(len(entries_before), 1)

        # Run 2: same prompt; should not crash, should reuse disk cache
        # We can't trivially observe "no prefill happened" from outside the
        # server, but we verify the cache state is preserved.
        proc2 = self._start_server(port)
        try:
            _post_chat(port, "what is 2+2?")
        finally:
            self._stop_server(proc2, signal.SIGTERM)
        entries_after = sorted(self.disk_dir.glob("models/*/entries/*.safetensors"))
        entries_after = [
            p for p in entries_after if not p.name.endswith(".tmp.safetensors")
        ]
        self.assertGreaterEqual(len(entries_after), 1)

    def test_sigterm_drains_writes(self):
        port = _free_port()
        proc = self._start_server(port)
        try:
            _post_chat(port, "hello")
            time.sleep(1.0)  # let writer thread finish
            self._stop_server(proc, signal.SIGTERM)
            entries = list(self.disk_dir.glob("models/*/entries/*.safetensors"))
            entries = [p for p in entries if not p.name.endswith(".tmp.safetensors")]
            self.assertGreaterEqual(len(entries), 1)
            # No leftover .tmp.safetensors debris from the SIGTERM drain
            tmps = list(self.disk_dir.glob("models/*/entries/*.tmp.safetensors"))
            self.assertEqual(tmps, [])
        finally:
            if proc.poll() is None:
                self._stop_server(proc, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
