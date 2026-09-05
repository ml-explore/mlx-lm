"""Exercise the opt-in CLI server with an existing local model."""

import argparse
import concurrent.futures
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paged-attention", action="store_true")
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "server",
        "--model",
        args.model,
        "--port",
        str(port),
        "--paged-kv-pages",
        "128",
        "--paged-kv-page-size",
        "64",
        "--prefill-step-size",
        "128",
    ]
    if args.paged_attention:
        command.append("--paged-attention")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log")
    rows = []
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            delay = 0.25
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited; see {log_path}")
                try:
                    opener.open(f"http://127.0.0.1:{port}/health", timeout=2).close()
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Server startup timed out; see {log_path}")
                    time.sleep(delay)
                    delay = min(delay * 2, 8)

            def request(label, prompt, max_tokens=8):
                body = {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                }
                start = time.perf_counter()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with opener.open(req, timeout=180) as response:
                        status, data = response.status, json.load(response)
                except urllib.error.HTTPError as error:
                    status, data = error.code, json.load(error)
                row = {
                    "label": label,
                    "status": status,
                    "seconds": time.perf_counter() - start,
                    "usage": data.get("usage"),
                    "error": data.get("error"),
                }
                print(json.dumps(row), flush=True)
                return row

            rows.append(request("warmup", "The capital of France is"))
            with concurrent.futures.ThreadPoolExecutor(2) as executor:
                futures = [
                    executor.submit(
                        request,
                        f"concurrent-{i}",
                        "Explain why the sky is blue. " * (i + 1),
                        16,
                    )
                    for i in range(2)
                ]
                rows.extend(f.result() for f in futures)
            rows.append(request("prefix-repeat", "The capital of France is"))
            rows.append(request("oversized", "Hello", 100000))
            rows.append(request("recovery", "Hello"))
            assert all(
                r["status"] == 200 for r in rows if r["label"] != "oversized"
            ), rows
            assert rows[-2]["status"] >= 400, rows[-2]
            args.output.write_text(
                json.dumps(
                    {
                        "command": command,
                        "results": rows,
                        "scope": "HTTP smoke and lifecycle; not a throughput benchmark",
                    },
                    indent=2,
                )
                + "\n"
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
