# ABOUTME: Ensures continuous batching matches batch_generate token outputs.
# ABOUTME: Runs a subprocess parity check on a small HF model.

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

MODEL_REPO = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
SCRIPT = f"""
import json, os
from mlx_lm import batch_generate, load
from mlx_lm.server_batched.engine import ModelRunner
from mlx_lm.server_batched.runtime import ContinuousBatchingRuntime

MODEL_REPO = "{MODEL_REPO}"
prompts = ["Hello world", "Goodbye moon"]
max_new_tokens = 4
model, tokenizer = load(path_or_hf_repo=MODEL_REPO)

prompt_token_batches = [tokenizer.encode(p) for p in prompts]
static = batch_generate(
    model,
    tokenizer,
    prompt_token_batches,
    max_tokens=max_new_tokens,
    verbose=False,
)
static_tokens = [tokenizer.encode(t, add_special_tokens=False) for t in static.texts]

runner = ModelRunner(
    model,
    tokenizer,
    max_num_seqs=len(prompts),
    prefill_chunk=64,
)
runtime = ContinuousBatchingRuntime(
    runner,
    max_num_seqs=len(prompts),
    max_tokens_per_step=512,
    prefill_chunk=64,
)

sampler_settings = {{
    "temp": 0.0,
    "top_p": 1.0,
    "min_p": 0.0,
    "top_k": 0,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.0,
    "xtc_special_tokens": [tokenizer.eos_token_id],
}}
stopping_settings = {{"eos_token_id": tokenizer.eos_token_id}}

generations = []
for prompt_tokens in prompt_token_batches:
    _, gen = runtime.submit_request(
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        sampler_settings=sampler_settings,
        stopping_settings=stopping_settings,
        logit_bias=None,
        repetition_penalty=None,
        repetition_context_size=None,
    )
    tokens = []
    for resp in gen:
        if resp.finish_reason != "stop":
            tokens.append(int(resp.token))
        if resp.finish_reason is not None:
            break
    generations.append(tokens)

runtime.shutdown()

ok = static_tokens == generations
print(json.dumps({{"ok": ok, "static": static_tokens, "cont": generations}}))
if not ok:
    raise SystemExit(1)
"""


class ContinuousParitySmallModelTest(unittest.TestCase):
    def test_continuous_matches_batch_generate_tokens(self):
        if not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
            self.skipTest("HUGGINGFACE_HUB_TOKEN not set")

        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
        proc = subprocess.run(
            [sys.executable, "-c", SCRIPT],
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"parity script failed: {proc.stderr}\nstdout: {proc.stdout}")
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as exc:  # noqa: BLE001
            self.fail(f"Malformed parity output: {proc.stdout}\nerror: {exc}")
        self.assertTrue(payload.get("ok"), f"tokens differ: {payload}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
