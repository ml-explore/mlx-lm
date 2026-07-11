# Copyright © 2026 Apple Inc.
"""Deterministic evidence probe for asymmetric KV-cache quantization.

The runner writes its JSON only after every arm completes. It intentionally
supports models no larger than 0.6B; representative-scale validation belongs
to a separate, coordinator-run gate.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.utils import load

ARMS = {
    "fp16": (None, None),
    "K8V8": (8, 8),
    "K8V4": (8, 4),
    "K4V8": (4, 8),
    "K4V4": (4, 4),
}
THROUGHPUT_SCHEDULE = [
    list(ARMS),
    list(reversed(ARMS)),
    list(ARMS),
]
DEFAULT_CORPUS_FILES = [
    "README.md",
    "mlx_lm/models/cache.py",
    "mlx_lm/models/base.py",
    "mlx_lm/generate.py",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen1.5-0.5B-Chat-4bit")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--eval-step-size", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus-file", action="append", dest="corpus_files")
    return parser.parse_args()


def build_corpus(tokenizer, paths, token_count):
    tokens = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        section = f"\n\n===== {path} =====\n\n{text}"
        tokens.extend(tokenizer.encode(section))
        if len(tokens) >= token_count:
            break
    if not tokens:
        raise ValueError("Corpus tokenization produced no tokens")
    if len(tokens) < token_count:
        tokens = (tokens * ((token_count + len(tokens) - 1) // len(tokens)))[
            :token_count
        ]
    else:
        tokens = tokens[:token_count]
    token_text = ",".join(map(str, tokens)).encode("ascii")
    return tokens, hashlib.sha256(token_text).hexdigest()


def cache_nbytes(cache):
    return sum(array.nbytes for _, array in tree_flatten([c.state for c in cache]))


def make_eval_cache(model, prefix, key_bits, value_bits, group_size=64):
    cache = make_prompt_cache(model)
    model(mx.array(prefix)[None], cache=cache)
    mx.eval([c.state for c in cache])
    source_dtype = str(cache[0].keys.dtype)
    if key_bits is not None:
        if not all(isinstance(c, KVCache) for c in cache):
            raise TypeError("Probe currently requires standard KVCache layers")
        cache = [
            c.to_quantized(
                group_size=group_size,
                bits=key_bits,
                key_bits=key_bits,
                value_bits=value_bits,
            )
            for c in cache
        ]
        mx.eval([c.state for c in cache])
    return cache, source_dtype


def measure_nll(model, tokens, key_bits, value_bits, step_size):
    split = len(tokens) // 2
    cache, source_dtype = make_eval_cache(
        model, tokens[: split - 1], key_bits, value_bits
    )
    bytes_at_split = cache_nbytes(cache)
    inputs = tokens[split - 1 : -1]
    targets = tokens[split:]
    loss_sum = 0.0
    count = 0
    for start in range(0, len(inputs), step_size):
        input_chunk = mx.array(inputs[start : start + step_size])[None]
        target_chunk = mx.array(targets[start : start + step_size])
        # Accumulate quality evidence in float32. The cache precision is the
        # variable under test; fp16 reduction noise must not decide parity.
        logits = model(input_chunk, cache=cache)[0].astype(mx.float32)
        selected = mx.take_along_axis(logits, target_chunk[:, None], axis=-1).squeeze(
            -1
        )
        losses = mx.logsumexp(logits, axis=-1) - selected
        mx.eval(losses)
        loss_sum += losses.sum().item()
        count += target_chunk.size
    return loss_sum / count, bytes_at_split / (split - 1), source_dtype


def measure_throughput(
    model, tokenizer, prompt, key_bits, value_bits, generation_tokens
):
    last = None
    kwargs = {}
    if key_bits is not None:
        kwargs.update(
            kv_key_bits=key_bits,
            kv_value_bits=value_bits,
            quantized_kv_start=0,
        )
    for last in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=generation_tokens,
        **kwargs,
    ):
        pass
    if last is None:
        raise RuntimeError("Generation produced no response")
    return last.generation_tps


def git_sha():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main():
    args = parse_args()
    if not 4096 <= args.tokens <= 8192:
        raise ValueError("--tokens must stay within the specified 4k-8k range")
    if "0.5B" not in args.model and "0.6B" not in args.model:
        raise ValueError("Probe model must be explicitly identifiable as <=0.6B")

    corpus_files = args.corpus_files or DEFAULT_CORPUS_FILES
    model, tokenizer = load(args.model)
    tokens, corpus_sha256 = build_corpus(tokenizer, corpus_files, args.tokens)

    results = {}
    for name, (key_bits, value_bits) in ARMS.items():
        nll, bytes_per_token, source_dtype = measure_nll(
            model,
            tokens,
            key_bits,
            value_bits,
            args.eval_step_size,
        )
        results[name] = {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "mean_nll": nll,
            "bytes_per_token_at_split": bytes_per_token,
            "source_cache_dtype": source_dtype,
            "generation_tps_reps": [],
        }
        mx.clear_cache()

    prompt = tokens[:512]
    for order in THROUGHPUT_SCHEDULE:
        for name in order:
            key_bits, value_bits = ARMS[name]
            tps = measure_throughput(
                model,
                tokenizer,
                prompt,
                key_bits,
                value_bits,
                args.generation_tokens,
            )
            results[name]["generation_tps_reps"].append(tps)
            mx.clear_cache()

    for result in results.values():
        result["generation_tps_median"] = statistics.median(
            result["generation_tps_reps"]
        )

    k8v8 = results["K8V8"]
    k8v4 = results["K8V4"]
    k4v8 = results["K4V8"]
    fp16 = results["fp16"]
    k4v4 = results["K4V4"]
    value_damage = k8v4["mean_nll"] - k8v8["mean_nll"]
    key_damage = k4v8["mean_nll"] - k8v8["mean_nll"]
    mixed_tps_ratio = k8v4["generation_tps_median"] / k8v8["generation_tps_median"]
    kv4_savings = fp16["bytes_per_token_at_split"] - k4v4["bytes_per_token_at_split"]
    mixed_savings = fp16["bytes_per_token_at_split"] - k8v4["bytes_per_token_at_split"]
    supports_prediction = (
        abs(value_damage) <= 0.01
        and key_damage >= 5 * max(abs(value_damage), 1e-6)
        and mixed_tps_ratio >= 0.9
    )

    payload = {
        "schema_version": 1,
        "verdict": "supports_pr" if supports_prediction else "kills_pr",
        "verdict_criteria": {
            "abs_K8V4_minus_K8V8_nll_lte": 0.01,
            "K4V8_damage_at_least_multiple_of_K8V4_damage": 5,
            "K8V4_generation_tps_ratio_vs_K8V8_gte": 0.9,
        },
        "interpretation": {
            "K8V4_minus_K8V8_nll": value_damage,
            "K4V8_minus_K8V8_nll": key_damage,
            "K8V4_generation_tps_ratio_vs_K8V8": mixed_tps_ratio,
            "K8V4_share_of_K4V4_memory_savings_vs_fp16": mixed_savings / kv4_savings,
        },
        "model": args.model,
        "model_size_gate": "<=0.6B",
        "git_sha": git_sha(),
        "runtime": {
            "mlx": mx.__version__,
            "platform": platform.platform(),
            "device": mx.device_info(),
        },
        "corpus": {
            "files": corpus_files,
            "token_count": len(tokens),
            "token_ids_sha256": corpus_sha256,
            "evaluation": "mean NLL of second half given first half cached",
        },
        "throughput": {
            "prompt_tokens": len(prompt),
            "generation_tokens": args.generation_tokens,
            "schedule": THROUGHPUT_SCHEDULE,
            "statistic": "median of 3 reps",
        },
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temp_path, output)
    except BaseException:
        os.unlink(temp_path)
        raise


if __name__ == "__main__":
    main()
