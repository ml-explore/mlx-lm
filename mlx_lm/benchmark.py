# Copyright © 2025 Apple Inc.

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx

from mlx_lm import batch_generate, load, stream_generate
from mlx_lm.generate import DEFAULT_MODEL
from mlx_lm.utils import sharded_load


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM benchmarking script")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "The path to the local model directory or Hugging Face repo. "
            f"If no model is specified, then {DEFAULT_MODEL} is used."
        ),
        default=None,
    )
    parser.add_argument(
        "--prompt-tokens",
        "-p",
        default=512,
        help="Length of prompt",
        type=int,
    )
    parser.add_argument(
        "--generation-tokens",
        "-g",
        default=1024,
        help="Length of completion",
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        default=1,
        help="Batch size",
        type=int,
    )
    parser.add_argument(
        "--num-trials",
        "-n",
        default=5,
        help="Number of timing trials",
        type=int,
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    parser.add_argument(
        "--quantize-activations",
        "-qa",
        action="store_true",
        help="Quantize activations using the same quantization config as the corresponding layer.",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Step size for prefill processing (default: 2048)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="Delay between each test in seconds (default: 0)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer/model loading.",
    )
    return parser


@dataclass
class TrialMetrics:
    prompt_tps: float
    generation_tps: float
    peak_memory: float  # GB
    total_time: float  # Seconds


@dataclass
class BenchmarkResult:
    model: str
    prompt_tokens: int
    generation_tokens: int
    batch_size: int
    prompt_tps: float
    generation_tps: float
    peak_memory: float  # GB
    trials: list[TrialMetrics]


def run_benchmark(
    model: str | None = None,
    *,
    prompt_tokens: int = 512,
    generation_tokens: int = 1024,
    batch_size: int = 1,
    num_trials: int = 5,
    pipeline: bool = False,
    quantize_activations: bool = False,
    prefill_step_size: int = 2048,
    delay: float = 0,
    trust_remote_code: bool = False,
    verbose: bool = False,
) -> BenchmarkResult:
    """Benchmark model inference using synthetic token IDs."""

    # Step 1: Validate the input parameters.
    for name, value in (
        ("prompt_tokens", prompt_tokens),
        ("generation_tokens", generation_tokens),
        ("batch_size", batch_size),
        ("num_trials", num_trials),
        ("prefill_step_size", prefill_step_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if delay < 0:
        raise ValueError("delay must be >= 0")

    # Step 2: Initialize MLX and distributed execution.
    mx.random.seed(0)

    group = mx.distributed.init()
    rank = group.rank()

    pipeline_group = group if pipeline else None
    tensor_group = None if pipeline else group

    model_path = model or DEFAULT_MODEL

    # Step 3: Load the pretrained LLM.
    if group.size() > 1:
        network, tokenizer, config = sharded_load(
            model_path,
            pipeline_group,
            tensor_group,
            return_config=True,
            trust_remote_code=trust_remote_code,
        )
    else:
        network, tokenizer, config = load(
            model_path,
            return_config=True,
            tokenizer_config={"trust_remote_code": True},
            model_config={"quantize_activations": quantize_activations},
            trust_remote_code=trust_remote_code,
        )

    # Avoid stopping generation early on EOS.
    tokenizer._eos_token_ids = {}

    # Step 4: Create synthetic input tokens.
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]

    prompts = mx.random.randint(
        0,
        vocab_size,
        (batch_size, prompt_tokens),
    ).tolist()

    # Step 5: Inference for a single prompt.
    def single_bench():
        last_response = None

        for response in stream_generate(
            network,
            tokenizer,
            prompts[0],
            max_tokens=generation_tokens,
            prefill_step_size=prefill_step_size,
        ):
            last_response = response

        if last_response is None:
            raise RuntimeError("The model produced no benchmark response")

        return last_response

    # Inference for multiple prompts.
    def batch_bench():
        return batch_generate(
            network,
            tokenizer,
            prompts,
            max_tokens=generation_tokens,
            prefill_step_size=prefill_step_size,
        ).stats

    bench = single_bench if batch_size == 1 else batch_bench

    # Only print when requested, on rank 0.
    def report(message):
        if verbose and rank == 0:
            print(message)

    # Step 6: Warm up the model.
    report("Running warmup..")
    bench()

    report(f"Timing with {prompt_tokens=}, " f"{generation_tokens=}, {batch_size=}.")

    # Step 7: Run measured trials.
    trials = []

    for index in range(num_trials):
        if delay:
            time.sleep(delay)

        start = time.perf_counter()

        response = bench()

        elapsed = time.perf_counter() - start

        trial = TrialMetrics(
            prompt_tps=float(response.prompt_tps),
            generation_tps=float(response.generation_tps),
            peak_memory=float(response.peak_memory),
            total_time=elapsed,
        )

        trials.append(trial)

        report(
            f"Trial {index + 1}:  "
            f"prompt_tps={trial.prompt_tps:.3f}, "
            f"generation_tps={trial.generation_tps:.3f}, "
            f"peak_memory={trial.peak_memory:.3f}, "
            f"total_time={trial.total_time:.3f}"
        )

    # Step 8: Calculate average metrics.
    def mean(attribute):
        return sum(getattr(trial, attribute) for trial in trials) / num_trials

    result = BenchmarkResult(
        model=model_path,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        batch_size=batch_size,
        prompt_tps=mean("prompt_tps"),
        generation_tps=mean("generation_tps"),
        peak_memory=mean("peak_memory"),
        trials=trials,
    )

    report(
        f"Averages: prompt_tps={result.prompt_tps:.3f}, "
        f"generation_tps={result.generation_tps:.3f}, "
        f"peak_memory={result.peak_memory:.3f}"
    )

    # Step 9: Return structured data to Python.
    return result


def main():
    """Preserve the existing command-line interface."""
    args = setup_arg_parser().parse_args()

    run_benchmark(
        model=args.model,
        prompt_tokens=args.prompt_tokens,
        generation_tokens=args.generation_tokens,
        batch_size=args.batch_size,
        num_trials=args.num_trials,
        pipeline=args.pipeline,
        quantize_activations=args.quantize_activations,
        prefill_step_size=args.prefill_step_size,
        delay=args.delay,
        trust_remote_code=args.trust_remote_code,
        verbose=True,
    )


if __name__ == "__main__":
    main()
