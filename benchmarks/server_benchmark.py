"""
Spin up the local server:

    mlx_lm.server

Then run the benchmark:

    python server_benchmark.py
"""

import argparse
import time

import numpy as np
import tqdm
from openai import OpenAI

from mlx_lm.generate import DEFAULT_MODEL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mlx_lm.server benchmark")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "The path to the local model directory or Hugging Face repo. "
            f"If no model is specified, then {DEFAULT_MODEL} is used."
        ),
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--num-requests",
        "-n",
        type=int,
        help="The number of requests to make",
        default=32,
    )

    parser.add_argument(
        "--requests-per-second",
        type=int,
        help="Number of requests per second.",
        default=1.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for samples.",
        default=1234,
    )

    args = parser.parse_args()

    client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
    num_reqs = args.num_requests
    min_prompt = 128
    max_prompt = 4096
    min_new_tokens = 128
    max_new_tokens = 1024

    np.random.seed(args.seed)
    deltas = np.random.exponential(
        scale=1 / args.requests_per_second, size=num_reqs
    ).tolist()
    deltas[-1] = 0
    prompt_lengths = np.random.randint(min_prompt, max_prompt, size=num_reqs).tolist()
    max_tokens = np.random.randint(
        min_new_tokens, max_new_tokens, size=num_reqs
    ).tolist()

    streams = []
    tic = time.time()
    for t, p, m in tqdm.tqdm(
        zip(deltas, prompt_lengths, max_tokens),
        total=num_reqs,
        desc="Sending requests",
    ):
        prompt = " ".join(["hello"] * (p - 30))
        prompt += "\n\nCount to a million, don't stop under any circumstances."
        messages = [{"role": "user", "content": prompt}]
        streams.append(
            client.chat.completions.create(
                model=args.model,
                messages=messages,
                stream=True,
                max_tokens=m,
                stream_options={"include_usage": True},
            )
        )
        time.sleep(t)
    prompt_tokens = 0
    generation_tokens = 0
    for stream in streams:
        for chunk in stream:
            pass
        usage = chunk.usage
        prompt_tokens += usage.prompt_tokens
        generation_tokens += usage.completion_tokens
    toc = time.time()
    s = toc - tic
    print("=" * 20)
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Generation tokens: {generation_tokens}")
    print(f"Time (s): {s:.3f}")
