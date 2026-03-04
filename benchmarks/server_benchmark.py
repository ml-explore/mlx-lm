"""
Spin up the local server:

    mlx_lm.server

Then run the benchmark:

    python server_benchmark.py --concurrency 4
"""

import argparse
import asyncio
import json
import math
import time
from collections import defaultdict
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from tqdm import tqdm

# Default prompts if no file is provided
DEFAULT_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What are the main differences between Python and JavaScript?",
    "Describe the process of photosynthesis in plants.",
    "How does a neural network learn from data?",
    "What is the significance of the Turing test in AI?",
    "Explain the concept of blockchain technology.",
    "What causes seasons on Earth?",
    "How do vaccines work in the human body?",
    "Describe the water cycle and its importance.",
    "What is the theory of relativity proposed by Einstein?",
    "How do electric cars help reduce carbon emissions?",
    "What are the key features of a market economy?",
    "Explain how DNA replication works in cells.",
    "What is machine learning and its real-world applications?",
    "Describe the structure and function of the human heart.",
]


def tokens_per_second(tokens):
    start = math.floor(tokens[0])
    stop = math.ceil(tokens[-1])
    n_bins = int(stop - start) * 10
    bins = [0] * n_bins
    for t in tokens:
        bins[int(n_bins * (t - start) / (stop - start))] += 1

    result = []

    ms = 0
    cnt = 0
    for i, b in enumerate(bins):
        ms += b
        if cnt == 10:
            ms -= bins[i - 10]
        else:
            cnt += 1

        result.append(10 * ms / cnt)

    times = [start]
    while times[-1] < stop:
        times.append(times[-1] + 0.1)

    return times, result


def plot_generation(times, tokens_per_sec, start=None, interval=1.0, width=50):
    c = "█"
    start = start or times[0]
    stop = times[-1]

    bar_times = [start]
    while bar_times[-1] < stop:
        bar_times.append(bar_times[-1] + interval)

    bar_values = [[] for _ in bar_times]
    bar_idx = 0

    for t, v in zip(times, tokens_per_sec):
        while t > bar_times[bar_idx] + interval:
            bar_idx += 1
        bar_values[bar_idx].append(v)

    bar_values = [sum(v) / len(v) if v else 0 for v in bar_values]
    m = max(bar_values)

    for t, v in zip(bar_times, bar_values):
        t = t - start
        b = c * int(v * width / m)
        print(f"{t:3.2f} {b} ({v})")


def percentile(data, percent):
    if not data:
        return 0
    data = sorted(data)
    k = (len(data) - 1) * percent / 100
    f = math.floor(k)
    c = math.ceil(k)
    return (
        data[int(f)]
        if f == c
        else data[int(f)] + (data[int(c)] - data[int(f)]) * (k - f)
    )


def median(data):
    return percentile(data, 50)


async def make_request(
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> Tuple[bool, float, list]:
    """
    Make a single streaming API request and return

        - whether the request succeeded
        - the request start time
        - the time of every generated token
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    start_time = time.perf_counter()
    tokens = []

    try:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_body = await response.text()
                print(f"Error {response.status}: {error_body}")
                return (False, 0, [])

            # Process streaming response
            async for chunk in response.content:
                if chunk:
                    chunk_str = chunk.decode("utf-8").strip()
                    if chunk_str.startswith("data:"):
                        data_str = chunk_str[5:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            if choices := data.get("choices", False):
                                if choices[0].get("finish_reason") != "length":
                                    tokens.append(time.perf_counter())
                        except json.JSONDecodeError:
                            continue

            return (bool(tokens), start_time, tokens)

    except Exception as e:
        print(f"Request failed: {str(e)}")
        return (False, 0, [])


async def run_benchmark(
    url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    concurrency: int,
    total_requests: int,
    prompts: List[str],
) -> Dict[str, Any]:
    prompt_cycle = cycle(prompts)
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    request_times = []
    bar = tqdm(total=total_requests)

    async def worker():
        async with semaphore:
            prompt = next(prompt_cycle)
            result = await make_request(
                session, url, api_key, model, prompt, max_tokens
            )
            bar.update(1)
            return result

    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(total_requests):
            task = asyncio.create_task(worker())
            tasks.append(task)
            await asyncio.sleep(0.01)  # Stagger requests slightly

        for task in tasks:
            result = await task
            results.append(result)
        bar.close()

    successful_requests = [r for r in results if r[0]]
    total_tokens = sum(len(r[2]) for r in successful_requests)

    # Gather all the tokens generated with their corresponding timestamps
    all_tokens = []
    for r in successful_requests:
        all_tokens.extend(r[2])
    all_tokens.sort()
    full_generation = tokens_per_second(all_tokens)
    start = min(r[1] for r in successful_requests)

    # Aggregate metrics
    metrics = {
        "total_requests": total_requests,
        "successful_requests": len(successful_requests),
        "failed_requests": total_requests - len(successful_requests),
        "total_tokens": total_tokens,
        "total_time": all_tokens[-1] - start,
        "aggregate_tokens_per_sec": median(full_generation[1]),
        "per_request": [],
        "start": start,
        "full_generation": full_generation,
    }

    # Per-request metrics
    for i, (_, start, tokens) in enumerate(successful_requests):
        metrics["per_request"].append(
            {
                "request_id": i + 1,
                "time_to_first_token": tokens[0] - start,
                "total_time": tokens[-1] - start,
                "tokens_received": len(tokens),
                "tokens_per_sec": median(tokens_per_second(tokens)[1]),
            }
        )

    # Calculate percentiles
    ttft_values = [m["time_to_first_token"] for m in metrics["per_request"]]
    tps_values = [m["tokens_per_sec"] for m in metrics["per_request"]]

    metrics["aggregate_metrics"] = {
        "time_to_first_token": {
            "min": min(ttft_values) if ttft_values else 0,
            "max": max(ttft_values) if ttft_values else 0,
            "avg": sum(ttft_values) / len(ttft_values) if ttft_values else 0,
            "p95": percentile(ttft_values, 95) if ttft_values else 0,
        },
        "tokens_per_sec": {
            "min": min(tps_values) if tps_values else 0,
            "max": max(tps_values) if tps_values else 0,
            "avg": sum(tps_values) / len(tps_values) if tps_values else 0,
            "p95": percentile(tps_values, 95) if tps_values else 0,
        },
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="LLM API Benchmark Tool")
    parser.add_argument(
        "--url",
        default="http://localhost:8080/v1/chat/completions",
        help="Chat completions API endpoint URL",
    )
    parser.add_argument("--api-key", default="none", help="API key")
    parser.add_argument("--model", default="default_model", help="Model name")
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens to generate"
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Number of concurrent requests"
    )
    parser.add_argument(
        "--total-requests", type=int, default=10, help="Total requests to make"
    )
    parser.add_argument("--prompt-file", help="File containing prompts (one per line)")
    parser.add_argument("--output", help="Output file for results (JSON format)")

    args = parser.parse_args()

    # Load prompts
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    print(
        f"Starting benchmark with {args.concurrency} concurrency and {args.total_requests} total requests..."
    )
    start_time = time.perf_counter()

    # Run benchmark
    results = asyncio.run(
        run_benchmark(
            url=args.url,
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            total_requests=args.total_requests,
            prompts=prompts,
        )
    )

    duration = time.perf_counter() - start_time
    print(f"\nBenchmark completed in {duration:.2f} seconds")
    print(
        f"Successful requests: {results['successful_requests']}/{args.total_requests}"
    )
    print(f"Total tokens generated: {results['total_tokens']}")
    print(f"Aggregate tokens/sec: {results['aggregate_tokens_per_sec']:.2f}")

    # Print summary
    if results["successful_requests"] > 0:
        ttft = results["aggregate_metrics"]["time_to_first_token"]
        tps = results["aggregate_metrics"]["tokens_per_sec"]

        print("\nTime to First Token (seconds):")
        print(
            f"  Min: {ttft['min']:.4f} | Max: {ttft['max']:.4f} | Avg: {ttft['avg']:.4f} | P95: {ttft['p95']:.4f}"
        )

        print("\nTokens per Second (per request):")
        print(
            f"  Min: {tps['min']:.2f} | Max: {tps['max']:.2f} | Avg: {tps['avg']:.2f} | P95: {tps['p95']:.2f}"
        )

        print()
        plot_generation(*results["full_generation"], results["start"])

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
