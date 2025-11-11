#!/usr/bin/env python3
"""
benchmark_ollama.py

Benchmark an Ollama backend (OpenAI-compatible /v1/chat/completions) by sending
concurrent requests and measuring per-request and overall metrics:
 - input tokens
 - output tokens
 - latency (time to full response)
 - TTFT (time to first token / first chunk)
 - TPOT (time per output token)
 - total throughput (tokens/sec)

Example:
    python benchmark_ollama.py --base-url localhost:11435 --batch-size 256 --input-size 4096 --output-len 128
"""

import argparse
import asyncio
import json
import time
import math
import sys
from typing import Optional, Dict, Any, List, Tuple

try:
    import httpx
except Exception as e:
    print("This script requires 'httpx' (async). Install with: pip install httpx", file=sys.stderr)
    raise

# Optional: tiktoken for accurate token counts. If absent we fallback to approximation.
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

# ---------- Utilities for tokenization ----------
def build_prompt_for_token_target(token_target: int, model_name: str = "gpt-4") -> str:
    """
    Build a prompt whose tokenized length is approximately `token_target`.
    Uses tiktoken if available; otherwise uses a conservative approximation (words).
    """
    if token_target <= 0:
        return ""
    if _HAS_TIKTOKEN:
        # choose encoding based on model name heuristically
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        # We will construct a prompt by repeating a filler until we reach target tokens.
        # Use a sentence that tokenizes to a known small number of tokens.
        sample = "This is a filler sentence."
        sample_tokens = len(enc.encode(sample))
        if sample_tokens == 0:
            sample_tokens = 4
        reps = max(1, token_target // sample_tokens)
        prompt = " ".join([sample] * reps)
        # adjust by trimming or appending words until close
        while len(enc.encode(prompt)) < token_target:
            prompt += " !"
        # if slightly over, it's OK
        return prompt
    else:
        # fallback: approximate 1 token ≈ 0.75 words (conservative), use words
        # We'll create repeated "word" tokens to reach approx token count.
        # Be conservative and create slightly more words.
        word_count = int(math.ceil(token_target / 0.75))
        words = ["word"] * word_count
        return " ".join(words)


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Return approximate or exact token count for text."""
    if _HAS_TIKTOKEN:
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    else:
        # approximate: split on whitespace and punctuation -> conservative: each word ~ 1 token
        # This is not exact but fine for benchmarking if tiktoken not available.
        return max(1, len(text.split()))


# ---------- HTTP / streaming helpers ----------
async def post_completion_stream(
    client: httpx.AsyncClient,
    url: str,
    json_payload: Dict[str, Any],
    request_timeout: Optional[float] = None,
) -> Tuple[str, float, float, float]:
    """
    POST with stream=True to read server-sent events or chunked responses.
    Returns:
      - full_text (concatenated)
      - ttft_seconds (time from start to first token/chunk)
      - latency_seconds (time from start to last chunk)
      - raw_duration_for_chunks (seconds between first and last chunk, maybe 0)
    """
    start = time.perf_counter()
    first_chunk_time = None
    last_chunk_time = None
    text_parts: List[str] = []

    # ensure streaming param is set as OpenAI expects
    # the server should support {"stream": True}. If not supported it may ignore it.
    try:
        async with client.stream("POST", url, json=json_payload, timeout=request_timeout) as resp:
            if resp.status_code >= 400:
                # read body
                body = await resp.aread()
                import pdb
                pdb.set_trace()
                raise RuntimeError(f"HTTP {resp.status_code}: {body.decode(errors='ignore')}")
            # read chunked response iteratively
            async for raw_chunk in resp.aiter_bytes():
                now = time.perf_counter()
                if not raw_chunk:
                    continue
                # consider first byte arrival as first token arrival if streaming content is not SSE
                if first_chunk_time is None:
                    first_chunk_time = now
                last_chunk_time = now
                try:
                    s = raw_chunk.decode(errors="ignore")
                except Exception as e:
                    print(f"Exception-1 (client ) : {e}")
                    import pdb
                    pdb.set_trace()
                    s = str(raw_chunk)
                # If server uses SSE style "data: ..." lines, extract them
                for line in s.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # SSE data lines typically start with "data: "
                    if line.startswith("data:"):
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]":
                            # done
                            continue
                        try:
                            j = json.loads(payload)
                            # try to extract delta content (chat format)
                            # choices -> ... -> delta -> content
                            if isinstance(j, dict):
                                choices = j.get("choices")
                                if choices and isinstance(choices, list):
                                    for ch in choices:
                                        delta = ch.get("delta", {})
                                        if isinstance(delta, dict):
                                            piece = delta.get("content")
                                            if piece:
                                                text_parts.append(piece)
                                        # fallback to text field for legacy
                                        text = ch.get("text")
                                        if text:
                                            text_parts.append(text)
                        except Exception as e:
                            # If not JSON, append raw payload
                            print(f"I don't know")
                            import pdb
                            pdb.set_trace()
                            text_parts.append(payload)
                    else:
                        # Not SSE-like, append raw line
                        text_parts.append(line)
            # reading finished
    except httpx.ReadError as e:
        # network read error; fallback to a normal non-streaming request
        print(f"httpx.ReadError : {e}")
        import pdb
        pdb.set_trace()
        raise
    end = time.perf_counter()
    ttft = (first_chunk_time - start) if first_chunk_time is not None else (end - start)
    latency = end - start
    chunk_duration = (last_chunk_time - first_chunk_time) if (first_chunk_time and last_chunk_time) else 0.0
    full_text = "".join(text_parts)
    return full_text, ttft, latency, chunk_duration


async def post_completion_nonstream(
    client: httpx.AsyncClient,
    url: str,
    json_payload: Dict[str, Any],
    request_timeout: Optional[float] = None,
) -> Tuple[str, float, float, float]:
    """
    POST normally and measure time to first byte and time to full response.
    Returns full_text, ttft, latency, chunk_duration (chunk_duration==latency here).
    """
    start = time.perf_counter()
    # we can get response iter_bytes to measure first byte: use stream to get first bytes
    try:
        resp = await client.post(url, json=json_payload, timeout=request_timeout)
        if resp.status_code >= 400:
            body = await resp.aread()

            import pdb
            pdb.set_trace()
            print(f"httpx.ReadError : {e}")

            raise RuntimeError(f"HTTP {resp.status_code}: {body.decode(errors='ignore')}")
        # read first chunk
        data_iter = resp.aiter_bytes()

        first_chunk = await data_iter.__anext__()  # may raise StopAsyncIteration quickly
        first_arrival = time.perf_counter()
        # read remaining
        rest = []
        async for chunk in data_iter:
            rest.append(chunk)
        body_bytes = first_chunk + b"".join(rest)
        end = time.perf_counter()
        try:
            j = json.loads(body_bytes.decode(errors="ignore"))
            # try to extract text from completion/chat response
            text_out = ""
            # chat completions structure
            if "choices" in j and isinstance(j["choices"], list):
                for c in j["choices"]:
                    # chat format
                    msg = c.get("message") or c.get("delta") or {}
                    if isinstance(msg, dict):
                        cont = msg.get("content") or msg.get("text")
                        if cont:
                            text_out += cont
                    # sometimes choice has 'text'
                    if "text" in c and c["text"]:
                        text_out += c["text"]
            else:
                # fallback: raw text
                text_out = body_bytes.decode(errors="ignore")
        except Exception as e:
            print(f"Exception : {e}")
            
            import pdb
            pdb.set_trace()

            text_out = body_bytes.decode(errors="ignore")
        ttft = first_arrival - start
        latency = end - start
        chunk_duration = latency  # approximate
        return text_out, ttft, latency, chunk_duration
    except StopAsyncIteration:
        # no body returned
        end = time.perf_counter()
        return "", end - start, end - start, 0.0


# ---------- Core benchmarking logic ----------
async def run_single_request(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
    model_for_token_count: str,
    request_timeout: Optional[float] = 360.0,
) -> Dict[str, Any]:
    """
    Send one request and measure stats.
    Returns dict with keys: input_tokens, output_tokens, latency, ttft, t_first_to_last, text
    """
    # Build payload in OpenAI chat format
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # include stream flag; some servers need query param or request body "stream": true
    }
    if stream:
        payload["stream"] = True

    input_tokens = count_tokens(prompt, model_for_token_count)

    try:
        if stream:
            text, ttft, latency, chunk_dur = await post_completion_stream(client, endpoint, payload, request_timeout)
        else:
            text, ttft, latency, chunk_dur = await post_completion_nonstream(client, endpoint, payload, request_timeout)
    except Exception as e:

        # import pdb
        # pdb.set_trace()

        return {
            "error": str(e),
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "latency": None,
            "ttft": None,
            "t_first_to_last": None,
            "text": "",
        }

    output_tokens = count_tokens(text, model_for_token_count) if text else 0
    # t_first_to_last: time between first chunk and last chunk
    t_first_to_last = chunk_dur

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency": latency,
        "ttft": ttft,
        "t_first_to_last": t_first_to_last,
        "text": text,
    }


async def run_benchmark(
    base_url: str,
    model: str,
    batch_size: int,
    input_size: int,
    output_len: int,
    concurrency: Optional[int] = None,
    use_streaming: bool = True,
    model_for_token_count: str = "gpt-4",
    timeout_per_request: float = 600.0,
) -> None:
    """
    Run the benchmark:
     - send `batch_size` requests concurrently (bounded by concurrency)
     - measure per-request metrics and overall throughput
    """
    if concurrency is None:
        concurrency = batch_size

    # Endpoint - following OpenAI format
    # If base_url contains port only (like localhost:11435) ensure we prefix http://
    if base_url.startswith("http://") or base_url.startswith("https://"):
        base = base_url
    else:
        base = "http://" + base_url
    endpoint = f"{base.rstrip('/')}/v1/chat/completions"
    # endpoint = f"{base.rstrip('/')}/v1/responses"

    # http://localhost:5001/v1/chat/completions 
    # print(f"endpoint : {endpoint}")

    prompt = build_prompt_for_token_target(input_size, model_for_token_count)

    # print(f"prompt : {prompt}")

    # prepare client
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        # create tasks with a semaphore to bound concurrency
        sem = asyncio.Semaphore(concurrency)

        results: List[Dict[str, Any]] = []

        async def worker(i: int):
            async with sem:
                return await run_single_request(
                    client,
                    endpoint,
                    model,
                    prompt,
                    max_tokens=output_len,
                    stream=use_streaming,
                    model_for_token_count=model_for_token_count,
                    request_timeout=timeout_per_request,
                )

        # schedule tasks
        start_wall = time.perf_counter()
        tasks = [asyncio.create_task(worker(i)) for i in range(batch_size)]
        # gather results as they finish
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
        end_wall = time.perf_counter()
        wall_time = end_wall - start_wall

    # Summarize results
    successes = [r for r in results if r.get("latency") is not None]
    # failures = [r for r in results if r.get("latency") is None]

    failures = []
    for r in results:
        if r.get("latency") is None:
            failures.append(r)
            pass

    total_input_tokens = sum(r.get("input_tokens", 0) for r in successes)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in successes)
    total_tokens = total_input_tokens + total_output_tokens

    # Latency statistics
    latencies = [r["latency"] for r in successes]
    ttfts = [r["ttft"] for r in successes if r["ttft"] is not None]
    first_to_last = [r["t_first_to_last"] for r in successes if r["t_first_to_last"] is not None and r["t_first_to_last"] > 0]

    def stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
        arr_sorted = sorted(arr)
        n = len(arr)
        def p(pct):
            idx = min(n - 1, int(math.floor(pct / 100.0 * n)))
            return arr_sorted[idx]
        mean = sum(arr) / n
        return {"min": arr_sorted[0], "p50": p(50), "p90": p(90), "max": arr_sorted[-1], "mean": mean}

    latency_stats = stats(latencies)
    ttft_stats = stats(ttfts)
    f2l_stats = stats(first_to_last)

    # Total throughput tokens/sec
    total_throughput = total_tokens / wall_time if wall_time > 0 else 0.0

    # TPOT: time per output token average across requests that had tokens
    # For each request: if output_tokens > 0, request_tpot = t_first_to_last / output_tokens
    per_req_tpot = []
    for r in successes:
        ot = r.get("output_tokens", 0)
        dur = r.get("t_first_to_last", 0.0)
        if ot > 0 and dur is not None:
            per_req_tpot.append(dur / ot)
    tpot_stats = stats(per_req_tpot)

    # TTFT definition: time from request start to first token
    # TPOT definition: time per output token (average across requests) computed above.

    # Print summary
    print("\n===== Ollama Benchmark Summary =====")
    print(f"Target model: {model}")
    print(f"Endpoint: {endpoint}")
    print(f"Batch size (total requests): {batch_size}")
    print(f"Concurrency (parallel requests): {concurrency}")
    print(f"Input size target (tokens): {input_size}")
    print(f"Output length requested (max_tokens): {output_len}")
    print(f"Use streaming: {use_streaming}")
    print(f"Wall-clock time for benchmark: {wall_time:.4f} s")
    print(f"Successful requests: {len(successes)}  Failed: {len(failures)}")
    print()
    print("Token totals (successful requests):")
    print(f"  Total input tokens:  {total_input_tokens}")
    print(f"  Total output tokens: {total_output_tokens}")
    print(f"  Grand total tokens:  {total_tokens}")
    print()
    print("Latency stats (seconds):")
    print(f"  Latency mean: {latency_stats['mean']:.4f}, p50: {latency_stats['p50']:.4f}, p90: {latency_stats['p90']:.4f}, max: {latency_stats['max']:.4f}")
    print("TTFT (time to first token) stats (seconds):")
    print(f"  mean: {ttft_stats['mean']:.4f}, p50: {ttft_stats['p50']:.4f}, p90: {ttft_stats['p90']:.4f}")
    print("TPOT (time per output token) stats (seconds/token):")
    print(f"  mean: {tpot_stats['mean']:.6f}, p50: {tpot_stats['p50']:.6f}, p90: {tpot_stats['p90']:.6f}")
    print()
    print(f"Total throughput: {total_throughput:.2f} tokens/sec (total tokens / wall-clock time)")
    print()
    # Provide per-request breakdown for first few requests
    print("Sample per-request results (first 10 successful requests):")
    for i, r in enumerate(successes[:10]):
        print(
            f" #{i+1:02d}: input_t={r['input_tokens']}, output_t={r['output_tokens']}, latency={r['latency']:.4f}s, "
            f"TTFT={r['ttft']:.4f}s, first->last={r['t_first_to_last']:.4f}s"
        )

    if failures:
        print("\nSome requests failed. Sample failures:")
        for i, f in enumerate(failures[:5]):
            print(f"  Failure {i+1}: error={f.get('error')}")


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Ollama (OpenAI-like) server.")
    p.add_argument("--base-url", required=True, help="Base URL (host[:port]) e.g. localhost:11435 or http://localhost:11435")
    p.add_argument("--model", default="gpt-oss:120b", help="Model name (server-specific). Default: gpt-oss:120b")
    p.add_argument("--batch-size", type=int, default=16, help="Number of requests to send (total).")
    p.add_argument("--concurrency", type=int, default=None, help="Max concurrent requests. Defaults to batch-size.")
    p.add_argument("--input-size", type=int, default=4096, help="Requested input size in tokens (approx).")
    p.add_argument("--output-len", type=int, default=128, help="max_tokens (requested output length).")
    p.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming; use non-streaming requests.")
    p.add_argument("--model-tokenizer", default="gpt-4", help="Model name to use for tokenizer (tiktoken).")
    p.add_argument("--timeout", type=float, default=1200.0, help="Timeout per request (seconds).")
    return p.parse_args()


def main():
    args = parse_args()
    print("Benchmark configuration:")
    print(json.dumps(vars(args), indent=2))
    # Run event loop
    try:
        asyncio.run(
            run_benchmark(
                base_url=args.base_url,
                model=args.model,
                batch_size=args.batch_size,
                input_size=args.input_size,
                output_len=args.output_len,
                concurrency=args.concurrency,
                use_streaming=args.stream,
                model_for_token_count=args.model_tokenizer,
                timeout_per_request=args.timeout,
            )
        )
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)


if __name__ == "__main__":
    main()