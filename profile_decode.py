# Copyright © 2026 Apple Inc.

import argparse
import time

import mlx.core as mx

# isort: split
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import distributed_argmax
from mlx_lm.utils import sharded_load


def main():
    p = argparse.ArgumentParser(description="Time decode forward passes.")
    p.add_argument("--model", required=True, help="Checkpoint directory.")
    p.add_argument("--prompt-tokens", type=int, default=1024)
    p.add_argument("--passes", type=int, default=5)
    p.add_argument("--prefill-step-size", type=int, default=2048)
    args = p.parse_args()

    group = mx.distributed.init()

    def log(msg):
        if group.rank() == 0:
            print(msg, flush=True)

    model, _ = sharded_load(args.model)
    # Wired as in generate_step and the server, so weights are not paged out.
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    # Same seed on every rank, so all ranks see the same prompt.
    mx.random.seed(0)
    prompt = mx.random.randint(0, model.args.vocab_size, (1, args.prompt_tokens))
    cache = make_prompt_cache(model)

    tic = time.perf_counter()
    prefill, step = prompt[:, :-1], args.prefill_step_size
    for i in range(0, prefill.shape[1], step):
        model(prefill[:, i : i + step], cache=cache)
        mx.eval([c.state for c in cache])
    log(f"prefill {prefill.shape[1]} tokens: {time.perf_counter() - tic:.2f} s")

    token = prompt[:, -1:]
    times = []
    for j in range(args.passes + 1):
        tic = time.perf_counter()
        logits = model(token, cache=cache)
        token = distributed_argmax(logits[:, -1:], model.vocab_group)
        mx.eval(token)
        times.append(time.perf_counter() - tic)
        log(f"pass {j}{' (warmup)' if j == 0 else ''}: {times[-1] * 1e3:.1f} ms")

    mean = sum(times[1:]) / args.passes
    log(
        f"mean {mean * 1e3:.1f} ms = {1 / mean:.1f} tok/s "
        f"(warmup {times[0] * 1e3:.1f} ms), peak memory {mx.get_peak_memory() / 1e9:.0f} GB"
    )

    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))


if __name__ == "__main__":
    main()
