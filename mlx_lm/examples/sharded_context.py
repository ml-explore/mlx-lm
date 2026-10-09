"""
Prepare a text once, then ask many short questions against its sharded KV cache.

Every machine keeps the whole model and only its own part of the KV cache.
The text file must be identical on all machines.

Try it with two local processes:

```
mlx.launch --hosts 127.0.0.1 -n 2 --backend ring -- \
    python /path/to/sharded_context.py prepare \
    --model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --file document.txt --cache-dir /tmp/context_cache

mlx.launch --hosts 127.0.0.1 -n 2 --backend ring -- \
    python /path/to/sharded_context.py ask \
    --model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --cache-dir /tmp/context_cache --question "What is the vault code?"
```

For real machines use `--hostfile hosts.json` instead of `--hosts`
and `--python /path/to/python` if the Python path differs between machines. The cache
needs the same number of machines when you ask as when you prepared it.
See mlx_lm/CONTEXT_SHARDING.md for details.
"""

import argparse
import os

import mlx.core as mx

from mlx_lm import load
from mlx_lm.sharded_prompt_cache import (
    block_owners,
    block_ranges,
    load_sharded_cache,
    make_sharded_cache,
    prefill_in_blocks,
    query_scope,
    save_sharded_cache,
)

MARK = "<<QUESTION>>"
END_TOKENS = ("<end_of_turn>", "<turn|>", "<|eot_id|>", "<|im_end|>")


def split_prompt(tokenizer, document):
    """Chat prompt with the document first: (text up to the question, text after it)."""
    messages = [{"role": "user", "content": f"{document}\n\nQuestion:{MARK}"}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    head, tail = text.split(MARK)
    return head, tail


def stop_tokens(tokenizer):
    stop = set(tokenizer.eos_token_ids)
    for name in END_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(name)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            stop.add(token_id)
    return stop


def check_same_on_all_ranks(ids, group):
    """Stop early if the machines read different texts."""
    mine = mx.array([float(len(ids)), float(sum(ids) % 1000003)])
    total = mx.distributed.all_sum(mine, group=group)
    mx.eval(total)
    if not mx.allclose(total, mine * group.size()).item():
        raise RuntimeError("The text differs between machines.")


def prepare(args, model, tokenizer, group):
    with open(args.file, encoding="utf-8") as f:
        head, tail = split_prompt(tokenizer, f.read())
    ids = tokenizer.encode(head, add_special_tokens=False)
    check_same_on_all_ranks(ids, group)

    weights = [float(w) for w in args.weights.split(",")] if args.weights else None
    n_blocks = len(block_ranges(len(ids), args.block_size))
    owners = block_owners(n_blocks, group.size(), weights)
    caches = make_sharded_cache(model, group, kv_bits=args.kv_bits or None)
    prefill_in_blocks(model, ids, caches, group, args.block_size, owners)

    path = save_sharded_cache(
        os.path.join(args.cache_dir, "context"),
        caches,
        group,
        len(ids),
        extra={"tail": tail},
    )
    print(f"[rank {group.rank()}] {len(ids)} tokens, shard saved to {path}", flush=True)


def answer(model, tokenizer, caches, question_ids, base, set_position, max_tokens):
    stop = stop_tokens(tokenizer)
    logits = model(mx.array(question_ids)[None], cache=caches)[:, -1, :]
    tokens = []
    for step in range(max_tokens):
        token = mx.argmax(logits, axis=-1).astype(mx.int32)
        mx.eval(token)  # evaluate every pass before the next one
        if token.item() in stop:
            break
        tokens.append(token.item())
        if step == max_tokens - 1:
            break
        set_position(base + len(question_ids) + step)
        logits = model(token[:, None], cache=caches)[:, -1, :]
    return tokenizer.decode(tokens)


def ask(args, model, tokenizer, group):
    caches, meta = load_sharded_cache(os.path.join(args.cache_dir, "context"), group)
    base, tail = meta["total_tokens"], meta["extra"]["tail"]
    for question in args.question:
        question_ids = tokenizer.encode(" " + question + tail, add_special_tokens=False)
        with query_scope(caches, base) as set_position:
            text = answer(
                model, tokenizer, caches, question_ids, base, set_position, args.max_tokens
            )
        if group.rank() == 0:
            print(f"Q: {question}\nA: {text.strip()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Sharded context example")
    parser.add_argument("mode", choices=["prepare", "ask"])
    parser.add_argument("--model", required=True, help="HF repo or path to local model.")
    parser.add_argument("--cache-dir", required=True, help="Folder for this machine's shard.")
    parser.add_argument("--file", help="Text file with the document (prepare).")
    parser.add_argument(
        "--question", action="append", help="Question (ask). Repeat for several."
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=0,
        choices=[0, 4, 8],
        help="Cache precision: 0 keeps 16-bit values, 8 or 4 quantizes them.",
    )
    parser.add_argument(
        "--weights",
        help="Relative share of the cache per machine, for example 3,1 (prepare).",
    )
    parser.add_argument("--backend", default="ring", help="Distributed backend.")
    args = parser.parse_args()

    group = mx.distributed.init(backend=args.backend)
    model, tokenizer = load(args.model)
    if args.mode == "prepare":
        if not args.file:
            parser.error("prepare needs --file")
        prepare(args, model, tokenizer, group)
    else:
        if not args.question:
            parser.error("ask needs --question")
        ask(args, model, tokenizer, group)


if __name__ == "__main__":
    main()
