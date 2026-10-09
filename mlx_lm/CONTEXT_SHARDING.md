# Context Sharding

Context sharding splits the KV cache of one long text across several Macs.
Each machine keeps the whole model and only its own part of the cache. You
prepare a big text once, keep the cache on disk, and then ask many short
questions. A question goes to all machines. Each machine computes attention
over its own part of the cache, and the results are merged.

> [!NOTE]
> This is an experimental branch. It is not part of upstream `mlx-lm`. Most
> checks ran on one machine with two local processes. Large parts of the code
> and of this text were written with the help of an AI assistant (Claude). Read
> [Limits and what is not verified](#limits-and-what-is-not-verified) before
> you rely on it.

## When to use it

Use it when:

- You have a long, fixed text (manuals, laws, contracts) and many questions.
- The model fits on one machine, but the KV cache of the text is large and
  generation is slow because of it.

Do not use it when:

- The context is short. Then reading the weights takes most of the time, and
  sharding gives no gain.
- The weights do not fit on one machine. The weights are copied to every
  machine, not split.

## Quick start

Every machine needs the model, the same text file, and the same package
version. Start with two local processes:

```
mlx.launch --hosts 127.0.0.1 -n 2 --backend ring -- \
    python mlx_lm/examples/sharded_context.py prepare \
    --model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --file document.txt --cache-dir /tmp/context_cache

mlx.launch --hosts 127.0.0.1 -n 2 --backend ring -- \
    python mlx_lm/examples/sharded_context.py ask \
    --model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --cache-dir /tmp/context_cache \
    --question "What is the vault code?" --question "Who is the harbour master?"
```

`prepare` runs once. Each machine writes its own shard to its own
`--cache-dir`. `ask` loads the shards and answers. After each answer the
question and the answer are removed from the cache, so the prepared cache stays
the same.

For real machines use `--hostfile hosts.json` instead of `--hosts`. Add
`--python /path/to/python` if the Python path differs between machines. See the
[MLX distributed documentation](https://ml-explore.github.io/mlx/build/html/usage/distributed.html).

Options of the example:

| Option | Meaning |
|---|---|
| `--kv-bits {0,8,4}` | Cache precision. `0` keeps 16-bit values (default). |
| `--weights 3,1` | Share of the cache for each machine. Use it for unequal machines. |
| `--block-size N` | Tokens per prefill block (default 2048). |
| `--backend` | Distributed backend (default `ring`). |

The cache belongs to the exact text and to the number of machines used in
`prepare`. If the cluster changes, prepare again.

## Python API

```python
import mlx.core as mx
from mlx_lm import load
from mlx_lm.sharded_prompt_cache import (
    block_owners, block_ranges, load_sharded_cache, make_sharded_cache,
    prefill_in_blocks, query_scope, save_sharded_cache,
)

group = mx.distributed.init(backend="ring")
model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")

# Prepare once. `ids` are the token ids of the text before the question.
caches = make_sharded_cache(model, group, kv_bits=None)  # or 8, or 4
owners = block_owners(len(block_ranges(len(ids), 2048)), group.size())
prefill_in_blocks(model, ids, caches, group, 2048, owners)
save_sharded_cache("cache/context", caches, group, len(ids))

# Ask later.
caches, meta = load_sharded_cache("cache/context", group)
base = meta["total_tokens"]
with query_scope(caches, base) as set_position:
    logits = model(mx.array(question_ids)[None], cache=caches)[:, -1, :]
    mx.eval(logits)
    # For each new token: set_position(base + number_of_tokens_added_so_far),
    # run the model on the token, and evaluate the result.
# Leaving the block removes the question and the answer from the cache.
```

See `mlx_lm/examples/sharded_context.py` for a complete generation loop.

Rules for callers:

1. All machines must produce the same token ids for the text. Some chat
   templates put the date in the prompt. Store the prompt tail in the cache
   metadata, as the example does.
2. Evaluate every forward pass (`mx.eval`) before you start the next one. A
   pass that is never evaluated can leave a delayed cache write that runs on
   one machine only. The machines then wait for different collectives and hang
   or give wrong results.
3. For a dialog, put the earlier questions and answers into the new question
   text. The prepared cache is never changed.

## How it works

Context sharding splits where the KV cache is stored, and the attention over it.
It does not split the rest of the work of a layer.

| Part | Split between the machines | Same on every machine |
|---|---|---|
| Storage of the keys and values | yes, each machine keeps its own part | |
| Attention over the stored text (a question, a generated token, or a new block during prepare) | yes, each machine reads only its own part | |
| Weights | | yes, every machine has all of them |
| The rest of each layer (projections, MLP, norms) | | yes, every machine computes it for every token |

- **Weights.** Every machine holds the full weights.
- **Prepare.** The text goes through the model in blocks (default 2048 tokens).
  Every machine runs every block through all layers. Two things are shared:
    - Storage. When a block is done, one machine keeps its keys and values. The
      other machines drop their copy. Blocks go to the machines in turn, or by
      the shares you give.
    - Attention over the earlier blocks. Each machine computes it only over the
      blocks it holds. The partial results are merged, as in the next item.
- **Prepare cost.** The work of computing a block is not shared, so prepare time
  does not fall when you add machines. The slowest machine sets the pace. You
  prepare once. Every later question reads the split cache, so each machine
  reads only its own part. The memory used during prepare depends on the block
  size, not on the text length. The score table is computed in tiles of a fixed
  size (`MLX_LM_SHARD_SCORE_BUDGET`).
- **Ring prefill (not used by the prepare flow).** The cache also supports a
  ring prefill: each machine takes its own part of the tokens, and the keys and
  values travel around the ring. This splits the work of the layers too. The
  example does not use it, and the tests in this branch do not cover it.
- **Question and generation.** Each machine computes a partial attention result
  over its own shard: the maximum score, the sum of the exponentials, and the
  weighted values. One `all_gather` per sharded layer collects these small
  results, and every machine merges them with the online softmax rule. The
  keys and values never cross the network.
- **Which layers are sharded.** Layers with a plain `KVCache`. Sliding-window
  layers (Gemma 3 and 4) keep their small cache on every machine. Gemma 4
  layers that reuse keys and values of an earlier layer merge through the cache
  of that earlier layer.
- **Where it plugs in.** `scaled_dot_product_attention` in `models/base.py`
  checks whether the cache has a `group`. If it has, it calls `cache.attend`.
  Models that use this function need no change. Gemma 2 computes attention by
  hand and has its own branch for the attention softcap.
- **Fast decode kernel.** For one query token, `fast_decode_attention.py` uses a
  Metal kernel that returns the partial result directly. It supports 16-bit
  and 32-bit values and head sizes 64 and 128. Other shapes use the normal
  path. `MLX_LM_SHARD_FAST_DECODE=0` turns it off.

Time for one generated token:

```
weight read + read of the own cache shard + one merge per sharded layer
```

Sharding only shortens the second part. It helps when the cache is large. For
short contexts the weights dominate.

## Supported models

| Family | Model types | Checked |
|---|---|---|
| Llama, Mistral | `llama`, `mistral` | Llama 3.2 1B, also on two real machines |
| Qwen 2 and 3 | `qwen2`, `qwen3` | Qwen2.5 0.5B, Qwen3 0.6B (two local processes) |
| Gemma 1, 2, 3 | `gemma`, `gemma2`, `gemma3_text`, `gemma3` | Gemma 3 1B and 4B; Gemma 1 and 2 only as tiny random models |
| Gemma 4 | `gemma4`, `gemma4_text` | E2B: text, pictures, video frames |

Not supported: Qwen 3.5 and Qwen Next (linear attention), Gemma 3n. Not tested:
Qwen3 MoE (same attention code, not run), Gemma 4 `gemma4_unified` (12B),
Ministral 3 and Mistral 3. The weights may be quantized. The weight format does
not depend on the cache format.

## Cache precision

| `kv_bits` | Size of the cache | Notes |
|---|---|---|
| none (16-bit) | 100% | No loss from the cache. Uses the fast decode kernel. |
| 8 | about 53% | `mx.quantize`, group size 64, `mx.quantized_matmul`. |
| 4 | about 28% | Optional. Loses facts in a small model. |

The head size must be a multiple of 64 for 8 and 4 bit.

What was measured:

- Speed of one attention layer at 100K keys (M3 Max, 40 query heads, 8 KV
  heads, head size 128): 1.83 ms for the 16-bit kernel, 1.80 ms for 8 bit, 1.44
  ms for 4 bit.
- A 1B model with a 4K and an 8K text, 80 questions for each length: 69 of 80
  answers were right with 16 and with 8 bit at 4K. At 8K the counts were 70
  and 68. The 8-bit cache lost two answers (one had a wrong digit) and won
  none. That is too few to judge. Measure on your own texts.
- A 1B model with 8192 tokens: the 4-bit cache lost the fact in all three
  answers.
- Gemma 4: stock `mlx-lm` cannot run the layers that reuse keys and values with
  a quantized cache. The sharded path can. Its 8 and 4 bit answers were
  compared with the exact 16-bit answer.

## Pictures and video (Gemma 3 and Gemma 4)

`mlx-lm` has no vision tower. Compute the image features with `mlx-vlm`, put
them into the embeddings, and pass them to the prefill:

```python
prefill_in_blocks(model, ids, caches, group, block_size, owners,
                  embeddings=embeddings, image_groups=groups)
```

- `embeddings` has the shape `(1, number_of_tokens, hidden_size)`. `mlx-lm`
  multiplies the embeddings by a scale inside the model. Divide the features
  from `mlx-vlm` by that scale first (`sqrt(hidden_size)`).
- `ids` are the token ids. For Gemma 4 use 0 at the image positions.
- `groups` has one number per token: -1 for text, and the same number for all
  tokens of one image. Tokens of one image see each other in both directions,
  in every layer type. Blocks never cut an image in two.
- The question must be text. The batch size is 1.
- Video is a list of frames. Each frame is one picture (256 tokens each in the
  tests).
- Gemma 4 writes a "thinking" text first. Use `enable_thinking=False` in the chat
  template. Import `mlx_vlm.models.gemma4.processing_gemma4` before you load its
  processor, because the processor in `transformers` needs `torch`.

Checked with Gemma 4 E2B on two local processes and 16, 8 and 4 bit caches:
two pictures with three questions gave three right answers in every format,
the same as one machine and `mlx-vlm`. With a clip of 4 frames the number in
the last frame was read right. The number in the first frame and the direction
of movement were not reliable. This is a limit of this small model. Each
process used about 3.1 GiB, including the weights.

## Measurements

| What | Result |
|---|---|
| Prepare memory, 1B model, 2 processes, blocks of 1024 | 1.54 GiB peak for texts from 2K to 16K tokens |
| Real pair (M3 Max and M1 MacBook Air over a Thunderbolt bridge), Llama 3.2 1B 4-bit, 8192 tokens, cache split 3:1 | prepare 34 s; a question takes 94 to 121 ms, from scratch about 1.9 s; all answers equal to one machine |
| Decode attention, one token, 40 layers, per machine cache of 62.5K / 100K / 200K keys | 48 / 71 / 127 ms with the kernel; 94 / 148 / 276 ms with the first version |
| Link between the two machines | about 0.2 ms per collective, about 3 GB/s one way |

The decode numbers are sums of single layer measurements. No full run with a
large model and a long text was measured.

## Networking

The examples and tests use the `ring` backend of `mx.distributed`. MLX always
has this backend. It sends data over TCP sockets, so the machines only need to
reach each other over a network: Ethernet, Wi-Fi or Thunderbolt. In a ring, each
machine talks only to its two neighbors.

- **Number of machines.** The MLX documentation does not give a maximum and
  shows a ring of 4 machines. In a Thunderbolt ring each machine uses two
  Thunderbolt ports, one for each neighbor, so the size of the ring does not
  depend on the number of ports. Over Ethernet, any machines that can reach each
  other work. We tested only 2 machines: an M3 Max and an M1 MacBook Air over a
  Thunderbolt bridge. Rings of 4 to 8 machines were not tested. For more than 2
  machines, MLX provides `mlx.distributed_config` to set up the links.
- **Latency.** On the 2-machine link we measured about 0.2 ms for one
  collective and about 3 GB/s in one direction. A ring collective passes data
  from neighbor to neighbor, so its time grows with the number of machines. We
  did not measure this growth.
- **JACCL** (RDMA over Thunderbolt 5). The MLX documentation says its latency is
  an order of magnitude lower than that of the ring backend. It needs macOS 26.2
  or later, Thunderbolt 5, and a fully connected mesh with a direct cable between
  every pair of machines, so N machines need N-1 ports each. We did not test
  JACCL. At the time of writing the MLX issue tracker lists open reports of
  crashes and hangs in JACCL, and its ring mode is new.

Context sharding sends small partial results, one collective per sharded layer,
and never the cache. So the latency of a collective matters more than the
bandwidth.

An idea that is not implemented: a two-level merge. Small groups of 3 or 4
machines could use an RDMA mesh inside the group, and the groups could use TCP
between them. The merge rule allows it, because merging partial results gives
the same answer in any order. MLX has one group per backend and no sub-groups
for JACCL and ring yet, and the merge code would need a second step. A rough
estimate for 8 machines and 40 layers is about 11 ms of synchronization per
token, against about 37 ms for one TCP ring. This is not measured. Models with
few global layers, such as Gemma 3 and 4, need few merges, so the gain is small.

## Debug switches

| Variable | Effect |
|---|---|
| `MLX_LM_SHARD_FAST_DECODE=0` | Turn off the Metal decode kernel. |
| `MLX_LM_SHARD_SCORE_BUDGET` | Size of the score tiles (default `2**26` values). |
| `MLX_LM_SHARD_UNFUSED=1` | Merge with three collectives instead of one. Same result. |
| `MLX_LM_SHARD_WEIGHTS=3,1` | Unequal shares for the older ring prefill path. |

## Tests

```
mlx.launch --hosts 127.0.0.1 -n 2 --backend ring -- \
    python tests/sharded_context_tests.py
```

The tests use tiny random models. For each model family they compare a sharded
run with the same model on one machine, using ordinary caches: the logits and
the top token after a question, after a second question (a rollback in
between), and after saving and loading the cache. They also cover 8-bit caches,
Gemma 4 layers that reuse keys and values, and pictures in Gemma 3 and 4. The
picture tests also compare with a run that has no cache at all. The tests run
with one process too, but then nothing is split.

## Limits and what is not verified

- Real two-machine runs used only Llama 3.2 1B. All other checks used two
  processes on one machine. They show that the results are right. They say
  nothing about speed, because both processes share one GPU.
- Speed with long contexts and big models is an estimate from single layer
  measurements. Nothing above two machines was measured.
- Prepare time does not fall with more machines. Every machine runs every block
  through all layers. Only attention over the stored past is split. The slowest
  machine sets the pace.
- The cache is bound to the exact token ids and to the number of machines.
- Results in `bfloat16` can differ from a single machine at very close token
  choices. The meaning stays the same.
- Small models often miss facts in long text. Sharding does not change that.
- There are no options for this in `mlx_lm.generate` and `mlx_lm.server`. Only
  the example script and the Python API exist. Reading a folder of documents is
  not implemented.
- The checks against real models (Qwen, Gemma 3 and 4, pictures, video frames)
  ran as scripts that are not part of this branch. The tests in `tests/` use
  tiny random models.
