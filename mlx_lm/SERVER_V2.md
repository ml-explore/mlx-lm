# Experimental HTTP Model Server

> [!NOTE]
> The MLX LM server is not recommended for production as it only implements
> basic security checks.

`mlx_lm.server_v2` is an experimental OpenAI-compatible server, separate from
`mlx_lm.server`. Its interface can change between releases. Start it with:

```shell
pip install "mlx-lm[server]"
mlx_lm.server_v2 --model mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

It serves the model given with `--model` at `/v1/chat/completions`,
`/v1/completions`, `/v1/models` and `/health`, with errors in the OpenAI
format. `/health` returns 200 with `{"status": "ok"}`, else 503 with the
status, e.g. `"stalled"`. For all options, run `mlx_lm.server_v2 --help`.

If the `generation_config.json` of the model sets `do_sample`, its temperature
(1.0 if it has none), top-p, top-k, min-p and repetition penalty are the
sampling defaults; else decoding is greedy. The flags `--temp`, `--top-p`,
`--top-k` and `--min-p` replace these defaults. A `generation_config.json`
with a bad value is ignored, with a warning. A request without `max_tokens`
generates at most `--max-tokens` tokens (default: 8192), within the context
length.

The decode batch pads each request to the longest one, so with long prompts,
lower `--decode-concurrency` to use less memory. Paged attention, which does
not pad, will be explored in the future.

A request gets a 400 error if its prompt and `max_tokens` do not fit in the
context length, if `n` is not 1, if a `/v1/completions` prompt is not a string,
or if `stop` has more than 16 strings or a string longer than 256 characters.

With `mlx.launch`, it serves one model from several machines, and rank 0 serves
HTTP. The ranks split the tensors of each layer, or the layers with
`--pipeline`.

These defaults and outputs differ from `mlx_lm.server`:

- `--max-tokens` is 8192 (`mlx_lm.server`: 512)
- `--prefill-step-size` is 512 (`mlx_lm.server`: 2048)
- CORS is off until `--allowed-origins` sets the origins (`mlx_lm.server`: `*`)
- Reasoning is sent as `reasoning` and also as `reasoning_content`

These features of `mlx_lm.server` are not supported yet:

- Speculative decoding (`--draft-model`)
- Images
- KV cache quantization (`--kv-bits`)
- A model, adapter or draft model in the request
- Models with a cache that cannot be batched
- Models without a chat template (`role_mapping`)
- `seed` with a temperature above 0
