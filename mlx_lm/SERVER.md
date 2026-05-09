# HTTP Model Server

You use `mlx-lm` to make an HTTP API for generating text with any supported
model. The HTTP API is intended to be similar to the [OpenAI chat
API](https://platform.openai.com/docs/api-reference).

> [!NOTE]  
> The MLX LM server is not recommended for production as it only implements
> basic security checks.

Start the server with: 

```shell
mlx_lm.server --model <path_to_model_or_hf_repo>
```

For example:

```shell
mlx_lm.server --model mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

This will start a text generation server on port `8080` of the `localhost`
using Mistral 7B instruct. The model will be downloaded from the provided
Hugging Face repo if it is not already in the local cache.

To see a full list of options run:

```shell
mlx_lm.server --help
```

You can make a request to the model by running:

```shell
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

You can also use the Open Responses API:

```shell
curl localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
     "model": "default_model",
     "input": "Say this is a test!",
     "max_output_tokens": 128
   }'
```

Streaming uses semantic Open Responses SSE events:

```shell
curl localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
     "input": [{"type": "message", "role": "user", "content": "Hello!"}],
     "stream": true
   }'
```

### Request Fields

- `messages`: An array of message objects representing the conversation
  history. Each message object should have a role (e.g. user, assistant) and
  content (the message text).

- `role_mapping`: (Optional) A dictionary to customize the role prefixes in
  the generated prompt. If not provided, the default mappings are used.

- `stop`: (Optional) An array of strings or a single string. These are
  sequences of tokens on which the generation should stop.

- `max_tokens`: (Optional) An integer specifying the maximum number of tokens
  to generate. Defaults to `512`.

- `stream`: (Optional) A boolean indicating if the response should be
  streamed. If true, responses are sent as they are generated. Defaults to
  false.

- `temperature`: (Optional) A float specifying the sampling temperature.
  Defaults to `0.0`.

- `top_p`: (Optional) A float specifying the nucleus sampling parameter.
  Defaults to `1.0`.

- `top_k`: (Optional) An integer specifying the top-k sampling parameter.
  Defaults to `0` (disabled).

- `min_p`: (Optional) A float specifying the min-p sampling parameter.
  Defaults to `0.0` (disabled).

- `repetition_penalty`: (Optional) Applies a multiplicative penalty to repeated
  tokens. Defaults to `0.0` (disabled).

- `repetition_context_size`: (Optional) The size of the context window for
  applying repetition penalty. Defaults to `20`.

- `presence_penalty`: (Optional) Applies an additive penalty to tokens
  that appeared before. Defaults to `0.0` (disabled).

- `presence_context_size`: (Optional) The size of the context window for
  applying presence penalty. Defaults to `20`.

- `frequency_penalty`: (Optional) Applies an additive penalty proportional to
  how many times a token appeared previously. Defaults to `0.0` (disabled).

- `frequency_context_size`: (Optional) The size of the context window for
  applying frequency penalty. Defaults to `20`.

- `logit_bias`: (Optional) A dictionary mapping token IDs to their bias
  values. Defaults to `None`.

- `logprobs`: (Optional) An integer specifying the number of top tokens and
  corresponding log probabilities to return for each output in the generated
  sequence. If set, this can be any value between 1 and 10, inclusive.

- `model`: (Optional) A string path to a local model or Hugging Face repo id.
  If the path is local is must be relative to the directory the server was
  started in.

- `adapters`: (Optional) A string path to low-rank adapters. The path must be
  relative to the directory the server was started in.

- `draft_model`: (Optional) Specifies a smaller model to use for speculative
  decoding. Set to `null` to unload.

- `num_draft_tokens`: (Optional) The number of draft tokens the draft model
  should predict at once. Defaults to `3`.

### Open Responses API

The server supports these Open Responses endpoints:

- `POST /v1/responses`: create a response with JSON or SSE output.
- `GET /v1/responses`: WebSocket transport for `response.create` messages.
- `POST /v1/responses/compact`: return an opaque compaction item.

Supported input items:

- `message` with `user`, `assistant`, `system`, or `developer` role.
- `function_call` and `function_call_output` for externally hosted tools.
- `reasoning`, `compaction`, and `item_reference` for replay/continuation.

Supported generation fields include `input`, `instructions`,
`previous_response_id`, `tools`, `tool_choice`, `temperature`, `top_p`,
`presence_penalty`, `frequency_penalty`, `max_output_tokens`, `top_logprobs`,
`stream`, `store`, and `max_tool_calls`.

Compatibility fields accepted and echoed or ignored include `metadata`,
`truncation`, `service_tier`, `safety_identifier`, `prompt_cache_key`,
`include`, `reasoning`, `parallel_tool_calls`, `text`, and `background`.

`previous_response_id` is backed by an in-memory store for `store: true`. The
WebSocket transport also keeps connection-local state, including `store: false`
responses, for continuation on the same socket.

Image, file, and video input parts are accepted and converted to textual markers
for models that do not have native multimodal support.

To run the upstream OpenResponses compliance suite against a local server:

```shell
bun run test:compliance \
  --base-url http://localhost:8080/v1 \
  --api-key test-key \
  --model default_model
```

### Response Fields

- `id`: A unique identifier for the chat.

- `system_fingerprint`: A unique identifier for the system.

- `object`: Any of "chat.completion", "chat.completion.chunk" (for
  streaming), or "text.completion".

- `model`: The model repo or path (e.g. `"mlx-community/Llama-3.2-3B-Instruct-4bit"`).

- `created`: A time-stamp for when the request was processed.

- `choices`: A list of outputs. Each output is a dictionary containing the fields:
    - `index`: The index in the list.
    - `logprobs`: A dictionary containing the fields:
        - `token_logprobs`: A list of the log probabilities for the generated
          tokens.
        - `tokens`: A list of the generated token ids.
        - `top_logprobs`: A list of lists. Each list contains the `logprobs`
          top tokens (if requested) with their corresponding probabilities.
    - `finish_reason`: The reason the completion ended. This can be either of
      `"stop"` or `"length"`.
    - `message`: The text response from the model.

- `usage`: A dictionary containing the fields:
    - `prompt_tokens`: The number of prompt tokens processed.
    - `completion_tokens`: The number of tokens generated.
    - `total_tokens`: The total number of tokens, i.e. the sum of the above two fields.

### List Models

Use the `v1/models` endpoint to list available models:

```shell
curl localhost:8080/v1/models -H "Content-Type: application/json"
```

This will return a list of locally available models where each model in the
list contains the following fields:

- `id`: The Hugging Face repo id.
- `created`: A time-stamp representing the model creation time.
