# MLX LM Examples

This directory contains advanced examples built using `mlx-lm`.

## Dynamic Batch Server

`mlx_lm.batch_server` is a lightweight inference server implemented based on Python's `http.server`. It utilizes `mlx-lm`'s `BatchGenerator` to implement dynamic batching, significantly improving throughput when handling concurrent requests.

The server provides an OpenAI API compatible interface (`/v1/chat/completions`), supporting both streaming and non-streaming responses.

### Features

- **Dynamic Batching**: Automatically merges multiple concurrent requests into a single batch for inference, maximizing GPU utilization.
- **OpenAI Compatible**: Supports the standard OpenAI Chat Completions API.
- **Streaming Output**: Supports Server-Sent Events (SSE) streaming output.

### Usage

Start the server:

```bash
mlx_lm.batch_server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080
```

### Arguments

| Argument | Description | Default |
|------|------|--------|
| `--model` | Model path or Hugging Face repo ID (Required) | - |
| `--adapter-path` | LoRA adapter path (Optional) | None |
| `--host` | Host address to bind | 127.0.0.1 |
| `--port` | Port to listen on | 8080 |
| `--batch-size` | Maximum batch size | 32 |
| `--log-level` | Log level (DEBUG, INFO, WARNING, ERROR) | INFO |

### API Call Example

You can use `curl` or the `openai` Python client to make calls:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

---

## Benchmark Client

`benchmark_client.py` is a tool for stress testing and performance evaluation. It can simulate multiple concurrent users sending requests to the server and calculate key performance metrics.

### Dependencies

Requires the `openai` library:

```bash
pip install openai
```

### Usage

Run the benchmark:

```bash
python benchmark_client.py --concurrency 8 --requests 50 --stream --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Arguments

| Argument | Description | Default |
|------|------|--------|
| `--base-url` | API Base URL | http://localhost:8080/v1 |
| `--api-key` | API Key (Can be anything if server doesn't verify) | EMPTY |
| `--model` | Model name | default_model |
| `--concurrency` | Number of concurrent threads (simulated users) | 4 |
| `--requests` | Total number of requests to send | 20 |
| `--stream` | Whether to use streaming requests (Recommended for calculating TTFT) | False |
| `--min-delay` | Minimum random delay between requests (seconds) | 0.1 |
| `--max-delay` | Maximum random delay between requests (seconds) | 2.0 |
| `--min-tokens` | Minimum tokens to generate (controlled via max_tokens) | 10 |
| `--max-tokens` | Maximum tokens to generate | 512 |

### Output Metrics Explanation

After the test completes, the client outputs the following statistics:

- **Successful requests**: Number of requests successfully completed.
- **Total tokens generated**: Total number of tokens generated across all requests.
- **Total duration**: Total duration of the test.
- **Throughput (TPS)**: System throughput (Tokens Per Second).
- **Avg Latency**: Average request latency (from sending to receiving full response).
- **Avg TTFT (Time To First Token)**: Average Time To First Token (Accurate only in streaming mode).
- **Avg ITL (Inter-Token Latency)**: Average Inter-Token Latency (Generation smoothness).
