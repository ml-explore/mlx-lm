from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8080/v1",
)

model = "mlx-community/Qwen3-4B-Thinking-2507-4bit"

messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]

# Non-streaming example

response = client.chat.completions.create(
    model=model, messages=messages, max_tokens=2048
)

reasoning = response.choices[0].message.reasoning
content = response.choices[0].message.content

print("=== reasoning ===\n")
print(f"\033[37m{reasoning}\033[0m")
print("=== content ===\n")
print(content)

# Streaming example

stream = client.chat.completions.create(
    model=model,
    messages=messages,
    stream=True,
    max_tokens=2048,
)

for chunk in stream:
    if (reasoning := chunk.choices[0].delta.reasoning) is not None:
        print(f"\033[37m{reasoning}\033[0m", end="")
    if (content := chunk.choices[0].delta.content) is not None:
        print(f"{content}", end="")
print()
