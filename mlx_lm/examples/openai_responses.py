# Copyright © 2025 Apple Inc.
"""
Examples using the OpenAI responses endpoint with mlx_lm.server.

To run, first start the server:

>>> mlx_lm.server

Then run this script.

More documentation on the API spec here:
https://platform.openai.com/docs/quickstart?api-mode=responses
"""
from openai import OpenAI

model = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

### Basic response example

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.responses.create(
    model=model, input="Write a one-sentence bedtime story about a unicorn."
)
print(response.output_text)

### Input with roles

response = client.responses.create(
    model=model,
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Write a one-sentence bedtime story about a unicorn.",
                },
            ],
        }
    ],
)
print(response.output_text)

### Streaming

stream = client.responses.create(
    model=model,
    input=[
        {
            "role": "user",
            "content": "Say 'double bubble bath' ten times fast.",
        },
    ],
    stream=True,
)

for event in stream:
    print(event)
