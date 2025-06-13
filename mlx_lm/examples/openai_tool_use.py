# Copyright © 2025 Apple Inc.
"""
This is an example of tool use with mlx_lm and the OpenAI client.

To run, first start the server:

>>> mlx_lm.server

Then run this script.
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

model = "mlx-community/qwen3-4b-4bit-DWQ"
messages = [{"role": "user", "content": "What's the weather in Boston?"}]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]


def get_current_weather(**kwargs):
    return "51 Farenheit, clear skies"


functions = {"get_current_weather": get_current_weather}

# The first query generates a tool call
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
)

# Call the function
function = response.choices[0].message.tool_calls[0].function
tool_result = functions[function.name](**function.arguments)

# Put the result of the function in the messages and generate the final
# response:
messages.append({"role": "tool", "name": function.name, "content": tool_result})
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
)
print(response.choices[0].message.content)
