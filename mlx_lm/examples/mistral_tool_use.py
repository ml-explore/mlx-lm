# Copyright © 2025 Apple Inc.

"""
Example demonstrating multi-turn tool calling with Mistral models using MLX-LM.

This example shows how to:
1. Define multiple tools in OpenAI format (weather and multiplication)
2. Use apply_chat_template() which automatically handles OpenAI-to-Mistral conversion
3. Parse tool calls from model output using a reusable parsing function
4. Execute tools and continue the conversation
5. Handle multiple turns with different tool calls

The conversation flow:
- First turn: User asks for weather in Paris → Model calls get_weather → Response about weather
- Second turn: User asks for multiplication → Model calls multiply → Response with calculation

Features:
- Factored parse_tool_calls() function that can handle both single and parallel tool calls
- OpenAI API compliant message format
- Proper error handling for missing tool calls

Key insight: The TokenizerWrapper.apply_chat_template() method handles the conversion
from OpenAI message format to Mistral format internally, so you only need to work
with standard OpenAI JSON messages.
"""

import json

from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

checkpoint = "graelo/Devstral-Small-2507-4bits"

model, tokenizer = load(path_or_hf_repo=checkpoint)


def get_weather(location: str, unit: str = "celsius"):
    """
    Get current weather information for a location.

    Args:
        location: The city and country, e.g. "Paris, France"
        unit: Temperature unit, either "celsius" or "fahrenheit"
    """
    weather_data = {
        "Paris, France": {"temperature": 22, "condition": "Sunny", "humidity": 65},
        "London, UK": {"temperature": 18, "condition": "Cloudy", "humidity": 78},
        "Tokyo, Japan": {
            "temperature": 25,
            "condition": "Partly Cloudy",
            "humidity": 70,
        },
    }

    data = weather_data.get(
        location, {"temperature": 15, "condition": "Unknown", "humidity": 50}
    )

    if unit == "fahrenheit":
        data["temperature"] = data["temperature"] * 9 / 5 + 32

    return json.dumps(
        {
            "location": location,
            "temperature": f"{data['temperature']}°{'F' if unit == 'fahrenheit' else 'C'}",
            "condition": data["condition"],
            "humidity": f"{data['humidity']}%",
            "unit": unit,
        }
    )


def multiply(a: float, b: float):
    """
    A function that multiplies two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b


tools = {"get_weather": get_weather, "multiply": multiply}

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g. Paris, France",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit, either celsius or fahrenheit",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "A function that multiplies two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number to multiply",
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number to multiply",
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
]


def parse_tool_calls(response_text):
    """
    Parse tool calls from model response text.

    Args:
        response_text: The model response containing tool calls

    Returns:
        List of tool call dictionaries with 'name' and 'arguments' keys
    """
    calls = []
    tool_open = "[TOOL_CALLS]"
    args_open = "[ARGS]"

    start_tool = response_text.find(tool_open)
    if start_tool == -1:
        return calls

    start_tool += len(tool_open)
    start_args = response_text.find(args_open, start_tool)
    if start_args == -1:
        return calls

    tool_name = response_text[start_tool:start_args].strip()
    args_section = response_text[start_args + len(args_open) :].strip()

    # Handle multiple argument sets (parallel tool calls)
    if "}, {" in args_section:
        json_parts = args_section.split("}, {")
        for i, part in enumerate(json_parts):
            if i == 0 and not part.endswith("}"):
                part = part + "}"
            elif i == len(json_parts) - 1 and not part.startswith("{"):
                part = "{" + part
            else:
                if not part.startswith("{"):
                    part = "{" + part
                if not part.endswith("}"):
                    part = part + "}"

            try:
                tool_args = json.loads(part)
                calls.append({"name": tool_name, "arguments": tool_args})
            except json.JSONDecodeError:
                pass
    else:
        # Single tool call
        try:
            tool_args = json.loads(args_section)
            calls.append({"name": tool_name, "arguments": tool_args})
        except json.JSONDecodeError:
            pass

    return calls


prompt = "What is the weather like in Paris?"
messages = [{"role": "user", "content": prompt}]

prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tools=tool_definitions
)
formatted_prompt = tokenizer._tokenizer.decode(
    prompt, special_token_policy=SpecialTokenPolicy.KEEP
)
print(f"Formatted prompt:\n{formatted_prompt}\n")

# Create prompt cache once for reuse across the conversation
# Note: Caching effectiveness depends on shared prefixes between prompts.
prompt_cache = make_prompt_cache(model)

response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=2048,
    verbose=True,
    prompt_cache=prompt_cache,
)

# Parse tool calls using the factored function
tool_calls = parse_tool_calls(response)
if tool_calls:
    tool_call = tool_calls[0]  # Get the first tool call
    tool_name = tool_call["name"]
    tool_args = tool_call["arguments"]
    tool_result = tools[tool_name](**tool_args)

    print(f"Tool result: {tool_result}")

    messages.append(
        {
            "role": "assistant",
            "content": None,  # Required by OpenAI API spec, even when making tool calls
            "tool_calls": [
                {
                    "id": "call00001",  # Must be exactly 9 alphanumeric characters
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": "call00001",  # Must match the tool call ID
            "name": tool_name,
            "content": tool_result,
        }
    )
else:
    print("No tool calls found in response")
    exit(1)

print(f"Messages after tool call: {json.dumps(messages, indent=2)}")

# Use apply_chat_template which handles the OpenAI-to-Mistral conversion internally
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)
formatted_prompt = tokenizer._tokenizer.decode(
    prompt, special_token_policy=SpecialTokenPolicy.KEEP
)
print(f"Formatted prompt:\n{formatted_prompt}\n")


response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=2048,
    verbose=True,
    prompt_cache=prompt_cache,
)

print(f"Final response:\n{response}\n")

# Second turn: Add a multiplication request
messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": "Now multiply 12234585 and 48838483920."})

print("=== SECOND TURN: MULTIPLICATION ===")

# Generate the second response with tools
# Note: This call reuses some cached computation from the shared prefix
# (the AVAILABLE_TOOLS section and initial conversation), but since
# the prompt is longer, the speedup is limited.
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tools=tool_definitions
)
formatted_prompt = tokenizer._tokenizer.decode(
    prompt, special_token_policy=SpecialTokenPolicy.KEEP
)
print(f"Formatted prompt for multiplication:\n{formatted_prompt}\n")

response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=2048,
    verbose=True,
    prompt_cache=prompt_cache,
)

# Parse the multiplication tool call using the factored function
tool_calls = parse_tool_calls(response)
if tool_calls:
    tool_call = tool_calls[0]  # Get the first tool call
    tool_name = tool_call["name"]
    tool_args = tool_call["arguments"]
    tool_result = tools[tool_name](**tool_args)

    print(f"Multiplication tool result: {tool_result}")

    # Add the multiplication tool call and result to conversation
    messages.append(
        {
            "role": "assistant",
            "content": None,  # Required by OpenAI API spec, even when making tool calls
            "tool_calls": [
                {
                    "id": "call00002",  # Different ID for second call
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": "call00002",
            "name": tool_name,
            "content": str(
                tool_result
            ),  # Convert to string since multiply returns a number
        }
    )
else:
    print("No tool calls found in multiplication response")
    exit(1)

# Generate final response with multiplication result
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)
formatted_prompt = tokenizer._tokenizer.decode(
    prompt, special_token_policy=SpecialTokenPolicy.KEEP
)
print(f"Final formatted prompt:\n{formatted_prompt}\n")

response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=2048,
    verbose=True,
    prompt_cache=prompt_cache,
)

print(f"Final multiplication response:\n{response}\n")
