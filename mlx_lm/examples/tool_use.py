# Copyright © 2025 Apple Inc.

import json

from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache

# Specify the checkpoint
checkpoint = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

# Load the corresponding model and tokenizer
model, tokenizer = load(path_or_hf_repo=checkpoint)


# An example tool, make sure to include a docstring and type hints
def multiply(a: float, b: float):
    """
    A function that multiplies two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b


tools = {"multiply": multiply}

# Specify the prompt and conversation history
prompt = "Multiply 12234585 and 48838483920."
messages = [{"role": "user", "content": prompt}]

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tools=list(tools.values()),
)

prompt_cache = make_prompt_cache(model)

# Generate the initial tool call:
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=2048,
    verbose=True,
    prompt_cache=prompt_cache,
)

# Parse the tool call:
# - The tool call format is model specific.
# - The tokenizer's tool parser expects tool call text to be already extracted.
start_tool = response.find(tokenizer.tool_call_start) + len(tokenizer.tool_call_start)
end_tool = response.find(tokenizer.tool_call_end)
tool_call = tokenizer.tool_parser(response[start_tool:end_tool].strip())
tool_result = tools[tool_call["name"]](**tool_call["arguments"])

# Put the tool result in the prompt
messages = [{"role": "tool", "name": tool_call["name"], "content": tool_result}]
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

# Generate the final response:
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=2048,
    verbose=True,
    prompt_cache=prompt_cache,
)
