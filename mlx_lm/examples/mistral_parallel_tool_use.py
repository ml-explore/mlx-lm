# Copyright © 2025 Apple Inc.

"""
Example demonstrating parallel tool calling with Mistral models using MLX-LM.

This example shows how to:
1. Define multiple tools in OpenAI format (weather, timezone, and population)
2. Use apply_chat_template() which automatically handles OpenAI-to-Mistral conversion
3. Parse parallel tool calls from model output using a reusable parsing function
4. Execute multiple tools with the same function name but different arguments
5. Handle the Mistral-specific parallel calling format

The conversation flow:
- Single turn: User asks for weather in multiple cities → Model calls get_weather multiple times in parallel → Results displayed

Features:
- Robust parse_tool_calls() function that handles parallel tool calls with "}, {" separation
- Multiple tool definitions (weather, timezone, population) available to the model
- OpenAI API compliant message format (though not using the full conversation in this example)
- Proper JSON parsing with error handling

Key insight: The TokenizerWrapper.apply_chat_template() method handles the conversion
from OpenAI message format to Mistral format internally, so you only need to work
with standard OpenAI JSON messages. The parallel calling format allows multiple
calls to the same function with different arguments in a single response.
"""

import json

from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache

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
        "New York, USA": {"temperature": 20, "condition": "Rainy", "humidity": 85},
        "Sydney, Australia": {"temperature": 28, "condition": "Sunny", "humidity": 60},
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


def get_time_zone(location: str):
    """
    Get timezone information for a location.

    Args:
        location: The city and country, e.g. "Paris, France"
    """
    timezone_data = {
        "Paris, France": {
            "timezone": "CET",
            "utc_offset": "+01:00",
            "current_time": "14:30",
        },
        "London, UK": {
            "timezone": "GMT",
            "utc_offset": "+00:00",
            "current_time": "13:30",
        },
        "Tokyo, Japan": {
            "timezone": "JST",
            "utc_offset": "+09:00",
            "current_time": "22:30",
        },
        "New York, USA": {
            "timezone": "EST",
            "utc_offset": "-05:00",
            "current_time": "08:30",
        },
        "Sydney, Australia": {
            "timezone": "AEDT",
            "utc_offset": "+11:00",
            "current_time": "00:30",
        },
    }

    data = timezone_data.get(
        location, {"timezone": "UTC", "utc_offset": "+00:00", "current_time": "13:30"}
    )

    return json.dumps(
        {
            "location": location,
            "timezone": data["timezone"],
            "utc_offset": data["utc_offset"],
            "current_time": data["current_time"],
        }
    )


def get_population(location: str):
    """
    Get population information for a city.

    Args:
        location: The city and country, e.g. "Paris, France"
    """
    population_data = {
        "Paris, France": {"population": 2161000, "metro_population": 12405426},
        "London, UK": {"population": 8982000, "metro_population": 9648110},
        "Tokyo, Japan": {"population": 13960000, "metro_population": 37832892},
        "New York, USA": {"population": 8336817, "metro_population": 20140470},
        "Sydney, Australia": {"population": 5312163, "metro_population": 5312163},
    }

    data = population_data.get(
        location, {"population": 1000000, "metro_population": 2000000}
    )

    return json.dumps(
        {
            "location": location,
            "city_population": f"{data['population']:,}",
            "metro_population": f"{data['metro_population']:,}",
        }
    )


tools = {
    "get_weather": get_weather,
    "get_time_zone": get_time_zone,
    "get_population": get_population,
}

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
            "name": "get_time_zone",
            "description": "Get timezone information for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g. Paris, France",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Get population information for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g. Paris, France",
                    }
                },
                "required": ["location"],
            },
        },
    },
]

prompt = "What's the weather like in Paris, London, and Tokyo? I need the temperature in Celsius."
messages = [{"role": "user", "content": prompt}]

prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tools=tool_definitions
)

prompt_cache = make_prompt_cache(model)

response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=2048,
    verbose=True,
    prompt_cache=prompt_cache,
)

tool_open = "[TOOL_CALLS]"
args_open = "[ARGS]"


def parse_tool_calls(response_text):
    calls = []
    start_tool = response_text.find(tool_open)
    if start_tool == -1:
        return calls

    start_tool += len(tool_open)
    start_args = response_text.find(args_open, start_tool)
    if start_args == -1:
        return calls

    tool_name = response_text[start_tool:start_args].strip()
    args_section = response_text[start_args + len(args_open) :].strip()

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
        try:
            tool_args = json.loads(args_section)
            calls.append({"name": tool_name, "arguments": tool_args})
        except json.JSONDecodeError:
            pass

    return calls


tool_calls = parse_tool_calls(response)

for call in tool_calls:
    tool_result = tools[call["name"]](**call["arguments"])
    print(f"Tool result for {call['name']}: {tool_result}")
