# Copyright © 2026 Apple Inc.

"""Response format of Muse Spark.

One turn holds several messages, each with a header:

    to=self<|message|>reasoning<|eom|>
    <|start|>assistant to=NAME<|message|><atem:function_calls>...
    <|start|>assistant to=user<|message|>answer<|eot|>

The text in a header is dropped. The state machine adds the tool call
transitions.
"""

RESPONSE_FORMAT = {
    "initial_state": "header",
    "transitions": {
        "header": [
            ("to=self<|message|>", "reasoning"),
            ("to=user<|message|>", "normal"),
            ("to=commentary<|message|>", "normal"),
            ("<atem:function_calls>", "tool"),
        ],
        "reasoning": [("<|eom|>", "header")],
        "normal": [("<|eom|>", "header"), ("<|eot|>", "normal")],
    },
}
