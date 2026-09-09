"""Parsing for OpenAI's "harmony" multi-channel response format (gpt-oss).

gpt-oss models separate internal reasoning from user-facing output with a
small set of control tokens rather than a single <think>/</think> pair:

    <|channel|>analysis<|message|>...internal reasoning...<|end|>
    <|start|>assistant<|channel|>final<|message|>...user-facing answer...

``<|channel|>`` is a single combined special token shared by every channel
boundary; the channel name ("analysis", "commentary", or "final") is
generated as ordinary text between ``<|channel|>`` and ``<|message|>``, and a
``commentary`` channel may additionally carry a ``<|constrain|>FORMAT``
segment (e.g. for a tool call's JSON payload). This does not fit the
existing think-token detection in ``tokenizer_utils._infer_thinking``, which
only matches a fixed single start/end token pair — see ``has_harmony_format``
for the dedicated detection this module pairs with.

Tracking: https://github.com/ml-explore/mlx-lm/issues/875 (channel token
leakage into ``message.content``, reported against 0.30.6, reproduced here
against 0.31.3 with mlx-community/gpt-oss-20b-MXFP4-Q8 via aider's
whole-file-edit client).
"""

import re

HARMONY_CONTROL_TOKENS = ("<|channel|>", "<|message|>", "<|start|>", "<|end|>")

_SEGMENT_RE = re.compile(
    r"<\|channel\|>(?P<name>[a-zA-Z_]+)"
    r"(?:<\|constrain\|>(?P<fmt>[^<]*))?"
    r"<\|message\|>(?P<body>.*?)"
    r"(?=<\|end\|>|<\|return\|>|<\|start\|>|<\|channel\|>|\Z)",
    re.DOTALL,
)


def parse_harmony_response(text: str) -> tuple[str, str]:
    """Split harmony-formatted text into (content, reasoning).

    Text with no channel markers at all is returned unchanged as content
    with empty reasoning — a safe no-op when called on non-harmony output.
    "analysis" channel bodies become reasoning; every other channel name
    ("final", "commentary", or anything unrecognized) contributes to
    content, concatenated in generation order — a commentary channel's
    tool-call payload is kept in content rather than dropped, matching the
    behavior requested in ml-explore/mlx-lm#875.

    A response truncated mid-channel (e.g. hit the completion length cap
    before a "final" channel ever started) correctly yields empty content
    and a partial reasoning string, rather than raising or fabricating an
    end marker that was never generated.
    """
    matches = list(_SEGMENT_RE.finditer(text))
    if not matches:
        return text, ""
    content_parts = []
    reasoning_parts = []
    for m in matches:
        (reasoning_parts if m.group("name") == "analysis" else content_parts).append(
            m.group("body")
        )
    return "".join(content_parts), "".join(reasoning_parts)


def has_harmony_format(tokenizer) -> bool:
    """True if `tokenizer`'s vocabulary declares the full harmony control set.

    Checked directly against the vocabulary rather than inferred from a
    model-family name, so this activates only for a tokenizer that actually
    has all four control tokens available — never a false positive for an
    unrelated model that merely happens to use one of these token strings
    for something else.
    """
    vocab = tokenizer.get_vocab()
    return all(t in vocab for t in HARMONY_CONTROL_TOKENS)
