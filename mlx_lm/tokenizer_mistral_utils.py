"""
Mistral-specific tokenizer utilities.

This module contains all Mistral-specific functionality that was previously embedded
in tokenizer_utils.py, providing clean separation between standard HuggingFace
tokenizer support and Mistral tokenizer support.
"""

try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer
    from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.protocol.instruct.messages import (
        UserMessage,
        AssistantMessage,
        SystemMessage,
        ToolMessage,
    )
    from mistral_common.protocol.instruct.tool_calls import (
        ToolCall,
        FunctionCall,
        Function,
        Tool,
    )

    MISTRAL_AVAILABLE = True
except ImportError:
    MistralTokenizer = None  # type: ignore
    Tekkenizer = None  # type: ignore
    SpecialTokenPolicy = None  # type: ignore
    ChatCompletionRequest = None  # type: ignore
    UserMessage = None  # type: ignore
    AssistantMessage = None  # type: ignore
    SystemMessage = None  # type: ignore
    TextChunk = None  # type: ignore
    ToolMessage = None  # type: ignore
    ToolCall = None  # type: ignore
    FunctionCall = None  # type: ignore
    Function = None  # type: ignore
    Tool = None  # type: ignore

    MISTRAL_AVAILABLE = False


class StreamingDetokenizer:
    """The streaming detokenizer interface so that we can detokenize one token at a time."""

    __slots__ = ("text", "tokens", "offset")

    def reset(self):
        raise NotImplementedError()

    def add_token(self, token):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()

    @property
    def last_segment(self):
        """Return the last segment of readable text since last time this property was accessed."""
        text = self.text
        segment = text[self.offset :]
        self.offset = len(text)
        return segment


class MistralStreamingDetokenizer(StreamingDetokenizer):
    """Efficient streaming detokenizer for MistralTokenizer with byte/unicode edge handling."""

    def __init__(self, tokenizer):
        # Extract the underlying Tekkenizer from MistralTokenizer
        if hasattr(tokenizer, "instruct_tokenizer") and hasattr(
            tokenizer.instruct_tokenizer, "tokenizer"
        ):
            self._tokenizer = tokenizer.instruct_tokenizer.tokenizer
        else:
            self._tokenizer = tokenizer
        if MISTRAL_AVAILABLE and Tekkenizer is not None:
            assert isinstance(self._tokenizer, Tekkenizer)
        self.reset()

    def reset(self):
        self.offset = 0
        self.tokens = []
        self._text = ""
        self._buffer = []
        self._current_text = ""

    def add_token(self, token):
        self._buffer.append(token)
        self.tokens.append(token)
        # Decode only the buffer to avoid unnecessary detokenization
        if MISTRAL_AVAILABLE and SpecialTokenPolicy is not None:
            decoded = self._tokenizer.decode(
                self._buffer, special_token_policy=SpecialTokenPolicy.KEEP
            )
        else:
            decoded = self._tokenizer.decode(self._buffer)
        # Heuristic: only flush if the decoded text is valid (no replacement
        # char) or ends with a space/newline
        if decoded and not decoded.endswith("\ufffd"):
            self._text += decoded
            self._buffer.clear()
            self._current_text = ""
        else:
            self._current_text = decoded

    def finalize(self):
        if self._buffer:
            decoded = self._tokenizer.decode(self._buffer)
            self._text += decoded
            self._buffer = []
            self._current_text = ""

    @property
    def text(self):
        return self._text + self._current_text


class MistralTokenizerWrapper:
    """Helper class that provides Mistral-specific tokenizer functionality."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def is_mistral_tokenizer(self, tokenizer) -> bool:
        """Check if tokenizer is a MistralTokenizer."""
        return hasattr(tokenizer, "instruct_tokenizer") and hasattr(
            tokenizer.instruct_tokenizer, "tokenizer"
        )

    def get_underlying_tokenizer(self, tokenizer):
        """Get the underlying Tekkenizer from MistralTokenizer if applicable."""
        if self.is_mistral_tokenizer(tokenizer):
            return tokenizer.instruct_tokenizer.tokenizer
        return tokenizer

    def get_vocab(self, tokenizer):
        """Get vocabulary from Mistral tokenizer."""
        if self.is_mistral_tokenizer(tokenizer):
            # For MistralTokenizer, get vocab from underlying tokenizer
            underlying_tokenizer = self.get_underlying_tokenizer(tokenizer)
            if hasattr(underlying_tokenizer, "vocab") and callable(
                underlying_tokenizer.vocab
            ):
                vocab_list = underlying_tokenizer.vocab()
                return {token: idx for idx, token in enumerate(vocab_list)}  # type: ignore
        return {}

    def has_mistral_chat_completion(self, tokenizer):
        """Check if tokenizer supports Mistral chat completion API."""
        return (
            hasattr(tokenizer, "encode_chat_completion")
            and ChatCompletionRequest is not None
            and UserMessage is not None
            and AssistantMessage is not None
            and SystemMessage is not None
        )

    def get_eos_token_id(self, tokenizer):
        """Get EOS token ID from Mistral tokenizer."""
        if self.is_mistral_tokenizer(tokenizer):
            underlying_tokenizer = self.get_underlying_tokenizer(tokenizer)
            return getattr(underlying_tokenizer, "eos_id", None)
        return None

    def encode(self, tokenizer, text, add_special_tokens=True, **kwargs):
        """Custom encode method for Mistral tokenizers."""
        if self.is_mistral_tokenizer(tokenizer):
            # For MistralTokenizer, use underlying Tekkenizer with bos/eos parameters
            underlying_tokenizer = self.get_underlying_tokenizer(tokenizer)
            return underlying_tokenizer.encode(
                text,
                bos=add_special_tokens,
                eos=False,  # Usually we don't want EOS during encoding
                **kwargs,
            )
        else:
            raise ValueError("Not a Mistral tokenizer")

    def convert_to_mistral_messages(self, messages):
        """Convert OpenAI-format messages to Mistral-common format."""
        if not MISTRAL_AVAILABLE:
            return []

        mistral_messages = []
        # Track tool calls to map IDs back to function names
        tool_call_map = {}

        for msg in messages:
            role = msg["role"]
            content = msg.get("content")

            if role == "system" and SystemMessage is not None:
                mistral_messages.append(SystemMessage(content=content))

            elif role == "user" and UserMessage is not None:
                mistral_messages.append(UserMessage(content=content))

            elif role == "assistant" and AssistantMessage is not None:
                if "tool_calls" in msg and msg["tool_calls"]:
                    try:
                        if ToolCall is not None and FunctionCall is not None:
                            tool_calls = []
                            for tool_call in msg["tool_calls"]:
                                if tool_call.get("type") == "function":
                                    function_call = tool_call["function"]
                                    call_id = tool_call["id"]
                                    func_name = function_call["name"]

                                    # Store mapping for later tool result messages
                                    tool_call_map[call_id] = func_name

                                    tool_calls.append(
                                        ToolCall(
                                            id=call_id,
                                            function=FunctionCall(
                                                name=func_name,
                                                arguments=function_call["arguments"],
                                            ),
                                        )
                                    )

                            mistral_messages.append(
                                AssistantMessage(content=content, tool_calls=tool_calls)
                            )
                        else:
                            mistral_messages.append(AssistantMessage(content=content))
                    except (ImportError, TypeError):
                        mistral_messages.append(AssistantMessage(content=content))
                else:
                    mistral_messages.append(AssistantMessage(content=content))

            elif role == "tool":
                try:
                    if ToolMessage is not None:
                        tool_call_id = msg["tool_call_id"]
                        name = msg.get("name", "")

                        # If name is missing, try to get it from our mapping
                        if not name and tool_call_id in tool_call_map:
                            name = tool_call_map[tool_call_id]

                        # If we still don't have a name, log a warning but continue
                        if not name:
                            print(
                                f"Warning: Tool message missing function name for call_id {tool_call_id}"
                            )

                        mistral_messages.append(
                            ToolMessage(
                                tool_call_id=tool_call_id,
                                name=name,
                                content=content,
                            )
                        )
                except (ImportError, TypeError):
                    pass

        return mistral_messages

    def convert_to_mistral_tools(self, tools):
        """Convert OpenAI-format tools to Mistral-common format."""
        if not tools or Tool is None or Function is None:
            return None

        mistral_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func_def = tool["function"]
                mistral_tool = Tool(
                    function=Function(
                        name=func_def["name"],
                        description=func_def.get("description", ""),
                        parameters=func_def.get("parameters", {}),
                    )
                )
                mistral_tools.append(mistral_tool)
        return mistral_tools

    def apply_mistral_chat_template(
        self, tokenizer, messages, add_generation_prompt=True, tools=None
    ):
        """Apply chat template using Mistral tokenizer."""
        if not MISTRAL_AVAILABLE or ChatCompletionRequest is None:
            raise ValueError("Mistral libraries not available")

        try:
            # Convert to Mistral-common format
            mistral_messages = self.convert_to_mistral_messages(messages)
            mistral_tools = self.convert_to_mistral_tools(tools)

            # Create ChatCompletionRequest
            request = ChatCompletionRequest(
                messages=mistral_messages, tools=mistral_tools
            )

            # Encode with MistralTokenizer
            result = tokenizer.encode_chat_completion(request)

            # Handle generation prompt - if we don't want generation prompt,
            # we might need to modify the tokens to remove the space at the end
            if not add_generation_prompt and result.text.endswith(" "):
                # Remove the last token if it's just a space for generation
                return (
                    result.tokens[:-1]
                    if result.tokens and result.tokens[-1] != result.tokens[0]
                    else result.tokens
                )

            return result.tokens

        except Exception:
            # Let the main tokenizer wrapper handle the fallback
            raise

    def get_bos_token(self, tokenizer):
        """Get BOS token from Mistral tokenizer."""
        if self.is_mistral_tokenizer(tokenizer):
            underlying_tokenizer = self.get_underlying_tokenizer(tokenizer)
            if hasattr(underlying_tokenizer, "bos_id"):
                try:
                    return underlying_tokenizer.decode([underlying_tokenizer.bos_id])
                except Exception:
                    return None
        return None

    def get_eos_token(self, tokenizer):
        """Get EOS token from Mistral tokenizer."""
        if self.is_mistral_tokenizer(tokenizer):
            underlying_tokenizer = self.get_underlying_tokenizer(tokenizer)
            if hasattr(underlying_tokenizer, "eos_id"):
                try:
                    return underlying_tokenizer.decode([underlying_tokenizer.eos_id])
                except Exception:
                    return None
        return None

    def save_pretrained(self, tokenizer, save_directory, **kwargs):
        """Save Mistral tokenizer."""
        from pathlib import Path

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.is_mistral_tokenizer(tokenizer):
            # For MistralTokenizer, check if the underlying tokenizer has a file_path
            underlying_tokenizer = self.get_underlying_tokenizer(tokenizer)
            if (
                hasattr(underlying_tokenizer, "file_path")
                and underlying_tokenizer.file_path
            ):
                # Copy the original tekken.json file
                import shutil

                tekken_file = Path(underlying_tokenizer.file_path)
                if tekken_file.exists():
                    shutil.copy2(tekken_file, save_path / "tekken.json")
                else:
                    print(f"Warning: Could not find tekken.json at {tekken_file}")
            else:
                print(
                    "Warning: MistralTokenizer has no file_path, cannot save tekken.json"
                )


def load_mistral_tokenizer(model_path, eos_token_ids=None):
    """Load a Mistral tokenizer if tekken.json exists."""
    tekken_file = model_path / "tekken.json"
    if tekken_file.exists() and MistralTokenizer is not None:
        tokenizer = MistralTokenizer.from_file(str(tekken_file))
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        return tokenizer, MistralStreamingDetokenizer, eos_token_ids
    return None, None, None
