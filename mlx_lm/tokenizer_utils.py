import json
from functools import partial
from json import JSONDecodeError
from typing import List

from transformers import AutoTokenizer, PreTrainedTokenizerFast

try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer
    from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.protocol.instruct.messages import (
        UserMessage,
        AssistantMessage,
        SystemMessage,
        TextChunk,
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

    MISTRAL_AVAILABLE = False


class StreamingDetokenizer:
    """The streaming detokenizer interface so that we can detokenize one token at a time.

    Example usage is as follows:

        detokenizer = ...

        # Reset the tokenizer state
        detokenizer.reset()

        for token in generate(...):
            detokenizer.add_token(token.item())

            # Contains the whole text so far. Some tokens may not be included
            # since it contains whole words usually.
            detokenizer.text

            # Contains the printable segment (usually a word) since the last
            # time it was accessed
            detokenizer.last_segment

            # Contains all the tokens added so far
            detokenizer.tokens

        # Make sure that we detokenize any remaining tokens
        detokenizer.finalize()

        # Now detokenizer.text should match tokenizer.decode(detokenizer.tokens)
    """

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


class NaiveStreamingDetokenizer(StreamingDetokenizer):
    """NaiveStreamingDetokenizer relies on the underlying tokenizer
    implementation and should work with every tokenizer.

    Its complexity is O(T^2) where T is the longest line since it will
    repeatedly detokenize the same tokens until a new line is generated.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        # Handle tokenizers that don't have clean_up_tokenization_spaces
        self._clean_up_spaces = getattr(tokenizer, "clean_up_tokenization_spaces", True)
        self._tokenizer.decode([0])
        self.reset()

    def reset(self):
        self.offset = 0
        self.tokens = []
        self._text = ""
        self._current_tokens = []
        self._current_text = ""

    def add_token(self, token):
        self._current_tokens.append(token)
        self.tokens.append(token)

    def finalize(self):
        self._text += self._tokenizer.decode(self._current_tokens)
        self._current_tokens = []
        self._current_text = ""

    @property
    def text(self):
        if self._current_tokens:
            self._current_text = self._tokenizer.decode(self._current_tokens)
            if self._current_text.endswith("\ufffd") or (
                self._clean_up_spaces
                and len(self._current_text) > 0
                and self._current_text[-1] == " "
            ):
                self._current_text = self._current_text[:-1]
        if self._current_text and self._current_text[-1] == "\n":
            self._text += self._current_text
            self._current_tokens.clear()
            self._current_text = ""
        return self._text + self._current_text


class SPMStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for SPM models.

    It adds tokens to the text if the next token starts with the special SPM
    underscore which results in linear complexity.
    """

    def __init__(self, tokenizer, trim_space=True):
        self.trim_space = trim_space
        self._sep = "\u2581".encode()

        # Extract the tokens in a list from id to text
        self.tokenmap = [""] * (max(tokenizer.vocab.values()) + 1)
        for value, tokenid in tokenizer.vocab.items():
            if value.startswith("<0x"):
                # Replace bytes with their value
                self.tokenmap[tokenid] = bytes([int(value[3:5], 16)])
            else:
                self.tokenmap[tokenid] = value.encode()

        self.reset()

    def reset(self):
        self.offset = 0
        self._unflushed = b""
        self.text = ""
        self.tokens = []

    def _try_flush(self, force=False):
        text = self._unflushed.replace(self._sep, b" ").decode("utf-8", "replace")
        if not force and text.endswith("\ufffd"):
            return
        if not self.text and self.trim_space and text and text[0] == " ":
            text = text[1:]
        self.text += text
        self._unflushed = b""

    def add_token(self, token):
        self.tokens.append(token)
        v = self.tokenmap[token]
        self._unflushed += v
        self._try_flush()

    def finalize(self):
        self._try_flush(force=True)
        self._unflushed = b""


class BPEStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for OpenAI style BPE models.

    It adds tokens to the text if the next token starts with a space similar to
    the SPM detokenizer.
    """

    _byte_decoder = None
    _space_matches = (".", "?", "!", ",", "n't", "'m", "'s", "'ve", "'re")

    def __init__(self, tokenizer):
        self.clean_spaces = tokenizer.clean_up_tokenization_spaces

        # Extract the tokens in a list from id to text
        self.tokenmap = [None] * len(tokenizer.vocab)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        self.reset()

        # Make the BPE byte decoder from
        # https://github.com/openai/gpt-2/blob/master/src/encoder.py
        self.make_byte_decoder()

    def reset(self):
        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens = []

    def _decode_bytes(self, seq):
        barr = bytearray()
        for c in seq:
            res = self._byte_decoder.get(c, False)
            if res:
                barr.append(res)
            else:
                barr.extend(bytes(c, "utf-8"))
        return barr.decode("utf-8", "replace")

    def _maybe_trim_space(self, current_text):
        if len(current_text) == 0:
            return current_text
        elif current_text[0] != " ":
            return current_text
        elif not self.text:
            return current_text[1:]
        elif self.clean_spaces and current_text[1:].startswith(self._space_matches):
            return current_text[1:]
        return current_text

    def add_token(self, token):
        self.tokens.append(token)
        v = self.tokenmap[token]
        self._unflushed += v
        text = self._decode_bytes(self._unflushed)

        # For multi-byte utf-8 wait until they are complete
        # For single spaces wait until the next token to clean it if needed
        if not text.endswith("\ufffd") and not (
            len(v) == 1 and self._byte_decoder[v[0]] == 32
        ):
            self.text += self._maybe_trim_space(text)
            self._unflushed = ""

    def finalize(self):
        current_text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode(
            "utf-8",
            "replace",
        )
        self.text += self._maybe_trim_space(current_text)
        self._unflushed = ""

    @classmethod
    def make_byte_decoder(cls):
        """See https://github.com/openai/gpt-2/blob/master/src/encoder.py for the rationale."""
        if cls._byte_decoder is not None:
            return

        char_to_bytes = {}
        limits = [
            0,
            ord("!"),
            ord("~") + 1,
            ord("¡"),
            ord("¬") + 1,
            ord("®"),
            ord("ÿ") + 1,
        ]
        n = 0
        for i, (start, stop) in enumerate(zip(limits, limits[1:])):
            if i % 2 == 0:
                for b in range(start, stop):
                    char_to_bytes[chr(2**8 + n)] = b
                    n += 1
            else:
                for b in range(start, stop):
                    char_to_bytes[chr(b)] = b
        cls._byte_decoder = char_to_bytes


class MistralStreamingDetokenizer(StreamingDetokenizer):
    """Efficient streaming detokenizer for MistralTokenizer with byte/unicode edge handling."""

    def __init__(self, tokenizer):
        # Extract the underlying Tekkenizer from MistralTokenizer
        # Use the same helper logic as TokenizerWrapper
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


class TokenizerWrapper:
    """A wrapper that combines an HF tokenizer and a detokenizer.

    Accessing any attribute other than the ``detokenizer`` is forwarded to the
    huggingface tokenizer.
    """

    def _is_mistral_tokenizer(self, tokenizer) -> bool:
        """Check if tokenizer is a MistralTokenizer."""
        return hasattr(tokenizer, "instruct_tokenizer") and hasattr(
            tokenizer.instruct_tokenizer, "tokenizer"
        )

    def _get_underlying_tokenizer(self, tokenizer):
        """Get the underlying Tekkenizer from MistralTokenizer if applicable."""
        if self._is_mistral_tokenizer(tokenizer):
            return tokenizer.instruct_tokenizer.tokenizer
        return tokenizer

    def _get_vocab(self, tokenizer):
        """Get vocabulary from tokenizer, handling different tokenizer APIs."""
        vocab = {}
        if hasattr(tokenizer, "get_vocab"):
            vocab = tokenizer.get_vocab()
        elif self._is_mistral_tokenizer(tokenizer):
            # For MistralTokenizer, get vocab from underlying tokenizer
            underlying_tokenizer = self._get_underlying_tokenizer(tokenizer)
            if hasattr(underlying_tokenizer, "vocab") and callable(
                underlying_tokenizer.vocab
            ):
                vocab_list = underlying_tokenizer.vocab()
                vocab = {token: idx for idx, token in enumerate(vocab_list)}  # type: ignore
        elif hasattr(tokenizer, "vocab"):
            # For standard tokenizers, vocab might be a dict
            if isinstance(tokenizer.vocab, dict):
                vocab = tokenizer.vocab
            elif callable(tokenizer.vocab):
                vocab_list = tokenizer.vocab()
                vocab = {token: idx for idx, token in enumerate(vocab_list)}  # type: ignore
            elif hasattr(tokenizer.vocab, "__iter__") and not isinstance(
                tokenizer.vocab, dict
            ):
                # Convert list of TokenInfo to dict
                vocab = {token.piece: idx for idx, token in enumerate(tokenizer.vocab)}
        return vocab

    def _has_mistral_chat_completion(self, tokenizer):
        """Check if tokenizer supports Mistral chat completion API."""
        return (
            hasattr(tokenizer, "encode_chat_completion")
            and ChatCompletionRequest is not None
            and UserMessage is not None
            and AssistantMessage is not None
            and SystemMessage is not None
        )

    def __init__(
        self, tokenizer, detokenizer_class=NaiveStreamingDetokenizer, eos_token_ids=None
    ):
        self._tokenizer = tokenizer
        self._detokenizer = detokenizer_class(tokenizer)

        # Handle different tokenizer APIs for eos_token_id
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None and self._is_mistral_tokenizer(tokenizer):
            # For MistralTokenizer, get from underlying tokenizer
            underlying_tokenizer = self._get_underlying_tokenizer(tokenizer)
            eos_token_id = getattr(underlying_tokenizer, "eos_id", None)

        self._eos_token_ids = (
            set(eos_token_ids)
            if eos_token_ids is not None
            else {eos_token_id}
            if eos_token_id is not None
            else set()
        )
        self._think_start = None
        self._think_end = None
        self._tool_call_start = None
        self._tool_call_end = None

        THINK_TOKENS = [("<think>", "</think>")]
        TOOL_CALL_TOKENS = [
            ("<tool_call>", "</tool_call>"),  # ChatML style
        ]
        MISTRAL_TOOL_CALL_START = (
            "[TOOL_CALLS]"  # MistralTokenizer style - no end token
        )

        # Handle different vocab APIs
        vocab = self._get_vocab(tokenizer)

        for think_start, think_end in THINK_TOKENS:
            if think_start in vocab and think_end in vocab:
                self._think_start = think_start
                self._think_end = think_end
                break

        # Check for tool calling support
        # For MistralTokenizer, we can detect tool calling by the presence of tool tokens
        # For other tokenizers, we also check the chat template
        has_chat_template_with_tools = (
            hasattr(tokenizer, "chat_template")
            and tokenizer.chat_template
            and '"tool"' in tokenizer.chat_template
        )

        # For MistralTokenizer, tool calling is supported if tool tokens exist
        is_mistral_tokenizer = self._is_mistral_tokenizer(tokenizer)

        if has_chat_template_with_tools or is_mistral_tokenizer:
            self._tool_call_start = ""
            self._tool_call_end = ""

            # Check for MistralTokenizer style first
            if is_mistral_tokenizer and MISTRAL_TOOL_CALL_START in vocab:
                self._tool_call_start = MISTRAL_TOOL_CALL_START
                self._tool_call_end = ""  # No end token for MistralTokenizer
            else:
                # Check for ChatML style tokens
                for tool_call_start, tool_call_end in TOOL_CALL_TOKENS:
                    if tool_call_start in vocab and tool_call_end in vocab:
                        self._tool_call_start = tool_call_start
                        self._tool_call_end = tool_call_end
                        break

    def add_eos_token(self, token: str):
        token_id = None
        try:
            token_id = int(token)
        except ValueError:
            token_id = self._tokenizer.convert_tokens_to_ids(token)

        if token_id is None:
            raise ValueError(f"'{token}' is not a token for this tokenizer")

        self._eos_token_ids.add(token_id)

    def encode(self, text, add_special_tokens=True, **kwargs):
        """Custom encode method that works with both HF and Mistral tokenizers."""
        # If it's a MistralTokenizer, use the underlying tokenizer
        if self._is_mistral_tokenizer(self._tokenizer):
            # For MistralTokenizer, use underlying Tekkenizer with bos/eos parameters
            underlying_tokenizer = self._get_underlying_tokenizer(self._tokenizer)
            return underlying_tokenizer.encode(
                text,
                bos=add_special_tokens,
                eos=False,  # Usually we don't want EOS during encoding
                **kwargs,
            )
        else:
            # For HuggingFace tokenizers, use the standard method
            return self._tokenizer.encode(
                text, add_special_tokens=add_special_tokens, **kwargs
            )

    def _convert_to_mistral_messages(self, messages):
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
                        from mistral_common.protocol.instruct.tool_calls import (
                            ToolCall,
                            FunctionCall,
                        )

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
                    except ImportError:
                        mistral_messages.append(AssistantMessage(content=content))
                else:
                    mistral_messages.append(AssistantMessage(content=content))

            elif role == "tool":
                try:
                    from mistral_common.protocol.instruct.messages import ToolMessage

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
                except ImportError:
                    pass

        return mistral_messages

    def _convert_to_mistral_tools(self, tools):
        """Convert OpenAI-format tools to Mistral-common format."""
        if not tools:
            return None

        from mistral_common.protocol.instruct.tool_calls import Function, Tool

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

    def _apply_mistral_chat_template(
        self, messages, add_generation_prompt=True, tools=None
    ):
        """Apply chat template using Mistral tokenizer."""
        if not MISTRAL_AVAILABLE or ChatCompletionRequest is None:
            raise ValueError("Mistral libraries not available")

        try:
            # Convert to Mistral-common format
            mistral_messages = self._convert_to_mistral_messages(messages)
            mistral_tools = self._convert_to_mistral_tools(tools)

            # Create ChatCompletionRequest
            request = ChatCompletionRequest(
                messages=mistral_messages, tools=mistral_tools
            )

            # Encode with MistralTokenizer
            result = self._tokenizer.encode_chat_completion(request)

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
            # Fallback to text concatenation if Mistral encoding fails
            return self._fallback_to_text_concatenation(messages, add_generation_prompt)

    def _preprocess_messages_for_hf(self, messages):
        """Preprocess messages for HuggingFace tokenizers that can't handle None content."""
        processed_messages = []
        for msg in messages:
            if msg["role"] == "assistant" and "tool_calls" in msg:
                # For assistant messages with tool calls, either use content or create a placeholder
                content = msg.get("content")
                if content is None:
                    # Some HF tokenizers need content, create a meaningful placeholder
                    if "tool_calls" in msg and msg["tool_calls"]:
                        tool_call = msg["tool_calls"][0]  # Take first tool call
                        if tool_call.get("type") == "function":
                            func_name = tool_call["function"]["name"]
                            content = f"I'll call the {func_name} function for you."
                    else:
                        content = ""

                processed_msg = {"role": "assistant", "content": content}
                # Some HF tokenizers support tool_calls, try to include them
                try:
                    processed_msg["tool_calls"] = msg["tool_calls"]
                except Exception:
                    pass
                processed_messages.append(processed_msg)
            elif msg["role"] == "tool":
                # Convert tool messages to a format HF tokenizers might understand
                processed_messages.append(msg)
            else:
                processed_messages.append(msg)
        return processed_messages

    def _apply_hf_chat_template(
        self, messages, add_generation_prompt=True, tools=None, **kwargs
    ):
        """Apply chat template using HuggingFace tokenizer."""
        hf_kwargs = kwargs.copy()
        if tools is not None:
            hf_kwargs["tools"] = tools

        # Preprocess messages for HF tokenizers
        processed_messages = self._preprocess_messages_for_hf(messages)

        try:
            return self._tokenizer.apply_chat_template(
                processed_messages,
                add_generation_prompt=add_generation_prompt,
                **hf_kwargs,
            )
        except (ValueError, TypeError) as e:
            if "add_generation_prompt" in str(e):
                # Fallback: remove the unsupported parameter
                return self._tokenizer.apply_chat_template(
                    processed_messages, **hf_kwargs
                )
            elif "tool" in str(e).lower() or "none" in str(e).lower():
                # Fallback: HF tokenizer doesn't support this format
                return self._fallback_to_text_concatenation(
                    messages, add_generation_prompt
                )
            raise

    def _fallback_to_text_concatenation(self, messages, add_generation_prompt=True):
        """Fallback method that concatenates message content as simple text."""
        text_parts = []
        for msg in messages:
            content = msg.get("content")
            if content:  # Only add non-empty content
                text_parts.append(content)
        text = " ".join(text_parts)
        return self.encode(text, add_special_tokens=add_generation_prompt)

    def apply_chat_template(
        self, messages, add_generation_prompt=True, tools=None, **kwargs
    ):
        """Apply chat template with automatic tokenizer detection and appropriate handling.

        This method automatically detects the tokenizer type and applies the most appropriate
        chat template formatting:
        - Mistral tokenizers: Uses native Mistral-common format with proper tool call support
        - HuggingFace tokenizers: Uses HF chat templates with preprocessing for tool calls
        - Fallback: Simple text concatenation for unsupported formats
        """
        # Route to appropriate implementation based on tokenizer type
        if self._has_mistral_chat_completion(self._tokenizer):
            return self._apply_mistral_chat_template(
                messages, add_generation_prompt, tools
            )

        elif hasattr(self._tokenizer, "apply_chat_template"):
            return self._apply_hf_chat_template(
                messages, add_generation_prompt, tools, **kwargs
            )

        else:
            # Final fallback for tokenizers without chat template support
            fallback_kwargs = kwargs.copy()
            if tools is not None:
                fallback_kwargs["tools"] = tools

            try:
                return self._tokenizer.apply_chat_template(messages, **fallback_kwargs)
            except (ValueError, TypeError) as e:
                if (
                    "none" in str(e).lower()
                    or "iterable" in str(e).lower()
                    or "tool" in str(e).lower()
                ):
                    return self._fallback_to_text_concatenation(
                        messages, add_generation_prompt
                    )
                raise

    @property
    def has_thinking(self):
        return self._think_start is not None

    @property
    def think_start(self):
        return self._think_start

    @property
    def think_end(self):
        return self._think_end

    @property
    def has_tool_calling(self):
        return self._tool_call_start is not None

    @property
    def tool_call_start(self):
        return self._tool_call_start

    @property
    def tool_call_end(self):
        return self._tool_call_end

    @property
    def bos_token(self):
        """Get BOS token, handling both HF and Mistral tokenizers."""
        if hasattr(self._tokenizer, "bos_token"):
            return self._tokenizer.bos_token
        elif self._is_mistral_tokenizer(self._tokenizer):
            # For MistralTokenizer, get from underlying tokenizer
            underlying_tokenizer = self._get_underlying_tokenizer(self._tokenizer)
            if hasattr(underlying_tokenizer, "bos_id"):
                try:
                    return underlying_tokenizer.decode([underlying_tokenizer.bos_id])
                except Exception:
                    return None
        return None

    @property
    def eos_token(self):
        """Get EOS token, handling both HF and Mistral tokenizers."""
        if hasattr(self._tokenizer, "eos_token"):
            return self._tokenizer.eos_token
        elif self._is_mistral_tokenizer(self._tokenizer):
            # For MistralTokenizer, get from underlying tokenizer
            underlying_tokenizer = self._get_underlying_tokenizer(self._tokenizer)
            if hasattr(underlying_tokenizer, "eos_id"):
                try:
                    return underlying_tokenizer.decode([underlying_tokenizer.eos_id])
                except Exception:
                    return None
        return None

    def save_pretrained(self, save_directory, **kwargs):
        """Save the tokenizer, handling both HF and Mistral tokenizers."""
        from pathlib import Path

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # If it's a MistralTokenizer, save the tekken.json file
        if self._is_mistral_tokenizer(self._tokenizer):
            # For MistralTokenizer, check if the underlying tokenizer has a file_path
            underlying_tokenizer = self._get_underlying_tokenizer(self._tokenizer)
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
        else:
            # For HuggingFace tokenizers, use the standard save_pretrained method
            return self._tokenizer.save_pretrained(save_directory, **kwargs)

    def __getattr__(self, attr):
        if attr == "detokenizer":
            return self._detokenizer
        elif attr == "eos_token_ids":
            return self._eos_token_ids
        elif attr.startswith("_"):
            return self.__getattribute__(attr)
        else:
            return getattr(self._tokenizer, attr)

    def __setattr__(self, attr, value):
        if attr in {"detokenizer", "eos_token_ids"}:
            if attr == "detokenizer":
                raise AttributeError("Cannot set the detokenizer.")
            elif attr == "eos_token_ids":
                self._eos_token_ids = set(value) if value is not None else set()
        elif attr.startswith("_"):
            super().__setattr__(attr, value)
        else:
            setattr(self._tokenizer, attr, value)


class NewlineTokenizer(PreTrainedTokenizerFast):
    """A tokenizer that replaces newlines with <n> and <n> with new line."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _preprocess_text(self, text):
        return text.replace("\n", "<n>")

    def _postprocess_text(self, text):
        return text.replace("<n>", "\n")

    def encode(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        padding=False,
        truncation=None,
        max_length=None,
        stride=0,
        return_tensors=None,
        **kwargs,
    ):  # type: ignore
        return super().encode(
            self._preprocess_text(text),
            text_pair,
            add_special_tokens,
            padding,
            truncation,
            max_length,
            stride,
            return_tensors,
            **kwargs,
        )

    def encode_batch(self, texts, add_special_tokens=True, **kwargs):
        return super().encode_batch(
            [self._preprocess_text(t) for t in texts],
            add_special_tokens=add_special_tokens,
            **kwargs,
        )  # type: ignore

    def decode(self, *args, **kwargs):
        return self._postprocess_text(super().decode(*args, **kwargs))

    def batch_decode(self, *args, **kwargs):
        decoded = super().batch_decode(*args, **kwargs)
        return [self._postprocess_text(d) for d in decoded]


AutoTokenizer.register("NewlineTokenizer", fast_tokenizer_class=NewlineTokenizer)


def _match(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and _match(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_match(ai, bi) for ai, bi in zip(a, b))

    return a == b


def _is_spm_decoder(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }
    return _match(_target_description, decoder)


def _is_spm_decoder_no_space(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    return _match(_target_description, decoder)


def _is_bpe_decoder(decoder):
    return isinstance(decoder, dict) and decoder.get("type", None) == "ByteLevel"


def load_tokenizer(
    model_path, tokenizer_config_extra={}, return_tokenizer=True, eos_token_ids=None
):
    """Load a huggingface or mistral tokenizer and try to infer the type of streaming
    detokenizer to use.

    Note, to use a fast streaming tokenizer, pass a local file path rather than
    a Hugging Face repo ID.
    """
    detokenizer_class = NaiveStreamingDetokenizer

    tekken_file = model_path / "tekken.json"
    if tekken_file.exists() and MistralTokenizer is not None:
        tokenizer = MistralTokenizer.from_file(str(tekken_file))
        detokenizer_class = MistralStreamingDetokenizer
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        if return_tokenizer:
            return TokenizerWrapper(
                tokenizer,
                detokenizer_class,  # type: ignore
                eos_token_ids=eos_token_ids,
            )
        else:
            return detokenizer_class

    tokenizer_file = model_path / "tokenizer.json"
    if tokenizer_file.exists():
        with open(tokenizer_file, "r", encoding="utf-8") as fid:
            try:
                tokenizer_content = json.load(fid)
            except JSONDecodeError as e:
                raise JSONDecodeError("Failed to parse tokenizer.json", e.doc, e.pos)

        if "decoder" in tokenizer_content:
            if _is_spm_decoder(tokenizer_content["decoder"]):
                detokenizer_class = SPMStreamingDetokenizer
            elif _is_spm_decoder_no_space(tokenizer_content["decoder"]):
                detokenizer_class = partial(SPMStreamingDetokenizer, trim_space=False)
            elif _is_bpe_decoder(tokenizer_content["decoder"]):
                detokenizer_class = BPEStreamingDetokenizer

    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    if return_tokenizer:
        return TokenizerWrapper(
            AutoTokenizer.from_pretrained(model_path, **tokenizer_config_extra),
            detokenizer_class,  # type: ignore
            eos_token_ids=eos_token_ids,
        )
    else:
        return detokenizer_class


def no_bos_or_eos(sequence: List, bos: int, eos: int) -> List:
    removed_bos = sequence if sequence[0] != bos else sequence[1:]
    return removed_bos[:-1] if removed_bos[-1] == eos else removed_bos
