import importlib
import json
import warnings
from functools import partial
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast


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
                self._tokenizer.clean_up_tokenization_spaces
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
        v = self.tokenmap[token] if token < len(self.tokenmap) else "!"
        self._unflushed += v
        text = self._decode_bytes(self._unflushed)

        # For multi-byte utf-8 wait until they are complete
        # For single spaces wait until the next token to clean it if needed
        if not text.endswith("\ufffd") and not (
            len(v) == 1 and self._byte_decoder.get(v[0]) == 32
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


class TokenizerWrapper:
    """A wrapper that combines an HF tokenizer and a detokenizer.

    Accessing any attribute other than the ``detokenizer`` is forwarded to the
    huggingface tokenizer.
    """

    def __init__(
        self,
        tokenizer,
        detokenizer_class=NaiveStreamingDetokenizer,
        eos_token_ids=None,
        chat_template=None,
        tool_call_start=None,
        tool_call_end=None,
        tool_parser=None,
    ):
        self._tokenizer = tokenizer
        self._detokenizer_class = detokenizer_class
        self._eos_token_ids = (
            set(eos_token_ids)
            if eos_token_ids is not None
            else {tokenizer.eos_token_id}
        )
        self._think_start = None
        self._think_end = None
        self._think_start_id = None
        self._think_end_id = None

        self._chat_template = chat_template
        self.has_chat_template = (
            tokenizer.chat_template is not None or chat_template is not None
        )
        self._tool_parser = tool_parser
        self._tool_call_start = tool_call_start
        self._tool_call_end = tool_call_end

        vocab = tokenizer.get_vocab()
        THINK_TOKENS = [
            ("<think>", "</think>"),
            ("<longcat_think>", "</longcat_think>"),
        ]
        for think_start, think_end in THINK_TOKENS:
            if think_start in vocab and think_end in vocab:
                self._think_start = think_start
                self._think_end = think_end
                self._think_start_id = vocab[think_start]
                self._think_end_id = vocab[think_end]
                break

        # Disable tool calling if tool call tokens aren't in vocab
        if (tool_call_start and tool_call_start not in vocab) or (
            tool_call_end and tool_call_end not in vocab
        ):
            self._tool_call_start = None
            self._tool_call_end = None
            self._tool_parser = None

    def apply_chat_template(self, *args, tokenize=True, **kwargs):
        if self._chat_template is not None:
            out = self._chat_template(*args, **kwargs)
            if tokenize:
                out = self._tokenizer.encode(out, add_special_tokens=False)
            return out

        kwargs["return_dict"] = False
        return self._tokenizer.apply_chat_template(*args, tokenize=tokenize, **kwargs)

    def add_eos_token(self, token: str):
        token_id = None
        try:
            token_id = int(token)
        except ValueError:
            token_id = self._tokenizer.convert_tokens_to_ids(token)

        if token_id is None:
            raise ValueError(f"'{token}' is not a token for this tokenizer")

        self._eos_token_ids.add(token_id)

    @property
    def has_thinking(self):
        return self._think_start is not None

    @property
    def think_start(self):
        return self._think_start

    @property
    def think_start_id(self):
        return self._think_start_id

    @property
    def think_end(self):
        return self._think_end

    @property
    def think_end_id(self):
        return self._think_end_id

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
    def tool_parser(self):
        return self._tool_parser

    @property
    def detokenizer(self):
        """
        Get a stateful streaming detokenizer.
        """
        return self._detokenizer_class(self)

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

    def encode(self, text, **kwargs):
        return super().encode(self._preprocess_text(text), **kwargs)

    def encode_batch(self, texts, **kwargs):
        return super().encode_batch([self._preprocess_text(t) for t in texts], **kwargs)

    def decode(self, *args, **kwargs):
        return self._postprocess_text(super().decode(*args, **kwargs))

    def batch_decode(self, *args, **kwargs):
        decoded = super().batch_decode(*args, **kwargs)
        return [self._postprocess_text(d) for d in decoded]


AutoTokenizer.register("NewlineTokenizer", fast_tokenizer_class=NewlineTokenizer)


def _match(a, b):
    if type(a) != type(b):
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


def _infer_tool_parser(chat_template):
    """Attempt to auto-infer a tool parser from the chat template."""
    if not isinstance(chat_template, str):
        return None
    elif "<minimax:tool_call>" in chat_template:
        return "minimax_m2"
    elif "<start_function_call>" in chat_template:
        return "function_gemma"
    elif "<longcat_tool_call>" in chat_template:
        return "longcat"
    elif "<arg_key>" in chat_template:
        return "glm47"
    elif "<|tool_list_start|>" in chat_template:
        return "pythonic"
    elif (
        "<tool_call>\\n<function=" in chat_template
        or "<tool_call>\n<function=" in chat_template
    ):
        return "qwen3_coder"
    elif "<|tool_calls_section_begin|>" in chat_template:
        return "kimi_k2"
    elif "<tool_call>" in chat_template and "tool_call.name" in chat_template:
        return "json_tools"
    return None


def load(
    model_path,
    tokenizer_config_extra: Optional[Dict[str, Any]] = None,
    eos_token_ids=None,
) -> TokenizerWrapper:
    """Load a huggingface tokenizer and try to infer the type of streaming
    detokenizer to use.

    Note, to use a fast streaming tokenizer, pass a local file path rather than
    a Hugging Face repo ID.
    """
    detokenizer_class = NaiveStreamingDetokenizer

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

    tokenizer_config_file = model_path / "tokenizer_config.json"
    chat_template = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, **(tokenizer_config_extra or {})
    )

    tokenizer_config = tokenizer.init_kwargs

    if chat_template_type := tokenizer_config.get("chat_template_type", False):
        chat_template = importlib.import_module(
            f"mlx_lm.chat_templates.{chat_template_type}"
        ).apply_chat_template

    tool_parser_type = tokenizer_config.get(
        "tool_parser_type", _infer_tool_parser(tokenizer.chat_template)
    )

    if tool_parser_type is not None:
        tool_module = importlib.import_module(f"mlx_lm.tool_parsers.{tool_parser_type}")
        tool_parser = tool_module.parse_tool_call
        tool_call_start = tool_module.tool_call_start
        tool_call_end = tool_module.tool_call_end
        tokenizer_config["tool_parser_type"] = tool_parser_type
    else:
        tool_parser = None
        tool_call_start = None
        tool_call_end = None

    return TokenizerWrapper(
        tokenizer,
        detokenizer_class,
        eos_token_ids=eos_token_ids,
        chat_template=chat_template,
        tool_parser=tool_parser,
        tool_call_start=tool_call_start,
        tool_call_end=tool_call_end,
    )


def no_bos_or_eos(sequence: List, bos: int, eos: int) -> List:
    removed_bos = sequence if sequence[0] != bos else sequence[1:]
    return removed_bos[:-1] if removed_bos[-1] == eos else removed_bos
