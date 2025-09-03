# Copyright © 2023-2024 Apple Inc.

import argparse
import json
import logging
import platform
import socket
import time
import uuid
import warnings
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import mlx.core as mx
from huggingface_hub import scan_cache_dir

from ._version import __version__
from .generate import stream_generate
from .models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from .sample_utils import make_logits_processors, make_sampler
from .utils import common_prefix_len, load


def get_system_fingerprint():
    gpu_arch = mx.metal.device_info()["architecture"] if mx.metal.is_available() else ""
    return f"{__version__}-{mx.__version__}-{platform.platform()}-{gpu_arch}"


class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int


def stopping_criteria(
    tokens: List[int],
    stop_id_sequences: List[List[int]],
    eos_token_id: Union[int, None],
) -> StopCondition:
    """
    Determines whether the token generation should stop based on predefined
    conditions.

    Args:
        tokens (List[int]): The current sequence of generated tokens.
        stop_id_sequences (List[List[[int]]): A list of integer lists, each
          representing a sequence of token IDs. If the end of the `tokens`
          list matches any of these sequences, the generation should stop.
        eos_token_id (Union[int, None]): The token ID that represents the
          end-of-sequence. If the last token in `tokens` matches this, the
          generation should stop.

    Returns:
        StopCondition: A named tuple indicating whether the stop condition has
          been met (`stop_met`) and how many tokens should be trimmed from the
          end if it has (`trim_length`).
    """
    if tokens and tokens[-1] == eos_token_id:
        return StopCondition(stop_met=True, trim_length=0)

    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids):
            if tokens[-len(stop_ids) :] == stop_ids:
                return StopCondition(stop_met=True, trim_length=len(stop_ids))

    return StopCondition(stop_met=False, trim_length=0)


def sequence_overlap(s1: Sequence, s2: Sequence) -> bool:
    """
    Checks if a suffix of s1 has overlap with a prefix of s2

    Args:
        s1 (Sequence): The first sequence
        s2 (Sequence): The second sequence

    Returns:
        bool: If the two sequences have overlap
    """
    max_overlap = min(len(s1), len(s2))
    return any(s1[-i:] == s2[:i] for i in range(1, max_overlap + 1))


def convert_chat(messages: List[dict], role_mapping: Optional[dict] = None):
    default_role_mapping = {
        "system_prompt": (
            "A chat between a curious user and an artificial intelligence "
            "assistant. The assistant follows the given rules no matter what."
        ),
        "system": "ASSISTANT's RULE: ",
        "user": "USER: ",
        "assistant": "ASSISTANT: ",
        "stop": "\n",
    }
    role_mapping = role_mapping if role_mapping is not None else default_role_mapping

    prompt = ""
    for line in messages:
        role_prefix = role_mapping.get(line["role"], "")
        stop = role_mapping.get("stop", "")
        content = line.get("content", "")
        prompt += f"{role_prefix}{content}{stop}"

    prompt += role_mapping.get("assistant", "")
    return prompt.rstrip()


def process_message_content(messages):
    """
    Convert message content to a format suitable for `apply_chat_template`.

    The function operates on messages in place. It converts the 'content' field
    to a string instead of a list of text fragments.

    Args:
        message_list (list): A list of dictionaries, where each dictionary may
          have a 'content' key containing a list of dictionaries with 'type' and
          'text' keys.

    Raises:
        ValueError: If the 'content' type is not supported or if 'text' is missing.

    """
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            text_fragments = [
                fragment["text"] for fragment in content if fragment["type"] == "text"
            ]
            if len(text_fragments) != len(content):
                raise ValueError("Only 'text' content type is supported.")
            message["content"] = "".join(text_fragments)
        elif content is None:
            message["content"] = ""


@dataclass
class PromptCache:
    cache: List[Any] = field(default_factory=list)
    model_key: Tuple[str, Optional[str]] = ("", None, None)
    tokens: List[int] = field(default_factory=list)


class ModelProvider:
    def __init__(self, cli_args: argparse.Namespace):
        """Load models on demand and persist them across the whole process."""
        self.cli_args = cli_args
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None

        # Preload the default model if it is provided
        self.default_model_map = {}
        if self.cli_args.model is not None:
            self.default_model_map[self.cli_args.model] = "default_model"
            self.load(self.cli_args.model, draft_model_path="default_model")

    def _validate_model_path(self, model_path: str):
        model_path = Path(model_path)
        if model_path.exists() and not model_path.is_relative_to(Path.cwd()):
            raise RuntimeError(
                "Local models must be relative to the current working dir."
            )

    # Added in adapter_path to load dynamically
    def load(self, model_path, adapter_path=None, draft_model_path=None):
        model_path, adapter_path, draft_model_path = map(
            lambda s: s.lower() if s else None,
            (model_path, adapter_path, draft_model_path),
        )

        model_path = self.default_model_map.get(model_path, model_path)
        if self.model_key == (model_path, adapter_path, draft_model_path):
            return self.model, self.tokenizer

        # Remove the old model if it exists.
        self.model = None
        self.tokenizer = None
        self.model_key = None
        self.draft_model = None

        # Building tokenizer_config
        tokenizer_config = {
            "trust_remote_code": True if self.cli_args.trust_remote_code else None
        }
        if self.cli_args.chat_template:
            tokenizer_config["chat_template"] = self.cli_args.chat_template

        if model_path == "default_model":
            if self.cli_args.model is None:
                raise ValueError(
                    "A model path has to be given as a CLI "
                    "argument or in the HTTP request"
                )
            adapter_path = adapter_path or self.cli_args.adapter_path
            model, tokenizer = load(
                self.cli_args.model,
                adapter_path=adapter_path,
                tokenizer_config=tokenizer_config,
            )
        else:
            self._validate_model_path(model_path)
            model, tokenizer = load(
                model_path, adapter_path=adapter_path, tokenizer_config=tokenizer_config
            )

        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template

        self.model_key = (model_path, adapter_path, draft_model_path)
        self.model = model
        self.tokenizer = tokenizer

        def validate_draft_tokenizer(draft_tokenizer):
            # Check if tokenizers are compatible
            if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                logging.warning(
                    "Draft model tokenizer does not match model tokenizer. "
                    "Speculative decoding may not work as expected."
                )

        # Load draft model if specified
        if (
            draft_model_path == "default_model"
            and self.cli_args.draft_model is not None
        ):
            self.draft_model, draft_tokenizer = load(self.cli_args.draft_model)
            validate_draft_tokenizer(draft_tokenizer)

        elif draft_model_path is not None and draft_model_path != "default_model":
            self._validate_model_path(draft_model_path)
            self.draft_model, draft_tokenizer = load(draft_model_path)
            validate_draft_tokenizer(draft_tokenizer)
        return self.model, self.tokenizer


class APIHandler(BaseHTTPRequestHandler):
    def __init__(
        self,
        model_provider: ModelProvider,
        *args,
        prompt_cache: Optional[PromptCache] = None,
        system_fingerprint: Optional[str] = None,
        **kwargs,
    ):
        """
        Create static request specific metadata
        """
        self.created = int(time.time())
        self.model_provider = model_provider
        self.prompt_cache = prompt_cache or PromptCache()
        self.system_fingerprint = system_fingerprint or get_system_fingerprint()
        super().__init__(*args, **kwargs)

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _set_completion_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self._set_cors_headers()

    def _set_stream_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._set_cors_headers()

    def do_OPTIONS(self):
        self._set_completion_headers(204)
        self.end_headers()

    def do_POST(self):
        """
        Respond to a POST request from a client.
        """
        endpoints = {
            "/v1/completions": self.handle_text_completions,
            "/v1/chat/completions": self.handle_chat_completions,
            "/chat/completions": self.handle_chat_completions,
        }

        if self.path not in endpoints:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Fetch and parse request body
        content_length = int(self.headers["Content-Length"])
        raw_body = self.rfile.read(content_length)
        try:
            self.body = json.loads(raw_body.decode())
        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError: {e} - Raw body: {raw_body.decode()}")
            # Set appropriate headers based on streaming requirement
            if self.stream:
                self._set_stream_headers(400)
                self.wfile.write(
                    f"data: {json.dumps({'error': f'Invalid JSON in request body: {e}'})}\n\n".encode()
                )
            else:
                self._set_completion_headers(400)
                self.wfile.write(
                    json.dumps({"error": f"Invalid JSON in request body: {e}"}).encode()
                )
            return

        indent = "\t"  # Backslashes can't be inside of f-strings
        logging.debug(f"Incoming Request Body: {json.dumps(self.body, indent=indent)}")
        assert isinstance(
            self.body, dict
        ), f"Request should be dict, but got {type(self.body)}"

        # Extract request parameters from the body
        self.stream = self.body.get("stream", False)
        self.stream_options = self.body.get("stream_options", None)
        self.requested_model = self.body.get("model", "default_model")
        self.requested_draft_model = self.body.get("draft_model", "default_model")
        self.num_draft_tokens = self.body.get(
            "num_draft_tokens", self.model_provider.cli_args.num_draft_tokens
        )
        self.adapter = self.body.get("adapters", None)
        self.max_tokens = self.body.get("max_completion_tokens", None)
        if self.max_tokens is None:
            self.max_tokens = self.body.get(
                "max_tokens", self.model_provider.cli_args.max_tokens
            )
        self.temperature = self.body.get(
            "temperature", self.model_provider.cli_args.temp
        )
        self.top_p = self.body.get("top_p", self.model_provider.cli_args.top_p)
        self.top_k = self.body.get("top_k", self.model_provider.cli_args.top_k)
        self.min_p = self.body.get("min_p", self.model_provider.cli_args.min_p)
        self.repetition_penalty = self.body.get("repetition_penalty", 1.0)
        self.repetition_context_size = self.body.get("repetition_context_size", 20)
        self.xtc_probability = self.body.get("xtc_probability", 0.0)
        self.xtc_threshold = self.body.get("xtc_threshold", 0.0)
        self.logit_bias = self.body.get("logit_bias", None)
        self.logprobs = self.body.get("logprobs", -1)
        self.seed = self.body.get("seed", None)
        self.validate_model_parameters()
        if self.seed is not None:
            mx.random.seed(self.seed)
        # Load the model if needed
        try:
            self.model, self.tokenizer = self.model_provider.load(
                self.requested_model,
                self.adapter,
                self.requested_draft_model,
            )
        except:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Get stop id sequences, if provided
        stop_words = self.body.get("stop")
        stop_words = stop_words or []
        stop_words = [stop_words] if isinstance(stop_words, str) else stop_words
        stop_id_sequences = [
            self.tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]

        # Send header type
        (
            self._set_stream_headers(200)
            if self.stream
            else self._set_completion_headers(200)
        )

        # Call endpoint specific method
        prompt = endpoints[self.path]()
        self.handle_completion(prompt, stop_id_sequences)

    def validate_model_parameters(self):
        """
        Validate the model parameters passed in the request for the correct types and values.
        """
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")

        if not isinstance(self.max_tokens, int) or self.max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")

        if not isinstance(self.temperature, (float, int)) or self.temperature < 0:
            raise ValueError("temperature must be a non-negative float")

        if not isinstance(self.top_p, (float, int)) or self.top_p < 0 or self.top_p > 1:
            raise ValueError("top_p must be a float between 0 and 1")

        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")

        if not isinstance(self.min_p, (float, int)) or self.min_p < 0 or self.min_p > 1:
            raise ValueError("min_p must be a float between 0 and 1")

        if not isinstance(self.num_draft_tokens, int) or self.num_draft_tokens < 0:
            raise ValueError("num_draft_tokens must be a non-negative integer")

        if (
            not isinstance(self.repetition_penalty, (float, int))
            or self.repetition_penalty < 0
        ):
            raise ValueError("repetition_penalty must be a non-negative float")

        if self.logprobs != -1 and not (0 < self.logprobs <= 10):
            raise ValueError(
                f"logprobs must be between 1 and 10 but got {self.logprobs:,}"
            )

        if (
            not isinstance(self.repetition_context_size, int)
            or self.repetition_context_size < 0
        ):
            raise ValueError("repetition_context_size must be a non-negative integer")

        if self.logit_bias is not None:
            if not isinstance(self.logit_bias, dict):
                raise ValueError("logit_bias must be a dict of int to float")

            try:
                self.logit_bias = {int(k): v for k, v in self.logit_bias.items()}
            except ValueError:
                raise ValueError("logit_bias must be a dict of int to float")
        if not (
            isinstance(self.xtc_probability, float)
            and 0.00 <= self.xtc_probability <= 1.00
        ):
            raise ValueError(f"xtc_probability must be a float between 0.00 and 1.00")
        if not (
            isinstance(self.xtc_threshold, float) and 0.00 <= self.xtc_threshold <= 0.50
        ):
            raise ValueError(f"xtc_threshold must be a float between 0.00 and 0.5")
        if not isinstance(self.requested_model, str):
            raise ValueError("model must be a string")
        if self.adapter is not None and not isinstance(self.adapter, str):
            raise ValueError("adapter must be a string")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

    def generate_response(
        self,
        text: str,
        finish_reason: Union[Literal["length", "stop"], None],
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        token_logprobs: Optional[List[float]] = None,
        top_tokens: Optional[List[Dict[int, float]]] = None,
        tokens: Optional[List[int]] = None,
        tool_calls: Optional[List[str]] = None,
    ) -> dict:
        """
        Generate a single response packet based on response type (stream or
        not), completion type and parameters.

        Args:
            text (str): Text generated by model
            finish_reason (Union[Literal["length", "stop"], None]): The reason the
              response is being sent: "length", "stop" or `None`.
            prompt_token_count (Optional[int]): The number of tokens in the prompt,
              used to populate the "usage" field (not used when stream).
            completion_token_count (Optional[int]): The number of tokens in the
              response, used to populate the "usage" field (not used when stream).
            token_logprobs (Optional[List[float]]): The log probabilities per token,
              in token order.
            top_tokens (Optional[List[Dict[int, float]]]): List of dictionaries mapping
              tokens to logprobs for the top N tokens at each token position.
            tokens (Optional[List[int]]): List of tokens to return with logprobs structure
            tool_calls (Optional[List[str]]): List of tool calls.

        Returns:
            dict: A dictionary containing the response, in the same format as
              OpenAI's API.
        """
        token_logprobs = token_logprobs or []
        top_logprobs = top_tokens or []
        tool_calls = tool_calls or []

        def parse_function(tool_text):
            tool_call = json.loads(tool_text.strip())
            return {
                "function": {
                    "name": tool_call.get("name", None),
                    "arguments": json.dumps(tool_call.get("arguments", "")),
                },
                "type": "function",
                "id": None,
            }

        # Static response
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": self.object_type,
            "model": self.requested_model,
            "created": self.created,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                },
            ],
        }

        if token_logprobs or top_logprobs or tokens:
            response["choices"][0]["logprobs"] = {
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
                "tokens": tokens,
            }

        if not self.stream:
            if not (
                isinstance(prompt_token_count, int)
                and isinstance(completion_token_count, int)
            ):
                raise ValueError(
                    "Response type is complete, but token counts not provided"
                )

            response["usage"] = {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            }

        choice = response["choices"][0]

        # Add dynamic response
        if self.object_type.startswith("chat.completion"):
            key_name = "delta" if self.stream else "message"
            choice[key_name] = {
                "role": "assistant",
                "content": text,
                "tool_calls": [parse_function(tool_text) for tool_text in tool_calls],
            }
        elif self.object_type == "text_completion":
            choice.update(text=text)
        else:
            raise ValueError(f"Unsupported response type: {self.object_type}")

        return response

    def reset_prompt_cache(self, prompt):
        """Resets the prompt cache and associated state.

        Args:
            prompt (List[int]): The tokenized new prompt which will populate the
                reset cache.
        """
        logging.debug(f"*** Resetting cache. ***")
        self.prompt_cache.model_key = self.model_provider.model_key
        self.prompt_cache.cache = make_prompt_cache(self.model_provider.model)
        if self.model_provider.draft_model is not None:
            self.prompt_cache.cache += make_prompt_cache(
                self.model_provider.draft_model
            )
        self.prompt_cache.tokens = list(prompt)  # Cache the new prompt fully

    def get_prompt_cache(self, prompt):
        """
        Determines the portion of the prompt that needs processing by comparing
        it to the cached prompt and attempting to reuse the common prefix.

        This function updates the internal prompt cache state (tokens and model cache)
        based on the comparison. If a common prefix exists, it attempts to trim
        the model cache (if supported) to match the common prefix length, avoiding
        recomputation.

        Args:
            prompt (List[int]): The tokenized new prompt.

        Returns:
            List[int]: The suffix of the prompt that actually needs to be processed
                       by the model. This will be the full prompt if the cache is
                       reset or cannot be effectively used.
        """
        cache_len = len(self.prompt_cache.tokens)
        prompt_len = len(prompt)
        com_prefix_len = common_prefix_len(self.prompt_cache.tokens, prompt)

        # Leave at least one token in the prompt
        com_prefix_len = min(com_prefix_len, len(prompt) - 1)

        # Condition 1: Model changed or no common prefix at all. Reset cache.
        if (
            self.prompt_cache.model_key != self.model_provider.model_key
            or com_prefix_len == 0
        ):
            self.reset_prompt_cache(prompt)

        # Condition 2: Common prefix exists and matches cache length. Process suffix.
        elif com_prefix_len == cache_len:
            logging.debug(
                f"*** Cache is prefix of prompt (cache_len: {cache_len}, prompt_len: {prompt_len}). Processing suffix. ***"
            )
            prompt = prompt[com_prefix_len:]
            self.prompt_cache.tokens.extend(prompt)

        # Condition 3: Common prefix exists but is shorter than cache length. Attempt trim.
        elif com_prefix_len < cache_len:
            logging.debug(
                f"*** Common prefix ({com_prefix_len}) shorter than cache ({cache_len}). Attempting trim. ***"
            )

            if can_trim_prompt_cache(self.prompt_cache.cache):
                num_to_trim = cache_len - com_prefix_len
                logging.debug(f"    Trimming {num_to_trim} tokens from cache.")
                trim_prompt_cache(self.prompt_cache.cache, num_to_trim)
                self.prompt_cache.tokens = self.prompt_cache.tokens[:com_prefix_len]
                prompt = prompt[com_prefix_len:]
                self.prompt_cache.tokens.extend(prompt)
            else:
                logging.debug(f"    Cache cannot be trimmed. Resetting cache.")
                self.reset_prompt_cache(prompt)

        # This case should logically not be reached if com_prefix_len <= cache_len
        else:
            logging.error(
                f"Unexpected cache state: com_prefix_len ({com_prefix_len}) > cache_len ({cache_len}). Resetting cache."
            )
            self.reset_prompt_cache(prompt)

        logging.debug(f"Returning {len(prompt)} tokens for processing.")
        return prompt

    def handle_completion(
        self,
        prompt: List[int],
        stop_id_sequences: List[List[int]],
    ):
        """
        Generate a response to a prompt and send it to the client in a single batch.

        Args:
            prompt (List[int]): The tokenized prompt.
            stop_id_sequences (List[List[int]]): A list of stop words passed
              to the stopping_criteria function
        """
        tokens = []
        finish_reason = "length"
        stop_sequence_suffix = None
        if self.stream:
            self.end_headers()
            logging.debug(f"Starting stream:")
        else:
            logging.debug(f"Starting completion:")
        token_logprobs = []
        top_tokens = []

        prompt = self.get_prompt_cache(prompt)

        text = ""
        tic = time.perf_counter()
        sampler = make_sampler(
            self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            xtc_probability=self.xtc_probability,
            xtc_threshold=self.xtc_threshold,
            xtc_special_tokens=[
                self.tokenizer.eos_token_id,
                self.tokenizer.encode("\n"),
            ],
        )
        logits_processors = make_logits_processors(
            self.logit_bias,
            self.repetition_penalty,
            self.repetition_context_size,
        )

        tool_calls = []
        tool_text = ""
        in_tool_call = False
        segment = ""

        # Create keepalive callback to send SSE comments during long prompt processing
        def keepalive_callback(processed_tokens, total_tokens):
            logging.info(
                f"Prompt processing progress: {processed_tokens}/{total_tokens}"
            )
            if self.stream:
                try:
                    # Send SSE comment for keepalive - invisible to clients but keeps connection alive
                    self.wfile.write(
                        f": keepalive {processed_tokens}/{total_tokens}\n\n".encode()
                    )
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    # Client disconnected, ignore
                    pass

        for gen_response in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=self.prompt_cache.cache,
            draft_model=self.model_provider.draft_model,
            num_draft_tokens=self.num_draft_tokens,
            prompt_progress_callback=keepalive_callback,
        ):
            logging.debug(gen_response.text)

            if (
                self.tokenizer.has_tool_calling
                and gen_response.text == self.tokenizer.tool_call_start
            ):
                in_tool_call = True
            elif in_tool_call:
                if gen_response.text == self.tokenizer.tool_call_end:
                    tool_calls.append(tool_text)
                    tool_text = ""
                    in_tool_call = False
                else:
                    tool_text += gen_response.text
            else:
                text += gen_response.text
                segment += gen_response.text
            token = gen_response.token
            logprobs = gen_response.logprobs
            tokens.append(token)
            self.prompt_cache.tokens.append(token)

            if self.logprobs > 0:
                sorted_indices = mx.argpartition(-logprobs, kth=self.logprobs - 1)
                top_indices = sorted_indices[: self.logprobs]
                top_logprobs = logprobs[top_indices]
                top_token_info = zip(top_indices.tolist(), top_logprobs.tolist())
                top_tokens.append(tuple(top_token_info))

            token_logprobs.append(logprobs[token].item())

            stop_condition = stopping_criteria(
                tokens, stop_id_sequences, self.tokenizer.eos_token_id
            )
            if stop_condition.stop_met:
                finish_reason = "stop"
                if stop_condition.trim_length:
                    stop_sequence_suffix = self.tokenizer.decode(
                        tokens[-stop_condition.trim_length :]
                    )
                    text = text[: -len(stop_sequence_suffix)]
                segment = ""
                break

            if self.stream and not in_tool_call:
                # If the end of tokens overlaps with a stop sequence, generate new
                # tokens until we know if the stop sequence is hit or not
                if any(
                    (
                        sequence_overlap(tokens, sequence)
                        for sequence in stop_id_sequences
                    )
                ):
                    continue
                elif segment or tool_calls:
                    response = self.generate_response(
                        segment, None, tool_calls=tool_calls
                    )
                    self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                    self.wfile.flush()
                    segment = ""
                    tool_calls = []

        if gen_response.finish_reason is not None:
            finish_reason = gen_response.finish_reason

        logging.debug(f"Prompt: {gen_response.prompt_tps:.3f} tokens-per-sec")
        logging.debug(f"Generation: {gen_response.generation_tps:.3f} tokens-per-sec")
        logging.debug(f"Peak memory: {gen_response.peak_memory:.3f} GB")

        if self.stream:
            response = self.generate_response(
                segment, finish_reason, tool_calls=tool_calls
            )
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()
            if self.stream_options is not None and self.stream_options["include_usage"]:
                original_prompt_length = (
                    len(self.prompt_cache.tokens) - len(tokens) + len(prompt)
                )
                response = self.completion_usage_response(
                    original_prompt_length, len(tokens)
                )
                self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write("data: [DONE]\n\n".encode())
            self.wfile.flush()
        else:
            response = self.generate_response(
                text,
                finish_reason,
                len(prompt),
                len(tokens),
                token_logprobs=token_logprobs,
                top_tokens=top_tokens,
                tokens=tokens,
                tool_calls=tool_calls,
            )
            response_json = json.dumps(response).encode()
            indent = "\t"  # Backslashes can't be inside of f-strings
            logging.debug(f"Outgoing Response: {json.dumps(response, indent=indent)}")

            # Send an additional Content-Length header when it is known
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json)
            self.wfile.flush()

    def completion_usage_response(
        self,
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
    ):
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": "chat.completion",
            "model": self.requested_model,
            "created": self.created,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            },
        }
        return response

    def handle_chat_completions(self) -> List[int]:
        """
        Handle a chat completion request.

        Returns:
            mx.array: A mx.array of the tokenized prompt from the request body
        """
        body = self.body
        assert "messages" in body, "Request did not contain messages"

        # Determine response type
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completion.chunk" if self.stream else "chat.completion"
        if self.tokenizer.chat_template:
            messages = body["messages"]
            process_message_content(messages)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                body.get("tools") or None,
                add_generation_prompt=True,
                **self.model_provider.cli_args.chat_template_args,
            )
        else:
            prompt = convert_chat(body["messages"], body.get("role_mapping"))
            prompt = self.tokenizer.encode(prompt)

        return prompt

    def handle_text_completions(self) -> List[int]:
        """
        Handle a text completion request.

        Returns:
            mx.array: A mx.array of the tokenized prompt from the request body
        """
        # Determine response type
        self.request_id = f"cmpl-{uuid.uuid4()}"
        self.object_type = "text_completion"
        assert "prompt" in self.body, "Request did not contain a prompt"
        return self.tokenizer.encode(self.body["prompt"])

    def do_GET(self):
        """
        Respond to a GET request from a client.
        """
        if self.path.startswith("/v1/models"):
            self.handle_models_request()
        elif self.path == "/health":
            self.handle_health_check()
        else:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def handle_health_check(self):
        """
        Handle a GET request for the /health endpoint.
        """
        self._set_completion_headers(200)
        self.end_headers()

        self.wfile.write('{"status": "ok"}'.encode())
        self.wfile.flush()

    def handle_models_request(self):
        """
        Handle a GET request for the /v1/models endpoint.
        """
        self._set_completion_headers(200)
        self.end_headers()

        files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]

        parts = self.path.split("/")
        filter_repo_id = None
        if len(parts) > 3:
            filter_repo_id = "/".join(parts[3:])

        def probably_mlx_lm(repo):
            if repo.repo_type != "model":
                return False
            if "main" not in repo.refs:
                return False
            if filter_repo_id is not None and repo.repo_id != filter_repo_id:
                return False
            file_names = {f.file_path.name for f in repo.refs["main"].files}
            return all(f in file_names for f in files)

        # Scan the cache directory for downloaded mlx models
        hf_cache_info = scan_cache_dir()
        downloaded_models = [
            repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)
        ]

        # Create a list of available models
        models = [
            {
                "id": repo.repo_id,
                "object": "model",
                "created": self.created,
            }
            for repo in downloaded_models
        ]

        response = {"object": "list", "data": models}

        response_json = json.dumps(response).encode()
        self.wfile.write(response_json)
        self.wfile.flush()


def run(
    host: str,
    port: int,
    model_provider: ModelProvider,
    server_class=HTTPServer,
    handler_class=APIHandler,
):
    server_address = (host, port)
    prompt_cache = PromptCache()
    infos = socket.getaddrinfo(
        *server_address, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )
    server_class.address_family, _, _, _, server_address = next(iter(infos))
    httpd = server_class(
        server_address,
        lambda *args, **kwargs: handler_class(
            model_provider,
            prompt_cache=prompt_cache,
            system_fingerprint=get_system_fingerprint(),
            *args,
            **kwargs,
        ),
    )
    warnings.warn(
        "mlx_lm.server is not recommended for production as "
        "it only implements basic security checks."
    )
    logging.info(f"Starting httpd at {host} on port {port}...")
    httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="MLX Http Server.")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="A model to be used for speculative decoding.",
        default=None,
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
        default=3,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Specify a chat template for the tokenizer",
        required=False,
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Default sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Default nucleus sampling top-p (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Default top-k sampling (default: 0, disables top-k)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Default min-p sampling (default: 0.0, disables min-p)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        help="""A JSON formatted string of arguments for the tokenizer's apply_chat_template, e.g. '{"enable_thinking":false}'""",
        default="{}",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run(args.host, args.port, ModelProvider(args))


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.server...` directly is deprecated."
        " Use `mlx_lm.server...` or `python -m mlx_lm server ...` instead."
    )
    main()
