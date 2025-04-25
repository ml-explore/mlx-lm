# mlx-lm/server.py (Modified for Qwen2.5-VL Support)
# Copyright © 2023-2024 Apple Inc.
# mlx-vlm: https://github.com/Blaizzy/mlx-vlm

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

# MLX-LM original imports
from ._version import __version__
# from .models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache # Commented out: Caching disabled for VLM
from .sample_utils import make_logits_processors, make_sampler
# from .utils import load # Replaced with mlx_vlm version

# VLM Integration Imports
from mlx_vlm.utils import (
    load as vlm_load,
    load_config as vlm_load_config,
    stream_generate as vlm_stream_generate,
    GenerationResult as VLMGenerationResult, # Assuming this class exists in mlx_vlm.utils
    get_model_path # Can use this helper
)
from mlx_vlm.prompt_utils import apply_chat_template

# HF Hub import for model listing
from huggingface_hub import scan_cache_dir


def get_system_fingerprint():
    # Assuming mlx_lm versioning is still relevant, maybe add vlm version?
    # from mlx_vlm.version import __version__ as vlm_version # If mlx_vlm has version file
    vlm_version = "unknown" # Placeholder
    gpu_arch = mx.metal.device_info()["architecture"] if mx.metal.is_available() else ""
    return f"lm-{__version__}-vlm-{vlm_version}-{mx.__version__}-{platform.platform()}-{gpu_arch}"


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
    conditions. NOTE: Primarily handled by processor's stopping_criteria now.
    This might serve as a fallback or secondary check if needed.
    """
    # Check for processor's EOS token ID first (most common case)
    if tokens and eos_token_id is not None and tokens[-1] == eos_token_id:
        return StopCondition(stop_met=True, trim_length=0)

    # Check for custom stop sequences
    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids):
            if tokens[-len(stop_ids) :] == stop_ids:
                return StopCondition(stop_met=True, trim_length=len(stop_ids))

    return StopCondition(stop_met=False, trim_length=0)


def sequence_overlap(s1: Sequence, s2: Sequence) -> bool:
    """
    Checks if a suffix of s1 has overlap with a prefix of s2
    """
    max_overlap = min(len(s1), len(s2))
    return any(s1[-i:] == s2[:i] for i in range(1, max_overlap + 1))


# convert_chat is likely unused for VLM template processing
# def convert_chat(messages: List[dict], role_mapping: Optional[dict] = None): ...

def process_message_content(messages):
    """
    Processes message content, extracting text and image URLs.
    Operates on messages list in place for text concatenation.
    Returns a tuple: (concatenated_text, list_of_image_sources).
    Only processes the *last* message assuming it's the user prompt with images.
    """
    if not messages:
        return "", []

    last_message = messages[-1]
    content = last_message.get("content")
    extracted_text = ""
    image_sources = []

    if isinstance(content, list):
        # Handle multimodal input list
        current_message_text = []
        for item in content:
            item_type = item.get("type")
            if item_type == "text":
                current_message_text.append(item.get("text", ""))
            elif item_type == "image_url":
                image_url_data = item.get("image_url", {})
                url = image_url_data.get("url")
                if url:
                    image_sources.append(url)
            # Ignore other types for now
        extracted_text = "".join(current_message_text)
        # Update the last message content for potential logging/debugging
        # (apply_chat_template will use the extracted text and image count)
        last_message["content"] = extracted_text # Simplification for processing later
    elif isinstance(content, str):
        # Handle text-only input
        extracted_text = content
    else:
        raise ValueError("Invalid message content format")

    return extracted_text, image_sources


# PromptCache is disabled for VLM Qwen2.5-VL path
# @dataclass
# class PromptCache:
#     cache: List[Any] = field(default_factory=list)
#     model_key: Tuple[str, Optional[str]] = ("", None, None)
#     tokens: List[int] = field(default_factory=list)


class ModelProvider:
    def __init__(self, cli_args: argparse.Namespace):
        """Load models on demand and persist them across the whole process."""
        self.cli_args = cli_args
        self.model_key = None
        self.model = None
        self.processor = None # Changed from tokenizer
        # self.draft_model = None # Draft model disabled for VLM

        # Preload the default model if it is provided
        if self.cli_args.model is not None:
            self.load("default_model") # Removed draft_model_path argument

    def _validate_model_path(self, model_path: str):
        # This check might need adjustment if models are *only* from HF Hub
        model_path_obj = Path(model_path)
        if model_path_obj.exists() and not model_path_obj.is_relative_to(Path.cwd()):
            # Allowing absolute paths if they exist locally
             if not model_path_obj.exists():
                 raise RuntimeError(
                    "Local models must be relative to the current working dir or an existing absolute path."
                 )
        # If it doesn't exist locally, assume it's an HF repo ID
        elif not model_path_obj.exists():
            pass # Assume HF repo

    # Removed adapter_path from signature for now, simplify load logic
    def load(self, model_path, adapter_path=None): # Removed draft_model_path
        # Re-evaluate model key - only model path and adapter path matter now
        current_key = (model_path, adapter_path)
        if self.model_key == current_key:
            return self.model, self.processor

        # Remove the old model if it exists.
        self.model = None
        self.processor = None
        self.model_key = None
        # self.draft_model = None

        # Determine the actual path (local or HF)
        if model_path == "default_model":
            if self.cli_args.model is None:
                raise ValueError(
                    "A model path has to be given as a CLI "
                    "argument or in the HTTP request"
                )
            actual_model_path_or_repo = self.cli_args.model
            # Use adapter from CLI only if model is default and request didn't specify one
            adapter_path = adapter_path or self.cli_args.adapter_path
        else:
            # Validate only if it looks like a local path attempt
            if Path(model_path).exists() or "/" in model_path or "\\" in model_path:
                 self._validate_model_path(model_path)
            actual_model_path_or_repo = model_path

        # Check config BEFORE loading to ensure it's Qwen2.5-VL
        try:
            resolved_path = get_model_path(actual_model_path_or_repo)
            config = vlm_load_config(resolved_path, trust_remote_code=self.cli_args.trust_remote_code)
            if config.get("model_type") != "qwen2_5_vl":
                 raise ValueError(f"Server currently only supports 'qwen2_5_vl' model type. Attempted to load: {config.get('model_type')}")
        except FileNotFoundError:
             # Assume HF repo, load will fail later if invalid
             logging.warning(f"Could not find local config for {actual_model_path_or_repo}. Assuming HF repo and proceeding with load attempt.")
             pass
        except Exception as e:
            logging.error(f"Error loading config for validation: {e}")
            raise e

        # Load model and processor using mlx_vlm.utils.load
        try:
            logging.info(f"Loading VLM model and processor from: {actual_model_path_or_repo}")
            # Pass trust_remote_code if required by the underlying AutoProcessor/AutoTokenizer
            load_kwargs = {}
            if self.cli_args.trust_remote_code:
                load_kwargs['trust_remote_code'] = True

            model, processor = vlm_load(
                actual_model_path_or_repo,
                adapter_path=adapter_path,
                 **load_kwargs
            )
            logging.info(f"Successfully loaded model: {actual_model_path_or_repo}")

        except Exception as e:
             logging.error(f"Failed to load VLM model or processor: {e}")
             # Log traceback for detailed debugging
             import traceback
             logging.error(traceback.format_exc())
             raise RuntimeError(f"Failed to load model from {actual_model_path_or_repo}") from e


        self.model_key = current_key
        self.model = model
        self.processor = processor

        # Draft model logic removed

        return self.model, self.processor


class APIHandler(BaseHTTPRequestHandler):
    def __init__(
        self,
        model_provider: ModelProvider,
        *args,
        # prompt_cache: Optional[PromptCache] = None, # Caching disabled
        system_fingerprint: Optional[str] = None,
        **kwargs,
    ):
        """
        Create static request specific metadata
        """
        self.created = int(time.time())
        self.model_provider = model_provider
        # self.prompt_cache = prompt_cache or PromptCache() # Caching disabled
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
        self.end_headers() # End headers here for non-streaming

    def _set_stream_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._set_cors_headers()
        self.end_headers() # End headers here for streaming start

    def do_OPTIONS(self):
        self._set_completion_headers(204)
        # No self.end_headers() here as _set_completion_headers does it

    def do_POST(self):
        """
        Respond to a POST request from a client. Handles multimodal chat completions.
        """
        endpoints = {
            # "/v1/completions": self.handle_text_completions, # Text completions might not make sense for VLM
            "/v1/chat/completions": self.handle_chat_completions,
            # "/chat/completions": self.handle_chat_completions, # Keep standard endpoint
        }

        if self.path not in endpoints:
            self._set_completion_headers(404)
            # self.end_headers() # Done by _set_completion_headers
            self.wfile.write(b"Not Found")
            return

        # Fetch and parse request body
        try:
            content_length = int(self.headers["Content-Length"])
            raw_body = self.rfile.read(content_length)
            self.body = json.loads(raw_body.decode())
            indent = "\t" # Backslashes can't be inside of f-strings
            logging.debug(f"Incoming Request Body: {json.dumps(self.body, indent=indent)}")
            assert isinstance(self.body, dict), f"Request should be dict, but got {type(self.body)}"

            # Extract messages early for processing
            messages = self.body.get("messages", [])
            if not messages:
                raise ValueError("Request body must contain 'messages' list.")

        except (json.JSONDecodeError, KeyError, AssertionError, ValueError) as e:
            logging.error(f"Bad request processing: {e}")
            self._set_completion_headers(400)
            # self.end_headers() # Done by _set_completion_headers
            self.wfile.write(json.dumps({"error": f"Bad Request: {e}"}).encode())
            return

        try:
            # Extract request parameters from the body
            self.stream = self.body.get("stream", False)
            self.stream_options = self.body.get("stream_options", None) # Keep for potential future use
            self.requested_model = self.body.get("model", "default_model")
            # self.requested_draft_model = self.body.get("draft_model", "default_model") # Draft disabled
            # self.num_draft_tokens = self.body.get("num_draft_tokens", 3) # Draft disabled
            self.adapter = self.body.get("adapters", None) # Keep adapter support
            self.max_tokens = self.body.get("max_completion_tokens", None)
            if self.max_tokens is None:
                 self.max_tokens = self.body.get("max_tokens", 512) # Default VLM might need more
            self.temperature = self.body.get("temperature", 0.0) # VLM often uses low temp
            self.top_p = self.body.get("top_p", 1.0)
            self.repetition_penalty = self.body.get("repetition_penalty", None) # Default to None if not supported well
            self.repetition_context_size = self.body.get("repetition_context_size", 20)
            # self.xtc_probability = self.body.get("xtc_probability", 0.0) # XTC likely not in VLM generate
            # self.xtc_threshold = self.body.get("xtc_threshold", 0.0) # XTC likely not in VLM generate
            self.logit_bias = self.body.get("logit_bias", None)
            self.logprobs = self.body.get("logprobs", -1) # Logprobs might not be supported by vlm generate_step

            self.validate_model_parameters()

            # Load the model (will raise ValueError if not Qwen2.5-VL)
            self.model, self.processor = self.model_provider.load(
                self.requested_model,
                self.adapter,
                # self.requested_draft_model, # Draft disabled
            )
            # Ensure processor is loaded
            if self.processor is None:
                 raise RuntimeError("Failed to load VLM processor.")


            # --- Multimodal Input Processing ---
            user_prompt_text, image_sources = process_message_content(messages)
            if not user_prompt_text and not image_sources:
                raise ValueError("Last message must contain text or image_url content.")
            # --- End Multimodal Input Processing ---


            # Get stop id sequences, if provided
            stop_words = self.body.get("stop", [])
            stop_words = [stop_words] if isinstance(stop_words, str) else stop_words

            # Configure StoppingCriteria on the processor
            try:
                # Reset to default EOS before adding custom ones
                tokenizer_obj = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
                default_eos = self.model.config.eos_token_id # Get default from loaded model config
                tokenizer_obj.stopping_criteria.reset(default_eos)
                if stop_words:
                    tokenizer_obj.stopping_criteria.add_eos_token_ids(stop_words)
            except Exception as e:
                 logging.warning(f"Could not configure custom stop words: {e}. Using default EOS.")
                 # Proceed with default EOS configured during load


            # Prepare headers (call *before* generating content)
            (
                self._set_stream_headers(200)
                if self.stream
                else self._set_completion_headers(200)
            )

            # Call endpoint specific method to get formatted prompt string
            formatted_prompt = endpoints[self.path](user_prompt_text, image_sources)

            # Call the VLM generation handler
            self.handle_completion(formatted_prompt, image_sources) # Pass images sources

        except ValueError as e:
             logging.error(f"Request validation error: {e}")
             # Potentially overwrite headers if they were already sent for streaming
             if not self.stream : self._set_completion_headers(400)
             # For stream, can't change header, just stop sending
             self.wfile.write(json.dumps({"error": f"Bad Request: {e}"}).encode())
             # Ensure headers are ended if non-streaming failed late
             # if not self.stream and not self.headers_sent: self.end_headers() # Headers already ended
        except RuntimeError as e:
            logging.error(f"Model loading or generation error: {e}")
            # Potentially overwrite headers if they were already sent for streaming
            if not self.stream : self._set_completion_headers(500)
             # For stream, can't change header, just stop sending
            self.wfile.write(json.dumps({"error": f"Internal Server Error: {e}"}).encode())
             # Ensure headers are ended if non-streaming failed late
             # if not self.stream and not self.headers_sent: self.end_headers() # Headers already ended
        except Exception as e:
            logging.error(f"Unexpected error in POST: {e}")
            import traceback
            logging.error(traceback.format_exc())
            if not self.headers_sent: # Check if headers sent before trying to set
                 self._set_completion_headers(500)
            self.wfile.write(json.dumps({"error": f"Unexpected Internal Server Error"}).encode())
            # Ensure headers are ended if non-streaming failed late
            # if not self.stream and not self.headers_sent: self.end_headers() # Headers already ended


    def validate_model_parameters(self):
        """
        Validate the model parameters passed in the request for the correct types and values.
        (Removed draft model validation)
        """
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")

        if not isinstance(self.max_tokens, int) or self.max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")

        if not isinstance(self.temperature, (float, int)) or self.temperature < 0:
            raise ValueError("temperature must be a non-negative float")

        if not isinstance(self.top_p, (float, int)) or not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be a float between 0 and 1")

        if self.repetition_penalty is not None and (
            not isinstance(self.repetition_penalty, (float, int))
            or self.repetition_penalty < 0
        ):
            raise ValueError("repetition_penalty must be a non-negative float")

        # Logprobs support might be limited in VLM, maybe warn or disable?
        if self.logprobs != -1:
             logging.warning("Logprobs might not be fully supported for VLM models.")
             # For now, allow it but it might return empty
             if not (0 < self.logprobs <= 10):
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
                self.logit_bias = {int(k): float(v) for k, v in self.logit_bias.items()}
            except (ValueError, TypeError):
                raise ValueError("logit_bias must be a dict of int to float")

        # XTC params removed
        # if not (
        #     isinstance(self.xtc_probability, float)
        #     and 0.00 <= self.xtc_probability <= 1.00
        # ): ...
        # if not (
        #     isinstance(self.xtc_threshold, float) and 0.00 <= self.xtc_threshold <= 0.50
        # ): ...

        if not isinstance(self.requested_model, str):
            raise ValueError("model must be a string")
        if self.adapter is not None and not isinstance(self.adapter, str):
            raise ValueError("adapter must be a string")


    def generate_response(
        self,
        text: str,
        finish_reason: Union[Literal["length", "stop"], None],
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        token_logprobs: Optional[List[float]] = None, # May be None from VLM
        top_tokens: Optional[List[Dict[int, float]]] = None, # May be None from VLM
        tokens: Optional[List[int]] = None, # May be None from VLM
    ) -> dict:
        """
        Generate a single response packet based on response type (stream or
        not), completion type and parameters.
        Adapted to handle potentially missing logprobs data from VLM.
        """
        # Provide defaults for potentially missing logprobs data
        token_logprobs = token_logprobs if token_logprobs is not None else []
        top_logprobs = top_tokens if top_tokens is not None else []
        tokens = tokens if tokens is not None else [] # VLM stream might not yield raw tokens easily

        # Basic response structure
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": self.object_type,
            "model": self.requested_model,
            "created": self.created,
            "choices": [
                {
                    "index": 0,
                    "logprobs": None, # Default to None unless logprobs were requested AND returned
                    "finish_reason": finish_reason,
                }
            ],
        }

        # Add logprobs structure ONLY if requested and available
        if self.logprobs > 0 and (token_logprobs or top_logprobs or tokens):
             response["choices"][0]["logprobs"] = {
                 "token_logprobs": token_logprobs if token_logprobs else None,
                 "top_logprobs": top_logprobs if top_logprobs else None,
                 "tokens": tokens if tokens else None,
             }


        # Add usage for non-streaming responses
        if not self.stream:
            if not (
                isinstance(prompt_token_count, int)
                and isinstance(completion_token_count, int)
            ):
                 # Log a warning instead of raising error, as prompt token count might be harder to get accurately for VLM
                 logging.warning(
                    "Token counts not fully provided for non-streaming response. Usage data might be incomplete."
                 )
                 prompt_token_count = prompt_token_count if isinstance(prompt_token_count, int) else -1 # Use -1 to indicate unknown
                 completion_token_count = completion_token_count if isinstance(completion_token_count, int) else -1

            response["usage"] = {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": (prompt_token_count + completion_token_count) if prompt_token_count !=-1 and completion_token_count !=-1 else -1 ,
            }

        choice = response["choices"][0]

        # Add message/delta content
        if self.object_type.startswith("chat.completion"):
            key_name = "delta" if self.stream else "message"
            choice[key_name] = {"role": "assistant", "content": text}
        elif self.object_type == "text_completion": # Keep for potential future non-chat VLM use
            choice.update(text=text)
        else:
            # Should not happen if path validation works
            raise ValueError(f"Unsupported response type: {self.object_type}")

        return response

    # get_prompt_cache method removed as caching is disabled

    def handle_completion(
        self,
        prompt_string: str, # Changed from prompt: List[int]
        image_sources: List[str], # Added image sources
    ):
        """
        Generate a response to a prompt string and optional images, sending results.
        Uses mlx_vlm.utils.stream_generate.
        Handles both streaming and non-streaming responses.
        """
        generation_args = {
            "model": self.model,
            "processor": self.processor,
            "prompt": prompt_string,
            "image": image_sources,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "repetition_context_size": self.repetition_context_size,
            "logit_bias": self.logit_bias,
            # 'logprobs': self.logprobs > 0 # Pass indicator if supported? Check vlm_stream_generate
        }
        # Filter out None values for optional args
        generation_args = {k: v for k, v in generation_args.items() if v is not None}


        # Cache is handled internally by vlm_stream_generate, no prompt cache logic here

        accumulated_text = ""
        last_response_obj = None
        all_tokens = [] # To collect tokens if possible
        all_token_logprobs = [] # To collect logprobs if possible

        try:
            logging.debug(f"Starting VLM generation:")
            for response_obj in vlm_stream_generate(**generation_args):
                last_response_obj = response_obj # Keep track of the last one for final stats
                segment = response_obj.text
                accumulated_text += segment

                # Store token info if logprobs are requested (and available)
                if self.logprobs > 0 and response_obj.token is not None and response_obj.logprobs is not None:
                     all_tokens.append(response_obj.token)
                     # Assuming logprobs is the full distribution, get the prob of the chosen token
                     # This might need adjustment based on VLMGenerationResult actual logprobs content
                     try:
                         token_logprob = response_obj.logprobs[response_obj.token].item()
                         all_token_logprobs.append(token_logprob)
                     except (IndexError, TypeError):
                         logging.warning(f"Could not extract logprob for token {response_obj.token}")
                         all_token_logprobs.append(None) # Placeholder if extraction fails

                if self.stream and segment:
                    # Send incremental update
                    response_payload = self.generate_response(
                        segment,
                        None, # Finish reason is None until the end
                        # Include partial logprobs if needed/possible here? Simpler to send all at end.
                    )
                    self.wfile.write(f"data: {json.dumps(response_payload)}\n\n".encode())
                    self.wfile.flush()

            # Ensure last_response_obj is not None if generation happened
            if last_response_obj is None:
                 # Handle case where nothing was generated (e.g., max_tokens=0 or immediate EOS)
                 logging.warning("No tokens were generated.")
                 # Create a minimal final response object if needed
                 last_response_obj = VLMGenerationResult(
                     text="", token=None, logprobs=None, prompt_tokens=-1, generation_tokens=0,
                     prompt_tps=0.0, generation_tps=0.0, peak_memory=mx.get_peak_memory() / 1e9, finish_reason="stop" # Assume stop if empty
                 )


            # --- Final response handling ---
            # Determine finish reason based on last token and max_tokens
            final_finish_reason = "stop" # Default reason

            if last_response_obj is not None: # Check if any generation actually happened
                last_token = last_response_obj.token
                num_generated = last_response_obj.generation_tokens
                tokenizer_obj = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor

                # Check if the last token triggered the stopping criteria used during generation
                if last_token is not None and tokenizer_obj.stopping_criteria(last_token):
                    final_finish_reason = "stop"
                # Check if the generation reached the token limit
                elif num_generated >= self.max_tokens:
                    final_finish_reason = "length"
                # Log if finished for some other unexpected reason (shouldn't normally occur)
                else:
                    logging.warning(f"Generation finished unexpectedly. Last token: {last_token}, Generated: {num_generated}/{self.max_tokens}. Defaulting finish_reason to 'stop'.")
                    final_finish_reason = "stop" # Default to stop if unclear
            else:
                # Handle case where nothing was generated (e.g., max_tokens=0 or immediate EOS in prompt)
                logging.warning("No tokens were generated. Setting finish_reason to 'stop'.")
                final_finish_reason = "stop"

            # Now use the determined final_finish_reason for response generation

            if self.stream:
                # Send the final (potentially empty) delta with the finish reason
                final_delta_payload = self.generate_response(
                    "", # Final delta has no new text
                    final_finish_reason,
                    # Include accumulated logprobs if needed
                     token_logprobs=all_token_logprobs if self.logprobs > 0 else None,
                     tokens=all_tokens if self.logprobs > 0 else None,
                     # top_tokens logic would need implementation if VLM provides top-k logprobs
                )
                self.wfile.write(f"data: {json.dumps(final_delta_payload)}\n\n".encode())
                self.wfile.flush()

                # Optionally send usage data if requested
                if self.stream_options and self.stream_options.get("include_usage"):
                    usage_payload = self.completion_usage_response(
                        last_response_obj.prompt_tokens,
                        last_response_obj.generation_tokens,
                    )
                    self.wfile.write(f"data: {json.dumps(usage_payload)}\n\n".encode())
                    self.wfile.flush()

                # Send the [DONE] marker
                self.wfile.write("data: [DONE]\n\n".encode())
                self.wfile.flush()
            else:
                # Send the single, complete response
                # Note: prompt_tokens from VLMGenerationResult might be approximate
                response_payload = self.generate_response(
                    accumulated_text,
                    final_finish_reason,
                    last_response_obj.prompt_tokens,
                    last_response_obj.generation_tokens,
                    token_logprobs=all_token_logprobs if self.logprobs > 0 else None,
                    tokens=all_tokens if self.logprobs > 0 else None,
                     # top_tokens logic would need implementation if VLM provides top-k logprobs
                )
                response_json = json.dumps(response_payload).encode()
                indent = "\t" # Backslashes can't be inside of f-strings
                logging.debug(f"Outgoing Response: {json.dumps(response_payload, indent=indent)}")

                # Send the response (headers already sent by _set_completion_headers)
                self.wfile.write(response_json)
                self.wfile.flush()

        except Exception as e:
             logging.error(f"Error during generation stream: {e}")
             import traceback
             logging.error(traceback.format_exc())
             # Try to send an error message if possible (might fail if stream headers sent)
             try:
                  if self.stream:
                      error_payload = {"error": {"message": "Error during generation", "type": "generation_error"}}
                      self.wfile.write(f"data: {json.dumps(error_payload)}\n\n".encode())
                      self.wfile.write("data: [DONE]\n\n".encode()) # Terminate stream
                      self.wfile.flush()
                  # else: # Non-streaming errors handled in do_POST
                      # pass
             except Exception as e_send:
                  logging.error(f"Failed to send error message to client: {e_send}")

    def completion_usage_response(
        self,
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
    ):
        # Helper to create the usage payload for streaming options
        prompt_tokens = prompt_token_count if isinstance(prompt_token_count, int) else -1
        completion_tokens = completion_token_count if isinstance(completion_token_count, int) else -1
        total_tokens = (prompt_tokens + completion_tokens) if prompt_tokens != -1 and completion_tokens != -1 else -1

        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": "chat.completion.chunk", # Usage is part of the chunk stream
            "model": self.requested_model,
            "created": self.created,
            "choices": [], # Usage chunk has no text delta
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        return response

    def handle_chat_completions(self, user_prompt_text: str, image_sources: List[str]) -> str:
        """
        Prepare the prompt string for VLM chat completion using apply_chat_template.

        Args:
            user_prompt_text (str): The text part of the user prompt.
            image_sources (List[str]): List of image URLs/paths provided.

        Returns:
            str: A formatted prompt string ready for the VLM.
        """
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completion.chunk" if self.stream else "chat.completion"

        num_images = len(image_sources)
        logging.debug(f"Applying chat template for text: '{user_prompt_text[:50]}...' and {num_images} images.")

        # Construct the messages list suitable for apply_chat_template if needed,
        # or just pass the text. Assuming passing text is enough based on vlm implementation.
        # Example if structure needed:
        # messages_for_template = [{"role": "user", "content": user_prompt_text}]
        # For multi-turn history, self.body["messages"] would need pre-processing.

        formatted_prompt = apply_chat_template(
            self.processor,
            self.model.config, # Pass the loaded model's config
            prompt=user_prompt_text, # Pass only the extracted text
            num_images=num_images,
            add_generation_prompt=True, # Important for generation
        )
        logging.debug(f"Formatted prompt string: {formatted_prompt[:100]}...")
        return formatted_prompt


    # handle_text_completions removed as focus is on chat
    # def handle_text_completions(self) -> str: ...

    def do_GET(self):
        """
        Respond to a GET request for listing models.
        """
        if self.path == "/v1/models":
            self.handle_models_request()
        else:
            self._set_completion_headers(404)
            # self.end_headers() # Done by _set_completion_headers
            self.wfile.write(b"Not Found")

    def handle_models_request(self):
        """
        Handle a GET request for the /v1/models endpoint.
        Lists locally available HF cache models compatible with Qwen2.5-VL (basic check).
        """
        self._set_completion_headers(200) # Also ends headers

        # Basic check for files often present in VLM repos
        # This is a heuristic and might not be perfectly accurate
        required_files = ["config.json", "model.safetensors", "tokenizer.json", "preprocessor_config.json"]

        models = []
        try:
            hf_cache_info = scan_cache_dir()
            for repo in hf_cache_info.repos:
                if repo.repo_type == "model":
                    repo_path = Path(repo.repo_path)
                    # Check if config exists and model_type is qwen2_5_vl
                    config_path = repo_path / "config.json"
                    if config_path.exists():
                        try:
                             with open(config_path, 'r') as f:
                                 config = json.load(f)
                             if config.get("model_type") == "qwen2_5_vl":
                                model_files = {f.name for f in repo_path.iterdir()} # Check files in the specific repo path
                                if all(f in model_files for f in ["model.safetensors.index.json", "tokenizer.json"]): # Check index and tokenizer too
                                     models.append({
                                         "id": repo.repo_id,
                                         "object": "model",
                                         "created": int(repo_path.stat().st_ctime), # Use folder creation time
                                         "owned_by": "huggingface", # Placeholder
                                     })
                        except Exception as e:
                             logging.warning(f"Could not read/parse config for {repo.repo_id}: {e}")

        except Exception as e:
            logging.error(f"Error scanning Hugging Face cache: {e}")
            # Return empty list or error response? Return empty for now.

        # Also add the currently loaded model if provided via CLI args
        if self.model_provider.cli_args.model and self.model_provider.model_key:
             loaded_model_id = self.model_provider.model_key[0] # Get the path/id used
             if loaded_model_id != 'default_model' and not any(m['id'] == loaded_model_id for m in models):
                  # Try to get creation time if it's a local path
                  created_time = int(time.time())
                  try:
                      created_time = int(Path(loaded_model_id).stat().st_ctime)
                  except: pass # Ignore if not a local path or error

                  models.append({
                      "id": loaded_model_id,
                      "object": "model",
                      "created": created_time,
                      "owned_by": "user",
                  })


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
    # prompt_cache = PromptCache() # Caching disabled
    system_fingerprint = get_system_fingerprint() # Get fingerprint once

    # Resolve server address correctly for IPv4/IPv6
    try:
         infos = socket.getaddrinfo(
            host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
         )
         # Try first address family returned
         server_class.address_family, _, _, _, server_address = infos[0]
    except socket.gaierror as e:
         logging.error(f"Error resolving server address {host}:{port} - {e}")
         return # Cannot start server


    httpd = server_class(
        server_address,
        # Use lambda to pass model_provider and fingerprint to each request handler instance
        lambda *args, **kwargs: handler_class(
            model_provider,
            # prompt_cache=prompt_cache, # Caching disabled
            system_fingerprint=system_fingerprint,
            *args,
            **kwargs,
        ),
    )
    warnings.warn(
        "mlx_lm.server (modified for VLM) is not recommended for production as "
        "it only implements basic security checks."
    )
    logging.info(f"Starting modified VLM httpd at {host} on port {port} (IPv{4 if server_class.address_family == socket.AF_INET else 6})...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logging.info("Server stopped by user.")
    finally:
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="MLX VLM HTTP Server (Qwen2.5-VL Only).")
    parser.add_argument(
        "--model",
        type=str,
        required=True, # Require model for this server
        help="The path or HF repo ID of the Qwen2.5-VL model.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained LoRA adapter weights.",
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
    # Draft model arguments removed
    # parser.add_argument(
    #     "--draft-model", ...
    # )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer/processor",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    # Chat template arguments removed (handled by processor)
    # parser.add_argument(
    #     "--chat-template", ...
    # )
    # parser.add_argument(
    #     "--use-default-chat-template", ...
    # )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", # Added name for better handler logging
    )

    # Instantiate ModelProvider with CLI args
    model_provider = ModelProvider(args)

    # Run the server
    run(args.host, args.port, model_provider)


# Keep the direct execution guard, but update the message
if __name__ == "__main__":
    print(
        "Running modified server for Qwen2.5-VL."
        " Use `python -m mlx_lm_vlm.server ...` (adjust module name if saved differently)."
    )
    main()