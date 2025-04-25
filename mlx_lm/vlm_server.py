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
import base64  # Added for Base64 decoding
import io  # Added for Base64 decoding
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
import requests  # Added for URL image loading
from PIL import Image, ImageOps  # Added for image handling

# MLX-LM original imports
from ._version import __version__
from .sample_utils import make_logits_processors, make_sampler

# VLM Integration Imports
from mlx_vlm.utils import (
    load as vlm_load,
    load_config as vlm_load_config,
    stream_generate as vlm_stream_generate,
    GenerationResult as VLMGenerationResult,
    get_model_path,
)
from mlx_vlm.prompt_utils import apply_chat_template

# HF Hub import for model listing
from huggingface_hub import scan_cache_dir


# --- Universal Image Loader Helper (with enhanced debugging) ---
def _load_image_universal(source_str: str, timeout: int = 10) -> Image.Image:
    """
    Loads an image from a URL, local file path, or Base64 data URI.
    Includes enhanced debugging for Base64.
    """
    image = None
    if source_str.startswith("data:image"):
        logging.debug(
            f"Attempting to load image from Base64 data URI (first 60 chars): {source_str[:60]}..."
        )
        try:
            # Split header and encoded data
            header, encoded = source_str.split(",", 1)
            logging.debug(f"  Base64 Header: {header}")
            logging.debug(f"  Encoded data length: {len(encoded)}")

            # Validate header structure minimally
            if not header.startswith("data:image/") or ";base64" not in header:
                raise ValueError(f"Invalid Base64 Data URI header format: {header}")

            # Decode Base64
            try:
                # Add padding if necessary, standard decoders might handle it, but being explicit can help
                missing_padding = len(encoded) % 4
                if missing_padding:
                    encoded += "=" * (4 - missing_padding)
                decoded_data = base64.b64decode(
                    encoded, validate=True
                )  # Use validate=True
                logging.debug(
                    f"  Successfully decoded {len(decoded_data)} bytes from Base64."
                )
            except base64.binascii.Error as b64_error:
                logging.error(f"  Base64 decoding error: {b64_error}")
                raise ValueError(
                    f"Failed to decode Base64 string: {b64_error}"
                ) from b64_error

            # Load image from decoded bytes using Pillow
            buffer = io.BytesIO(decoded_data)
            image = Image.open(buffer)
            logging.debug(
                f"  Successfully opened image from buffer. Format: {image.format}, Mode: {image.mode}, Size: {image.size}"
            )

        except Exception as e:
            # Log the full traceback for any exception during Base64 processing
            logging.error(f"Error processing Base64 URI: {e}", exc_info=True)
            raise ValueError(f"Failed to decode/load Base64 image: {e}") from e

    elif source_str.startswith(("http://", "https://")):
        # Handle URL
        logging.debug(f"Attempting to load image from URL: {source_str}")
        try:
            response = requests.get(source_str, stream=True, timeout=timeout)
            response.raise_for_status()  # Check for HTTP errors
            # Read into a buffer first to avoid issues with some servers/image types
            img_buffer = io.BytesIO(response.content)
            image = Image.open(img_buffer)
            logging.debug(
                f"  Successfully opened image from URL. Format: {image.format}, Mode: {image.mode}, Size: {image.size}"
            )
        except Exception as e:
            logging.error(
                f"Error loading image from URL '{source_str}': {e}", exc_info=True
            )
            raise ValueError(
                f"Failed to load image from URL '{source_str}': {e}"
            ) from e
    else:
        # Handle Local File Path
        logging.debug(f"Attempting to load image from path: {source_str}")
        image_path = Path(source_str)
        if image_path.is_file():
            try:
                image = Image.open(image_path)
                logging.debug(
                    f"  Successfully opened image from path. Format: {image.format}, Mode: {image.mode}, Size: {image.size}"
                )
            except Exception as e:
                # Log specific file loading errors
                logging.error(
                    f"Error loading image from path '{source_str}': {e}", exc_info=True
                )
                raise ValueError(
                    f"Failed to load image from path '{source_str}': {e}"
                ) from e
        else:
            # Log if path doesn't exist before raising generic error
            logging.warning(
                f"Invalid image source: Path '{source_str}' does not exist or is not a file."
            )
            raise ValueError(
                f"Invalid image source: '{source_str}' is not a valid URL, existing local path, or Base64 URI."
            )

    # --- Post-processing ---
    if image:
        try:
            # Ensure image orientation is correct (from EXIF data)
            image = ImageOps.exif_transpose(image)
            # Convert to RGB format for consistency
            if image.mode != "RGB":
                image = image.convert("RGB")
                logging.debug(f"  Converted image to RGB mode.")
            return image
        except Exception as e:
            # Catch errors during post-processing (e.g., converting corrupted image)
            logging.error(
                f"Error post-processing image (EXIF/convert): {e}", exc_info=True
            )
            raise ValueError(
                f"Failed to post-process image from '{source_str}': {e}"
            ) from e
    else:
        # This should technically be unreachable if logic above is correct
        raise ValueError(f"Image loading resulted in None for source: {source_str}")


# --- Rest of the Server Code ---

def get_system_fingerprint():
    vlm_version = "unknown"
    try:
        from mlx_vlm.version import __version__ as vlm_ver

        vlm_version = vlm_ver
    except ImportError:
        pass
    gpu_arch = mx.metal.device_info()["architecture"] if mx.metal.is_available() else ""
    return f"lm-{__version__}-vlm-{vlm_version}-{mx.__version__}-{platform.platform()}-{gpu_arch}"

class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int


# stopping_criteria and sequence_overlap remain unchanged

def process_message_content(messages):
    if not messages:
        return "", []
    last_message = messages[-1]
    content = last_message.get("content")
    extracted_text, image_sources = "", []
    if isinstance(content, list):
        current_message_text = []
        for item in content:
            item_type = item.get("type")
            if item_type == "text":
                current_message_text.append(item.get("text", ""))
            elif item_type == "image_url":
                url = item.get("image_url", {}).get("url")
                if url:
                    image_sources.append(url)
        extracted_text = "".join(current_message_text)
        last_message["content"] = extracted_text  # Update for logging
    elif isinstance(content, str):
        extracted_text = content
    elif last_message.get("role") in ["system", "assistant"]:
        extracted_text = content or ""
    else:
        raise ValueError(
            f"Invalid message content format for role '{last_message.get('role')}': {type(content)}"
        )
    return extracted_text, image_sources


class ModelProvider:
    # __init__, _validate_model_path, load methods remain the same as previous version
    def __init__(self, cli_args: argparse.Namespace):
        self.cli_args = cli_args
        self.model_key = None
        self.model = None
        self.processor = None
        if self.cli_args.model is not None:
            self.load("default_model")

    def _validate_model_path(self, model_path: str):
        model_path_obj = Path(model_path)
        if (
            model_path_obj.exists()
            and not model_path_obj.is_absolute()
            and not model_path_obj.resolve().is_relative_to(Path.cwd().resolve())
        ):
            logging.warning(
                f"Local model path {model_path} exists but is not relative to CWD."
            )
        elif (
            not model_path_obj.exists()
            and "/" not in model_path
            and "\\" not in model_path
        ):
            pass
        elif not model_path_obj.exists():
            logging.warning(f"Provided model path {model_path} does not exist locally.")

    def load(self, model_path, adapter_path=None):
        current_key = (model_path, adapter_path)
        if self.model_key == current_key:
            return self.model, self.processor
        self.model, self.processor, self.model_key = None, None, None
        if model_path == "default_model":
            if self.cli_args.model is None:
                raise ValueError("A model path must be provided via --model.")
            actual_model_path_or_repo = self.cli_args.model
            adapter_path = adapter_path or self.cli_args.adapter_path
        else:
            self._validate_model_path(model_path)
            actual_model_path_or_repo = model_path
        try:
            resolved_path = get_model_path(actual_model_path_or_repo)
            config = vlm_load_config(
                resolved_path, trust_remote_code=self.cli_args.trust_remote_code
            )
            model_type = config.get("model_type")
            if model_type != "qwen2_5_vl":
                raise ValueError(
                    f"Server only supports 'qwen2_5_vl'. Attempted: '{model_type}'"
                )
        except FileNotFoundError:
            logging.warning(
                f"Config file not found for {actual_model_path_or_repo}. Assuming HF repo."
            )
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            raise e
        try:
            logging.info(
                f"Loading Qwen2.5-VL model/processor from: {actual_model_path_or_repo}"
            )
            load_kwargs = (
                {"trust_remote_code": True} if self.cli_args.trust_remote_code else {}
            )
            model, processor = vlm_load(
                actual_model_path_or_repo, adapter_path=adapter_path, **load_kwargs
            )
            logging.info(f"Successfully loaded: {actual_model_path_or_repo}")
        except Exception as e:
            logging.error(f"Failed to load VLM model or processor: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to load model from {actual_model_path_or_repo}"
            ) from e
        self.model_key, self.model, self.processor = current_key, model, processor
        return self.model, self.processor


class APIHandler(BaseHTTPRequestHandler):
    # __init__, _set_cors_headers, _set_completion_headers, _set_stream_headers, do_OPTIONS remain the same
    def __init__(
        self,
        model_provider: ModelProvider,
        *args,
        system_fingerprint: Optional[str] = None,
        **kwargs,
    ):
        self.created = int(time.time())
        self.model_provider = model_provider
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
        self.end_headers()

    def _set_stream_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._set_cors_headers()
        self.end_headers()

    def do_OPTIONS(self):
        self._set_completion_headers(204)

    # do_POST remains the same as previous version
    def do_POST(self):
        endpoints = {"/v1/chat/completions": self.handle_chat_completions}
        if self.path not in endpoints:
            self._set_completion_headers(404)
            self.wfile.write(b"Not Found")
            return
        try:
            content_length = int(self.headers["Content-Length"])
            raw_body = self.rfile.read(content_length)
            self.body = json.loads(raw_body.decode())
            indent = "\t"
            logging.debug(f"Incoming Request Body: {json.dumps(self.body, indent=indent)}")
            assert isinstance(self.body, dict), (
                f"Request should be dict, got {type(self.body)}"
            )
            messages = self.body.get("messages", [])
            assert messages, "Messages list required."
        except Exception as e:
            logging.error(f"Bad request: {e}")
            self._set_completion_headers(400)
            self.wfile.write(json.dumps({"error": f"Bad Request: {e}"}).encode())
            return
        try:
            self.stream = self.body.get("stream", False)
            self.stream_options = self.body.get("stream_options", None)
            self.requested_model = self.model_provider.cli_args.model
            self.adapter = self.body.get("adapters", None)
            self.max_tokens = self.body.get("max_tokens", 512)
            self.temperature = self.body.get("temperature", 0.7)
            self.top_p = self.body.get("top_p", 1.0)
            self.repetition_penalty = self.body.get("repetition_penalty", None)
            self.repetition_context_size = self.body.get("repetition_context_size", 20)
            self.logit_bias = self.body.get("logit_bias", None)
            self.logprobs = self.body.get("logprobs", -1)
            self.validate_model_parameters()
            self.model, self.processor = self.model_provider.load(
                "default_model", self.adapter
            )
            assert self.processor, "Failed to load VLM processor."
            (
                system_prompt_content,
                user_prompt_text,
                image_sources,
                processed_messages_for_template,
            ) = None, "", [], []
            for msg in messages:
                role = msg.get("role")
                if role == "system":
                    system_prompt_content = msg.get("content")
                    processed_messages_for_template.append(msg)
                elif role == "user":
                    user_prompt_text, image_sources = process_message_content([msg])
                    processed_messages_for_template.append(
                        {"role": "user", "content": user_prompt_text}
                    )
                elif role == "assistant":
                    processed_messages_for_template.append(
                        {"role": "assistant", "content": msg.get("content", "")}
                    )
            assert user_prompt_text or image_sources or system_prompt_content, (
                "Request needs content."
            )
            stop_words = self.body.get("stop", [])
            stop_words = [stop_words] if isinstance(stop_words, str) else stop_words
            try:
                tokenizer_obj = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
                default_eos = getattr(
                    self.model.config,
                    "eos_token_id",
                    getattr(tokenizer_obj, "eos_token_id", None),
                )
                tokenizer_obj.stopping_criteria.reset(default_eos)
                if stop_words:
                    tokenizer_obj.stopping_criteria.add_eos_token_ids(stop_words)
                logging.debug(
                    f"Stopping criteria EOS IDs: {tokenizer_obj.stopping_criteria.eos_token_ids}"
                )
            except Exception as e:
                logging.warning(f"Could not configure stop words: {e}")
            if self.stream:
                self._set_stream_headers(200)
            else:
                self._set_completion_headers(200)
            formatted_prompt_string = endpoints[self.path](
                processed_messages_for_template, image_sources
            )
            self.handle_completion(formatted_prompt_string, image_sources)
        except ValueError as e:
            logging.error(f"Request error: {e}")
            self.send_error_response(400, f"Bad Request: {e}")
        except RuntimeError as e:
            logging.error(f"Server error: {e}", exc_info=True)
            self.send_error_response(500, f"Internal Server Error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}", exc_info=True)
            self.send_error_response(500, "Unexpected Internal Server Error")

    def response_sent(self):
        # Simple heuristic, might not be perfect in all edge cases
        return hasattr(self, "_headers_buffer") and bool(self._headers_buffer)

    def send_error_response(self, code: int, message: str):
        """Sends a JSON error response, trying not to break existing streams."""
        try:
            if not self.response_sent():
                # If headers not sent, send proper HTTP error
                if code == 400:
                    self._set_completion_headers(400)
                else:
                    self._set_completion_headers(500)
                self.wfile.write(json.dumps({"error": message}).encode())
            elif self.stream:
                # If stream started, send error in stream data
                error_payload = {"error": {"message": message, "type": "server_error"}}
                self.wfile.write(f"data: {json.dumps(error_payload)}\n\n".encode())
                self.wfile.write("data: [DONE]\n\n".encode())  # Terminate stream
                self.wfile.flush()
            # If non-streaming and headers sent, can't do much, error is logged
        except Exception as e:
            logging.error(f"Failed to send error response to client: {e}")

    # validate_model_parameters remains the same as previous version
    def validate_model_parameters(self):
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(self.temperature, (float, int)) or not (
            0.0 <= self.temperature <= 2.0
        ):
            raise ValueError("temperature must be a float between 0.0 and 2.0")
        if not isinstance(self.top_p, (float, int)) or not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be a float between 0.0 and 1.0")
        if self.repetition_penalty is not None and (
            not isinstance(self.repetition_penalty, (float, int))
            or self.repetition_penalty <= 0
        ):
            raise ValueError("repetition_penalty must be a positive float")
        if self.logprobs != -1:
            logging.warning(
                "Logprobs requested, but may not be provided by VLM backend."
            )
            if not (0 < self.logprobs <= 10):
                raise ValueError(f"logprobs must be between 1 and 10")
        if (
            not isinstance(self.repetition_context_size, int)
            or self.repetition_context_size < 0
        ):
            raise ValueError("repetition_context_size must be non-negative int")
        if self.logit_bias is not None:
            if not isinstance(self.logit_bias, dict):
                raise ValueError("logit_bias must be a dict")
            try:
                self.logit_bias = {int(k): float(v) for k, v in self.logit_bias.items()}
            except Exception:
                raise ValueError("logit_bias keys must be int, values float")

    # generate_response remains the same as previous version
    def generate_response(
        self,
        text: str,
        finish_reason: Union[Literal["length", "stop"], None],
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        token_logprobs: Optional[List[float]] = None,
        top_tokens: Optional[List[Dict[int, float]]] = None,
        tokens: Optional[List[int]] = None,
    ) -> dict:
        token_logprobs = token_logprobs or []
        top_logprobs = top_tokens or []
        tokens = tokens or []
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": self.object_type,
            "model": self.requested_model,
            "created": self.created,
            "choices": [{"index": 0, "logprobs": None, "finish_reason": finish_reason}],
        }
        if self.logprobs > 0 and (token_logprobs or top_logprobs or tokens):
            response["choices"][0]["logprobs"] = {
                "token_logprobs": token_logprobs or None,
                "top_logprobs": top_logprobs or None,
                "tokens": tokens or None,
            }
        if not self.stream:
            prompt_tokens_est = (
                prompt_token_count if isinstance(prompt_token_count, int) else -1
            )
            completion_tokens_est = (
                completion_token_count
                if isinstance(completion_token_count, int)
                else -1
            )
            if prompt_tokens_est == -1 or completion_tokens_est == -1:
                logging.warning("Usage token counts may be inaccurate.")
            response["usage"] = {
                "prompt_tokens": prompt_tokens_est,
                "completion_tokens": completion_tokens_est,
                "total_tokens": (prompt_tokens_est + completion_tokens_est)
                if prompt_tokens_est != -1 and completion_tokens_est != -1
                else -1,
            }
        choice = response["choices"][0]
        key_name = "delta" if self.stream else "message"
        choice[key_name] = {"role": "assistant", "content": text}
        return response

    # handle_completion remains the same as previous version (with indent fix)
    def handle_completion(self, prompt_string: str, image_sources: List[str]):
        original_load_image = None
        try:
            # --- Monkey-patch image loading ---
            import mlx_vlm.utils

            if hasattr(mlx_vlm.utils, "load_image"):
                original_load_image = mlx_vlm.utils.load_image
                mlx_vlm.utils.load_image = _load_image_universal
                logging.debug("Monkey-patched mlx_vlm.utils.load_image")
            # --- Prepare arguments ---
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
            }
            generation_args = {
                k: v for k, v in generation_args.items() if v is not None
            }
            accumulated_text, last_response_obj, all_tokens, all_token_logprobs = (
                "",
                None,
                [],
                [],
            )
            logging.debug(f"Starting VLM generation...")
            # --- Generation Loop ---
            for response_obj in vlm_stream_generate(**generation_args):
                last_response_obj = response_obj
                segment = response_obj.text
                accumulated_text += segment
                if (
                    self.logprobs > 0
                    and response_obj.token is not None
                    and response_obj.logprobs is not None
                ):
                    all_tokens.append(response_obj.token)
                    try:
                        all_token_logprobs.append(
                            response_obj.logprobs[response_obj.token].item()
                        )
                    except Exception:
                        all_token_logprobs.append(None)
                if self.stream and segment:
                    response_payload = self.generate_response(segment, None)
                    try:
                        self.wfile.write(
                            f"data: {json.dumps(response_payload)}\n\n".encode()
                        )
                        self.wfile.flush()
                    except BrokenPipeError:
                        logging.warning("Client disconnected.")
                        return
            # --- Final response handling ---
            final_finish_reason = "stop"
            if last_response_obj is not None:
                last_token, num_generated = (
                    last_response_obj.token,
                    last_response_obj.generation_tokens,
                )
                tokenizer_obj = (
                    self.processor.tokenizer
                    if hasattr(self.processor, "tokenizer")
                    else self.processor
                )
                if last_token is not None and tokenizer_obj.stopping_criteria(
                    last_token
                ):
                    final_finish_reason = "stop"
                elif num_generated >= self.max_tokens:
                    final_finish_reason = "length"
                else:
                    logging.warning(
                        f"Generation finished unexpectedly. Defaulting finish_reason to 'stop'."
                    )
            else:
                logging.warning("No tokens generated. Setting finish_reason to 'stop'.")
                last_response_obj = VLMGenerationResult(
                    "", None, None, -1, 0, 0.0, 0.0, mx.get_peak_memory() / 1e9
                )
            # --- Send final response ---
            if self.stream:
                final_delta_payload = self.generate_response(
                    "",
                    final_finish_reason,
                    token_logprobs=(
                        all_token_logprobs or None if self.logprobs > 0 else None
                    ),
                    tokens=(all_tokens or None if self.logprobs > 0 else None),
                )
                try:
                    self.wfile.write(
                        f"data: {json.dumps(final_delta_payload)}\n\n".encode()
                    )
                    if self.stream_options and self.stream_options.get("include_usage"):
                        usage_payload = self.completion_usage_response(
                            last_response_obj.prompt_tokens,
                            last_response_obj.generation_tokens,
                        )
                        self.wfile.write(
                            f"data: {json.dumps(usage_payload)}\n\n".encode()
                        )
                    self.wfile.write("data: [DONE]\n\n".encode())
                    self.wfile.flush()
                except BrokenPipeError:
                    logging.warning("Client disconnected before final stream.")
            else:  # Non-streaming
                response_payload = self.generate_response(
                    accumulated_text,
                    final_finish_reason,
                    last_response_obj.prompt_tokens,
                    last_response_obj.generation_tokens,
                    token_logprobs=(
                        all_token_logprobs or None if self.logprobs > 0 else None
                    ),
                    tokens=(all_tokens or None if self.logprobs > 0 else None),
                )
                response_json = json.dumps(response_payload).encode()
                indent = "\t"
                logging.debug(
                    f"Outgoing Response: {json.dumps(response_payload, indent=indent)}"
                )  # Define indent here
                self.wfile.write(response_json)
                self.wfile.flush()
        except Exception as e:
            logging.error(f"Error during generation: {e}", exc_info=True)
            if self.stream and self.response_sent():
                try:
                    error_payload = {
                        "error": {
                            "message": "Error during generation",
                            "type": "generation_error",
                        }
                    }
                    self.wfile.write(f"data: {json.dumps(error_payload)}\n\n".encode())
                    self.wfile.write("data: [DONE]\n\n".encode())
                    self.wfile.flush()
                except Exception as e_send:
                    logging.error(f"Failed to send error stream: {e_send}")
            # Error response for non-streaming is handled in do_POST
        finally:
            # --- Restore original image loader ---
            if original_load_image is not None:
                try:
                    import mlx_vlm.utils

                    mlx_vlm.utils.load_image = original_load_image
                    logging.debug("Restored original load_image")
                except Exception as e:
                    logging.error(f"Failed restore load_image: {e}")

    # completion_usage_response remains the same
    def completion_usage_response(
        self,
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
    ):
        prompt_tokens = (
            prompt_token_count if isinstance(prompt_token_count, int) else -1
        )
        completion_tokens = (
            completion_token_count if isinstance(completion_token_count, int) else -1
        )
        total_tokens = (
            (prompt_tokens + completion_tokens)
            if prompt_tokens != -1 and completion_tokens != -1
            else -1
        )
        return {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": "chat.completion.chunk",
            "model": self.requested_model,
            "created": self.created,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    # handle_chat_completions remains the same
    def handle_chat_completions(
        self, processed_messages: list, image_sources: List[str]
    ) -> str:
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completion.chunk" if self.stream else "chat.completion"
        num_images = len(image_sources)
        logging.debug(
            f"Applying template for {len(processed_messages)} messages, {num_images} images."
        )
        formatted_prompt = apply_chat_template(
            self.processor,
            self.model.config,
            prompt=processed_messages,
            num_images=num_images,
            add_generation_prompt=True,
        )
        logging.debug(f"Formatted prompt string (start): {formatted_prompt[:100]}...")
        return formatted_prompt

    # do_GET and handle_models_request remain the same
    def do_GET(self):
        if self.path == "/v1/models":
            self.handle_models_request()
        else:
            self._set_completion_headers(404)
            self.wfile.write(b"Not Found")

    def handle_models_request(self):
        self._set_completion_headers(200)
        models = []
        if self.model_provider.model_key:
            loaded_model_id = self.model_provider.model_key[0]
            created_time = int(time.time())
            if loaded_model_id == "default_model":
                loaded_model_id = self.model_provider.cli_args.model
            try:
                created_time = int(Path(loaded_model_id).stat().st_ctime)
            except Exception:
                pass
            models.append(
                {
                    "id": loaded_model_id,
                    "object": "model",
                    "created": created_time,
                    "owned_by": "user",
                }
            )
        response = {"object": "list", "data": models}
        response_json = json.dumps(response).encode()
        self.wfile.write(response_json)
        self.wfile.flush()


# run function remains the same
def run(
    host: str,
    port: int,
    model_provider: ModelProvider,
    server_class=HTTPServer,
    handler_class=APIHandler,
):
    server_address = (host, port)
    system_fingerprint = get_system_fingerprint()
    try:
        infos = socket.getaddrinfo(
            host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            flags=socket.AI_PASSIVE,
        )
        server_class.address_family, _, _, _, server_address = infos[0]
    except socket.gaierror as e:
        logging.error(f"Error resolving {host}:{port} - {e}")
        return
    httpd = server_class(
        server_address,
        lambda *args, **kwargs: handler_class(
            model_provider, system_fingerprint=system_fingerprint, *args, **kwargs
        ),
    )
    warnings.warn(
        "mlx_lm server (modified for VLM) is not recommended for production.",
        stacklevel=2,
    )
    logging.info(
        f"Starting modified VLM httpd at {host}:{port} (IPv{4 if server_class.address_family == socket.AF_INET else 6})..."
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logging.info("Server stopped.")
    finally:
        logging.info("Closing server.")
        httpd.server_close()


# main function remains the same
def main():
    parser = argparse.ArgumentParser(
        description="MLX VLM HTTP Server (Qwen2.5-VL Only)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HF repo ID of the Qwen2.5-VL model.",
    )
    parser.add_argument(
        "--adapter-path", type=str, help="Optional path for LoRA adapter."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    model_provider = ModelProvider(args)
    run(args.host, args.port, model_provider)


if __name__ == "__main__":
    print("Running modified server for Qwen2.5-VL.")
    main()