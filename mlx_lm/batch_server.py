import argparse
import json
import time
import threading
import queue
import uuid
import logging
import sys
import os
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any, Union, Literal

from .generate import BatchGenerator
from .sample_utils import make_sampler
from .utils import load

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class GenerationRequest:
    def __init__(self, prompt_tokens: List[int], max_tokens: int, response_queue: queue.Queue, tokenizer: Any, created: int):
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.response_queue = response_queue
        self.id = f"chatcmpl-{uuid.uuid4()}"
        self.created = created
        self.detokenizer = tokenizer.detokenizer
        self.detokenizer.reset()

class GenerationResult:
    def __init__(self, text_segment: str, finish_reason: Optional[str] = None, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.text_segment = text_segment
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

class PendingRequest:
    def __init__(self, input_data: Union[str, List[Dict[str, str]]], max_tokens: int, response_queue: queue.Queue, created: int):
        self.input_data = input_data
        self.max_tokens = max_tokens
        self.response_queue = response_queue
        self.created = created

class ModelWorker(threading.Thread):
    def __init__(self, model_path: str, adapter_path: str = None, 
                 max_batch_size: int = 32,
                 trust_remote_code: bool = False,
                 chat_template: str = "",
                 use_default_chat_template: bool = False,
                 chat_template_args: Dict = {},
                 temp: float = 0.0,
                 top_p: float = 1.0,
                 top_k: int = 0,
                 min_p: float = 0.0,
                 default_max_tokens: int = 512):
        super().__init__()
        self.request_queue = queue.Queue()
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_batch_size = max_batch_size
        self.trust_remote_code = trust_remote_code
        self.chat_template = chat_template
        self.use_default_chat_template = use_default_chat_template
        self.chat_template_args = chat_template_args
        self.temp = temp
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.default_max_tokens = default_max_tokens
        
        self.daemon = True
        self.running = True
        self.tokenizer = None
        self.model = None

    def run(self):
        logger.info(f"Loading model from {self.model_path}...")
        tokenizer_config = {"trust_remote_code": True} if self.trust_remote_code else {}
        self.model, self.tokenizer = load(self.model_path, adapter_path=self.adapter_path, tokenizer_config=tokenizer_config)
        
        # Setup chat template
        if self.use_default_chat_template:
            if self.tokenizer.chat_template is None:
                self.tokenizer.chat_template = self.tokenizer.default_chat_template
        elif self.chat_template:
            self.tokenizer.chat_template = self.chat_template
            
        logger.info("Model loaded. Starting worker loop.")

        # Create sampler
        sampler = make_sampler(
            temp=self.temp,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p
        )

        gen = BatchGenerator(
            self.model, 
            stop_tokens=self.tokenizer.eos_token_ids,
            sampler=sampler,
            completion_batch_size=self.max_batch_size,
        )

        uid_map: Dict[int, GenerationRequest] = {}
        
        last_stats_time = time.time()
        stats_interval = 10.0  # Log stats every 10 seconds
        
        # For instantaneous TPS calculation
        last_prompt_tokens = 0
        last_prompt_time = 0.0
        last_gen_tokens = 0
        last_gen_time = 0.0

        while self.running:
            # 1. Fetch new requests
            new_prompts = []
            new_max_tokens = []
            pending_requests_map = [] # To map index back to request object

            # Get all available requests from queue
            while not self.request_queue.empty():
                try:
                    pending = self.request_queue.get_nowait()
                    
                    # Handle templating and tokenization
                    if isinstance(pending.input_data, str):
                        prompt_text = pending.input_data
                    else:
                        # Apply chat template
                        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                            prompt_text = self.tokenizer.apply_chat_template(
                                pending.input_data, 
                                tokenize=False, 
                                add_generation_prompt=True,
                                **self.chat_template_args
                            )
                        else:
                            # Fallback if no template
                            prompt_text = pending.input_data[-1]["content"] if pending.input_data else ""
                    
                    prompt_tokens = self.tokenizer.encode(prompt_text)
                    
                    req = GenerationRequest(prompt_tokens, pending.max_tokens, pending.response_queue, self.tokenizer, pending.created)
                    
                    new_prompts.append(prompt_tokens)
                    new_max_tokens.append(req.max_tokens)
                    pending_requests_map.append(req)
                except queue.Empty:
                    break

            # 2. Insert into BatchGenerator
            if new_prompts:
                internal_uids = gen.insert(new_prompts, max_tokens=new_max_tokens)
                for i_uid, req in zip(internal_uids, pending_requests_map):
                    uid_map[i_uid] = req
                logger.debug(f"Added {len(new_prompts)} requests. Active: {len(uid_map)}")

            # 3. Step generation
            # If no active batch and no new requests, sleep briefly
            if gen.active_batch is None and not new_prompts and not gen.unprocessed_prompts:
                time.sleep(0.001) 
                continue
                
            responses = gen.next()

            # Log stats periodically
            current_time = time.time()
            if current_time - last_stats_time > stats_interval:
                stats = gen.stats()
                
                # Calculate instantaneous TPS
                delta_prompt_tokens = stats.prompt_tokens - last_prompt_tokens
                delta_prompt_time = stats.prompt_time - last_prompt_time
                
                delta_gen_tokens = stats.generation_tokens - last_gen_tokens
                delta_gen_time = stats.generation_time - last_gen_time
                
                inst_prompt_tps = delta_prompt_tokens / delta_prompt_time if delta_prompt_time > 0 else 0.0
                inst_gen_tps = delta_gen_tokens / delta_gen_time if delta_gen_time > 0 else 0.0
                
                logger.info(f"Instant Stats (Last {stats_interval}s) - Prompt TPS: {inst_prompt_tps:.2f}, Gen TPS: {inst_gen_tps:.2f}, Active Requests: {len(uid_map)}")
                
                # Update last stats
                last_stats_time = current_time
                last_prompt_tokens = stats.prompt_tokens
                last_prompt_time = stats.prompt_time
                last_gen_tokens = stats.generation_tokens
                last_gen_time = stats.generation_time

            # 4. Process responses
            for resp in responses:
                if resp.uid in uid_map:
                    req = uid_map[resp.uid]
                    
                    # Streaming detokenization
                    req.detokenizer.add_token(resp.token)
                    segment = req.detokenizer.last_segment
                    
                    # If finished, finalize detokenizer to get any remaining text
                    if resp.finish_reason:
                        req.detokenizer.finalize()
                        segment += req.detokenizer.last_segment
                        
                    # Calculate usage if finished
                    prompt_tokens_count = len(req.prompt_tokens) if resp.finish_reason else 0
                    # Note: We don't track exact completion tokens count in GenerationRequest easily without counting, 
                    # but we can approximate or the client counts. 
                    # For now, we send 0 for completion_tokens in intermediate chunks.
                    
                    result = GenerationResult(segment, resp.finish_reason, prompt_tokens=prompt_tokens_count)
                    req.response_queue.put(result)

                    if resp.finish_reason:
                        del uid_map[resp.uid]

    def submit(self, input_data: Union[str, List[Dict[str, str]]], max_tokens: int) -> queue.Queue:
        q = queue.Queue()
        created = int(time.time())
        req = PendingRequest(input_data, max_tokens, q, created)
        self.request_queue.put(req)
        return q

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    pass

class APIHandler(BaseHTTPRequestHandler):
    def __init__(self, worker: ModelWorker, *args, **kwargs):
        self.worker = worker
        super().__init__(*args, **kwargs)

    def _set_headers(self, status_code=200, content_type='application/json'):
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        if self.path == "/v1/models":
            self.handle_models()
        elif self.path == "/health":
            self.handle_health()
        else:
            self.send_error(404)

    def do_POST(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/v1/chat/completions":
            self.handle_chat_completions()
        else:
            self.send_error(404)

    def handle_models(self):
        response = {
            "object": "list",
            "data": [
                {
                    "id": self.worker.model_path,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mlx-lm",
                }
            ]
        }
        self._set_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def handle_health(self):
        self._set_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode('utf-8'))

    def handle_chat_completions(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self.send_error(400, "Missing body")
            return
            
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        messages = data.get("messages")
        if not messages:
             self.send_error(400, "Missing 'messages' field")
             return

        max_tokens = data.get("max_tokens", self.worker.default_max_tokens)
        stream = data.get("stream", False)
        
        # Submit task
        response_queue = self.worker.submit(messages, max_tokens)
        
        # We need to wait for at least the first item to get the ID and created time
        # But since submit is async, we don't have the ID yet. 
        # The worker generates the ID. We'll get it with the first result.
        
        if stream:
            self._set_headers(200, 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')

            request_id = None
            created = int(time.time())
            
            while True:
                result = response_queue.get()
                
                # In a real implementation we would get the ID from the result
                # For now we generate one if missing (though worker generates one, we don't pass it back in Result object currently)
                # Let's fix GenerationResult to carry ID if needed, or just generate one here.
                # The worker generates an ID but we don't easily get it back until the first token.
                # For simplicity, we'll generate a request ID here if we don't have one, 
                # but ideally it should come from the worker to be consistent.
                if request_id is None:
                    request_id = f"chatcmpl-{uuid.uuid4()}"

                if result.finish_reason:
                    # Send final chunk if there is remaining text
                    if result.text_segment:
                         chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": self.worker.model_path,
                            "choices": [{"index": 0, "delta": {"content": result.text_segment}, "finish_reason": None}]
                        }
                         self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode('utf-8'))
                    
                    # Send done
                    final_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.worker.model_path,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": result.finish_reason}]
                    }
                    self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode('utf-8'))
                    self.wfile.write(b"data: [DONE]\n\n")
                    break
                
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.worker.model_path,
                    "choices": [{"index": 0, "delta": {"content": result.text_segment}, "finish_reason": None}]
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode('utf-8'))
                self.wfile.flush()
        else:
            full_text = ""
            finish_reason = None
            prompt_tokens = 0
            completion_tokens = 0
            
            while True:
                result = response_queue.get()
                full_text += result.text_segment
                completion_tokens += 1 # Approximate
                if result.finish_reason:
                    finish_reason = result.finish_reason
                    prompt_tokens = result.prompt_tokens
                    break
            
            response = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.worker.model_path,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                } 
            }
            
            self._set_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))

def run(host: str, port: int, worker: ModelWorker):
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, lambda *args, **kwargs: APIHandler(worker, *args, **kwargs))
    logger.info(f"Server started at http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

def main():
    parser = argparse.ArgumentParser(description="MLX Dynamic Batching Server")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
        required=True,
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
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="A model to be used for speculative decoding. (Not supported in dynamic batching yet)",
        default=None,
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding. (Not supported in dynamic batching yet)",
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
        help="Set the logging level",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Max completion batch size",
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

    # Update logging level
    logger.setLevel(getattr(logging, args.log_level.upper()))

    worker = ModelWorker(
        model_path=args.model, 
        adapter_path=args.adapter_path, 
        max_batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
        chat_template=args.chat_template,
        use_default_chat_template=args.use_default_chat_template,
        chat_template_args=args.chat_template_args,
        temp=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        default_max_tokens=args.max_tokens
    )
    worker.start()

    run(args.host, args.port, worker)

if __name__ == "__main__":
    main()
