
import argparse
import logging
import time
import uuid
import json
import asyncio
from typing import List, Optional, Union, Dict, Any, Literal
from contextlib import asynccontextmanager
from queue import Queue

import mlx.core as mx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from .server import (
    ModelProvider, 
    ResponseGenerator, 
    LRUPromptCache, 
    GenerationArguments, 
    SamplingArguments,
    LogitsProcessorArguments,
    ModelDescription,
    CompletionRequest,
    CompletionRequest,
    get_system_fingerprint
)
from .logits_processors import OutlinesLogitsProcessor

# --- Pydantic Models for Input Validation ---

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]], None] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0.0 
    min_p: Optional[float] = 0.0
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 100
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    repetition_penalty: Optional[float] = 1.0
    repetition_context_size: Optional[int] = 20
    logit_bias: Optional[Dict[int, float]] = None
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    regex: Optional[str] = None

class CompletionRequestModel(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    repetition_penalty: Optional[float] = 1.0
    repetition_context_size: Optional[int] = 20
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None

# --- Global State ---

model_provider: Optional[ModelProvider] = None
response_generator: Optional[ResponseGenerator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs when the app starts
    yield
    # This runs when the app shuts down
    if response_generator:
        response_generator.stop_and_join()

def create_app(cli_args: argparse.Namespace):
    global model_provider, response_generator
    
    # Initialize Core Components
    model_provider = ModelProvider(cli_args)
    prompt_cache = LRUPromptCache(max_size=cli_args.max_kv_size or 10)
    response_generator = ResponseGenerator(model_provider, prompt_cache)

    app = FastAPI(title="MLX-LM Server", version="v1", lifespan=lifespan)

    @app.get("/v1/models")
    async def list_models():
        model_id = model_provider.model_key[0] if model_provider.model_key else "default_model"
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mlx-lm",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        req_id = f"chatcmpl-{uuid.uuid4()}"
        
        stop_words = request.stop or []
        if isinstance(stop_words, str):
            stop_words = [stop_words]
            
        gen_args = GenerationArguments(
            model=ModelDescription(
                model=request.model,
                draft=cli_args.draft_model,
                adapter=cli_args.adapter_path
            ),
            sampling=SamplingArguments(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k, 
                min_p=request.min_p,
                xtc_probability=0.0,
                xtc_threshold=0.0
            ),
            logits=LogitsProcessorArguments(
                logit_bias=request.logit_bias,
                repetition_penalty=request.repetition_penalty,
                repetition_context_size=request.repetition_context_size
            ),
            stop_words=stop_words,
            max_tokens=request.max_tokens,
            num_draft_tokens=cli_args.num_draft_tokens,
            logprobs=0,
            seed=request.seed
        )

        logits_processors = []
        if request.response_format and request.response_format.get("type") == "json_object":
             # Basic JSON schema support if provided in schema
             schema = request.response_format.get("schema")
             logits_processors.append(OutlinesLogitsProcessor(model_provider.tokenizer, schema_str=schema))
        elif request.regex:
             logits_processors.append(OutlinesLogitsProcessor(model_provider.tokenizer, regex_str=request.regex))

        # We need to pass these processors to the generation Logic. 
        # The current ResponseGenerator/GenerationArguments structure expects LogitsProcessorArguments 
        # which only supports basic penalties. 
        # We need to hack or extend.
        # Since `_make_logits_processors` in server.py is used by ResponseGenerator, it builds lists from args.
        # We can't easily inject custom object processors via arguments without modifying server.py or sub-classing.
        
        # Only feasible way without modifying server.py heavily is ensuring LogitsProcessorArguments 
        # can carry extra processors or we patch `_make_logits_processors`.
        
        # Ideally, we modify `GenerationArguments` to accept a list of `extra_logits_processors`.
        gen_args.extra_logits_processors = logits_processors

        messages = [m.model_dump() for m in request.messages]
        
        comp_request = CompletionRequest(
            request_type="chat",
            prompt="", 
            messages=messages,
            tools=request.tools,
            role_mapping=None
        )

        rqueue = Queue()
        response_generator.requests.put((rqueue, comp_request, gen_args))

        if request.stream:
            return StreamingResponse(
                stream_generator(rqueue, req_id, request.model), 
                media_type="text/event-stream"
            )
        else:
            return await collect_response(rqueue, req_id, request.model)

    def stream_generator(rqueue, req_id, model_name):
        system_fingerprint = get_system_fingerprint()
        
        while True:
            response = rqueue.get()
            if response is None:
                yield f"data: [DONE]\n\n"
                break
                
            if isinstance(response, Exception):
                logging.error(f"Error during generation: {response}")
                break

            if isinstance(response, tuple): 
                continue 
            
            if hasattr(response, "text"):
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "system_fingerprint": system_fingerprint,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": response.text} if response.text else {},
                            "logprobs": None,
                            "finish_reason": response.finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
    async def collect_response(rqueue, req_id, model_name):
        system_fingerprint = get_system_fingerprint()
        full_content = ""
        finish_reason = None
        
        while True:
            try:
                # Polling the queue in a non-blocking way for the asyncio loop
                response = rqueue.get_nowait()
            except:
                await asyncio.sleep(0.01)
                continue
                
            if response is None:
                break
            if isinstance(response, Exception):
                raise HTTPException(status_code=500, detail=str(response))
            
            if hasattr(response, "text"):
                full_content += response.text
                if response.finish_reason:
                    finish_reason = response.finish_reason
            
            # Flush progress info if any (tuples)
            if isinstance(response, tuple):
                continue

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "system_fingerprint": system_fingerprint,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_content,
                    },
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    
    return app

def main():
    parser = argparse.ArgumentParser(description="MLX LM FastAPI Server")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--sys-prompt", type=str, default=None)
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--num-draft-tokens", type=int, default=5)
    parser.add_argument("--max-kv-size", type=int, default=None)
    # Add other necessary args from original server if needed
    parser.add_argument("--trust-remote-code", action="store_true")
    # Missing args compatibility might be needed
    # ...
    # We can use the existing setup_arg_parser from generate or similar if available, 
    # but here we just add what we use.
    
    # Mocking missing args for ModelProvider compliance if necessary
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--use-default-chat-template", action="store_true")

    args = parser.parse_args()
    
    # ModelProvider expects 'args' to have certain attributes
    if not hasattr(args, 'chat_template_args'):
        args.chat_template_args = {}

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
