# Copyright © 2026 Apple Inc.

import http
import os
import threading

import pytest
import requests

from mlx_lm.server import APIHandler, LRUPromptCache, ResponseGenerator
from mlx_lm.utils import load

MODEL_ENV = "MLX_LM_TEST_MODEL"
RUN_ENV = "MLX_LM_INTEGRATION"
DEFAULT_MODEL = "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-8bit"


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def _integration_enabled() -> bool:
    return _truthy_env(RUN_ENV)


@pytest.fixture(scope="module")
def integration_model():
    if not _integration_enabled():
        pytest.skip(f"Set {RUN_ENV}=1 to run integration tests.")
    model_id = os.getenv(MODEL_ENV, DEFAULT_MODEL)
    model, tokenizer = load(model_id)
    return model_id, model, tokenizer


@pytest.fixture(scope="module")
def integration_server(integration_model):
    model_id, model, tokenizer = integration_model

    class IntegrationModelProvider:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.model_key = (model_id, None)
            self.draft_model = None
            self.draft_model_key = None
            self.is_batchable = True
            self.cli_args = type(
                "obj",
                (object,),
                {
                    "adapter_path": None,
                    "chat_template": None,
                    "use_default_chat_template": False,
                    "trust_remote_code": False,
                    "draft_model": None,
                    "num_draft_tokens": 3,
                    "temp": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "min_p": 0.0,
                    "max_tokens": 512,
                    "chat_template_args": {},
                    "model": None,
                },
            )

        def load(self, model_name, adapter=None, draft_model=None):
            return self.model, self.tokenizer

    response_generator = ResponseGenerator(
        IntegrationModelProvider(model, tokenizer), LRUPromptCache()
    )
    server_address = ("localhost", 0)
    httpd = http.server.HTTPServer(
        server_address,
        lambda *args, **kwargs: APIHandler(response_generator, *args, **kwargs),
    )
    port = httpd.server_port
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    try:
        yield f"http://localhost:{port}", model_id
    finally:
        httpd.shutdown()
        httpd.server_close()
        server_thread.join()
        response_generator.stop_and_join()


def test_load_model_qwen3(integration_model):
    model_id, model, tokenizer = integration_model
    assert model is not None
    assert tokenizer is not None
    assert model_id


def test_responses_generation_qwen3(integration_server):
    base_url, model_id = integration_server
    url = f"{base_url}/v1/responses"
    post_data = {
        "model": model_id,
        "input": "Give a short greeting.",
        "max_output_tokens": 32,
        "temperature": 0.0,
        "store": False,
    }

    response = requests.post(url, json=post_data, timeout=180)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["model"] == model_id
    assert body["status"] == "completed"
    assert body["output"]
