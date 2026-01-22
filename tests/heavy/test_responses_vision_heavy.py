# Copyright © 2026 Apple Inc.

import http
import os
import threading

import pytest
import requests

from mlx_lm.server import APIHandler, LRUPromptCache, ResponseGenerator

RUN_HEAVY = os.getenv("RUN_HEAVY_TESTS") == "1"
VISION_MODEL_ENV = "HEAVY_VISUAL_MODEL"
DEFAULT_VISION_MODEL = "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-8bit"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not RUN_HEAVY,
        reason="Set RUN_HEAVY_TESTS=1 to run heavy vision tests.",
    ),
]


def _vision_deps_available() -> bool:
    try:
        import mlx_vlm  # noqa: F401
    except Exception:
        return False
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        return False
    return True


@pytest.fixture(scope="module")
def vision_model_id():
    if not _vision_deps_available():
        pytest.skip("mlx-vlm + Pillow are required for heavy vision tests.")
    return os.getenv(VISION_MODEL_ENV, DEFAULT_VISION_MODEL)


@pytest.fixture(scope="module")
def integration_server(vision_model_id):
    class IntegrationModelProvider:
        def __init__(self):
            self.model = None
            self.tokenizer = None
            self.model_key = (None, None)
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

    response_generator = ResponseGenerator(IntegrationModelProvider(), LRUPromptCache())
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
        yield f"http://localhost:{port}", vision_model_id
    finally:
        httpd.shutdown()
        httpd.server_close()
        server_thread.join()
        response_generator.stop_and_join()


def test_responses_generation_qwen3_vl_vision(integration_server):
    base_url, model_id = integration_server
    url = f"{base_url}/v1/responses"
    post_data = {
        "model": model_id,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What animal is this?"},
                    {
                        "type": "input_image",
                        "image_url": "https://picsum.photos/id/237/200/300",
                    },
                ],
            }
        ],
        "max_output_tokens": 96,
        "temperature": 0.0,
        "store": False,
    }

    response = requests.post(url, json=post_data, timeout=300)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["model"] == model_id
    assert body["status"] == "completed"
    assert body["output"]
