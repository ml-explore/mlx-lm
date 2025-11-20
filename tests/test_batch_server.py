import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json
import queue
import io
import threading
import time

from mlx_lm import batch_server

class TestBatchServer(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        # Mock detokenizer behavior
        self.mock_tokenizer.detokenizer.last_segment = "test"
        self.mock_tokenizer.detokenizer.reset = MagicMock()
        self.mock_tokenizer.detokenizer.add_token = MagicMock()
        self.mock_tokenizer.detokenizer.finalize = MagicMock()
        self.mock_tokenizer.eos_token_ids = {2}

    def test_worker_processing(self):
        model_path = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        
        worker = batch_server.ModelWorker(
            model_path=model_path,
            max_batch_size=1,
            temp=0.0
        )
        
        worker.start()
        
        try:
            q = worker.submit("Hello", max_tokens=10)
            
            full_text = ""
            finish_reason = None
            
            # Wait for results. First result might take longer due to model loading.
            while True:
                try:
                    result = q.get(timeout=120.0) # Allow time for model download/load
                    full_text += result.text_segment
                    if result.finish_reason:
                        finish_reason = result.finish_reason
                        break
                except queue.Empty:
                    self.fail("Timed out waiting for generation result")
            
            self.assertTrue(len(full_text) > 0)
            self.assertIsNotNone(finish_reason)
            
        finally:
            worker.running = False
            worker.join()

    def test_api_handler_chat_completions(self):
        worker = MagicMock()
        worker.default_max_tokens = 100
        worker.model_path = "test_model"
        
        mock_queue = queue.Queue()
        worker.submit.return_value = mock_queue
        
        mock_result = batch_server.GenerationResult(
            text_segment="Hello",
            finish_reason="stop",
            prompt_tokens=2,
            completion_tokens=1
        )
        mock_queue.put(mock_result)
        
        request_body = json.dumps({
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50
        }).encode('utf-8')
        
        mock_socket = MagicMock()
        
        response_buffer = io.BytesIO()
        
        def side_effect(mode, *args, **kwargs):
            if 'w' in mode:
                return response_buffer
            # Return a new BytesIO for each read to simulate a stream that doesn't close immediately
            # But makefile returns a file object.
            return io.BytesIO(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Content-Length: " + str(len(request_body)).encode('utf-8') + b"\r\n"
                b"\r\n" + 
                request_body
            )
            
        mock_socket.makefile.side_effect = side_effect
        
        # Suppress logging during test
        with patch('sys.stderr', new=io.StringIO()):
            class MockSocket:
                def makefile(self, mode, *args, **kwargs):
                    if 'w' in mode:
                        return response_buffer
                    return io.BytesIO(
                        b"POST /v1/chat/completions HTTP/1.1\r\n"
                        b"Content-Length: " + str(len(request_body)).encode('utf-8') + b"\r\n"
                        b"\r\n" + 
                        request_body
                    )
                
                def sendall(self, data):
                    response_buffer.write(data)

            # Instantiating APIHandler triggers handle() -> handle_one_request()
            handler = batch_server.APIHandler(worker, MockSocket(), ('127.0.0.1', 12345), MagicMock())
        
        response = response_buffer.getvalue().decode('utf-8')
        self.assertIn('"content": "Hello"', response)
        self.assertIn('"finish_reason": "stop"', response)

    def test_api_handler_models(self):
        worker = MagicMock()
        worker.model_path = "test_model"
        
        response_buffer = io.BytesIO()
        
        with patch('sys.stderr', new=io.StringIO()):
            class MockSocket:
                def makefile(self, mode, *args, **kwargs):
                    if 'w' in mode:
                        return response_buffer
                    return io.BytesIO(b"GET /v1/models HTTP/1.1\r\n\r\n")
                
                def sendall(self, data):
                    response_buffer.write(data)

            handler = batch_server.APIHandler(worker, MockSocket(), ('127.0.0.1', 12345), MagicMock())
            
        response = response_buffer.getvalue().decode('utf-8')
        self.assertIn('"id": "test_model"', response)

if __name__ == '__main__':
    unittest.main()
