import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GOOD_ENTITIES = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes.",
}
GOOD_TEXT = json.dumps(GOOD_ENTITIES, ensure_ascii=False)
MISSING_FIELD_TEXT = json.dumps({"tecnologias": ["FastAPI"], "nivel_de_criticidad": "alta"})
EMPTY_LIST_TEXT = json.dumps({**GOOD_ENTITIES, "tecnologias": []})
MALFORMED_TEXT = '{"tecnologias": ["FastAPI", "Redis"'


def http_failure(status):
    return {"status": status}


def completion(text=GOOD_TEXT, truncated=False):
    return {"text": text, "truncated": truncated}


class FakeAPI:
    def __init__(self):
        self.requests: list[dict] = []
        self.script: list[dict] = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _make_handler(self):
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body: bytes):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                api.requests.append({"path": self.path, "body": body})
                step = api.script.pop(0) if api.script else completion()
                if "status" in step:
                    return self._fail(step["status"])
                if self.path.endswith("/chat/completions"):
                    return self._openai(step)
                if self.path.endswith("/v1/messages"):
                    return self._anthropic(body, step)
                self._send(404, b"{}")

            def _fail(self, status):
                error = {"error": {"message": f"fallo {status}", "type": "x"}}
                if self.path.endswith("/v1/messages"):
                    error = {"type": "error", "error": {"type": "x", "message": f"fallo {status}"}}
                self._send(status, json.dumps(error).encode())

            def _openai(self, step):
                response = {
                    "id": "c1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-test",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "length" if step["truncated"] else "stop",
                            "message": {"role": "assistant", "content": step["text"]},
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                }
                self._send(200, json.dumps(response).encode())

            def _anthropic(self, body, step):
                response = {
                    "id": "m1",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [{"type": "text", "text": step["text"]}],
                    "stop_reason": "max_tokens" if step["truncated"] else "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 7, "output_tokens": 2},
                }
                self._send(200, json.dumps(response).encode())

        return Handler
