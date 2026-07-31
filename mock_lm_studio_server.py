"""
AXUM ROVER - Mock LM Studio server.

WHAT: A minimal stand-in for a real LM Studio instance, implementing just
enough of the OpenAI-compatible /v1/chat/completions endpoint that
_try_gpu_tier_model() (or any code calling LM_STUDIO_BASE_URL) can
actually be executed and observed, for the first time, without needing a
real LM Studio install or GPU.

WHY this exists: per Systems Integration Engineer's readiness report, the
restoration LLM hook has been written and compiles cleanly, but has NEVER
been run — not once, on any hardware — because nothing in any test
environment has a live LM Studio server to talk to. A clean compile
doesn't rule out a wrong endpoint path, a wrong request body shape, or an
exception path that's simply never fired. This lets that code actually
execute tonight and surface those problems now instead of live tomorrow.

WHAT THIS DOES NOT TEST: model quality, real inference behavior, GPU
offload, or anything about whether the actual restoration output is good.
It only proves the HTTP contract — does the calling code send a request
LM Studio would accept, and does it correctly parse a response in the
shape LM Studio would send back. That's a real, meaningful gap to close,
but it's not the whole picture.

USAGE:
    python mock_lm_studio_server.py                  # normal mode, port 1234
    python mock_lm_studio_server.py --fail-mode       # returns HTTP 500 for every request,
                                                       # to exercise the caller's error-handling path
    python mock_lm_studio_server.py --fail-mode timeout   # sleeps past any reasonable timeout instead
    python mock_lm_studio_server.py --port 1234
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def make_handler(fail_mode: str | None):
    class MockLMStudioHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Default logging is noisy for a quick test session — keep it minimal.
            print(f"[mock-lm-studio] {self.command} {self.path}")

        def do_GET(self):
            if self.path == "/v1/models":
                self._send_json(200, {
                    "data": [{"id": "mock-restoration-model", "object": "model"}]
                })
            else:
                self._send_json(404, {"error": "not found in mock server"})

        def do_POST(self):
            if not self.path.endswith("/chat/completions"):
                self._send_json(404, {"error": f"mock server only implements /chat/completions, got {self.path}"})
                return

            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length) if length else b""
            try:
                request_body = json.loads(raw_body) if raw_body else {}
            except json.JSONDecodeError:
                request_body = None

            print(f"[mock-lm-studio] request body: {request_body}")

            if fail_mode == "timeout":
                # Sleep well past any reasonable client timeout (config.py's
                # LM_STUDIO_TIMEOUT_SEC is 30.0) to exercise the caller's
                # timeout-handling path specifically, not just a 500 error.
                time.sleep(60)
                return

            if fail_mode == "error":
                self._send_json(500, {"error": {"message": "mock server: simulated failure", "type": "mock_error"}})
                return

            if request_body is None:
                self._send_json(400, {"error": {"message": "mock server: could not parse request body as JSON"}})
                return

            # Real OpenAI-compatible chat completion response shape.
            self._send_json(200, {
                "id": "mock-completion-001",
                "object": "chat.completion",
                "model": request_body.get("model", "mock-restoration-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "[MOCK RESTORATION OUTPUT] This is a canned response from the mock "
                                       "LM Studio server, not a real model. If your calling code parses this "
                                       "correctly, the HTTP contract between it and a real LM Studio instance "
                                       "is sound.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            })

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return MockLMStudioHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock LM Studio server for testing the restoration LLM hook")
    parser.add_argument("--port", type=int, default=1234, help="matches LM_STUDIO_BASE_URL's default port")
    parser.add_argument("--fail-mode", nargs="?", const="error", default=None,
                        choices=["error", "timeout"],
                        help="'error' returns HTTP 500 for every request; 'timeout' hangs past any reasonable client timeout")
    args = parser.parse_args()

    handler = make_handler(args.fail_mode)
    server = HTTPServer(("127.0.0.1", args.port), handler)

    mode_desc = f"FAIL MODE: {args.fail_mode}" if args.fail_mode else "normal mode"
    print(f"Mock LM Studio server running on http://127.0.0.1:{args.port} ({mode_desc})")
    print("Point LM_STUDIO_BASE_URL at this address and run the real restoration hook against it.")
    print("Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()