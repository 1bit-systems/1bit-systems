#!/usr/bin/env python3
# unified-router.py — Content-aware routing proxy for 1bit systems
#
# Routes requests between NPU (small model) and GPU (large model) backends
# based on content complexity analysis.
#
# Usage:
#   python3 unified-router.py --port 18181 --npu-backend http://127.0.0.1:18101 --gpu-backend http://127.0.0.1:18102

import json
import os
import sys
import time
import signal
import argparse
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

MAX_BODY_SIZE = 16 * 1024 * 1024  # 16 MB

# Default backend URLs (override with --npu-backend / --gpu-backend)
NPU_BACKEND = os.environ.get("NPU_BACKEND_URL", "http://127.0.0.1:18101")
GPU_BACKEND = os.environ.get("GPU_BACKEND_URL", "http://127.0.0.1:18102")

# Model names as known to the inference backends
SMALL_MODEL = os.environ.get("SMALL_MODEL", "zaya-small")
BIG_MODEL = os.environ.get("BIG_MODEL", "zaya-large")
ROUTER_NAME = os.environ.get("ROUTER_NAME", "unified-router")

# Keyword-based routing: if a message contains any of these keywords, route to GPU
GPU_KEYWORDS = [
    "explain", "summarize", "write", "create", "design", "review", "analyze",
    "code", "function", "class", "implement", "refactor", "debug", "architecture",
    "compare", "evaluate", "recommend", "translate",
]


def should_route_to_gpu(body: dict) -> bool:
    """Check if the request should be routed to the GPU (large) model."""
    # Check for explicit model routing
    model = body.get("model", "")
    if model == BIG_MODEL or model == "gpu":
        return True
    if model == SMALL_MODEL or model == "npu":
        return False

    # Check for content-based routing via messages
    messages = body.get("messages", [])
    if not messages:
        # Check prompt field too
        prompt = body.get("prompt", "")
        if prompt:
            messages = [{"content": prompt}]

    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        content_lower = content.lower()
        # Use word-boundary matching to avoid substring false positives
        words = set(content_lower.split())
        for kw in GPU_KEYWORDS:
            if kw in words:
                return True
            # Also check for the keyword as a word prefix (e.g. "explaining")
            for word in words:
                if word.startswith(kw):
                    return True
    return False


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server — handles concurrent requests."""
    daemon_threads = True
    # Limit pending connections to avoid resource exhaustion
    request_queue_size = 128


class RouterHandler(BaseHTTPRequestHandler):
    """HTTP request handler that routes between NPU and GPU backends."""

    # Backend URLs (set by main)
    backend_url = GPU_BACKEND
    npu_url = NPU_BACKEND
    gpu_url = GPU_BACKEND

    def log_message(self, format, *args):
        # Suppress default logging to stderr (use our own format)
        pass

    def _send_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "127.0.0.1")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _proxy_request(self, method: str, url: str = None, body: bytes = None,
                       headers: dict = None, timeout: int = 300):
        """Proxy a request to a backend and return (status, headers_dict, body_bytes)."""
        if url is None:
            url = self.backend_url
        target = url.rstrip("/") + self.path
        req = urllib.request.Request(target, data=body, method=method)
        if headers:
            for k, v in headers.items():
                if k.lower() not in ("host", "content-length"):
                    req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_headers = dict(resp.headers)
                resp_body = resp.read()
                return resp.status, resp_headers, resp_body
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()
        except urllib.error.URLError as e:
            return 502, {}, json.dumps({"error": f"Backend unreachable: {e.reason}"}).encode()

    def _proxy_request_with_body(self, method: str, body: bytes,
                                  url: str = None, timeout: int = 300):
        """Proxy a request with body to a backend."""
        return self._proxy_request(method, url, body, timeout=timeout)

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self._send_json(200, {"status": "ok", "router": ROUTER_NAME})
        elif self.path == "/v1/models":
            status, headers, data = self._proxy_request("GET")
            self.send_response(status)
            self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
            self.end_headers()
            self.wfile.write(data)
        else:
            # Exact path matching for safety
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", 0))
        if cl > MAX_BODY_SIZE:
            self.send_error(413, "Payload too large")
            return
        body_bytes = self.rfile.read(cl) if cl > 0 else b"{}"

        # EXACT path matching — no substring "in" operator (fixes path injection)
        if self.path == "/v1/chat/completions":
            try:
                body = json.loads(body_bytes)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[{time.strftime('%H:%M:%S')}] WARNING: invalid JSON in /chat/completions: {e}",
                      file=sys.stderr)
                body = {}
            self._handle_chat_completion(body)
        elif self.path == "/v1/completions":
            try:
                body = json.loads(body_bytes) if body_bytes else {}
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[{time.strftime('%H:%M:%S')}] WARNING: invalid JSON in /completions: {e}",
                      file=sys.stderr)
                self._send_json(400, {"error": f"Invalid JSON: {e}"})
                return
            model = body.get("model", "")
            if model == ROUTER_NAME or model == "auto":
                body["model"] = BIG_MODEL if should_route_to_gpu(body) else SMALL_MODEL
            elif model == "npu":
                body["model"] = SMALL_MODEL
            elif model == "gpu":
                body["model"] = BIG_MODEL
            body_bytes = json.dumps(body).encode()
            status, headers, data = self._proxy_request_with_body("POST", body_bytes)
            self.send_response(status)
            self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._send_json(404, {"error": "not found"})

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "127.0.0.1")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def _handle_chat_completion(self, body: dict):
        """Handle /v1/chat/completions with content-based routing."""
        model = body.get("model", "")
        stream = body.get("stream", False)

        # Determine target backend
        if model == SMALL_MODEL or model == "npu":
            target = self.npu_url
        elif model == BIG_MODEL or model == "gpu":
            target = self.gpu_url
        elif model == ROUTER_NAME or model == "auto" or not model:
            if should_route_to_gpu(body):
                target = self.gpu_url
                body["model"] = BIG_MODEL
            else:
                target = self.npu_url
                body["model"] = SMALL_MODEL
        else:
            # Pass-through: send to default backend with original model name
            target = self.backend_url

        body_bytes = json.dumps(body).encode()

        if stream:
            # Streaming: proxy the SSE stream
            target_url = target.rstrip("/") + "/v1/chat/completions"
            req = urllib.request.Request(target_url, data=body_bytes, method="POST")
            req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    self.send_response(resp.status)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "127.0.0.1")
                    self.end_headers()
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except urllib.error.URLError as e:
                self._send_json(502, {"error": f"Backend unreachable: {e.reason}"})
        else:
            status, headers, data = self._proxy_request_with_body("POST", body_bytes, url=target)
            self.send_response(status)
            self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
            self.send_header("Access-Control-Allow-Origin", "127.0.0.1")
            self.end_headers()
            self.wfile.write(data)


def main():
    global SMALL_MODEL, BIG_MODEL
    parser = argparse.ArgumentParser(description="1bit Unified Router — NPU+GPU content-aware proxy")
    parser.add_argument("--port", type=int, default=18180, help="Listen port (default: 18180)")
    parser.add_argument("--bind", type=str, default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1; use 0.0.0.0 to expose publicly)")
    parser.add_argument("--npu-backend", type=str, default=NPU_BACKEND,
                        help=f"NPU backend URL (default: {NPU_BACKEND})")
    parser.add_argument("--gpu-backend", type=str, default=GPU_BACKEND,
                        help=f"GPU backend URL (default: {GPU_BACKEND})")
    parser.add_argument("--small-model", type=str, default=SMALL_MODEL,
                        help=f"Small model name (default: {SMALL_MODEL})")
    parser.add_argument("--large-model", type=str, default=BIG_MODEL,
                        help=f"Large model name (default: {BIG_MODEL})")
    args = parser.parse_args()

    RouterHandler.backend_url = args.gpu_backend
    RouterHandler.npu_url = args.npu_backend
    RouterHandler.gpu_url = args.gpu_backend

    SMALL_MODEL = args.small_model
    BIG_MODEL = args.large_model

    if args.bind != "127.0.0.1":
        print("⚠️  WARNING: Binding to non-localhost address. Ensure firewall rules are in place.",
              file=sys.stderr)

    print("=" * 56)
    print("  Unified NPU+GPU Router")
    print("=" * 56)
    print(f"  Listen:  http://{args.bind}:{args.port}")
    print(f"  NPU:     {args.npu_backend} ({args.small_model})")
    print(f"  GPU:     {args.gpu_backend} ({args.large_model})")
    print(f"  Routing: keyword-based → GPU; default → NPU")
    print(f"    npu                  → {args.small_model} (NPU)")
    print(f"    gpu                  → {args.large_model} (GPU)")
    print(f"    <any other>          → pass-through to inference backend")
    print()

    server = ThreadingHTTPServer((args.bind, args.port), RouterHandler)

    def shutdown(sig, frame):
        print("\nShutting down...")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
