#!/usr/bin/env python3
"""
Unified NPU+GPU Router — proxies requests to the NPU or GPU inference backend.

Routes requests to NPU (qwen3-0.6b-FLM) or GPU (MLX/Qwen3-8B) based on policy
(see should_route_to_gpu). Sits in front of the upstream server so apps see one
endpoint (fixes #70).
"""

import argparse
import json
import os
import signal
import sys
import time
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

# ── Routing policy ──────────────────────────────────────────────────

SMALL_MODEL = "qwen3-0.6b-FLM"           # NPU (210ms/tok, efficient)
BIG_MODEL   = "mlx-engine.mlx-community/Qwen3-8B-4bit"  # GPU (fast compute)
ROUTER_NAME = "user.Unified"

# Keywords that trigger GPU routing
GPU_KEYWORDS = [
    "code", "explain", "analyze", "write", "implement", "debug",
    "refactor", "function", "algorithm", "bug", "error", "review",
    "optimize", "design", "architecture", "test",
]

def should_route_to_gpu(body: dict) -> bool:
    """Decide NPU vs GPU based on request content."""
    messages = body.get("messages", [])
    max_chars = body.get("max_tokens", 0)

    # Check user message content for GPU keywords
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            # Long context → GPU
            if len(content) > 800:
                return True
            # GPU keywords
            for kw in GPU_KEYWORDS:
                if kw in content.lower():
                    return True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    if len(text) > 800:
                        return True
                    for kw in GPU_KEYWORDS:
                        if kw in text.lower():
                            return True

    # Tool calls → GPU (needs more capability)
    if body.get("tools"):
        return True

    return False

# ── HTTP handler ────────────────────────────────────────────────────

class RouterHandler(BaseHTTPRequestHandler):
    backend_url = "http://127.0.0.1:13305"

    def _proxy_request(self, method: str):
        """Forward request to the inference backend."""
        path = self.path
        url = f"{self.backend_url}{path}"
        body_bytes = None

        if method == "POST":
            cl = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(cl) if cl > 0 else b"{}"

        req = urllib.request.Request(
            url, data=body_bytes, method=method,
            headers={"Content-Type": "application/json"},
        )

        try:
            resp = urllib.request.urlopen(req, timeout=600)
            return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()
        except Exception as e:
            return 502, {}, json.dumps({"error": str(e)}).encode()

    def _handle_chat_completion(self, body: dict):
        """Route chat completion to NPU or GPU."""
        model = body.get("model", "")

        if model == ROUTER_NAME or model == "auto":
            # Auto-route
            target = BIG_MODEL if should_route_to_gpu(body) else SMALL_MODEL
            body["model"] = target
        elif model == "npu":
            body["model"] = SMALL_MODEL
        elif model == "gpu":
            body["model"] = BIG_MODEL

        body_bytes = json.dumps(body).encode()
        url = f"{self.backend_url}/v1/chat/completions"
        req = urllib.request.Request(
            url, data=body_bytes, method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            resp = urllib.request.urlopen(req, timeout=600)
            data = resp.read()
            self.send_response(resp.status)
            ct = resp.headers.get("Content-Type", "application/json")
            self.send_header("Content-Type", ct)
            self.send_header("X-Route", body["model"])
            self.end_headers()
            self.wfile.write(data)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_GET(self):
        status, headers, data = self._proxy_request("GET")
        self.send_response(status)
        self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(cl) if cl > 0 else b"{}"

        if "/chat/completions" in self.path:
            try:
                body = json.loads(body_bytes)
            except json.JSONDecodeError:
                body = {}
            self._handle_chat_completion(body)
        elif "/completions" in self.path:
            body = json.loads(body_bytes) if body_bytes else {}
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
            status, headers, data = self._proxy_request("POST")
            self.send_response(status)
            self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
            self.end_headers()
            self.wfile.write(data)

    def _proxy_request_with_body(self, method: str, body_bytes: bytes):
        url = f"{self.backend_url}{self.path}"
        req = urllib.request.Request(
            url, data=body_bytes, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
            return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()
        except Exception as e:
            return 502, {}, json.dumps({"error": str(e)}).encode()

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]} {args[1]} -> {args[2]}")

# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified NPU+GPU Router")
    parser.add_argument("--port", type=int, default=13305, help="Router listen port")
    parser.add_argument("--backend", type=str, default="http://127.0.0.1:13305",
                        help="inference backend backend URL")
    args = parser.parse_args()

    RouterHandler.backend_url = args.backend.rstrip("/")

    print("=" * 56)
    print("  Unified NPU+GPU Router")
    print(f"  Listen:  http://0.0.0.0:{args.port}")
    print(f"  Backend: {RouterHandler.backend_url}")
    print("=" * 56)
    print()
    print("  Model routing:")
    print(f"    user.Unified / auto  → auto-route NPU ↔ GPU")
    print(f"    npu                  → {SMALL_MODEL} (NPU)")
    print(f"    gpu                  → {BIG_MODEL} (GPU)")
    print(f"    <any other>          → pass-through to inference backend")
    print()

    server = HTTPServer(("0.0.0.0", args.port), RouterHandler)

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
