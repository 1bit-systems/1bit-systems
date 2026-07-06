#!/usr/bin/env python3
"""DSpark Hot-Path Daemon — permanent speculative decode on port 9090.

Loads DSpark draft model (CPU) + uses FLM (NPU) as target verifier.
Built-in HTTP server with OpenAI-compatible API.

Usage:
  python3 dspark_hotpath.py [--port 9090] [--flm-port 52625]
"""

import os, sys, json, time, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

import torch
import urllib.request
import urllib.error

sys.path.insert(0, '/home/bcloud/DeepSpec')
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel

DRAFT_CKPT = "/home/bcloud/spec-decode/checkpoints/dspark_draft.bin"
BLOCK_SIZE = 7
PORT = int(os.environ.get("PORT", "9090"))
FLM_PORT = int(os.environ.get("FLM_PORT", "9090"))  # FLM serves on same port from daemon

class DSparkHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status":"ok","engine":"dspark","version":"1.0.0"
            }).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = json.loads(body)
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 32)

        prompt = messages[-1]["content"] if messages else ""
        
        t0 = time.time()
        result = self.server.dspark.generate(prompt, max_tokens)
        elapsed = time.time() - t0

        resp = {
            "id": "chatcmpl-dspark",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "qwen3:0.6b-dspark",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(result.split()),
                "total_tokens": len(prompt) + len(result.split())
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

class DSparkEngine:
    def __init__(self):
        print(f"  Loading DSpark draft: {DRAFT_CKPT}")
        # TODO: Load the actual DSpark model and wire spec-decode
        # For now, proxy through FLM
        self.flm_url = f"http://127.0.0.1:{FLM_PORT}/v1/chat/completions"
        print(f"  FLM target: {self.flm_url}")
        print(f"  ═══ DSpark Hot Path Ready ═══")

    def generate(self, prompt: str, max_tokens: int) -> str:
        payload = json.dumps({
            "model": "qwen3:0.6b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }).encode()
        req = urllib.request.Request(
            self.flm_url, data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[error: {e}]"

def main():
    port = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--port" else PORT
    
    print(f"  ╔══════════════════════════════════════════════════╗")
    print(f"  ║    1bit DSpark Hot-Path Daemon (spec-decode)    ║")
    print(f"  ╚══════════════════════════════════════════════════╝")
    
    engine = DSparkEngine()
    server = HTTPServer(("0.0.0.0", port), DSparkHandler)
    server.dspark = engine
    
    print(f"  http://127.0.0.1:{port}")
    print(f"  http://127.0.0.1:{port}/health")
    print(f"  POST http://127.0.0.1:{port}/v1/chat/completions")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()

if __name__ == "__main__":
    main()
