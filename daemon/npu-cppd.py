#!/usr/bin/env python3
"""
npu-cppd — C++ NPU Engine Daemon (FLM replacement)

Supports two backends:
  1. torch2aie:  INT8 NPU via Python fused xclbin runner (default, proven)
  2. ternary:    Native ternary 2-bit packed NPU via C++ npu_ternaryd (new!)

Usage:
  sudo ./daemon/npu-cppd.py [--port PORT] [--backend torch2aie|ternary]

Environment:
  NPU_XCLBIN_DIR       Override xclbin directory
  NPU_MODEL_PATH       Override model path
  NPU_TERNARY_MODEL    Override ternary model dir (default: model.ternary/)
  NPU_TERNARY_XCLBIN   Override ternary xclbin dir
"""

import argparse
import json
import os
import subprocess
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class Torch2AIEBackend:
    """INT8 NPU inference via torch2aie fused xclbin Python runner."""

    def __init__(self):
        self.ready = False
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = (
            "/home/bcloud/torch2aie/toolchain/mlir_aie/python:"
            "/home/bcloud/torch2aie/examples/qwen3-decode-layer:"
            "/home/bcloud/torch2aie/examples/qwen3-decode-layer/cases"
        )

    def start(self):
        result = subprocess.run(
            ["/home/bcloud/torch2aie/.venv/bin/python3", "-c",
             "import sys; sys.path.insert(0, '/home/bcloud/1bit-systems'); "
             "from tools.npu_runner import NPURunner; print('OK')"],
            capture_output=True, text=True, timeout=120, env=self.env,
            cwd="/home/bcloud/1bit-systems")
        if result.returncode == 0:
            self.ready = True
            print(f"  INT8 NPU runner ready", flush=True)
        else:
            print(f"  INT8 NPU runner failed: {result.stderr[:200]}", flush=True)

    def stop(self):
        self.ready = False

    def run(self, tokens: list, max_new: int) -> dict:
        """Returns {"tokens": [...], "finished": bool} or {"error": str}."""
        if not self.ready:
            return {"error": "NPU engine not ready"}

        req = json.dumps({"tokens": tokens, "max_new_tokens": min(max_new, 64)})
        proc = subprocess.run(
            ["/home/bcloud/torch2aie/.venv/bin/python3",
             "/home/bcloud/1bit-systems/tools/npu_runner.py"],
            input=req, capture_output=True, text=True, timeout=300,
            env=self.env, cwd="/home/bcloud/1bit-systems")

        if proc.returncode != 0:
            return {"error": f"Runner failed: {proc.stderr[:200]}"}
        try:
            return json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            return {"error": f"Bad JSON: {proc.stdout[:200]}"}


class NativeTernaryBackend:
    """Native ternary (2-bit packed) NPU inference via C++ npu_ternaryd."""

    def __init__(self, model_dir=None, xclbin_dir=None):
        self.model_dir = model_dir or os.environ.get(
            "NPU_TERNARY_MODEL", str(REPO_ROOT / "model.ternary"))
        self.xclbin_dir = xclbin_dir or os.environ.get(
            "NPU_TERNARY_XCLBIN",
            str(REPO_ROOT / "engine/npu/build/ternary_m52_k64"))
        self.ready = False
        self.proc = None

    def start(self):
        binary = str(REPO_ROOT / "engine/npu/build/npu_ternaryd")
        if not os.path.exists(binary):
            print(f"  npu_ternaryd not found at {binary}", flush=True)
            print(f"  Build: bash engine/npu/build/build_ternary_daemon.sh", flush=True)
            return

        # Quick check: can we import the model?
        manifest = os.path.join(self.model_dir, "manifest.json")
        if not os.path.exists(manifest):
            print(f"  Ternary model not found at {self.model_dir}/", flush=True)
            print(f"  Convert: python tools/q2_0_to_packed.py model.gguf model.ternary/", flush=True)
            return

        # Check xclbin dir
        if not os.path.isdir(self.xclbin_dir):
            print(f"  Xclbin dir not found: {self.xclbin_dir}", flush=True)
            return

        print(f"  Native ternary model:  {self.model_dir}", flush=True)
        print(f"  Native ternary xclbin: {self.xclbin_dir}", flush=True)
        self.ready = True

    def stop(self):
        self.ready = False

    def run(self, tokens: list, max_new: int) -> dict:
        """Returns {"tokens": [...], "finished": bool} or {"error": str}."""
        if not self.ready:
            return {"error": "Native ternary engine not ready"}

        binary = str(REPO_ROOT / "engine/npu/build/npu_ternaryd")
        req = json.dumps({"tokens": tokens, "max_new_tokens": min(max_new, 64)})

        try:
            proc = subprocess.run(
                [binary, self.model_dir, self.xclbin_dir],
                input=req, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return {"error": "Timeout"}

        if proc.returncode != 0:
            return {"error": f"npu_ternaryd failed: {proc.stderr[:200]}"}
        try:
            return json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            return {"error": f"Bad JSON: {proc.stdout[:200]}"}


# ── Tokenizer helpers (shared) ──────────────────────────

TOKENIZER_PATH = str(REPO_ROOT / ".config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json")


def tokenize(text: str) -> list:
    r = subprocess.run(
        [str(REPO_ROOT / "engine/npu/tokenizer/tokenize"), TOKENIZER_PATH],
        input=text.encode(), capture_output=True, timeout=10)
    return [int(t) for t in r.stdout.decode().strip().split(",") if t.strip()]


def detokenize(tokens: list) -> str:
    inp = " ".join(str(t) for t in tokens)
    r = subprocess.run(
        [str(REPO_ROOT / "engine/npu/tokenizer/detokenize"), TOKENIZER_PATH],
        input=inp.encode(), capture_output=True, timeout=10)
    return r.stdout.decode().strip()


# ── HTTP API ────────────────────────────────────────────

class RequestHandler(BaseHTTPRequestHandler):
    backend = None   # set by main
    model_name = "qwen3-0.6b-npu"

    def log_message(self, format, *args):
        print(f"  {self.address_string()} — {format % args}", flush=True)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self.handle_chat()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            ready = self.backend.ready if self.backend else False
            self.wfile.write(json.dumps({"status": "ok" if ready else "not ready"}).encode())
        else:
            self.send_error(404)

    def handle_chat(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages", [])
        model = body.get("model", self.model_name)
        max_tokens = body.get("max_tokens", 64)

        # Build prompt
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        full_prompt = "".join(parts)

        prompt_tokens = tokenize(full_prompt)
        if not prompt_tokens:
            self.send_json({"error": "Empty prompt"}, 400)
            return

        # Run backend
        result = self.backend.run(prompt_tokens, max_tokens)
        if "error" in result:
            self.send_json({"error": result["error"]}, 502)
            return

        out_tokens = result.get("tokens", [])
        out_text = detokenize(out_tokens)

        response = {
            "id": f"chatcmpl-npu-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": out_text},
                "finish_reason": "stop" if result.get("finished") else "length",
            }],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(out_tokens),
                "total_tokens": len(prompt_tokens) + len(out_tokens),
            },
        }
        self.send_json(response)

    def send_json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="NPU C++ Engine Daemon")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend", choices=["torch2aie", "ternary"], default="torch2aie")
    parser.add_argument("--model", default="qwen3-0.6b-npu")
    args = parser.parse_args()

    print(f"NPU Daemon starting on port {args.port} (backend={args.backend})", flush=True)

    if args.backend == "ternary":
        backend = NativeTernaryBackend()
    else:
        backend = Torch2AIEBackend()

    backend.start()
    RequestHandler.backend = backend
    RequestHandler.model_name = args.model

    server = HTTPServer(("0.0.0.0", args.port), RequestHandler)
    print(f"  Listening on :{args.port}", flush=True)
    print(f"  POST /v1/chat/completions", flush=True)
    print(f"  GET  /health", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...", flush=True)
    finally:
        backend.stop()
        server.server_close()


if __name__ == "__main__":
    main()
