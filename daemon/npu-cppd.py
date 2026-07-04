#!/usr/bin/env python3
"""
npu-cppd — C++ NPU Engine Daemon (FLM replacement)

Starts the C++ NPU inference engine as a subprocess and
provides an OpenAI-compatible HTTP API for chat completions.

Usage:
  sudo ./daemon/npu-cppd.py [--port PORT] [--model MODEL]

Environment:
  NPU_XCLBIN_DIR    Override xclbin directory
  NPU_MODEL_PATH    Override model path
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
ENGINE_BIN = REPO_ROOT / "engine" / "npu" / "build" / "npu_engine_server"
TOKENIZER_BIN = REPO_ROOT / "engine" / "npu" / "tokenizer" / "tokenize"
DETOKENIZER_BIN = REPO_ROOT / "engine" / "npu" / "tokenizer" / "detokenize"
MODEL_DIR = os.environ.get("NPU_MODEL_DIR", os.path.expanduser("~/.config/flm/models/Qwen3-0.6B-NPU2"))
TOKENIZER_JSON = os.path.join(MODEL_DIR, "tokenizer.json")
MODEL_PATH = os.environ.get("NPU_MODEL_PATH", os.path.join(MODEL_DIR, "model.q4nx"))


class CppEngineBackend:
    """C++ NPU inference engine via stdin/stdout JSON protocol."""

    def __init__(self, port: int):
        self.port = port
        self.process = None
        self.ready = False

    def start(self):
        env = os.environ.copy()
        print(f"  Starting C++ NPU engine: {ENGINE_BIN}", flush=True)
        self.process = subprocess.Popen(
            [str(ENGINE_BIN)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        deadline = time.time() + 60
        while time.time() < deadline:
            line = self.process.stderr.readline()
            if not line:
                break
            print(f"  [engine] {line.strip()}", flush=True)
            if "Ready" in line:
                self.ready = True
                break
        if self.ready:
            print(f"  C++ NPU engine ready (pid={self.process.pid})", flush=True)
        else:
            print(f"  Engine failed to start within 60s", flush=True)
            self.stop()

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        self.ready = False

    def tokenize(self, text: str) -> list:
        result = subprocess.run([str(TOKENIZER_BIN), TOKENIZER_JSON],
            input=text.encode(), capture_output=True, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(f"Tokenizer failed")
        line = result.stdout.decode().strip()
        return [int(t) for t in line.split(",") if t.strip()]

    def detokenize(self, tokens: list) -> str:
        inp = " ".join(str(t) for t in tokens)
        result = subprocess.run([str(DETOKENIZER_BIN), TOKENIZER_JSON],
            input=inp.encode(), capture_output=True, timeout=10)
        return result.stdout.decode().strip()

    def chat(self, model: str, messages: list, **kwargs) -> dict:
        if not self.ready:
            return {"error": "NPU engine not ready"}
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m["content"]
                break
        prompt_tokens = self.tokenize(last_user_msg)
        if not prompt_tokens:
            return {"error": "Empty prompt"}
        max_new_tokens = kwargs.get("max_tokens", 256)
        req = json.dumps({"tokens": prompt_tokens, "max_new_tokens": max_new_tokens})
        self.process.stdin.write(req + "\n")
        self.process.stdin.flush()
        resp_line = self.process.stdout.readline()
        if not resp_line:
            return {"error": "Engine closed connection"}
        try:
            resp = json.loads(resp_line)
        except json.JSONDecodeError:
            return {"error": f"Bad JSON: {resp_line[:200]}"}
        if resp.get("error"):
            return {"error": resp["error"]}
        out_tokens = resp.get("tokens", [])
        out_text = self.detokenize(out_tokens)
        return {
            "id": "chatcmpl-npu-cpp",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": out_text},
                "finish_reason": "stop" if resp.get("finished") else "length",
            }],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(out_tokens),
                "total_tokens": len(prompt_tokens) + len(out_tokens),
            },
        }


def make_handler(backend):
    class Handler(BaseHTTPRequestHandler):
        def _respond(self, status, data):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            if self.path == "/v1/models":
                self._respond(200, {
                    "object": "list",
                    "data": [{"id": "Qwen3-0.6B-NPU2", "object": "model", "owned_by": "npu-cpp"}]
                })
            elif self.path == "/health":
                self._respond(200, {"status": "ok", "ready": backend.ready})
            else:
                self._respond(404, {"error": "Not found"})

        def do_POST(self):
            if self.path == "/v1/chat/completions":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode()
                try:
                    req = json.loads(body)
                except json.JSONDecodeError:
                    self._respond(400, {"error": "Invalid JSON"})
                    return
                model = req.get("model", "Qwen3-0.6B-NPU2")
                messages = req.get("messages", [])
                kwargs = {k: req[k] for k in ("max_tokens", "temperature", "top_p") if k in req}
                result = backend.chat(model, messages, **kwargs)
                self._respond(200, result)
            else:
                self._respond(404, {"error": "Not found"})

        def log_message(self, format, *args):
            print(f"  [http] {args[0]} {args[1]} {args[2]}", flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="C++ NPU Engine Daemon")
    parser.add_argument("--port", type=int, default=9090, help="HTTP API port")
    args = parser.parse_args()
    print(f"=== 1bit C++ NPU Daemon ===", flush=True)
    print(f"  Engine: {ENGINE_BIN}", flush=True)
    print(f"  Model: {MODEL_PATH}", flush=True)
    print(f"  Port: {args.port}", flush=True)
    backend = CppEngineBackend(args.port)
    backend.start()
    if not backend.ready:
        print(f"  Engine failed to start, exiting", flush=True)
        sys.exit(1)
    server = HTTPServer(("0.0.0.0", args.port), make_handler(backend))
    print(f"  HTTP server on :{args.port}", flush=True)
    print(f"  Ready.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n  Shutting down...", flush=True)
        backend.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
