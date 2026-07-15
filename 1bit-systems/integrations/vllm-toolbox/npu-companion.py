#!/usr/bin/env python3
"""
NPU Companion — XDNA 2 speculative decoding draft model for vLLM.

Runs alongside the vLLM server (kyuz0/amd-strix-halo-vllm-toolboxes) 
as a sidecar, providing an OpenAI-compatible /generate endpoint that 
vLLM's speculative decoding can consume for NPU-accelerated inference.

Uses the native 1bit-systems NPU engine or FLM proxy to run a small 
draft model (Qwen3-0.6B at 69-94 tok/s), proposing tokens that the 
GPU verifies in batch against the target model.

Requires: amdxdna kernel module loaded, /dev/accel/accel0 accessible.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

DEFAULT_MODEL = "qwen3:0.6b"
DEFAULT_PORT = 18089
NPU_ENGINE_BIN = os.path.join(os.path.dirname(__file__), 
                               "../../engine/npu/build/npu_engine_qwen3_0_6b_32v5")


class NPUEngine:
    """Wrapper around the 1bit-systems NPU engine."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.flm_port = 52625
        self.flm_url = f"http://127.0.0.1:{self.flm_port}"
        self.flm_proc: Optional[subprocess.Popen] = None
        self._find_engine()
        if self.engine_type == "flm":
            self._start_flm_server()

    def _find_engine(self):
        """Find the best available NPU engine."""
        # Prefer native 1bit-systems NPU binary
        candidates = [
            NPU_ENGINE_BIN,
            os.path.expanduser("~/1bit-systems/engine/npu/build/npu_engine_qwen3_0_6b_32v5"),
        ]
        for path in candidates:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                self.engine_path = path
                self.engine_type = "native"
                return

        # Fall back to FLM proxy
        if subprocess.run(["which", "flm"], capture_output=True).returncode == 0:
            self.engine_type = "flm"
            return

        raise RuntimeError("No NPU engine found. Build native engine or install FLM.")

    def generate(self, prompt: str, max_tokens: int = 5, temperature: float = 0.0) -> str:
        if self.engine_type == "flm":
            return self._generate_flm(prompt, max_tokens, temperature)
        else:
            return self._generate_native(prompt, max_tokens, temperature)

    def _generate_native(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            result = subprocess.run(
                [self.engine_path, "--prompt", prompt, "--max-tokens", str(max_tokens)],
                capture_output=True, text=True, timeout=10,
                env={**os.environ, "OMP_NUM_THREADS": "16",
                     "OMP_WAIT_POLICY": "active", "OMP_PROC_BIND": "close",
                     "OMP_PLACES": "cores"}
            )
            return result.stdout.strip()
        except Exception as e:
            print(f"[npu-companion] Native engine error: {e}", file=sys.stderr)
            return ""

    def _generate_flm(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            req = urllib.request.Request(
                f"{self.flm_url}/v1/chat/completions",
                data=json.dumps({
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[npu-companion] FLM error: {e}", file=sys.stderr)
            return ""

    def _start_flm_server(self):
        try:
            urllib.request.urlopen(f"{self.flm_url}/v1/models", timeout=1)
            return
        except Exception:
            pass
        self.flm_proc = subprocess.Popen(
            ["flm", "serve", self.model, "--pmode", "turbo"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        for _ in range(30):
            try:
                urllib.request.urlopen(f"{self.flm_url}/v1/models", timeout=1)
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("FLM server failed to start")

    def shutdown(self):
        if self.flm_proc:
            self.flm_proc.terminate()
            self.flm_proc = None


class NPUHandler(BaseHTTPRequestHandler):
    engine: NPUEngine = None

    def log_message(self, format, *args):
        print(f"[npu-companion] {args[0]}", file=sys.stderr)

    def do_POST(self):
        if self.path not in ("/generate", "/v1/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 5)
        temperature = body.get("temperature", 0.0)

        start = time.time()
        text = self.engine.generate(prompt, max_tokens, temperature)
        elapsed_ms = int((time.time() - start) * 1000)
        tokens = len(text.split()) if text else 0

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "text": text, "tokens": tokens, "ms": elapsed_ms,
            "backend": "NPU XDNA 2",
        }).encode())

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok", "backend": "NPU XDNA 2",
                "model": NPUHandler.engine.model,
                "device": "/dev/accel/accel0",
            }).encode())
        else:
            self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description="NPU Companion for vLLM speculative decoding")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not os.path.exists("/dev/accel/accel0"):
        print("[npu-companion] ERROR: /dev/accel/accel0 not found. Is amdxdna loaded?", file=sys.stderr)
        sys.exit(1)

    engine = NPUEngine(model=args.model)
    NPUHandler.engine = engine
    server = HTTPServer(("127.0.0.1", args.port), NPUHandler)
    print(f"[npu-companion] NPU draft model ready — {engine.engine_type} engine")
    print(f"[npu-companion] Listening on http://127.0.0.1:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[npu-companion] Shutting down.")
        engine.shutdown()


if __name__ == "__main__":
    main()
